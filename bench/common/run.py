"""Run ONE solver over a directory of SAT/PB instances -> per-instance CSV.

Solver-agnostic: the command is a template with a `{f}` placeholder for the
(decompressed) instance path, e.g.

    qayd-sat : ../../target/release/qayd-sat -t {t} {f}
    cadical  : cadical {f}
    qayd-pb  : ../../target/release/qayd-pb -t {t} {f}
    sat4j-pb : java -cp solvers/s4j-pb.jar:solvers/s4j-core.jar \
                    org.sat4j.pb.LanceurPseudo2007 Default {t} {f}

Instances may be xz/bz2/gz compressed or bare; we transparently decompress each
to a scratch file before the run (uniform across solvers that can't read
compressed input). The external wall limit is always enforced on the complete
process group. Commands using the `{t}` placeholder may additionally receive a
short, explicitly recorded finalization window after their internal search
deadline so a completed assignment is not truncated while being printed.

Output CSV columns include status, objective, timing, validation, hashes, and
optional raw log paths (one row per instance).
`status` is normalized to SAT / UNSAT / OPTIMUM / UNKNOWN / ERROR.
"""
import argparse
import bz2
import csv
import gzip
import hashlib
import json
import lzma
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from .competitive import machine_provenance, sha256_file
    from .satpb_verify import verify_solution
except ImportError:
    from competitive import machine_provenance, sha256_file
    from satpb_verify import verify_solution

# .xz (fd 37 7a) and raw-.lzma (5d 00 00) both go through the lzma module
# (lzma.open auto-detects the container); bz2 and gzip by their own magic.
MAGIC = {b"\xfd7zXZ": lzma, b"\x5d\x00\x00": lzma, b"BZh": bz2, b"\x1f\x8b": gzip}
ACTIVE_PROCESSES = set()
ACTIVE_PROCESSES_LOCK = threading.Lock()


def register_process(process):
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES.add(process)


def unregister_process(process):
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES.discard(process)


def kill_active_processes():
    with ACTIVE_PROCESSES_LOCK:
        processes = list(ACTIVE_PROCESSES)
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def opener(path):
    """Return a decompressing open() for path based on magic bytes, else plain."""
    with open(path, "rb") as f:
        head = f.read(6)
    for magic, mod in MAGIC.items():
        if head.startswith(magic):
            return lambda: mod.open(path, "rb")
    return lambda: open(path, "rb")


def materialize(path, scratch, tag):
    """Decompress/copy instance into scratch dir, return the plain file path.

    `tag` keeps the scratch name unique so parallel workers don't clobber
    each other's instance file.

    For CNF, strip any SATLIB-style trailer (a line that is just `%`, then `0`)
    -- strict parsers like CaDiCaL reject it, competition parsers ignore it.
    """
    base = os.path.basename(path)
    is_cnf = ".cnf" in base
    if is_cnf:
        ext = ".cnf"
    elif ".xml" in base or ".lzma" in base:
        ext = ".xml"        # XCSP3 (CSP/COP)
    elif ".wbo" in base:
        ext = ".wbo"
    else:
        ext = ".opb"
    dst = os.path.join(scratch, f"inst{tag}{ext}")
    with opener(path)() as src, open(dst, "wb") as out:
        if ext == ".cnf":
            for line in src:
                if line.strip() == b"%":
                    break
                out.write(line)
        else:
            shutil.copyfileobj(src, out)
    return dst


def normalize_status(raw):
    raw = raw.upper()
    if "OPTIMUM" in raw:
        return "OPTIMUM"
    if "UNSAT" in raw:
        return "UNSAT"
    if "SATISF" in raw:  # SAT / SATISFIABLE
        return "SAT"
    if "UNSUPP" in raw:
        return "UNSUPPORTED"
    return "UNKNOWN"


def rlimit_preexec(mem_mb):
    """Cap the child's address space (solver-agnostic OOM guard, like the
    wall-clock kill): the kernel fails its allocations instead of taking the
    machine down. Enforced on Linux; macOS ignores RLIMIT_AS (best effort)."""
    if not mem_mb:
        return None
    import resource

    def preexec():
        limit = mem_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ValueError, OSError):
            pass

    return preexec


def run_one(
    cmd_tmpl,
    inst,
    timeout,
    scratch,
    tag=0,
    mem_mb=0,
    verify_kind="none",
    grace_seconds=1.0,
    finalization_seconds=0.0,
):
    f = materialize(inst, scratch, tag)
    cmd = cmd_tmpl.replace("{f}", f).replace("{t}", str(timeout))
    argv = shlex.split(cmd)
    t0 = time.time()
    timedout = False
    out = ""
    stderr = ""
    returncode = None
    try:
        try:
            p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, start_new_session=True,
                                 preexec_fn=rlimit_preexec(mem_mb))
            register_process(p)
            try:
                out, stderr = p.communicate(timeout=timeout + finalization_seconds)
            except subprocess.TimeoutExpired:
                timedout = True
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                try:
                    out, stderr = p.communicate(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    out, stderr = p.communicate()
            except KeyboardInterrupt:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
                raise
            finally:
                unregister_process(p)
            returncode = p.returncode
        except Exception as error:
            return (
                "ERROR",
                None,
                time.time() - t0,
                False,
                str(error),
                returncode,
                {"attempted": False, "valid": None, "reason": "execution-error"},
                argv,
                out,
                stderr,
            )

        dt = time.time() - t0
        status, obj = "UNKNOWN", None
        for line in out.splitlines():
            if line.startswith("s "):
                status = normalize_status(line[2:])
            elif line.startswith("o "):
                match = re.match(r"o\s+(-?\d+)", line)
                if match:
                    obj = int(match.group(1))
        validation = verify_solution(verify_kind, Path(f), out, status, obj)
        return status, obj, dt, timedout, "", returncode, validation, argv, out, stderr
    finally:
        # Parallel runs materialize one file per in-flight instance; drop it
        # eagerly so scratch holds at most `jobs` decompressed instances.
        try:
            os.remove(f)
        except OSError:
            pass


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_identity(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"artifact does not exist: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def write_json_atomic(path, value):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def write_solver_logs(log_dir, relative_instance, stdout, stderr):
    if not log_dir:
        return "", ""
    root = Path(log_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(relative_instance.encode("utf-8")).hexdigest()[:16]
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(relative_instance).name)
    stem = f"{digest}-{base}"
    stdout_path = root / f"{stem}.stdout.log"
    stderr_path = root / f"{stem}.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return str(stdout_path), str(stderr_path)


def direction(path):
    """min/max/? objective sense: OPB `min:`/`max:` or XCSP `<minimize>`/`<maximize>`.

    Scans decompressed content (XCSP objective blocks can sit far from the top),
    capped so a huge instance never blows memory.
    """
    try:
        with opener(path)() as f:
            blob = f.read(8 << 20).decode("utf-8", "replace").lower()
    except Exception:
        return "?"
    if "max:" in blob or "<maximize" in blob:
        return "max"
    if "min:" in blob or "<minimize" in blob:
        return "min"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory of instances (recursed)")
    ap.add_argument("--cmd", required=True, help="solver command template with {f} and optional {t}")
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="cap instances (0 = all)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="instances run concurrently (solver itself stays as templated); "
                    "co-running instances add timing noise, so compare runs made with the SAME value")
    ap.add_argument("--mem-mb", type=int, default=0,
                    help="address-space cap per instance in MB (0 = none): the OS kills an "
                    "over-consuming solver instead of the machine dying; scored like any abnormal exit")
    ap.add_argument("--grace-seconds", type=float, default=1.0,
                    help="post-timeout SIGTERM grace used only for final output")
    ap.add_argument("--finalization-seconds", type=float, default=0.0,
                    help="extra external wall time for a solver bounded internally by {t} to print output")
    ap.add_argument("--verify-kind", choices=("none", "sat", "pb"), default="none",
                    help="independently replay feasible SAT or linear OPB assignments")
    ap.add_argument("--solver", default="solver", help="stable solver id recorded in provenance")
    ap.add_argument("--artifact", action="append", default=[],
                    help="solver artifact to hash; repeat for multi-file Java solvers")
    ap.add_argument("--provenance-out",
                    help="JSON sidecar path (default: <out>.provenance.json)")
    ap.add_argument("--log-dir", help="directory for per-instance stdout and stderr logs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.grace_seconds < 0:
        ap.error("--grace-seconds must be non-negative")
    if args.timeout <= 0:
        ap.error("--timeout must be positive")
    if args.finalization_seconds < 0:
        ap.error("--finalization-seconds must be non-negative")
    if args.finalization_seconds and "{t}" not in args.cmd:
        ap.error("--finalization-seconds requires an internal {t} command placeholder")
    if args.jobs <= 0:
        ap.error("--jobs must be positive")
    if args.mem_mb < 0:
        ap.error("--mem-mb must be non-negative")

    exts = (".cnf", ".opb", ".wbo", ".xml", ".lzma", ".xz", ".bz2", ".gz")
    files = []
    for root, _, names in os.walk(args.dir):
        for n in sorted(names):
            if n.endswith(exts):
                files.append(os.path.join(root, n))
    files.sort()
    if args.limit:
        files = files[: args.limit]
    total = len(files)
    if total == 0:
        print(f"no instances under {args.dir}", file=sys.stderr)
        sys.exit(1)

    root = Path(__file__).resolve().parents[2]
    instance_identities = [
        {
            "path": Path(path).resolve().relative_to(Path(args.dir).resolve()).as_posix(),
            "sha256": sha256_file(Path(path)),
        }
        for path in files
    ]
    artifacts = [artifact_identity(path) for path in args.artifact]
    configuration = {
        "solver": args.solver,
        "command_template": args.cmd,
        "timeout_seconds": args.timeout,
        "jobs": args.jobs,
        "memory_mb": args.mem_mb,
        "grace_seconds": args.grace_seconds,
        "finalization_seconds": args.finalization_seconds,
        "verify_kind": args.verify_kind,
        "log_dir": str(Path(args.log_dir).resolve()) if args.log_dir else None,
        "artifacts": artifacts,
        "instances": instance_identities,
    }
    provenance_path = args.provenance_out or args.out + ".provenance.json"

    scratch = tempfile.mkdtemp(prefix="satpb_")
    invalid = 0
    inconclusive = 0

    def solve_row(tag_inst):
        tag, inst = tag_inst
        status, obj, dt, to, err, returncode, validation, argv, stdout, stderr = run_one(
            args.cmd,
            inst,
            args.timeout,
            scratch,
            tag,
            args.mem_mb,
            args.verify_kind,
            args.grace_seconds,
            args.finalization_seconds,
        )
        relative = Path(inst).resolve().relative_to(Path(args.dir).resolve()).as_posix()
        instance_hash = instance_identities[tag - 1]["sha256"]
        stdout_log, stderr_log = write_solver_logs(args.log_dir, relative, stdout, stderr)
        return (
            relative, status, obj, dt, to, err, direction(inst), returncode,
            validation, argv, instance_hash, stdout_log, stderr_log,
        )

    try:
        Path(args.out).resolve().parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow([
                "instance", "status", "obj", "time", "timedout", "dir",
                "valid", "validation_reason", "validation_message", "returncode",
                "instance_sha256", "stdout_log", "stderr_log",
            ])

            def emit(done, row):
                nonlocal invalid, inconclusive
                (
                    name, status, obj, dt, to, err, d, returncode, validation,
                    _argv, instance_hash, stdout_log, stderr_log,
                ) = row
                valid = validation["valid"]
                invalid += valid is False
                inconclusive += validation["attempted"] and valid is None
                wr.writerow([
                    name,
                    status,
                    "" if obj is None else obj,
                    f"{dt:.3f}",
                    int(to),
                    d,
                    "" if valid is None else int(valid),
                    validation["reason"],
                    validation.get("message", ""),
                    "" if returncode is None else returncode,
                    instance_hash,
                    stdout_log,
                    stderr_log,
                ])
                fh.flush()
                extra = f" obj={obj}" if obj is not None else ""
                if status == "ERROR" and err:
                    extra += f" ({err})"
                if valid is False:
                    extra += f" INVALID({validation['reason']})"
                elif validation["attempted"] and valid is None:
                    extra += f" UNVERIFIED({validation['reason']})"
                print(f"[{done}/{total}] {name}: {status}{extra} {dt:.2f}s{' TO' if to else ''}", flush=True)

            if args.jobs <= 1:
                for i, inst in enumerate(files, 1):
                    emit(i, solve_row((i, inst)))
            else:
                # CSV rows land in completion order; compare.py keys by name.
                pool = ThreadPoolExecutor(max_workers=args.jobs)
                futures = [pool.submit(solve_row, (i, inst)) for i, inst in enumerate(files, 1)]
                try:
                    for done, fut in enumerate(as_completed(futures), 1):
                        emit(done, fut.result())
                except KeyboardInterrupt:
                    kill_active_processes()
                    for future in futures:
                        future.cancel()
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
                else:
                    pool.shutdown(wait=True)
    finally:
        kill_active_processes()
        shutil.rmtree(scratch, ignore_errors=True)

    sidecar = {
        "schema": "qayd.satpb.campaign/v1",
        "complete": True,
        "configuration": configuration,
        "configuration_sha256": canonical_sha256(configuration),
        "machine": machine_provenance(root),
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "validator_sha256": sha256_file(Path(__file__).with_name("satpb_verify.py")),
        },
        "results": {
            "path": str(Path(args.out).resolve()),
            "sha256": sha256_file(Path(args.out)),
            "records": total,
            "invalid": invalid,
            "inconclusive_validations": inconclusive,
        },
    }
    write_json_atomic(provenance_path, sidecar)


if __name__ == "__main__":
    main()
