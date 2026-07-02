"""Run ONE solver over a directory of SAT/PB instances -> per-instance CSV.

Solver-agnostic: the command is a template with a `{f}` placeholder for the
(decompressed) instance path, e.g.

    qayd-sat : ../../target/release/qayd-sat -t {t} {f}
    cadical  : cadical {f}
    qayd-pb  : ../../target/release/qayd-pb -t {t} {f}
    sat4j-pb : java -cp solvers/s4j-pb.jar:solvers/s4j-core.jar \
                    org.sat4j.pb.LanceurPseudo2007 {f}

Instances may be xz/bz2/gz compressed or bare; we transparently decompress each
to a scratch file before the run (uniform across solvers that can't read
compressed input). Each solver gets the SAME wall-clock timeout, enforced here
by killing the whole process group -- so a solver that ignores an internal `-t`
is still bounded and scored as a timeout.

Output CSV columns: instance,status,obj,time,timedout  (one row per instance).
`status` is normalized to SAT / UNSAT / OPTIMUM / UNKNOWN / ERROR.
"""
import argparse
import bz2
import csv
import gzip
import lzma
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

# .xz (fd 37 7a) and raw-.lzma (5d 00 00) both go through the lzma module
# (lzma.open auto-detects the container); bz2 and gzip by their own magic.
MAGIC = {b"\xfd7zXZ": lzma, b"\x5d\x00\x00": lzma, b"BZh": bz2, b"\x1f\x8b": gzip}


def opener(path):
    """Return a decompressing open() for path based on magic bytes, else plain."""
    with open(path, "rb") as f:
        head = f.read(6)
    for magic, mod in MAGIC.items():
        if head.startswith(magic):
            return lambda: mod.open(path, "rb")
    return lambda: open(path, "rb")


def materialize(path, scratch):
    """Decompress/copy instance into scratch dir, return the plain file path.

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
    dst = os.path.join(scratch, "inst" + ext)
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


def run_one(cmd_tmpl, inst, timeout, scratch):
    f = materialize(inst, scratch)
    cmd = cmd_tmpl.replace("{f}", f).replace("{t}", str(timeout))
    argv = cmd.split()
    t0 = time.time()
    timedout = False
    out = ""
    try:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, start_new_session=True)
        try:
            out, _ = p.communicate(timeout=timeout + 5)
        except subprocess.TimeoutExpired:
            timedout = True
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            out, _ = p.communicate()
    except Exception as e:
        return "ERROR", None, time.time() - t0, False, str(e)
    dt = time.time() - t0

    status, obj = "UNKNOWN", None
    for line in out.splitlines():
        if line.startswith("s "):
            status = normalize_status(line[2:])
        elif line.startswith("o "):
            m = re.match(r"o\s+(-?\d+)", line)
            if m:
                obj = int(m.group(1))
    if timedout and status == "UNKNOWN":
        status = "UNKNOWN"  # explicit: timed out with no answer
    return status, obj, dt, timedout, ""


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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

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

    scratch = tempfile.mkdtemp(prefix="satpb_")
    try:
        with open(args.out, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["instance", "status", "obj", "time", "timedout", "dir"])
            for i, inst in enumerate(files, 1):
                status, obj, dt, to, err = run_one(args.cmd, inst, args.timeout, scratch)
                d = direction(inst)
                name = os.path.basename(inst)
                wr.writerow([name, status, "" if obj is None else obj, f"{dt:.3f}", int(to), d])
                fh.flush()
                extra = f" obj={obj}" if obj is not None else ""
                if status == "ERROR" and err:
                    extra += f" ({err})"
                print(f"[{i}/{total}] {name}: {status}{extra} {dt:.2f}s{' TO' if to else ''}", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
