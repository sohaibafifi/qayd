"""Download SAT Competition instances from the Global Benchmark Database (GBD).

GBD (https://benchmark-database.de) exposes every SAT-Competition CNF as an
individually addressable, xz-compressed file, so we can pull a bounded slice
instead of a multi-TB monolithic tarball.

  - `--track main_2024` selects a competition track (main_2020 .. main_2024,
    also parallel_*, cloud_*, incremental_* etc. exist on GBD).
  - Files download as `<hash>.cnf.xz` into <out>/<track>/ and are skipped on
    re-run (resumable).
  - `--limit N` caps the instance count; `--max-mb M` skips any file whose
    *compressed* size exceeds M MB (crafted competition CNFs can be hundreds of
    MB and hopeless under a short timeout). Use `--limit 0` for the whole track.

Full track (all 400) example:
    python fetch_sat.py --track main_2024 --limit 0 --out instances/sat
"""
import argparse
import os
import subprocess

GBD = "https://benchmark-database.de"

# The macOS python.org runtime ships without a usable CA bundle, so urllib's TLS
# verification fails here; curl (system, uses the keychain) is reliable. Shell
# out to it for every network call.


def curl(*extra, timeout=180):
    return subprocess.run(["curl", "-sL", "--max-time", str(timeout), *extra],
                          capture_output=True)


def list_instances(track):
    r = curl(f"{GBD}/getinstances?query=track%3D{track}")
    body = r.stdout.decode("utf-8", "replace")
    return [ln.strip() for ln in body.splitlines() if "/file/" in ln]


def head_size(url):
    r = curl("-I", url, timeout=60)
    for line in r.stdout.decode("utf-8", "replace").splitlines():
        if line.lower().startswith("content-length:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="main_2024")
    ap.add_argument("--out", default="instances/sat")
    ap.add_argument("--limit", type=int, default=40, help="max instances (0 = all)")
    ap.add_argument("--max-mb", type=float, default=5.0, help="skip files larger than this (compressed); 0 = no cap")
    args = ap.parse_args()

    outdir = os.path.join(args.out, args.track)
    os.makedirs(outdir, exist_ok=True)
    urls = list_instances(args.track)
    print(f"track {args.track}: {len(urls)} instances available", flush=True)
    cap = args.max_mb * 1e6

    got = 0
    for url in urls:
        if args.limit and got >= args.limit:
            break
        h = url.rsplit("/", 1)[-1]
        dst = os.path.join(outdir, f"{h}.cnf.xz")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            got += 1
            continue
        if cap:
            sz = head_size(url)
            if sz is not None and sz > cap:
                print(f"  skip {h} ({sz/1e6:.1f} MB > {args.max_mb} MB)", flush=True)
                continue
        r = curl("-o", dst, url)
        if r.returncode != 0 or not (os.path.exists(dst) and os.path.getsize(dst) > 0):
            print(f"  FAIL {h}: curl rc={r.returncode}", flush=True)
            if os.path.exists(dst):
                os.remove(dst)
            continue
        got += 1
        print(f"[{got}] {h} ({os.path.getsize(dst)/1e6:.2f} MB)", flush=True)
    print(f"done: {got} instances in {outdir}", flush=True)


if __name__ == "__main__":
    main()
