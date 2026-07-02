"""Download Pseudo-Boolean Competition instances from CRIL (Artois).

The PB competition publishes per-year tarballs at
https://www.cril.univ-artois.fr/PB24/benchs/ :

  - `selected-PB<YY>.tar`   -> instances actually used in that competition round
                               (best default: small, curated, representative).
  - `normalized-PB<YY>.tar` -> every submitted instance for that year (larger).

Instances inside are `.opb` (linear PB), some `.wbo`; frequently `.bz2`/`.gz`
compressed. We fetch the tarball, extract, and leave OPB files under
<out>/<archive>/ . `--keep-tar` retains the download for reuse.

Whole-history example (all normalized years):
    for y in 06 07 09 10 11 12 16 24; do
        python fetch_pb.py --archive normalized-PB$y --out instances/pb ; done
"""
import argparse
import os
import subprocess
import tarfile

BASE = "https://www.cril.univ-artois.fr/PB24/benchs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="selected-PB24", help="tar basename, e.g. selected-PB24, normalized-PB16")
    ap.add_argument("--out", default="instances/pb")
    ap.add_argument("--keep-tar", action="store_true")
    ap.add_argument("--max-files", type=int, default=0, help="extract at most N members (0 = all)")
    ap.add_argument("--kind", default="opb", choices=["opb", "wbo", "all"],
                    help="opb = linear PB (qayd-pb supported, default); wbo = weighted; all = everything")
    args = ap.parse_args()

    # qayd-pb solves linear OPB (PBS/PBO tracks). The selected tar bundles OPB
    # under DEC-LIN/OPT-LIN and separate WBO files -- filter to the ones the
    # solver actually accepts so the comparison is meaningful.
    def wanted(name):
        low = name.lower().split(".xz")[0].split(".bz2")[0].split(".gz")[0]
        if args.kind == "all":
            return low.endswith((".opb", ".wbo"))
        return low.endswith("." + args.kind)

    os.makedirs(args.out, exist_ok=True)
    tar = os.path.join(args.out, args.archive + ".tar")
    url = f"{BASE}/{args.archive}.tar"

    if not (os.path.exists(tar) and os.path.getsize(tar) > 0):
        print(f"downloading {url}", flush=True)
        # curl over urllib: system curl uses the keychain CA store (python.org
        # runtime lacks a usable CA bundle on macOS).
        rc = subprocess.run(["curl", "-sL", "--max-time", "1800", "-o", tar, url]).returncode
        if rc != 0 or not (os.path.exists(tar) and os.path.getsize(tar) > 0):
            raise SystemExit(f"download failed (curl rc={rc}): {url}")
    print(f"tar: {os.path.getsize(tar)/1e6:.1f} MB", flush=True)

    dest = os.path.join(args.out, args.archive)
    os.makedirs(dest, exist_ok=True)
    n = 0
    with tarfile.open(tar) as tf:
        for m in tf:
            if not m.isfile() or not wanted(m.name):
                continue
            if args.max_files and n >= args.max_files:
                break
            tf.extract(m, dest, filter="data")
            n += 1
    print(f"extracted {n} members into {dest}", flush=True)
    if not args.keep_tar:
        os.remove(tar)


if __name__ == "__main__":
    main()
