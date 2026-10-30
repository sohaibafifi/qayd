"""Shared XCSP3 instance fetcher for the csp/ (decision) and cop/ (optimization)
benchmark dirs.

The XCSP competition ships one unified zip per year
(https://www.cril.univ-artois.fr/~lecoutre/compets/instancesXCSP<YY>.zip) whose
members are already split into track subtrees: `CSP<YY>/`, `COP<YY>/`,
`MiniCSP<YY>/`, `MiniCOP<YY>/`. We extract only the requested track's `.lzma`
instances. If the repo already has them under `data/XCSP<YY>/<TRACK><YY>/`, we
reuse that instead of downloading.
"""
import os
from pathlib import PurePosixPath
import subprocess
import zipfile

BASE = "https://www.cril.univ-artois.fr/~lecoutre/compets"


def fetch_xcsp(track, year, out, limit=0, mini=False, repo_root=None):
    """track: 'CSP' or 'COP'.  year: two digits e.g. '25'.  out: dest dir."""
    subdir = ("Mini" if mini else "") + track + year
    dest = os.path.join(out, subdir)
    os.makedirs(dest, exist_ok=True)

    # fast path: copy from an already-extracted repo tree
    if repo_root:
        local = os.path.join(repo_root, "data", f"XCSP{year}", subdir)
        if os.path.isdir(local):
            names = sorted(n for n in os.listdir(local) if n.endswith((".lzma", ".xml", ".xz")))
            # An empty or capped cache does not establish complete coverage.
            # Full-track requests must check the archive membership below.
            if limit and len(names) >= limit:
                names = names[:limit]
                for n in names:
                    dst = os.path.join(dest, n)
                    if not os.path.exists(dst):
                        with open(os.path.join(local, n), "rb") as s, open(dst, "wb") as d:
                            d.write(s.read())
                print(f"reused {len(names)} instances from {local}", flush=True)
                return dest

    # download + extract only this track's members
    zpath = os.path.join(out, f"instancesXCSP{year}.zip")
    if not (os.path.exists(zpath) and os.path.getsize(zpath) > 0):
        url = f"{BASE}/instancesXCSP{year}.zip"
        print(f"downloading {url}", flush=True)
        rc = subprocess.run(["curl", "-sL", "--max-time", "1800", "-o", zpath, url]).returncode
        if rc != 0 or not os.path.getsize(zpath):
            raise SystemExit(f"download failed (curl rc={rc})")

    n = 0
    with zipfile.ZipFile(zpath) as z:
        # Official archives may wrap the track in an XCSP<YY>/ directory.
        members = [m for m in z.namelist()
                   if subdir in PurePosixPath(m).parts and m.endswith((".lzma", ".xml", ".xz"))]
        members.sort()
        if not members:
            raise SystemExit(f"archive contains no {subdir} instances: {zpath}")
        if len({os.path.basename(m) for m in members}) != len(members):
            raise SystemExit(f"duplicate instance basenames in {subdir}: {zpath}")
        if limit:
            members = members[:limit]
        for m in members:
            data = z.read(m)
            dst = os.path.join(dest, os.path.basename(m))
            with open(dst, "wb") as f:
                f.write(data)
            n += 1
    print(f"extracted {n} {subdir} instances into {dest}", flush=True)
    return dest
