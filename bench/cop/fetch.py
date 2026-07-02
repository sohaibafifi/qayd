"""Fetch XCSP3 COP (optimization) competition instances.

    python fetch.py --year 25 --limit 30
    python fetch.py --year 24 --mini            # MiniCOP track
    python fetch.py --year 25 --limit 0         # whole COP track
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from xcsp_fetch import fetch_xcsp  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="25")
    ap.add_argument("--limit", type=int, default=30, help="cap instances (0 = all)")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "instances"))
    args = ap.parse_args()
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    fetch_xcsp("COP", args.year, args.out, limit=args.limit, mini=args.mini, repo_root=repo)
