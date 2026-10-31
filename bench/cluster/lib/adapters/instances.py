"""Cluster helpers retained from bench/common/run.py; experimental semantics preserved."""
from __future__ import annotations

import bz2
import gzip
import lzma
import os
import shutil


MAGIC = {b"\xfd7zXZ": lzma, b"\x5d\x00\x00": lzma, b"BZh": bz2, b"\x1f\x8b": gzip}


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


