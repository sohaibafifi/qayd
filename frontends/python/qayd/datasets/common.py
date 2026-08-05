"""Shared helpers for benchmark instance parsers."""

from __future__ import annotations

import bz2
import gzip
import lzma
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


class DatasetParseError(ValueError):
    """A malformed benchmark instance with source and line information."""

    def __init__(
        self, message: str, *, source: str = "<string>", line: Optional[int] = None
    ):
        self.source = source
        self.line = line
        location = source if line is None else f"{source}:{line}"
        super().__init__(f"{location}: {message}")


def read_text(path: object) -> Tuple[str, str]:
    """Read one UTF-8 benchmark file and return its text and display name."""

    resolved = Path(path)  # type: ignore[arg-type]
    suffix = resolved.suffix.lower()
    opener = {
        ".bz2": bz2.open,
        ".gz": gzip.open,
        ".lzma": lzma.open,
        ".xz": lzma.open,
    }.get(suffix)
    if opener is None:
        text = resolved.read_text(encoding="utf-8-sig")
    else:
        with opener(resolved, "rt", encoding="utf-8-sig") as stream:
            text = stream.read()
    return text, str(resolved)


def numbered_lines(text: str) -> List[Tuple[int, str]]:
    return [
        (line_number, raw.rstrip("\r\n"))
        for line_number, raw in enumerate(text.splitlines(), 1)
    ]


def nonempty_lines(text: str, *, comments: Iterable[str] = ()) -> List[Tuple[int, str]]:
    prefixes = tuple(comments)
    return [
        (line_number, raw.strip())
        for line_number, raw in numbered_lines(text)
        if raw.strip() and not raw.lstrip().startswith(prefixes)
    ]


def integer_tokens(line: str, *, source: str, line_number: int) -> List[int]:
    values: List[int] = []
    for token in line.split():
        try:
            values.append(int(token))
        except ValueError as exc:
            raise DatasetParseError(
                f"expected an integer, got {token!r}", source=source, line=line_number
            ) from exc
    return values


def require(
    condition: bool, message: str, *, source: str, line: Optional[int] = None
) -> None:
    if not condition:
        raise DatasetParseError(message, source=source, line=line)
