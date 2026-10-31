"""Cluster helpers retained from bench/fastcop/manifest.py; experimental semantics preserved."""
from __future__ import annotations

import gzip
import lzma
import xml.sax
import xml.sax.handler
from pathlib import Path
from typing import BinaryIO


FAMILIES = (
    ("FlexibleJobshopScen", "FlexibleJobshopScen"),
    ("ChampionsLeague-strict", "ChampionsLeague"),
    ("LowAutocorrelation", "LowAutocorrelation"),
    ("MetabolicNetwork", "MetabolicNetwork"),
    ("BlockModeling", "BlockModeling"),
    ("ButtonsScissors", "ButtonsScissors"),
    ("ChampionsLeague", "ChampionsLeague"),
    ("AlteredStates-bis", "AlteredStates"),
    ("AlteredStates", "AlteredStates"),
    ("BusScheduling", "BusScheduling"),
    ("RoadefPlaning2", "RoadefPlaning2"),
    ("TankAllocation1", "TankAllocation"),
    ("TankAllocation2", "TankAllocation"),
    ("SchedulingOS", "SchedulingOS"),
    ("RollerSplat", "RollerSplat"),
    ("Fortress1", "Fortress"),
    ("Fortress2", "Fortress"),
    ("Cutstock", "Cutstock"),
    ("Coprime", "Coprime"),
    ("FAPP", "FAPP"),
    ("IHTC", "IHTC"),
)


class ManifestError(RuntimeError):
    """The corpus or manifest is incomplete or inconsistent."""


def open_instance(path: Path) -> BinaryIO:
    """Open an XCSP instance, detecting compression from magic bytes."""
    with path.open("rb") as source:
        magic = source.read(6)
    if magic.startswith((b"\xfd7zXZ", b"\x5d\x00\x00")):
        return lzma.open(path, "rb")
    if magic.startswith(b"\x1f\x8b"):
        return gzip.open(path, "rb")
    return path.open("rb")


def detect_objective_sense(path: Path) -> str:
    """Return min or max after streaming the complete XML objective section."""
    class ObjectiveFound(Exception):
        def __init__(self, sense: str) -> None:
            self.sense = sense

    class ObjectiveHandler(xml.sax.handler.ContentHandler):
        def startElement(self, name: str, attrs) -> None:
            del attrs
            local_name = name.rsplit(":", 1)[-1].lower()
            if local_name == "minimize":
                raise ObjectiveFound("min")
            if local_name == "maximize":
                raise ObjectiveFound("max")

    try:
        with open_instance(path) as source:
            parser = xml.sax.make_parser()
            parser.setFeature(xml.sax.handler.feature_external_ges, False)
            parser.setContentHandler(ObjectiveHandler())
            parser.parse(source)
    except ObjectiveFound as found:
        return found.sense
    except (xml.sax.SAXException, EOFError, lzma.LZMAError, OSError) as error:
        raise ManifestError(f"cannot parse objective in {path}: {error}") from error
    raise ManifestError(f"no minimize/maximize objective in {path}")


def family_for(filename: str) -> tuple[str, str]:
    for family, group in FAMILIES:
        if filename == family or filename.startswith(family + "-"):
            return family, group
    raise ManifestError(f"unknown XCSP25 FAST COP family: {filename}")


