"""Guards for the packaged type stubs (``__init__.pyi`` + ``py.typed``).

Covers three things:
1. the stub file is valid Python (``ast.parse``);
2. the *installed* package ships both marker files, so ``pip install qayd``
   gives downstream users working types;
3. every public name in ``qayd.__all__`` is defined in the stub, so adding an
   API symbol without a stub fails here.
"""
import ast
from pathlib import Path

import pytest

# Source-tree stub (always present, even without a build).
_STUB = Path(__file__).resolve().parents[2] / "frontends" / "python" / "qayd" / "__init__.pyi"


def _defined_names(source: str) -> set:
    """Top-level class/function/assignment names in a stub source."""
    tree = ast.parse(source)
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def test_stub_source_parses():
    assert _STUB.is_file(), f"stub missing: {_STUB}"
    ast.parse(_STUB.read_text())  # raises SyntaxError on malformed stub


def test_installed_package_ships_markers():
    qayd = pytest.importorskip("qayd")
    pkg_dir = Path(qayd.__file__).parent
    assert (pkg_dir / "py.typed").is_file(), f"py.typed not installed in {pkg_dir}"
    assert (pkg_dir / "__init__.pyi").is_file(), f"__init__.pyi not installed in {pkg_dir}"


def test_stub_covers_all_public_names():
    qayd = pytest.importorskip("qayd")
    # Prefer the installed stub (what users actually get); fall back to source.
    installed = Path(qayd.__file__).parent / "__init__.pyi"
    stub = installed if installed.is_file() else _STUB
    defined = _defined_names(stub.read_text())
    missing = set(qayd.__all__) - defined
    assert not missing, f"public names absent from stub: {sorted(missing)}"
