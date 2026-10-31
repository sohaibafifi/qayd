"""A Rust panic must surface as a Python exception, never abort the interpreter.

Requires the extension built with the `pyext` profile (the pyproject default):
`release` has panic=abort, under which this test would kill pytest outright.
"""
import pytest

qayd = pytest.importorskip("qayd")


def test_rust_panic_raises_python_exception():
    from qayd import _core

    with pytest.raises(BaseException, match="intentional test panic"):
        _core._rust_panic()
    # The interpreter survived; the extension still works.
    assert isinstance(qayd.Model(), object)
