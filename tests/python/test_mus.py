"""MUS enumeration returns a named result (muses / msses / complete)."""
import pytest

cp = pytest.importorskip("qayd")


def test_enumerate_mus_returns_named_result():
    # A disequality triangle x != y != z != x over a 2-value domain: colouring a
    # triangle needs 3 colours, so all three disequalities together are the unique
    # minimal unsatisfiable core, and dropping any one is satisfiable.
    model = cp.Model()
    x = model.int_var(0, 1, name="x")
    y = model.int_var(0, 1, name="y")
    z = model.int_var(0, 1, name="z")
    with model.soft(name="xy"):
        model.add(x != y)
    with model.soft(name="yz"):
        model.add(y != z)
    with model.soft(name="xz"):
        model.add(x != z)

    result = model.enumerate_mus()
    assert result.complete is True
    assert [sorted(m) for m in result.muses] == [["xy", "xz", "yz"]], result.muses
    assert sorted(sorted(m) for m in result.msses) == [["xy", "xz"], ["xy", "yz"], ["xz", "yz"]]
    assert "MusEnumeration" in repr(result)
