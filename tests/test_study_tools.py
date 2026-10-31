import pytest

from core.tool_registry import ToolRegistry
from tools.study_tools import calculate, convert_units, lookup_formula, register


def test_calculate_basic_and_functions():
    assert calculate("37*48")["result"] == 1776
    assert calculate("sqrt(144)")["result"] == 12
    assert calculate("2^10")["result"] == 1024
    assert calculate("pi*7^2")["result"] == pytest.approx(153.938, abs=1e-3)


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "2**99999", "factorial(100000)"])
def test_calculate_rejects_unsafe_or_huge(expr):
    with pytest.raises(ValueError):
        calculate(expr)


def test_convert_units():
    assert convert_units(5, "kilometers", "miles")["result"] == pytest.approx(3.10686)
    assert convert_units(100, "F", "C")["result"] == pytest.approx(37.7778)
    assert convert_units(25, "kg", "pounds")["result"] == pytest.approx(55.1156)
    with pytest.raises(ValueError):
        convert_units(5, "kg", "m")


def test_lookup_formula_aliases_and_fuzzy():
    assert lookup_formula("Kinetic Energy")["formula"].startswith("KE")
    assert lookup_formula("ohms law")["formula"] == "V = I * R"
    assert lookup_formula("kinetik energy")["found"] is True
    assert lookup_formula("quantum chromodynamics")["found"] is False


def test_registry_never_raises():
    reg = ToolRegistry()
    register(reg)
    assert set(reg.names()) == {"calculate", "convert_units", "lookup_formula"}
    assert "error" in reg.call("calculate", '{"expression": "__import__(\'os\')"}')
    assert "error" in reg.call("nope", "{}")
    assert reg.call("calculate", '{"expression": "6*7"}')["result"] == 42
