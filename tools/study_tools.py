"""Study-helper tools: calculate, convert_units, lookup_formula.

Domain-specific and pluggable. The runtime loads this module by name and calls
register(registry); nothing in core/ or voice_io/ imports it.
"""
from __future__ import annotations

import ast
import difflib
import math
import operator

SYSTEM_PROMPT = """\
You are Vaak, a friendly study helper for school and college students.
- Use calculate for any arithmetic or math expression.
- Use convert_units when the user wants a quantity in a different unit.
- Use lookup_formula only when the user asks for a formula or equation. \
Concept explanations ("explain", "what is", "why") are answered directly.
- If a question needs a formula AND a number, look up the formula, then use calculate.
"""


# --- calculate -----------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt, "cbrt": math.cbrt, "abs": abs, "round": round,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "radians": math.radians, "degrees": math.degrees,
    "log": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp,
    "factorial": lambda n: math.factorial(n) if n <= 1000 else _too_big(), "floor": math.floor, "ceil": math.ceil,
}
_CONSTS = {"pi": math.pi, "e": math.e}
_MAX_EXPONENT = 1000


def _too_big():
    raise ValueError("number too large")


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")


def calculate(expression: str) -> dict:
    expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    value = _eval(ast.parse(expr, mode="eval").body)
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    elif isinstance(value, float):
        value = round(value, 10)
    return {"expression": expression, "result": value}


# --- convert_units ---------------------------------------------------------------

# unit -> (dimension, factor to SI base unit)
_UNITS = {
    # length (m)
    "m": ("length", 1), "km": ("length", 1000), "cm": ("length", 0.01), "mm": ("length", 0.001),
    "um": ("length", 1e-6), "nm": ("length", 1e-9),
    "mi": ("length", 1609.344), "yd": ("length", 0.9144), "ft": ("length", 0.3048),
    "in": ("length", 0.0254),
    # mass (kg)
    "kg": ("mass", 1), "g": ("mass", 0.001), "mg": ("mass", 1e-6), "t": ("mass", 1000),
    "lb": ("mass", 0.45359237), "oz": ("mass", 0.028349523125), "quintal": ("mass", 100),
    # time (s)
    "s": ("time", 1), "ms": ("time", 0.001), "min": ("time", 60), "h": ("time", 3600),
    "day": ("time", 86400), "week": ("time", 604800), "year": ("time", 31557600),
    # speed (m/s)
    "m/s": ("speed", 1), "km/h": ("speed", 1000 / 3600), "mph": ("speed", 0.44704),
    # area (m^2)
    "m2": ("area", 1), "km2": ("area", 1e6), "cm2": ("area", 1e-4), "ft2": ("area", 0.09290304),
    "hectare": ("area", 1e4), "acre": ("area", 4046.8564224),
    # volume (m^3)
    "m3": ("volume", 1), "l": ("volume", 0.001), "ml": ("volume", 1e-6), "cm3": ("volume", 1e-6),
    "gal": ("volume", 0.003785411784),
    # energy (J)
    "j": ("energy", 1), "kj": ("energy", 1000), "cal": ("energy", 4.184), "kcal": ("energy", 4184),
    "kwh": ("energy", 3.6e6), "ev": ("energy", 1.602176634e-19),
    # pressure (Pa)
    "pa": ("pressure", 1), "kpa": ("pressure", 1000), "atm": ("pressure", 101325),
    "bar": ("pressure", 1e5), "mmhg": ("pressure", 133.322387415),
}
_TEMPS = {"c", "f", "k"}
_ALIASES = {
    "meter": "m", "metre": "m", "kilometer": "km", "kilometre": "km", "centimeter": "cm",
    "centimetre": "cm", "millimeter": "mm", "millimetre": "mm", "micrometer": "um",
    "nanometer": "nm", "mile": "mi", "yard": "yd", "foot": "ft", "feet": "ft", "inch": "in",
    "inches": "in", "kilogram": "kg", "kilo": "kg", "gram": "g", "milligram": "mg",
    "tonne": "t", "ton": "t", "pound": "lb", "lbs": "lb", "ounce": "oz",
    "second": "s", "sec": "s", "millisecond": "ms", "minute": "min", "hour": "h", "hr": "h",
    "kmph": "km/h", "kph": "km/h", "km/hr": "km/h", "mps": "m/s", "miles per hour": "mph",
    "sq m": "m2", "square meter": "m2", "square metre": "m2", "sq km": "km2",
    "square kilometer": "km2", "sq cm": "cm2", "sq ft": "ft2", "square foot": "ft2",
    "square feet": "ft2", "m^2": "m2", "km^2": "km2", "cm^2": "cm2",
    "liter": "l", "litre": "l", "milliliter": "ml", "millilitre": "ml", "cubic meter": "m3",
    "m^3": "m3", "cc": "cm3", "gallon": "gal",
    "joule": "j", "kilojoule": "kj", "calorie": "cal", "kilocalorie": "kcal",
    "electronvolt": "ev", "pascal": "pa", "kilopascal": "kpa", "atmosphere": "atm",
    "celsius": "c", "centigrade": "c", "°c": "c", "degc": "c", "fahrenheit": "f", "°f": "f",
    "degf": "f", "kelvin": "k",
}


def _norm_unit(unit: str) -> str:
    u = unit.strip().lower()
    if u in _UNITS or u in _TEMPS:
        return u
    if u in _ALIASES:
        return _ALIASES[u]
    if u.endswith("s") and u[:-1] in _ALIASES:  # plurals: "meters", "hours"
        return _ALIASES[u[:-1]]
    raise ValueError(f"unknown unit: {unit!r}")


def _convert_temp(value: float, src: str, dst: str) -> float:
    kelvin = {"c": value + 273.15, "f": (value - 32) * 5 / 9 + 273.15, "k": value}[src]
    return {"c": kelvin - 273.15, "f": (kelvin - 273.15) * 9 / 5 + 32, "k": kelvin}[dst]


def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
    src, dst = _norm_unit(from_unit), _norm_unit(to_unit)
    if src in _TEMPS or dst in _TEMPS:
        if not (src in _TEMPS and dst in _TEMPS):
            raise ValueError(f"cannot convert {from_unit} to {to_unit}")
        converted = _convert_temp(value, src, dst)
    else:
        (dim_a, fa), (dim_b, fb) = _UNITS[src], _UNITS[dst]
        if dim_a != dim_b:
            raise ValueError(f"cannot convert {dim_a} ({from_unit}) to {dim_b} ({to_unit})")
        converted = value * fa / fb
    return {"value": value, "from_unit": from_unit, "to_unit": to_unit,
            "result": float(f"{converted:.6g}")}


# --- lookup_formula ------------------------------------------------------------------

_FORMULAS = {
    "newton's second law": ("F = m * a", "F force (N), m mass (kg), a acceleration (m/s^2)"),
    "kinetic energy": ("KE = (1/2) * m * v^2", "m mass (kg), v velocity (m/s)"),
    "potential energy": ("PE = m * g * h", "m mass (kg), g = 9.8 m/s^2, h height (m)"),
    "momentum": ("p = m * v", "m mass (kg), v velocity (m/s)"),
    "work": ("W = F * d * cos(theta)", "F force (N), d displacement (m), theta angle between them"),
    "power": ("P = W / t", "W work (J), t time (s)"),
    "speed": ("speed = distance / time", ""),
    "acceleration": ("a = (v - u) / t", "u initial velocity, v final velocity, t time"),
    "equations of motion": ("v = u + a*t;  s = u*t + (1/2)*a*t^2;  v^2 = u^2 + 2*a*s",
                            "u initial velocity, v final velocity, a acceleration, t time, s displacement"),
    "density": ("rho = m / V", "m mass, V volume"),
    "pressure": ("P = F / A", "F force (N), A area (m^2)"),
    "ohm's law": ("V = I * R", "V voltage (V), I current (A), R resistance (ohm)"),
    "electric power": ("P = V * I", "V voltage, I current"),
    "gravitational force": ("F = G * m1 * m2 / r^2", "G = 6.674e-11 N m^2/kg^2, r distance"),
    "mass-energy equivalence": ("E = m * c^2", "c = 3e8 m/s"),
    "ideal gas law": ("P * V = n * R * T", "n moles, R = 8.314 J/(mol K), T temperature (K)"),
    "wave speed": ("v = f * lambda", "f frequency (Hz), lambda wavelength (m)"),
    "area of circle": ("A = pi * r^2", "r radius"),
    "circumference of circle": ("C = 2 * pi * r", "r radius"),
    "area of triangle": ("A = (1/2) * b * h", "b base, h height"),
    "area of rectangle": ("A = l * w", "l length, w width"),
    "volume of sphere": ("V = (4/3) * pi * r^3", "r radius"),
    "volume of cylinder": ("V = pi * r^2 * h", "r radius, h height"),
    "volume of cone": ("V = (1/3) * pi * r^2 * h", "r radius, h height"),
    "pythagorean theorem": ("a^2 + b^2 = c^2", "c hypotenuse"),
    "quadratic formula": ("x = (-b ± sqrt(b^2 - 4ac)) / (2a)", "for a*x^2 + b*x + c = 0"),
    "simple interest": ("SI = P * R * T / 100", "P principal, R rate (% per year), T time (years)"),
    "compound interest": ("A = P * (1 + R/100)^T", "P principal, R rate (% per period), T periods"),
}
_FORMULA_ALIASES = {
    "f=ma": "newton's second law", "newton second law": "newton's second law",
    "newtons second law": "newton's second law", "force": "newton's second law",
    "ke": "kinetic energy", "pe": "potential energy", "gravitational potential energy": "potential energy",
    "ohms law": "ohm's law", "e=mc2": "mass-energy equivalence", "e=mc^2": "mass-energy equivalence",
    "einstein": "mass-energy equivalence", "pythagoras theorem": "pythagorean theorem",
    "pythagoras": "pythagorean theorem", "circle area": "area of circle",
    "circumference": "circumference of circle", "velocity": "speed", "si": "simple interest",
    "ci": "compound interest", "kinematics": "equations of motion", "suvat": "equations of motion",
}


def lookup_formula(topic: str) -> dict:
    key = topic.strip().lower()
    key = _FORMULA_ALIASES.get(key, key)
    if key not in _FORMULAS:
        close = difflib.get_close_matches(key, list(_FORMULAS) + list(_FORMULA_ALIASES), n=1, cutoff=0.6)
        if not close:
            return {"found": False, "topic": topic, "available": sorted(_FORMULAS)}
        key = _FORMULA_ALIASES.get(close[0], close[0])
    formula, variables = _FORMULAS[key]
    return {"found": True, "topic": key, "formula": formula, "variables": variables}


# --- registration --------------------------------------------------------------------

def register(registry) -> None:
    registry.register_tool(
        "calculate",
        {
            "description": "Evaluate a math expression exactly. Supports + - * / // % ** "
            "parentheses, sqrt, cbrt, sin/cos/tan (radians), radians(), degrees(), log, "
            "log10, log2, exp, factorial, pi, e.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '37*48' or 'sqrt(144)'"}
                },
                "required": ["expression"],
            },
        },
        calculate,
    )
    registry.register_tool(
        "convert_units",
        {
            "description": "Convert a quantity between units of length, mass, time, speed, "
            "area, volume, energy, pressure or temperature (C/F/K).",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string", "description": "e.g. 'km', 'lb', 'C'"},
                    "to_unit": {"type": "string", "description": "e.g. 'mi', 'kg', 'F'"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
        convert_units,
    )
    registry.register_tool(
        "lookup_formula",
        {
            "description": "Look up a standard physics/maths/finance formula by topic name "
            "(English), e.g. 'kinetic energy', 'area of circle', 'ohm's law'.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
        lookup_formula,
    )
