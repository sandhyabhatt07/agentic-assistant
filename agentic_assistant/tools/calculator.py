"""Safe arithmetic evaluator (an AST whitelist, never ``eval``)."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from .base import Tool, ToolError

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_MAX_EXPONENT = 1000  # keeps "9 ** 9 ** 9" from hanging the process
_MAX_EXPRESSION_LENGTH = 500


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ToolError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*[_eval_node(arg) for arg in node.args])
    raise ToolError("unsupported expression")


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return repr(round(value, 10))
    return str(value)


def calculate(expression: str) -> str:
    """Evaluate ``expression`` and return the result as text; raises ``ToolError`` on bad input."""
    expression = expression.strip()
    if not expression:
        raise ToolError("empty expression")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
        return _format_number(_eval_node(tree))
    except ZeroDivisionError:
        raise ToolError("division by zero") from None
    except (SyntaxError, ValueError, TypeError, OverflowError) as exc:
        raise ToolError(f"could not evaluate '{expression}' ({exc})") from None


calculator_tool = Tool(
    name="calculator",
    description=(
        "Evaluates mathematical expressions. Supports +, -, *, /, //, %, **, parentheses, "
        "sqrt, abs, round, sin, cos, tan, log, exp, pi, e. Input must be a plain expression, e.g. '5 + 3 * 2'."
    ),
    func=calculate,
)
