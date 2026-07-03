"""Regression test for issue #33: CLI append arguments must not use list defaults.

An argparse argument with ``action="append"`` and a pre-populated list default
appends explicit CLI values onto the default instead of replacing it. That
silently double-loads input splits (observed as 15908 = 2 x 7954 rows in strict
robustness artifacts). Scripts must use ``default=None`` and apply the fallback
at the usage site instead.
"""

import ast
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def iter_add_argument_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            yield node


def test_no_append_argument_uses_list_default():
    offenders = []
    for script in sorted(SCRIPTS_DIR.glob("*.py")):
        tree = ast.parse(script.read_text(), filename=str(script))
        for call in iter_add_argument_calls(tree):
            keywords = {
                keyword.arg: keyword.value for keyword in call.keywords if keyword.arg
            }
            action = keywords.get("action")
            default = keywords.get("default")
            if (
                isinstance(action, ast.Constant)
                and action.value == "append"
                and isinstance(default, (ast.List, ast.Tuple))
            ):
                offenders.append(f"{script.name}:{call.lineno}")
    assert not offenders, (
        "append-action arguments with list defaults double-load explicit CLI "
        f"inputs; use default=None with a usage-site fallback: {offenders}"
    )
