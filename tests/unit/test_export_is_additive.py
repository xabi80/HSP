"""The FloatFEA export must remain unreachable from any solve path.

FloatFEA gate **G1.5** requires HSP's regression suite to be bit-identical with
and without the export work. The export modules satisfy that *structurally*:
nothing in ``floatsim/`` imports them, so they cannot execute during a solve and
cannot move a number. That is stronger evidence than a pass count, which shows
only that no assertion moved outside tolerance.

But a structural argument that is merely true is not a guard. **A convenience
import added six months from now would break G1.5 silently**, and no pass count
would reveal it -- the suite would stay green while the property it rests on had
quietly gone. So the argument is asserted here.

Recorded in ``docs/hsp-coupling.md`` as the reason the "single permitted touch on
the solve loop" was never needed; this test is what keeps the document true.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_FLOATSIM = Path(__file__).resolve().parents[2] / "floatsim"

# Modules that exist only to serve the FloatFEA interchange. Nothing that runs
# during a solve may reach them.
_EXPORT_MODULES: frozenset[str] = frozenset({"flr_export", "flr_strips"})


def _imported_names(source: str) -> set[str]:
    """Every module name an ``import`` or ``from ... import`` statement names."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def _solve_path_modules() -> list[Path]:
    """Every module in the package except the export modules themselves."""
    return [
        p
        for p in sorted(_FLOATSIM.rglob("*.py"))
        if p.stem not in _EXPORT_MODULES and "__pycache__" not in p.parts
    ]


def test_no_floatsim_module_imports_the_export() -> None:
    """The structural basis of G1.5, asserted rather than assumed."""
    offenders: list[str] = []
    for path in _solve_path_modules():
        imported = _imported_names(path.read_text(encoding="utf-8"))
        for name in imported:
            tail = name.rsplit(".", maxsplit=1)[-1]
            parts = set(name.split("."))
            if tail in _EXPORT_MODULES or parts & _EXPORT_MODULES:
                offenders.append(f"{path.relative_to(_FLOATSIM.parent)} imports {name}")
    assert not offenders, (
        "the FloatFEA export is reachable from a solve path, which breaks the "
        "structural basis of G1.5 (docs/hsp-coupling.md):\n  " + "\n  ".join(offenders)
    )


def test_the_export_modules_are_actually_present() -> None:
    """Guard against the previous test passing because the modules were renamed.

    A test that checks 'nothing imports X' passes trivially when X no longer
    exists. If these move, the import test above must be updated with them.
    """
    found = {p.stem for p in _FLOATSIM.rglob("*.py")} & _EXPORT_MODULES
    assert found == _EXPORT_MODULES, (
        f"export modules missing or renamed: expected {sorted(_EXPORT_MODULES)}, "
        f"found {sorted(found)}. Update _EXPORT_MODULES so the G1.5 guard keeps "
        "checking the right thing."
    )


@pytest.mark.parametrize("module", sorted(_EXPORT_MODULES))
def test_export_modules_do_not_touch_the_solver_at_import_time(module: str) -> None:
    """They may *read* solver types, but must not run anything on import.

    ``flr_export`` imports ``_generalized_alpha_coefficients`` -- a pure
    function of ``rho_inf`` -- and the retardation types. That is reading, not
    executing, and it is what lets the record be self-describing.
    """
    path = next(_FLOATSIM.rglob(f"{module}.py"))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_level_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert not module_level_calls, (
        f"{module} executes a call at import time; the export must be inert "
        "until explicitly invoked."
    )
