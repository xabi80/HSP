"""Every declared subpackage is importable — guards against layout drift (ARCHITECTURE.md §4)."""

import importlib


def test_top_level_version_is_defined() -> None:
    import floatsim

    assert isinstance(floatsim.__version__, str)
    assert floatsim.__version__  # non-empty


def test_all_declared_subpackages_import() -> None:
    expected = [
        "floatsim.bodies",
        "floatsim.hydro",
        "floatsim.hydro.readers",
        "floatsim.waves",
        "floatsim.mooring",
        "floatsim.solver",
        "floatsim.io",
        "floatsim.post",
        "floatsim.validation",
    ]
    for name in expected:
        importlib.import_module(name)
