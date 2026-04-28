"""BEM database readers (OrcaFlex YAML, WAMIT, Capytaine) — see CLAUDE.md §7.

All readers produce a :class:`floatsim.hydro.database.HydroDatabase`. The
:func:`load_hydro_database` dispatcher selects a concrete reader by the
``format`` argument; see ARCHITECTURE.md §8 M5 for the locked decision to
keep dispatch as a flat ``if/elif`` (Q6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.hydro.readers.orcaflex_vessel_yaml import read_orcaflex_vessel_yaml
from floatsim.hydro.readers.wamit import read_wamit

__all__ = [
    "BEMFormat",
    "load_hydro_database",
    "read_capytaine",
    "read_orcaflex_vessel_yaml",
    "read_wamit",
]

BEMFormat = Literal["orcaflex", "wamit", "capytaine"]


def load_hydro_database(
    path: str | Path,
    *,
    format: BEMFormat,
    **reader_kwargs: object,
) -> HydroDatabase:
    """Load a BEM database from disk in the requested format.

    Parameters
    ----------
    path
        Path to the BEM file (or stem, for the WAMIT reader).
    format
        One of ``"orcaflex"`` (VesselType YAML export of an OrcaWave run),
        ``"wamit"`` (WAMIT plain-text outputs ``.1`` + ``.3`` + ``.hst``),
        or ``"capytaine"`` (NetCDF written by ``capytaine.io.xarray``).
    **reader_kwargs
        Forwarded to the selected reader. See each reader's signature for
        accepted keywords.

    Returns
    -------
    HydroDatabase
        Validated single-body BEM database.

    Raises
    ------
    ValueError
        If ``format`` is not one of the supported formats.
    """
    if format == "orcaflex":
        return read_orcaflex_vessel_yaml(path, **reader_kwargs)  # type: ignore[arg-type]
    if format == "wamit":
        return read_wamit(path, **reader_kwargs)  # type: ignore[arg-type]
    if format == "capytaine":
        return read_capytaine(path, **reader_kwargs)  # type: ignore[arg-type]
    raise ValueError(
        f"Unknown BEM format {format!r}; expected one of " f"'orcaflex', 'wamit', 'capytaine'."
    )
