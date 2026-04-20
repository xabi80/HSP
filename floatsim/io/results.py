"""HDF5 writer/reader stub for FloatSim simulation outputs.

Milestone 0 scope: plumbing only. This module exposes a minimal, explicit API
for opening and closing an HDF5 file, attaching run-level metadata (deck
hash, FloatSim version, timestamp) as file attributes, and reading/writing
named numeric datasets at the file root. The per-channel time-series schema
is intentionally not committed yet — it will be defined when the solver
actually produces data.

All functions take a live ``h5py.File`` handle rather than a path. This keeps
write patterns explicit (callers must close the handle) and lets tests
reason about lifecycle without hidden state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from numpy.typing import NDArray

_METADATA_KEYS = ("deck_hash", "floatsim_version", "created_at")


@dataclass(frozen=True)
class RunMetadata:
    """Minimum metadata stamped onto every results file.

    Attributes
    ----------
    deck_hash
        Content hash of the deck that produced this run (e.g. ``"sha256:..."``).
        Used downstream to detect result/deck drift.
    floatsim_version
        The ``floatsim.__version__`` at write time.
    created_at
        ISO-8601 UTC timestamp (``"YYYY-MM-DDTHH:MM:SSZ"``).
    """

    deck_hash: str
    floatsim_version: str
    created_at: str


def open_results(path: str | Path, mode: Literal["r", "w", "a"] = "r") -> h5py.File:
    """Open an HDF5 results file.

    Parameters
    ----------
    path
        Filesystem path. Parent directory must exist.
    mode
        One of ``"r"`` (read-only), ``"w"`` (truncate), ``"a"`` (append).

    Returns
    -------
    h5py.File
        Open file handle. Caller is responsible for closing it with
        :func:`close_results`.
    """
    return h5py.File(str(path), mode)


def close_results(handle: h5py.File) -> None:
    """Close an HDF5 file handle opened via :func:`open_results`."""
    handle.close()


def write_run_metadata(handle: h5py.File, meta: RunMetadata) -> None:
    """Stamp run metadata onto the file as root attributes."""
    for key, value in asdict(meta).items():
        handle.attrs[key] = value


def read_run_metadata(handle: h5py.File) -> RunMetadata:
    """Read run metadata written by :func:`write_run_metadata`.

    Raises
    ------
    KeyError
        If any of the required metadata attributes are missing.
    """
    for key in _METADATA_KEYS:
        if key not in handle.attrs:
            raise KeyError(f"missing run metadata attribute: {key!r}")
    values = {key: str(handle.attrs[key]) for key in _METADATA_KEYS}
    return RunMetadata(**values)


def write_dataset(handle: h5py.File, name: str, data: NDArray[np.floating]) -> None:
    """Write a floating-point array as a dataset at the file root.

    Overwrites any existing dataset of the same name. Storage dtype matches
    the input array (no silent downcasting).
    """
    if name in handle:
        del handle[name]
    handle.create_dataset(name, data=np.asarray(data))


def read_dataset(handle: h5py.File, name: str) -> NDArray[np.floating]:
    """Read a floating-point dataset from the file root.

    Raises
    ------
    KeyError
        If the named dataset does not exist.
    """
    if name not in handle:
        raise KeyError(f"dataset not found: {name!r}")
    node = handle[name]
    assert isinstance(node, h5py.Dataset)
    return np.asarray(node[()])
