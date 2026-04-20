"""HDF5 writer/reader round-trip — Milestone 0 stub.

We only exercise the plumbing: open/close, write run metadata, write a single
numeric dataset, read it back. The time-series schema (per-channel groups,
sampling metadata, etc.) is deferred to later milestones when there is actual
physics producing data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from floatsim.io.results import (
    RunMetadata,
    close_results,
    open_results,
    read_dataset,
    read_run_metadata,
    write_dataset,
    write_run_metadata,
)


def _sample_metadata() -> RunMetadata:
    return RunMetadata(
        deck_hash="sha256:deadbeef",
        floatsim_version="0.0.0",
        created_at="2026-04-19T00:00:00Z",
    )


def test_metadata_and_dataset_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "results.h5"

    handle = open_results(path, mode="w")
    write_run_metadata(handle, _sample_metadata())
    arr = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    write_dataset(handle, "time", arr)
    close_results(handle)

    assert path.exists()

    handle = open_results(path, mode="r")
    try:
        meta = read_run_metadata(handle)
        back = read_dataset(handle, "time")
    finally:
        close_results(handle)

    assert meta == _sample_metadata()
    np.testing.assert_allclose(back, arr, rtol=1e-12, atol=1e-12)


def test_writing_to_readonly_handle_raises(tmp_path: Path) -> None:
    path = tmp_path / "results.h5"
    # Create an empty valid file first.
    handle = open_results(path, mode="w")
    close_results(handle)

    handle = open_results(path, mode="r")
    try:
        with pytest.raises(Exception):  # noqa: B017 -- HDF5 raises its own type
            write_dataset(handle, "time", np.zeros(3))
    finally:
        close_results(handle)


def test_read_missing_dataset_raises(tmp_path: Path) -> None:
    path = tmp_path / "results.h5"
    handle = open_results(path, mode="w")
    close_results(handle)

    handle = open_results(path, mode="r")
    try:
        with pytest.raises(KeyError):
            read_dataset(handle, "does_not_exist")
    finally:
        close_results(handle)


def test_metadata_missing_raises(tmp_path: Path) -> None:
    path = tmp_path / "results.h5"
    handle = open_results(path, mode="w")
    close_results(handle)

    handle = open_results(path, mode="r")
    try:
        with pytest.raises(KeyError):
            read_run_metadata(handle)
    finally:
        close_results(handle)
