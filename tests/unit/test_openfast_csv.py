"""Unit tests for :mod:`tests.support.openfast_csv` (M6 PR1).

Round-trip: write a hand-authored canonical CSV + JSON sidecar in a
``tmp_path``, load it via :func:`load_openfast_history`, verify the
parsed structure matches what was written. Plus systematic
error-path coverage for every validation the loader performs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.support.openfast_csv import OpenFASTHistory, load_openfast_history


def _write_canonical_pair(
    tmp_path: Path,
    *,
    scenario_id: str = "s1_static_eq",
    n_samples: int = 11,
    dt_s: float = 0.05,
    extra_columns: dict[str, list[float]] | None = None,
    metadata_overrides: dict[str, object] | None = None,
) -> Path:
    """Author a canonical CSV + JSON sidecar in ``tmp_path``; return the CSV path."""
    csv_path = tmp_path / f"{scenario_id}.csv"
    json_path = tmp_path / f"{scenario_id}.json"

    times = np.arange(n_samples, dtype=np.float64) * dt_s
    surge = 0.1 * np.cos(2.0 * np.pi * times / 5.0)
    sway = 0.05 * np.sin(2.0 * np.pi * times / 5.0)
    heave = 0.5 * np.cos(2.0 * np.pi * times / 17.0)
    roll = 0.001 * np.cos(2.0 * np.pi * times / 22.0)
    pitch = 0.002 * np.cos(2.0 * np.pi * times / 27.0)
    yaw = np.zeros_like(times)

    headers = ["time_s", "surge_m", "sway_m", "heave_m", "roll_rad", "pitch_rad", "yaw_rad"]
    columns: list[np.ndarray] = [times, surge, sway, heave, roll, pitch, yaw]
    if extra_columns is not None:
        for name, values in extra_columns.items():
            assert len(values) == n_samples
            headers.append(name)
            columns.append(np.asarray(values, dtype=np.float64))

    data = np.column_stack(columns)
    csv_path.write_text(
        ",".join(headers)
        + "\n"
        + "\n".join(",".join(f"{v:.10e}" for v in row) for row in data)
        + "\n",
        encoding="utf-8",
    )

    metadata: dict[str, object] = {
        "scenario_id": scenario_id,
        "openfast_version": "v3.5.3",
        "dt_s": dt_s,
        "duration_s": (n_samples - 1) * dt_s,
        "unit_system": "SI_canonical",
        "extracted_by": "scripts/extract_openfast_fixtures.py (test fixture)",
        "source_inputs": [
            "tests/fixtures/openfast/oc4_deepcwind/inputs/OC4Semi.fst",
        ],
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)

    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return csv_path


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_round_trip_loads_required_columns(tmp_path: Path) -> None:
    csv_path = _write_canonical_pair(tmp_path, n_samples=21, dt_s=0.1)
    history = load_openfast_history(csv_path)

    assert isinstance(history, OpenFASTHistory)
    assert history.n_samples == 21
    assert history.t.shape == (21,)
    assert history.xi.shape == (21, 6)
    np.testing.assert_allclose(history.t[0], 0.0, atol=1e-15)
    np.testing.assert_allclose(history.t[-1], 2.0, rtol=1e-12)
    assert history.dt_s == pytest.approx(0.1, rel=1e-12)
    assert history.duration_s == pytest.approx(2.0, rel=1e-12)


def test_round_trip_extra_columns_round_trip(tmp_path: Path) -> None:
    """Tensions/extra columns flow through into ``extra_columns``."""
    extras = {
        "tension_line1_N": list(np.linspace(1.0e5, 1.5e5, 11)),
        "tension_line2_N": list(np.linspace(0.9e5, 1.4e5, 11)),
    }
    csv_path = _write_canonical_pair(tmp_path, extra_columns=extras, n_samples=11)
    history = load_openfast_history(csv_path)
    assert set(history.extra_columns.keys()) == {"tension_line1_N", "tension_line2_N"}
    np.testing.assert_allclose(
        history.extra_columns["tension_line1_N"], extras["tension_line1_N"], rtol=1e-9
    )


def test_round_trip_metadata_pass_through(tmp_path: Path) -> None:
    """Non-required metadata keys flow through into ``history.metadata``."""
    csv_path = _write_canonical_pair(
        tmp_path,
        metadata_overrides={
            "compfast_flags": {"CompElast": 1, "CompAero": 0},
            "notes": "no-wave free-decay run, IC heave 0.5 m",
        },
    )
    history = load_openfast_history(csv_path)
    assert history.metadata["openfast_version"] == "v3.5.3"
    assert history.metadata["compfast_flags"] == {"CompElast": 1, "CompAero": 0}
    assert "no-wave" in str(history.metadata["notes"])


# ---------------------------------------------------------------------------
# missing-file errors
# ---------------------------------------------------------------------------


def test_missing_csv_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="OpenFAST CSV not found"):
        load_openfast_history(tmp_path / "does_not_exist.csv")


def test_missing_json_sidecar_raises_filenotfound(tmp_path: Path) -> None:
    csv_path = _write_canonical_pair(tmp_path)
    csv_path.with_suffix(".json").unlink()
    with pytest.raises(FileNotFoundError, match="JSON sidecar not found"):
        load_openfast_history(csv_path)


# ---------------------------------------------------------------------------
# JSON-sidecar validation
# ---------------------------------------------------------------------------


def test_json_missing_required_key_raises(tmp_path: Path) -> None:
    csv_path = _write_canonical_pair(tmp_path)
    json_path = csv_path.with_suffix(".json")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    raw.pop("openfast_version")
    json_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing required keys.*openfast_version"):
        load_openfast_history(csv_path)


def test_json_scenario_id_mismatch_raises(tmp_path: Path) -> None:
    csv_path = _write_canonical_pair(tmp_path, scenario_id="s2_free_decay")
    json_path = csv_path.with_suffix(".json")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    raw["scenario_id"] = "s99_typo"
    json_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=r"scenario_id=.*does not match the CSV stem"):
        load_openfast_history(csv_path)


def test_json_non_canonical_unit_system_raises(tmp_path: Path) -> None:
    csv_path = _write_canonical_pair(
        tmp_path, metadata_overrides={"unit_system": "openfast_native"}
    )
    with pytest.raises(ValueError, match=r"unit_system=.*requires 'SI_canonical'"):
        load_openfast_history(csv_path)


def test_json_dt_must_be_positive(tmp_path: Path) -> None:
    csv_path = _write_canonical_pair(tmp_path, metadata_overrides={"dt_s": -0.05})
    with pytest.raises(ValueError, match=r"dt_s must be a positive number"):
        load_openfast_history(csv_path)


def test_json_source_inputs_must_be_list(tmp_path: Path) -> None:
    csv_path = _write_canonical_pair(tmp_path, metadata_overrides={"source_inputs": "single"})
    with pytest.raises(ValueError, match=r"source_inputs must be a list"):
        load_openfast_history(csv_path)


# ---------------------------------------------------------------------------
# CSV-shape and column validation
# ---------------------------------------------------------------------------


def test_csv_missing_required_column_raises(tmp_path: Path) -> None:
    csv_path = _write_canonical_pair(tmp_path)
    # Strip the heave_m column from the CSV, keep the JSON intact.
    text = csv_path.read_text(encoding="utf-8").splitlines()
    headers = text[0].split(",")
    heave_idx = headers.index("heave_m")
    new_headers = [h for i, h in enumerate(headers) if i != heave_idx]
    new_rows = []
    for row in text[1:]:
        values = row.split(",")
        new_rows.append(",".join(v for i, v in enumerate(values) if i != heave_idx))
    csv_path.write_text(",".join(new_headers) + "\n" + "\n".join(new_rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing required columns.*heave_m"):
        load_openfast_history(csv_path)


def test_csv_duplicate_column_raises(tmp_path: Path) -> None:
    csv_path = _write_canonical_pair(tmp_path)
    text = csv_path.read_text(encoding="utf-8").splitlines()
    # Insert a duplicated heave_m column.
    headers = text[0].split(",")
    headers.append("heave_m")
    new_rows = [row + "," + row.split(",")[3] for row in text[1:]]
    csv_path.write_text(",".join(headers) + "\n" + "\n".join(new_rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"duplicate column names"):
        load_openfast_history(csv_path)


def test_csv_non_monotonic_time_raises(tmp_path: Path) -> None:
    csv_path = _write_canonical_pair(tmp_path)
    text = csv_path.read_text(encoding="utf-8").splitlines()
    # Swap rows 3 and 4 to break monotonicity in time_s.
    swapped = [*text[:3], text[4], text[3], *text[5:]]
    csv_path.write_text("\n".join(swapped) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"strictly monotonically increasing"):
        load_openfast_history(csv_path)


def test_csv_dt_disagreement_with_metadata_raises(tmp_path: Path) -> None:
    """If the JSON's dt_s claims 0.05 but the CSV is sampled at 0.1, fail loudly."""
    # Generate a CSV at dt=0.1 but write metadata claiming dt=0.05.
    csv_path = _write_canonical_pair(tmp_path, dt_s=0.1, metadata_overrides={"dt_s": 0.05})
    with pytest.raises(ValueError, match=r"observed mean dt_s.*disagrees with sidecar"):
        load_openfast_history(csv_path)


def test_csv_too_few_rows_raises(tmp_path: Path) -> None:
    csv_path = _write_canonical_pair(tmp_path, n_samples=2, dt_s=0.05)
    # Strip down to 1 data row.
    text = csv_path.read_text(encoding="utf-8").splitlines()
    csv_path.write_text("\n".join(text[:2]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"only 1 data rows; need at least 2"):
        load_openfast_history(csv_path)
