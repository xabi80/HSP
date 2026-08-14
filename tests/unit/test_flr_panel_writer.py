"""The panel writer stores COEFFICIENTS and refuses an unjustified bound.

Coefficient storage is a decision with arithmetic behind it (schema §5.0.2), and
it changes what the record must carry: because the reader reconstructs the time
domain from what is stored, the reader applies the time convention. These tests
pin the metadata that makes that safe.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from floatsim.io.flr_panels import (  # noqa: E402
    TIME_CONVENTION,
    PanelGeometry,
    write_panel_fields,
)

_P, _NW, _NDOF = 5, 3, 12


def _geom() -> PanelGeometry:
    return PanelGeometry(
        centroid=np.zeros((_P, 3)),
        area=np.ones(_P),
        normal=np.tile([0.0, 0.0, 1.0], (_P, 1)),
    )


def _write(tmp_path, **over):
    kw = dict(
        body="buoy1",
        geometry=_geom(),
        omega=np.linspace(1.0, 2.0, _NW),
        froude_krylov=np.ones((_NW, _P), dtype=complex),
        diffraction=np.ones((_NW, _P), dtype=complex) * 2,
        radiation=np.ones((_NW, _NDOF, _P), dtype=complex) * 3,
        reference_pose=np.zeros(7),
        body_pose=np.zeros((4, 7)),
        validity_bound=1.5,
        validity_bound_basis="0.18 x spar diameter; linear-theory small-motion assumption",
    )
    kw.update(over)
    path = tmp_path / "p.h5"
    with h5py.File(path, "w") as h:
        write_panel_fields(h, **kw)
    return path


def test_round_trip_preserves_the_coefficients(tmp_path) -> None:
    path = _write(tmp_path)
    with h5py.File(path, "r") as h:
        g = h["panels/buoy1"]
        assert g["radiation"].shape == (_NW, _NDOF, _P)
        assert g.attrs["storage"] == "complex_coefficients"
        assert g.attrs["time_convention"] == TIME_CONVENTION
        assert g.attrs["n_radiating_dof"] == _NDOF
        np.testing.assert_allclose(g["diffraction"][:], 2.0)


def test_an_unjustified_validity_bound_is_refused(tmp_path) -> None:
    """A threshold whose reasoning is absent is a number nobody can re-check."""
    with pytest.raises(ValueError, match="re-check"):
        _write(tmp_path, validity_bound_basis="   ")


def test_an_unknown_time_convention_is_refused(tmp_path) -> None:
    """With coefficients stored, the reader applies the convention; a wrong one
    is a 180 degree phase error on every damping term."""
    with pytest.raises(ValueError, match="time_convention"):
        _write(tmp_path, time_convention="whatever")


def test_radiation_must_carry_the_full_radiating_dof_dimension(tmp_path) -> None:
    """The database is a multi-body solve; a per-body 6-DOF array is wrong."""
    with pytest.raises(ValueError, match="n_radiating_dof"):
        _write(tmp_path, radiation=np.ones((_NW, _P), dtype=complex))
