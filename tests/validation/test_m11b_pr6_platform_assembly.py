"""M11b PR6 -- 12-buoy platform mesh generator + 17-body assembly.

Promotes the M11b Phase-1 measurements (STEP 2 mesh, STEP 4 assembly) to
PERMANENT gates, so PR6's de-risking is asserted, not re-measured ad hoc:

- the mesh generator builds 17,856 panels at the re-derived platform draft with
  0 inward panels and the expected 0.620 m closest cross-cluster pair;
- the 17-body / 16-joint platform assembles through the real ``build_system``
  path with all M10-PR1 preconditions (rank(M+A_inf)=102, rank(G)=64,
  n_constraints=64, free=38, phi(rest)=0, mass=402.04, refs threaded).

The assembly uses SYNTHETIC hydro (the committed 18-DOF cluster database tiled
x4 -> 72-DOF), since the real 12-buoy BEM is PR7. The checks are structural and
independent of the hydro values.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
_PLAT = REPO / "studies" / "platform-12buoy"
sys.path.insert(0, str(_PLAT))

import platform_common as pc  # noqa: E402

from floatsim.driver import build_system  # noqa: E402
from floatsim.hydro.database import HydroDatabase  # noqa: E402
from floatsim.hydro.readers.capytaine import read_capytaine  # noqa: E402
from floatsim.io.deck import HydroDatabaseRef  # noqa: E402

_NC = REPO / "studies" / "cluster-3buoy-rigid" / "capytaine_multibody_18dof.nc"
_REF = REPO / "studies" / "cluster-3buoy-rigid" / "reference_single_bem.nc"


def _tiled_hdb() -> HydroDatabase:
    """18-DOF (3 buoys) tiled block-diagonal x4 -> 72-DOF (buoy1..12)."""
    h = read_capytaine(_NC)
    w = np.asarray(h.omega)
    keep = np.array(
        [
            k
            for k in range(w.size)
            if k not in {int(np.argmin(np.abs(w - c))) for c in (4.934, 20.909)}
        ]
    )
    A18, B18, Ai18, C18 = h.A[:, :, keep], h.B[:, :, keep], h.A_inf, h.C
    RAO18 = h.RAO[:, keep, :]
    nw = keep.size
    A = np.zeros((72, 72, nw), dtype=A18.dtype)
    B = np.zeros((72, 72, nw), dtype=B18.dtype)
    Ai = np.zeros((72, 72))
    C = np.zeros((72, 72))
    RAO = np.zeros((72, nw, RAO18.shape[2]), dtype=RAO18.dtype)
    for c in range(4):
        s = slice(18 * c, 18 * c + 18)
        A[s, s, :] = A18
        B[s, s, :] = B18
        Ai[s, s] = Ai18
        C[s, s] = C18
        RAO[s, :, :] = RAO18
    return HydroDatabase(
        omega=h.omega[keep],
        heading_deg=h.heading_deg,
        A=A,
        B=B,
        A_inf=Ai,
        C=C,
        RAO=RAO,
        reference_point=h.reference_point,
        C_source=h.C_source,
        metadata=dict(h.metadata),
        body_labels=tuple(f"buoy{i + 1}" for i in range(12)),
    )


# ---------------------------------------------------------------------------
# Draft + geometry (fast)
# ---------------------------------------------------------------------------


def test_draft_and_geometry() -> None:
    """Platform draft re-derived on the mesh matches the cached PLATFORM_DZ, is
    deeper than the cluster DZ2, and the mass balance / closest pair hold."""
    assert pc.M_TOTAL == pytest.approx(402.04, abs=0.01)
    assert pc.M_PER_BUOY == pytest.approx(33.5033, abs=1e-3)
    dz = pc.derive_draft()
    assert dz == pytest.approx(pc.PLATFORM_DZ, rel=1e-3)
    assert dz > pc.cc.DZ2  # deeper than the cluster (carries more per-buoy mass)
    assert pc.closest_cross_cluster_gap() == pytest.approx(0.620, abs=0.001)  # geom §3.6


# ---------------------------------------------------------------------------
# Mesh generator (slow: mesh_hygiene on 17,856 panels)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mesh_build_and_hygiene() -> None:
    """The generator builds 17,856 panels with 0 inward (copies of a validated
    hull cannot flip winding) and 1152 = 12x96 open edges."""
    import build_platform_mesh as bpm

    from floatsim.hydro.mesh_hygiene import validate_panel_normals

    mesh = bpm.build()
    assert mesh.n_panels == 17856  # 12 x 1488
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = validate_panel_normals(mesh, return_report=True)
    assert report.inward_indices.size == 0
    assert report.indeterminate_indices.size == 0
    assert report.n_open_edges == 1152


# ---------------------------------------------------------------------------
# Assembly preconditions (slow: build_system builds the 72-DOF kernel)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_assembly_preconditions() -> None:
    """The 17-body / 16-joint platform assembles at n=102 with every M10-PR1
    precondition (STEP 4 promoted to a permanent gate)."""
    deck = pc.build_platform_deck(
        HydroDatabaseRef(format="capytaine", path=str(_NC)),
        HydroDatabaseRef(format="capytaine", path=str(_REF)),
    )
    assert len(deck.bodies) == 17
    assert len(deck.joints) == 16
    assert sum(b.mass for b in deck.bodies) == pytest.approx(402.04, abs=0.01)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        setup = build_system(
            deck,
            bem_databases={},
            dt=0.01,
            t_max_kernel=30.0,
            solve_equilibrium=False,
            shared_hydro_database=_tiled_hdb(),
            hydrostatic_database=read_capytaine(_REF),
            asymptote_check_override="M11b PR6 platform assembly (small-body hulls)",
        )

    ma = np.asarray(setup.lhs.M_plus_Ainf)
    assert ma.shape == (102, 102)
    assert np.linalg.matrix_rank(ma) == 102
    assert len(setup.body_name_to_index) == 17  # refs threaded (PR0.75)

    js = setup.constraints
    assert js is not None
    assert js.n_constraints == 64  # 16 yaw_locked x 4 rows
    g = js.jacobian(np.zeros(102))
    assert g.shape == (64, 102)
    assert np.linalg.matrix_rank(g) == 64
    assert 102 - js.n_constraints == 38  # free DOF
    assert np.max(np.abs(js.phi(np.zeros(102)))) < 1e-9  # phi(rest) = 0
