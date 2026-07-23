"""M8 PR4 — the two terminal condensation gates (plan Q4/Q5, Step C).

Both gates condense the committed 18-DOF coupled BEM fixture through the
rigid-body map ``T`` (built BY LABEL per the Q5 contract,
``tests/support/condensation.py``) and compare against the committed
composite (rigid single-body) results of the cluster study.

HONESTY CLAUSE (plan Q4 — verbatim in both gate docstrings): these
gates validate the ingestion and assembly path, not the underlying BEM
physics — the two models share an influence matrix, so agreement is a
linear-algebra identity. Independent validation of coupled
hydrodynamics does not exist in this program before M10.

Fixtures (all committed):
- ``studies/cluster-3buoy-rigid/capytaine_multibody_18dof.nc`` — the
  coupled 18-DOF database, production grid geomspace(0.1, 30, 80).
  Retained UNMODIFIED including the contaminated frequency slice at
  omega~4.934 (tracker BEM-CONTAMINATED-FREQUENCY-SLICE-CLUSTER-DRAFT).
- ``studies/cluster-3buoy-rigid/composite_bem.nc`` — the rigid cluster
  solved as ONE 6-DOF body, SAME grid (widened 40 -> 80 at PR4
  precisely so both sides sit on identical grids BY CONSTRUCTION — the
  Q4 lock; no interpolation path exists in these tests).
- ``studies/cluster-3buoy-rigid/reference_single_bem.nc`` — single hull
  at cluster draft; supplies the per-hull 6x6 hydrostatic stiffness
  (the 18-DOF fixture carries no hydrostatic block — see the PR4 Step-0
  determination in the closure doc).

Locked constants below are cited to their committed sources; the
standing rule is re-derive-from-the-record, never carry from
conversation.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import find_peaks

from floatsim.bodies.mass_properties import rigid_body_mass_matrix
from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.radiation import assemble_cummins_lhs
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.hydro.retardation import compute_retardation_kernel
from floatsim.solver.equilibrium import solve_static_equilibrium
from floatsim.solver.newmark import integrate_cummins
from tests.support.condensation import build_rigid_condensation_map

_STUDY = Path(__file__).resolve().parents[2] / "studies" / "cluster-3buoy-rigid"
_NC_18DOF = _STUDY / "capytaine_multibody_18dof.nc"
_NC_COMPOSITE = _STUDY / "composite_bem.nc"
_NC_REF_SINGLE = _STUDY / "reference_single_bem.nc"

# --- Locked cluster geometry / mass (studies/cluster-3buoy-rigid/
#     cluster_common.py + results/mass_properties.json, committed at the
#     cluster-study closeout) ---
_CLUSTER_RADIUS = 0.5  # m; hulls at angles 0/120/240 deg (cluster_common.py)
_BUOY_ANGLES_DEG = (0.0, 120.0, 240.0)
# Reference points at FULL committed precision (results/
# mass_properties.json — the composite BEM was solved about cog_m, so a
# hand-truncated CoG here shows up as a spurious ~1e-4 moment-arm error
# on pitch; measured during PR4 Phase A and fixed by loading the json).
_MASS_PROPS = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "studies"
        / "cluster-3buoy-rigid"
        / "results"
        / "mass_properties.json"
    ).read_text()
)
_HULL_COG_Z = float(_MASS_PROPS["z_buoy_cog"])  # -1.1956674320...
_CLUSTER_COG = np.array(_MASS_PROPS["cog_m"])  # z = -0.9888676769...
_M_CLUSTER = 98.01  # kg = 3 x 28.67 (hulls) + 12.0 (arms), cluster_common.py
_I_HULL = np.diag([24.0, 24.0, 0.114])  # kg m^2 about the hull CoM
# Reference decay period: rigid cluster study, WITH hydrodynamic
# interaction (results/interaction.json T_n_with_interaction = 3.10609;
# FloatSim composite run measured 3.1067 via peaks / 3.10533 via
# zero-crossings — all within the 1e-2 gate band around 3.106).
_T_N_REFERENCE_S = 3.106
_A33_COMPOSITE_INF = 64.0738  # kg, interaction.json A33_composite_inf
_C33_COMPOSITE = 663.2420  # N/m, interaction.json C33_composite

# Decay setup mirrors the composite study run exactly
# (cluster_study_common.py: DT/DURATION/KERNEL_TMAX/IC_HEAVE).
_DT = 0.01
_DURATION_S = 50.0
_KERNEL_TMAX_S = 30.0
_IC_HEAVE_M = 0.10
_G = 9.81

_OVERRIDE = (
    "M8 PR4 condensation gate: small-body cluster hulls (L~1.85 m); "
    "1/omega^4 regime not reached at omega_max=30; see "
    "ITEM25-SMALL-BODY-APPLICABILITY"
)

# The two contaminated frequencies EXCLUDED from the decay gate's grid
# (grid SELECTION at test level, NOT a value modification — the fixture
# on disk is unmodified; the PR3 negative gate reads it whole). The
# EXCITATION gate deliberately KEEPS omega=4.934 — see its docstring.
_CONTAMINATED_OMEGA = (4.934, 20.909)

# Excitation-gate magnitude floor (Phase-1 MA convention, plan risk
# register "noise/noise" row): relative error is only meaningful where
# the excitation is physically nonzero. The gate compares the
# PHYSICALLY-EXCITED DOFs only (surge/heave/pitch at beta=0; the
# y-mirror symmetry forbids sway/roll/yaw a priori), with the floor
# applied on top.
_EXC_FLOOR_REL = 1.0e-6
_EXCITED_DOF = (0, 2, 4)  # surge, heave, pitch at beta = 0
# Identity tolerances, TWO-TIER (measured at PR4 Phase A, margins ~7x):
# the two models share panels and influence matrix, so T^T F_18 =
# F_comp is a linear-algebra identity to solver round-off — but the
# round-off is CONDITIONING-DEPENDENT. Below Capytaine's
# mesh-resolution flag (omega ~ 15.665, "largest panel radius >
# wavelength/8" solver warning) the identity is machine-grade
# (measured worst 1.46e-5 magnitude / 0.0005 deg phase); above it the
# ill-conditioned shared operator amplifies the two solves' different
# RHS noise differently (measured worst 8.7e-4 / 0.015 deg).
_MESH_RESOLUTION_BAND_RAD_S = 15.665
_EXC_RTOL_RESOLVED = 1.0e-4
_EXC_PHASE_DEG_RESOLVED = 0.01
_EXC_RTOL_FLAGGED = 5.0e-3
_EXC_PHASE_DEG_FLAGGED = 0.1


def _body_reference_points() -> dict[str, np.ndarray]:
    ang = np.deg2rad(_BUOY_ANGLES_DEG)
    return {
        f"buoy{i + 1}": np.array(
            [_CLUSTER_RADIUS * np.cos(a), _CLUSTER_RADIUS * np.sin(a), _HULL_COG_Z]
        )
        for i, a in enumerate(ang)
    }


def _load_18dof_excluding_contaminated() -> HydroDatabase:
    """The committed 18-DOF fixture minus the two contaminated slices.

    Mirrors ``tests/unit/test_retardation_kernel.py``'s PR3 positive-gate
    helper: a grid selection, not a value edit.
    """
    hdb = read_capytaine(_NC_18DOF)
    w = np.asarray(hdb.omega)
    drop = {int(np.argmin(np.abs(w - wc))) for wc in _CONTAMINATED_OMEGA}
    keep = np.array([k for k in range(w.size) if k not in drop])
    return HydroDatabase(
        omega=hdb.omega[keep],
        heading_deg=hdb.heading_deg,
        A=hdb.A[:, :, keep],
        B=hdb.B[:, :, keep],
        A_inf=hdb.A_inf,
        C=hdb.C,
        RAO=hdb.RAO[:, keep, :],
        reference_point=hdb.reference_point,
        C_source=hdb.C_source,
        metadata=dict(hdb.metadata),
        body_labels=hdb.body_labels,
    )


# ---------------------------------------------------------------------------
# Step A — the Q5 label contract raise paths (permanent)
# ---------------------------------------------------------------------------


def test_condensation_map_label_mismatch_raises() -> None:
    """Q5 label contract: a missing or unknown label must raise, never
    fall back to positional mapping."""
    pts = _body_reference_points()
    labels = ("buoy1", "buoy2", "buoy3")
    missing = dict(pts)
    del missing["buoy3"]
    with pytest.raises(ValueError, match=r"missing.*buoy3"):
        build_rigid_condensation_map(labels, missing, _CLUSTER_COG)
    extra = dict(pts)
    extra["buoy4"] = np.zeros(3)
    with pytest.raises(ValueError, match=r"unknown.*buoy4"):
        build_rigid_condensation_map(labels, extra, _CLUSTER_COG)


def test_condensation_map_duplicate_label_raises() -> None:
    """Q5 label contract: duplicate labels make the label->block map
    ambiguous and must raise."""
    pts = _body_reference_points()
    with pytest.raises(ValueError, match=r"duplicate.*buoy1"):
        build_rigid_condensation_map(("buoy1", "buoy1", "buoy3"), pts, _CLUSTER_COG)


def test_condensation_map_is_label_keyed_not_positional() -> None:
    """Supplying the SAME positions dict regardless of iteration order
    yields the same T — the map follows body_labels, not dict order."""
    hdb = read_capytaine(_NC_18DOF)
    pts = _body_reference_points()
    t_fwd = build_rigid_condensation_map(hdb.body_labels, pts, _CLUSTER_COG)
    shuffled = {k: pts[k] for k in reversed(list(pts))}
    t_rev = build_rigid_condensation_map(hdb.body_labels, shuffled, _CLUSTER_COG)
    np.testing.assert_array_equal(t_fwd, t_rev)
    assert t_fwd.shape == (18, 6)


# ---------------------------------------------------------------------------
# Step B/C — DECAY GATE (permanent)
# ---------------------------------------------------------------------------


def test_decay_gate_condensed_18dof_reproduces_cluster_period() -> None:
    """DECAY GATE: the coupled 18-DOF database, condensed through T
    (``T^T M T``, ``T^T A(w) T``, ``T^T B(w) T``, ``T^T C T``),
    reproduces the rigid cluster study's heave decay period
    T_n = 3.106 s at rtol 1e-2.

    This gate validates the ingestion and assembly path, not the
    underlying BEM physics — the two models share an influence matrix,
    so agreement is a linear-algebra identity. Independent validation
    of coupled hydrodynamics does not exist in this program before M10.

    Grid: the two contaminated frequencies (4.934, 20.909) are EXCLUDED
    at test level (grid selection, not value modification) — a
    contaminated slice would corrupt the retardation kernel (PR3
    finding). The fixture on disk is unmodified.

    Mass model: per-hull blocks carry M_CLUSTER/3 = 32.67 kg (hull
    28.67 kg + one 4 kg arm lumped at the hull reference) and the hull
    inertia about its own CoM. Heave is EXACT by construction
    (sum of masses = 98.01 kg = the composite study's mass);
    the rotational blocks lump the arm mass at the hull CoG and are
    approximate — they are NOT exercised by this heave-only gate,
    mirroring the composite study's own declaration
    (cluster_study_common.py: "heave is reference-independent...
    pitch/roll are not exercised").

    Hydrostatics: the 18-DOF fixture carries no hydrostatic block, so
    C_18 is assembled block-diagonally from the committed single-hull
    hydrostatic (reference_single_bem.nc, C33 = 221.0807 N/m at cluster
    draft); T^T C_18 T reproduces the composite C33 = 663.2420 N/m
    exactly (asserted).
    """
    hdb18 = _load_18dof_excluding_contaminated()
    ref = read_capytaine(_NC_REF_SINGLE)
    t_map = build_rigid_condensation_map(hdb18.body_labels, _body_reference_points(), _CLUSTER_COG)

    # --- condense the coupled assembly ---
    m_hull = rigid_body_mass_matrix(
        mass=_M_CLUSTER / 3.0, inertia_at_reference=_I_HULL, cog_offset_body=None
    )
    m_18 = np.kron(np.eye(3), m_hull)
    c_single = np.asarray(ref.C, dtype=np.float64)
    c_18 = np.kron(np.eye(3), c_single)

    m_c = t_map.T @ m_18 @ t_map
    c_c = t_map.T @ c_18 @ t_map
    a_c = np.einsum("ip,ijw,jq->pqw", t_map, np.asarray(hdb18.A), t_map)
    b_c = np.einsum("ip,ijw,jq->pqw", t_map, np.asarray(hdb18.B), t_map)
    a_inf_c = t_map.T @ np.asarray(hdb18.A_inf) @ t_map
    rao_c = np.einsum("ip,iwh->pwh", t_map, np.asarray(hdb18.RAO))

    # Condensation pins against the committed study record.
    assert m_c[2, 2] == pytest.approx(_M_CLUSTER, rel=1e-12)  # heave mass exact
    # T^T C_18 T heave = 3 x single-hull C33: an exact construction
    # identity (heave column of T carries no moment arm into C33)...
    assert c_c[2, 2] == pytest.approx(3.0 * float(c_single[2, 2]), rel=1e-12)
    # ...and it lands on the study record (4-decimal constant -> rel 1e-6).
    assert c_c[2, 2] == pytest.approx(_C33_COMPOSITE, rel=1e-6)
    assert a_inf_c[2, 2] == pytest.approx(_A33_COMPOSITE_INF, rel=1e-3)

    hdb_c = HydroDatabase(
        omega=hdb18.omega,
        heading_deg=hdb18.heading_deg,
        A=a_c,
        B=b_c,
        A_inf=a_inf_c,
        C=c_c,
        RAO=rao_c,
        reference_point=np.asarray(_CLUSTER_COG),
        C_source=ref.C_source,
        metadata={"condensed_from": "capytaine_multibody_18dof.nc via T (M8 PR4)"},
    )

    # --- run the condensed 6x6 decay, mirroring the composite study ---
    lhs = assemble_cummins_lhs(
        rigid_body_mass=m_c,
        hdb=hdb_c,
        mass=_M_CLUSTER,
        cog_offset_from_bem_origin=np.zeros(3),
        gravity=_G,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernel = compute_retardation_kernel(
            hdb_c, t_max=_KERNEL_TMAX_S, dt=_DT, asymptote_check_override=_OVERRIDE
        )
    eq = solve_static_equilibrium(lhs=lhs, state_force=None)
    assert eq.converged
    xi0 = eq.xi_eq.copy()
    xi0[2] += _IC_HEAVE_M
    res = integrate_cummins(
        lhs=lhs, kernel=kernel, xi0=xi0, xi_dot0=np.zeros(6), duration=_DURATION_S, dt=_DT
    )

    # Period via heave peaks (cluster_analyze.py convention).
    pk, _ = find_peaks(res.xi[:, 2], height=1e-4)
    assert pk.size >= 3, "decay produced too few peaks to measure a period"
    t_n = float(np.mean(np.diff(res.t[pk])))
    assert t_n == pytest.approx(_T_N_REFERENCE_S, rel=1e-2), (
        f"condensed decay period {t_n:.5f} s vs cluster study " f"{_T_N_REFERENCE_S} s (rtol 1e-2)"
    )


# ---------------------------------------------------------------------------
# Step B/C — EXCITATION GATE (permanent)
# ---------------------------------------------------------------------------


def test_excitation_gate_condensation_identity_matched_grids() -> None:
    """EXCITATION GATE: ``T^T F_exc,18(w)`` equals the composite
    ``F_exc(w)`` on every physically-excited DOF, at every frequency of
    the IDENTICAL 80-point production grid — including the contaminated
    slice at omega = 4.934, DELIBERATELY.

    This gate validates the ingestion and assembly path, not the
    underlying BEM physics — the two models share an influence matrix,
    so agreement is a linear-algebra identity. Independent validation
    of coupled hydrodynamics does not exist in this program before M10.

    Why omega = 4.934 is INCLUDED here (and excluded from the decay
    gate): at that frequency the entire shared BEM solve is contaminated
    (whole-matrix, ~5 % on the large DOFs — tracker
    BEM-CONTAMINATED-FREQUENCY-SLICE-CLUSTER-DRAFT). Both models share
    the contaminated influence matrix, so the condensation identity
    must hold there REGARDLESS. Its passing at 4.934 is the closure
    doc's worked example, measured rather than argued: every identity
    gate in this milestone would pass on a wholly wrong BEM solve.
    That is what "no independent reference for coupled hydrodynamics"
    means concretely.

    Grid identity is BY CONSTRUCTION and asserted: both fixtures'
    generators evaluate geomspace(0.1, 30, 80); the test requires the
    stored omega arrays to be exactly equal (no interpolation path
    exists in this test).

    Compared DOFs: the PHYSICALLY-EXCITED set at beta = 0 — surge,
    heave, pitch. The y-mirror symmetry forbids sway / roll / yaw a
    priori; both models carry only numerical noise there (which in the
    high-omega mesh-resolution band rises ABOVE the magnitude floor, so
    comparing it would reintroduce the cbc0dc1 noise/noise artifact —
    measured 17.2 "relative error" on roll noise at omega=27.9 during
    Phase A). Magnitude floor on top: 1e-6 x max|F_comp| per omega.

    Tolerances are TWO-TIER because even a construction identity has a
    noise floor set by operator conditioning: below Capytaine's
    mesh-resolution flag (omega ~ 15.665) the identity is machine-grade
    (measured worst 1.46e-5 / 0.0005 deg); above it, the ill-conditioned
    shared operator amplifies the two solves' different right-hand-side
    noise differently (measured worst 8.7e-4 / 0.015 deg).
    """
    hdb18 = read_capytaine(_NC_18DOF)  # FULL grid, contaminated slice included
    comp = read_capytaine(_NC_COMPOSITE)

    # Identical grids by construction — asserted, not assumed.
    np.testing.assert_array_equal(np.asarray(hdb18.omega), np.asarray(comp.omega))
    assert hdb18.n_headings == comp.n_headings == 1

    t_map = build_rigid_condensation_map(hdb18.body_labels, _body_reference_points(), _CLUSTER_COG)
    f_18 = np.asarray(hdb18.RAO)[:, :, 0]  # (18, n_w) complex
    f_comp = np.asarray(comp.RAO)[:, :, 0]  # (6, n_w) complex
    f_cond = t_map.T @ f_18  # (6, n_w)

    w = np.asarray(hdb18.omega)
    k_contam = int(np.argmin(np.abs(w - 4.934)))
    assert abs(w[k_contam] - 4.934) < 1e-2  # the slice is in this grid

    worst = {"resolved": [0.0, 0.0], "flagged": [0.0, 0.0]}  # [mag rel, phase deg]
    compared = 0
    rel_at_contaminated: dict[int, float] = {}
    for k in range(w.size):
        floor = _EXC_FLOOR_REL * float(np.max(np.abs(f_comp[:, k])))
        band = "resolved" if w[k] <= _MESH_RESOLUTION_BAND_RAD_S else "flagged"
        for j in _EXCITED_DOF:
            if abs(f_comp[j, k]) <= floor:
                continue  # below floor at this omega: not compared
            rel = abs(f_cond[j, k] - f_comp[j, k]) / abs(f_comp[j, k])
            dphase = np.degrees(np.angle(f_cond[j, k]) - np.angle(f_comp[j, k]))
            dphase = abs(float((dphase + 180.0) % 360.0 - 180.0))
            worst[band][0] = max(worst[band][0], rel)
            worst[band][1] = max(worst[band][1], dphase)
            compared += 1
            if k == k_contam:
                rel_at_contaminated[j] = rel

    assert compared > 0, "no excited DOF found above the floor -- floor misconfigured?"
    # The identity must hold everywhere, at the band-appropriate grade...
    assert (
        worst["resolved"][0] < _EXC_RTOL_RESOLVED
    ), f"resolved-band worst magnitude rel-diff {worst['resolved'][0]:.3e}"
    assert (
        worst["resolved"][1] < _EXC_PHASE_DEG_RESOLVED
    ), f"resolved-band worst phase diff {worst['resolved'][1]:.5f} deg"
    assert (
        worst["flagged"][0] < _EXC_RTOL_FLAGGED
    ), f"flagged-band worst magnitude rel-diff {worst['flagged'][0]:.3e}"
    assert (
        worst["flagged"][1] < _EXC_PHASE_DEG_FLAGGED
    ), f"flagged-band worst phase diff {worst['flagged'][1]:.5f} deg"
    # ...and specifically AT the contaminated frequency (the worked
    # example: the identity is blind to the shared solve being wrong;
    # omega = 4.934 sits in the RESOLVED band, so the tight tier
    # applies -- measured 1.8e-8 / 2.4e-8 / 4.2e-6 on surge/heave/pitch).
    assert rel_at_contaminated, "no excited DOF above floor at omega=4.934"
    for j, rel in rel_at_contaminated.items():
        assert rel < _EXC_RTOL_RESOLVED, (
            f"identity broke at the contaminated slice (DOF {j}: {rel:.3e}) -- "
            "significant finding, see plan risk register"
        )
