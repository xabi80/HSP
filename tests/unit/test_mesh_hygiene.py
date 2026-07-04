"""M7.5 PR3 unit tests for floatsim.hydro.mesh_hygiene.

Q2 final amendment (2026-07-03) — the red gate is recalibrated
to the MEASURED ground truth from the Step 0 diagnostic gate:

  ORIGINAL fixture:  216 inward panels (192 horizontal plate faces
                     + 24 strip panels)
  CORRECTED fixture: 24 inward panels (all strips; the study's
                     z-band heuristic never touched them)

The study's `fix_mesh_normals.py` fixed exactly 192 horizontal
plate panels; the 24 outer-edge strip panels of the plate are
objectively misoriented in BOTH ORIGINAL and CORRECTED, but
Capytaine's A_inf(heave) calculation was insensitive to them
(quantified via tier-2 `check_hydrostatic_volume`). The
sixth-amendment ray-parity algorithm detects and can fix all
216 automatically; the `panel_mask` escape hatch reproduces
the study's 192-panel fix byte-for-byte.

Two-tier coverage:

- Tier 1: validate_panel_normals + fix_panel_normals via
  per-panel ray-parity (assumption-free). Recalibrated red
  gate against measured ground truth on both fixtures.
- Tier 2: check_hydrostatic_volume physics screen. Measured
  values documented inline in the tier-2 tests.

See:
  docs/m7.5-reader-hygiene-plan.md §Q2 (final amendment 2026-07-03)
  docs/multibody-conventions.md Item 5 (final amendment)
  docs/phase2-followups.md BEM-MESH-THIN-SURFACE-ORIENTATION
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from floatsim.hydro.mesh_hygiene import (
    GdfMesh,
    OrientationReport,
    VolumeReport,
    _panel_normal,
    _ray_parity_inward_flags,
    check_hydrostatic_volume,
    fix_panel_normals,
    load_gdf_panels,
    validate_panel_normals,
)

_FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "bem"
    / "mesh_hygiene"
)
_FIXTURE_ORIGINAL = _FIXTURE_DIR / "test2_spar_fin_ORIGINAL.gdf"
_FIXTURE_CORRECTED = _FIXTURE_DIR / "test2_spar_fin_corrected.gdf"

# Locked physical inputs from the spar-fin study (also
# documented at studies/spar-fin-decay/README.md).
_RHO_SEAWATER = 1025.0
_BODY_MASS_KG = 28.67


# ---------------------------------------------------------------------------
# Fixture presence sanity check
# ---------------------------------------------------------------------------


def test_fixture_meshes_committed() -> None:
    """Both fixture meshes must exist on disk."""
    assert _FIXTURE_ORIGINAL.is_file(), (
        f"Missing fixture {_FIXTURE_ORIGINAL}; regenerate via the "
        f"scaffolding commit sequence."
    )
    assert _FIXTURE_CORRECTED.is_file(), (
        f"Missing fixture {_FIXTURE_CORRECTED}; regenerate via the "
        f"scaffolding commit sequence."
    )


# ---------------------------------------------------------------------------
# Fixture-based red gate (Q2 sixth-amendment recalibration)
# ---------------------------------------------------------------------------


def _horiz_strip_crosstab(
    mesh: GdfMesh, indices: np.ndarray
) -> tuple[int, int]:
    """Return (n_horizontal, n_strip) counts for the given panel indices.

    Horizontal = ``|n_z| > 0.9`` on the as-stored panel normal; strip =
    ``|n_z| <= 0.9`` (or degenerate normal, counted as strip). Used to
    verify the 192/24 split in the ORIGINAL fixture and the 0/24
    split in the CORRECTED fixture.
    """
    n_horiz = 0
    n_strip = 0
    for pi in indices:
        n = _panel_normal(mesh.panels[int(pi)])
        if n is None:
            n_strip += 1
            continue
        if abs(n[2]) > 0.9:
            n_horiz += 1
        else:
            n_strip += 1
    return n_horiz, n_strip


def test_original_mesh_216_inward_with_horiz_strip_crosstab() -> None:
    """Red gate (i): ORIGINAL fixture has EXACTLY 216 inward panels
    per per-panel ray-parity, split as 192 horizontal (|n_z|>0.9) plate
    faces + 24 strip (|n_z|<=0.9) panels. Zero indeterminate. This is
    the measured ground truth from the Step 0 diagnostic gate
    (2026-07-03). Any deviation is a finding, not a tolerance failure.

    Also pins the fixture's open-boundary property: 96 open edges
    (measured 2026-07-03) — the plate-spar junction and the disk
    edge form the boundary. The open-boundary UserWarning is
    expected to fire once per validate call.
    """
    mesh = load_gdf_panels(_FIXTURE_ORIGINAL)
    with pytest.warns(UserWarning, match="open boundary"):
        report = validate_panel_normals(mesh, return_report=True)
    assert isinstance(report, OrientationReport)
    assert report.inward_indices.size == 216, (
        f"Expected 216 inward panels on ORIGINAL; got "
        f"{report.inward_indices.size}."
    )
    assert report.indeterminate_indices.size == 0, (
        f"Expected 0 indeterminate on ORIGINAL; got "
        f"{report.indeterminate_indices.size}. Indices: "
        f"{report.indeterminate_indices[:10].tolist()}"
    )
    assert report.n_open_edges == 96, (
        f"Pinned fixture property: ORIGINAL has 96 open boundary edges "
        f"(measured 2026-07-03); got {report.n_open_edges}."
    )
    n_horiz, n_strip = _horiz_strip_crosstab(mesh, report.inward_indices)
    assert n_horiz == 192, (
        f"Expected 192 horizontal (|n_z|>0.9) inward panels; got {n_horiz}."
    )
    assert n_strip == 24, (
        f"Expected 24 strip (|n_z|<=0.9) inward panels; got {n_strip}."
    )


def test_corrected_mesh_24_strip_inward() -> None:
    """Red gate (ii): CORRECTED fixture has exactly 24 inward panels,
    all strips, zero horizontal. Documents the study fixture's known
    deficiency: the study's `fix_mesh_normals.py` z-band + radius +
    |n_z|>0.9 filter deliberately excluded the strip panels; they
    remain misoriented in the "corrected" fixture. Capytaine's
    A_inf(heave) result was insensitive to this per tier-2
    measurement. Also pins the 96 open-edge fixture property; the
    open-boundary UserWarning is expected.
    """
    mesh = load_gdf_panels(_FIXTURE_CORRECTED)
    with pytest.warns(UserWarning, match="open boundary"):
        report = validate_panel_normals(mesh, return_report=True)
    assert isinstance(report, OrientationReport)
    assert report.inward_indices.size == 24, (
        f"Expected 24 inward panels on CORRECTED; got "
        f"{report.inward_indices.size}."
    )
    assert report.indeterminate_indices.size == 0
    assert report.n_open_edges == 96, (
        f"Pinned fixture property: CORRECTED has 96 open boundary edges "
        f"(measured 2026-07-03); got {report.n_open_edges}."
    )
    n_horiz, n_strip = _horiz_strip_crosstab(mesh, report.inward_indices)
    assert n_horiz == 0, f"Expected 0 horizontal inward on CORRECTED; got {n_horiz}."
    assert n_strip == 24, f"Expected 24 strip inward on CORRECTED; got {n_strip}."


def test_full_fix_from_original_produces_zero_inward() -> None:
    """Red gate (iii): fix_panel_normals(ORIGINAL) (auto path) yields
    a mesh with 0 inward panels on re-validation. This full_fix mesh
    is NOT byte-equal to the study's corrected fixture — it differs
    in exactly the 24 strip panels (array-equal on the complement;
    the strip panels have reversed winding vs the corrected fixture).
    """
    original = load_gdf_panels(_FIXTURE_ORIGINAL)
    corrected = load_gdf_panels(_FIXTURE_CORRECTED)
    with pytest.warns(UserWarning, match="open boundary"):
        full_fix = fix_panel_normals(original)
    # Re-validation is clean (still emits the open-boundary warning:
    # topology unchanged by vertex-order flips).
    with pytest.warns(UserWarning, match="open boundary"):
        ff_report = validate_panel_normals(full_fix, return_report=True)
    assert ff_report.inward_indices.size == 0, (
        f"Expected 0 inward after full auto-fix; got "
        f"{ff_report.inward_indices.size}."
    )
    # Panels differing between full_fix and the corrected fixture:
    # exactly the strip panels the study did not flip.
    panel_diffs = np.any(
        np.any(full_fix.panels != corrected.panels, axis=-1), axis=-1
    )
    n_diff = int(panel_diffs.sum())
    assert n_diff == 24, (
        f"Expected exactly 24 panels differing between full_fix and "
        f"CORRECTED; got {n_diff}."
    )
    # The differing panels are exactly the strip panels (|n_z|<=0.9).
    diff_indices = np.where(panel_diffs)[0]
    n_horiz, n_strip = _horiz_strip_crosstab(corrected, diff_indices)
    assert n_horiz == 0, (
        f"Panels differing from CORRECTED should be strip-only; "
        f"got {n_horiz} horizontal, {n_strip} strip."
    )
    assert n_strip == 24
    # For each strip panel, full_fix's panel is the vertex-reversed
    # form of the corrected fixture's panel.
    for pi in diff_indices:
        assert np.array_equal(
            full_fix.panels[pi], corrected.panels[pi, ::-1, :]
        ), f"strip panel {pi} not vertex-reversed vs CORRECTED"


def test_fix_idempotent_on_full_fix() -> None:
    """Red gate (iv): applying fix_panel_normals to full_fix (already
    outward everywhere) returns a bit-identical mesh. Open-boundary
    warning fires on both calls (fixture topology has 96 open edges;
    fixing doesn't change edge adjacency).
    """
    original = load_gdf_panels(_FIXTURE_ORIGINAL)
    with pytest.warns(UserWarning, match="open boundary"):
        full_fix = fix_panel_normals(original)
    with pytest.warns(UserWarning, match="open boundary"):
        twice_fixed = fix_panel_normals(full_fix)
    assert np.array_equal(twice_fixed.panels, full_fix.panels)


def test_panel_mask_reproduces_study_fixture() -> None:
    """Red gate (v): the panel_mask escape hatch, given the 192
    horizontal-inward indices from the ORIGINAL report, produces a
    mesh byte-equal to the study's corrected fixture. Reproducing
    the study's output remains available for byte-compatibility;
    it's no longer the default auto-fix path.
    """
    original = load_gdf_panels(_FIXTURE_ORIGINAL)
    corrected = load_gdf_panels(_FIXTURE_CORRECTED)
    with pytest.warns(UserWarning, match="open boundary"):
        report = validate_panel_normals(original, return_report=True)
    # Extract exactly the 192 horizontal inward indices from the report.
    horiz_indices = []
    for pi in report.inward_indices:
        n = _panel_normal(original.panels[int(pi)])
        if n is not None and abs(n[2]) > 0.9:
            horiz_indices.append(int(pi))
    assert len(horiz_indices) == 192, (
        f"Expected 192 horizontal inward indices; got {len(horiz_indices)}."
    )
    reproduced = fix_panel_normals(original, panel_mask=horiz_indices)
    assert np.array_equal(reproduced.panels, corrected.panels), (
        "panel_mask reproduction of study's corrected fixture failed "
        "byte-equality on the panels array."
    )


# ---------------------------------------------------------------------------
# Tier-2 hydrostatic-volume screen (physics-first protection)
# ---------------------------------------------------------------------------
#
# Measured values on the terminal fixtures (Step B.3 diagnostic,
# 2026-07-03, rho=1025 kg/m^3, mass=28.67 kg):
#
#             signed_volume  displaced_mass  residual_fraction
#   ORIGINAL  +3.882e-02     39.79 kg        +0.388
#   CORRECTED +3.914e-02     40.12 kg        +0.399
#   full_fix  +3.989e-02     40.89 kg        +0.426
#
# Interpretation:
# - All three variants have POSITIVE signed_volume — the majority of
#   panels contribute outward. The 192 reversed horizontal plate
#   panels in ORIGINAL barely perturb V because the plate is thin
#   (annular area ~O(0.1 m^2), thickness ~O(4 mm), so volume
#   contribution ~O(1e-3) — smaller than the ~4e-2 hull volume).
# - residual_fraction is ~+40% on all three because the body has
#   significant reserve buoyancy (fully-submerged displaced mass ~40
#   kg vs body mass 28.67 kg). This is NOT a "buoyancy mismatch"
#   signal in the usual sense; it's a "reserve buoyancy" measurement
#   for a spar-buoy geometry.
# - The tier-2 signal for detecting the plate-face pathology is the
#   SIGN of signed_volume (a wholly-inverted mesh would flip it) and
#   the relative delta between variants (0.8% ORIGINAL vs CORRECTED,
#   1.9% full_fix vs CORRECTED). The magnitude is small because the
#   plate is thin -- confirming quantitatively that Capytaine's
#   A_inf(heave) 16x change did NOT come from displaced-volume
#   perturbation.


def test_hydrostatic_volume_synthetic_unit_cube() -> None:
    """Tier-2 analytic check: unit cube [0,1]^3 with outward faces has
    signed_volume == 1.0 exactly.
    """
    cube = _unit_cube_outward()
    vr = check_hydrostatic_volume(cube, rho=1000.0)
    assert isinstance(vr, VolumeReport)
    assert vr.signed_volume == pytest.approx(1.0, abs=1.0e-12), (
        f"Expected unit cube volume == 1.0; got {vr.signed_volume}."
    )
    assert vr.displaced_mass == pytest.approx(1000.0, abs=1.0e-9)
    assert vr.residual_fraction is None


def test_hydrostatic_volume_all_inward_cube_is_negated() -> None:
    """Tier-2 sign check: reversing every panel on the unit cube
    negates signed_volume. This is exactly the property that catches
    a wholly-inverted mesh.
    """
    cube = _unit_cube_outward()
    inward_panels = cube.panels[:, ::-1, :].copy()
    inward_cube = GdfMesh(header_lines=cube.header_lines, panels=inward_panels)
    vr = check_hydrostatic_volume(inward_cube, rho=1000.0)
    assert vr.signed_volume == pytest.approx(-1.0, abs=1.0e-12)


def test_hydrostatic_volume_fixture_measurements_positive() -> None:
    """Tier-2 fixture measurements: all three variants (ORIGINAL,
    CORRECTED, full_fix) have positive signed_volume within 5% of
    each other. The 0.8% ORIG-vs-CORR and 1.9% FULLFIX-vs-CORR
    deltas are the quantitative confirmation that the plate-face
    pathology which motivated M7.5 (A_inf 1.30 vs 21.11 kg = 16x)
    could NOT come from displaced-volume perturbation — the thin
    plate contributes O(2%) to V and cannot account for a 16x A_inf
    change.

    Measured values 2026-07-03 (Step B.3 diagnostic,
    rho=1025 kg/m^3, mass=28.67 kg):

        signed_volume  displaced_mass  residual_fraction
      ORIGINAL   +3.882e-02   39.79 kg      +0.388
      CORRECTED  +3.914e-02   40.12 kg      +0.399
      full_fix   +3.989e-02   40.89 kg      +0.426

      |V(ORIG)    - V(CORR)| / V(CORR) = 0.82%
      |V(FULLFIX) - V(CORR)| / V(CORR) = 1.92%
    """
    orig = load_gdf_panels(_FIXTURE_ORIGINAL)
    corr = load_gdf_panels(_FIXTURE_CORRECTED)
    with pytest.warns(UserWarning, match="open boundary"):
        full_fix = fix_panel_normals(orig)

    v_orig = check_hydrostatic_volume(orig, rho=_RHO_SEAWATER, mass=_BODY_MASS_KG)
    v_corr = check_hydrostatic_volume(corr, rho=_RHO_SEAWATER, mass=_BODY_MASS_KG)
    v_ff = check_hydrostatic_volume(full_fix, rho=_RHO_SEAWATER, mass=_BODY_MASS_KG)

    # All positive (no wholly-inverted mesh).
    # Measured 2026-07-03: +3.882e-02, +3.914e-02, +3.989e-02 m^3.
    assert v_orig.signed_volume > 0.0
    assert v_corr.signed_volume > 0.0
    assert v_ff.signed_volume > 0.0

    # Pairwise differences within 5% of CORRECTED volume. Actual
    # measured deltas (2026-07-03): |ORIG - CORR|/CORR = 0.82%,
    # |FULLFIX - CORR|/CORR = 1.92%. Tolerance 5% chosen to allow
    # future minor geometry re-mesh without breaking the test; if
    # this ratio ever exceeds 5%, that's a topology / area change
    # worth investigating.
    delta_orig = abs(v_orig.signed_volume - v_corr.signed_volume) / v_corr.signed_volume
    delta_ff = abs(v_ff.signed_volume - v_corr.signed_volume) / v_corr.signed_volume
    assert delta_orig < 0.05, (
        f"|V(ORIG) - V(CORR)| / V(CORR) = {delta_orig:.4e} > 0.05; "
        f"topology / area change worth investigating."
    )
    assert delta_ff < 0.05, (
        f"|V(FULLFIX) - V(CORR)| / V(CORR) = {delta_ff:.4e} > 0.05; "
        f"contradicts the 'Capytaine insensitive to strips' premise."
    )

    # Records a known mesh-buoyancy-vs-mass inconsistency of the study
    # fixture (~+40%): at the meshed waterline the mesh displaces
    # ~40.1 kg against a 28.67 kg body. This is a documented property,
    # not a target of ~0. See BEM-MESH-THIN-SURFACE-ORIENTATION
    # sub-item in phase2-followups.md. If this assertion fails after a
    # mesh or mass change, the tracker entry must be updated, not the
    # tolerance.
    # Measured 2026-07-03 on CORRECTED: residual_fraction = +0.399.
    assert v_corr.residual_fraction is not None
    assert 0.35 <= v_corr.residual_fraction <= 0.45, (
        f"CORRECTED residual_fraction {v_corr.residual_fraction:.4f} "
        f"outside [0.35, 0.45]; measured +0.399 (2026-07-03). "
        f"See BEM-MESH-THIN-SURFACE-ORIENTATION in phase2-followups.md."
    )


# ---------------------------------------------------------------------------
# Synthetic small-mesh tests
# ---------------------------------------------------------------------------


def _unit_cube_outward() -> GdfMesh:
    """Return a GdfMesh for the unit cube [0, 1]^3 with all 6 quad
    faces oriented outward (right-hand-rule normal pointing away from
    the cube interior).
    """
    corners = np.array(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [0, 1, 0],  # 2
            [1, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [0, 1, 1],  # 6
            [1, 1, 1],  # 7
        ],
        dtype=np.float64,
    )
    face_indices = np.array(
        [
            [4, 5, 7, 6],  # z=1 top,    normal +z
            [0, 2, 3, 1],  # z=0 bottom, normal -z
            [1, 3, 7, 5],  # x=1 right,  normal +x
            [0, 4, 6, 2],  # x=0 left,   normal -x
            [0, 1, 5, 4],  # y=0 front,  normal -y
            [2, 6, 7, 3],  # y=1 back,   normal +y
        ]
    )
    panels = corners[face_indices]
    header = (
        "Synthetic unit cube",
        "  1.000000  9.810000",
        "  0  0",
        "  6",
    )
    return GdfMesh(header_lines=header, panels=panels)


def test_closed_cube_consistent_passes() -> None:
    """A consistent outward unit cube passes validation."""
    cube = _unit_cube_outward()
    result = validate_panel_normals(cube)
    assert result is None
    inward, indet = _ray_parity_inward_flags(cube.panels)
    assert int(inward.sum()) == 0
    assert int(indet.sum()) == 0


def test_closed_cube_one_flipped_detected() -> None:
    """Reversing one panel's vertex order on the outward cube gets
    detected by per-panel ray-parity; the fix restores the original
    bit-identically.
    """
    cube = _unit_cube_outward()
    corrupted = cube.panels.copy()
    corrupted[0] = corrupted[0, ::-1, :]
    mesh = GdfMesh(header_lines=cube.header_lines, panels=corrupted)
    inward, indet = _ray_parity_inward_flags(mesh.panels)
    assert int(inward.sum()) == 1
    assert bool(inward[0])
    assert int(indet.sum()) == 0
    fixed = fix_panel_normals(mesh)
    assert np.array_equal(fixed.panels, cube.panels)


def test_all_inward_cube_flagged_and_fixed() -> None:
    """Every panel reversed on the outward cube: ray-parity flags all
    6 as inward (they all point into the cube interior); the fix
    produces the outward cube.
    """
    cube = _unit_cube_outward()
    inward_panels = cube.panels[:, ::-1, :].copy()
    mesh = GdfMesh(header_lines=cube.header_lines, panels=inward_panels)
    inward, indet = _ray_parity_inward_flags(mesh.panels)
    assert int(inward.sum()) == 6
    assert int(indet.sum()) == 0
    fixed = fix_panel_normals(mesh)
    assert np.array_equal(fixed.panels, cube.panels)


def _closed_pyramid_outward() -> GdfMesh:
    """Return a GdfMesh for a closed square pyramid with all 5 quad
    faces oriented outward. The 4 triangular side faces are stored
    as DEGENERATE QUADS with the apex vertex repeated (``v3 == v0``
    pattern, matching WAMIT's convention for triangular panels).

    Base is a 2x2 square at z=-1; apex at z=+1. The origin is
    inside the pyramid at height z=0 (halfway up) — chosen so that
    no vertex sits at the origin, otherwise the
    ``v0 · (v1 × v2)`` signed-volume contribution would be zero
    for many panels and the volume screen would return zero
    trivially.
    """
    A = np.array([-1.0, -1.0, -1.0])
    B = np.array([+1.0, -1.0, -1.0])
    C = np.array([+1.0, +1.0, -1.0])
    D = np.array([-1.0, +1.0, -1.0])
    T = np.array([0.0, 0.0, +1.0])
    panels = np.stack(
        [
            np.stack([A, D, C, B]),
            np.stack([A, B, T, A]),
            np.stack([B, C, T, B]),
            np.stack([C, D, T, C]),
            np.stack([D, A, T, D]),
        ],
        axis=0,
    )
    header = (
        "Synthetic closed pyramid with degenerate-quad triangles",
        "  1.000000  9.810000",
        "  0  0",
        "  5",
    )
    return GdfMesh(header_lines=header, panels=panels)


def test_pyramid_with_degenerate_quads() -> None:
    """A closed pyramid with 4 triangular side faces stored as
    degenerate quads (v3 == v0) passes ray-parity validation, and
    flipping one side face triggers the reversed-normal ValueError
    (not any spurious non-manifold raise).
    """
    pyramid = _closed_pyramid_outward()
    # (a) Consistent-outward pyramid passes.
    validate_panel_normals(pyramid)
    inward, indet = _ray_parity_inward_flags(pyramid.panels)
    assert int(inward.sum()) == 0
    assert int(indet.sum()) == 0

    # (b) Flip one side face.
    corrupted = pyramid.panels.copy()
    corrupted[1] = corrupted[1, ::-1, :]
    mesh = GdfMesh(header_lines=pyramid.header_lines, panels=corrupted)
    with pytest.raises(
        ValueError, match=r"panels with inward-facing normals"
    ) as excinfo:
        validate_panel_normals(mesh)
    assert "non-watertight" not in str(excinfo.value), (
        "Should not raise non-manifold on a degenerate-quad mesh."
    )
    fixed = fix_panel_normals(mesh)
    assert np.array_equal(fixed.panels, pyramid.panels)


def test_t_junction_raises() -> None:
    """Three panels sharing a single edge (T-junction) creates
    genuinely ambiguous orientation; validate raises with a message
    citing the tracker.
    """
    v_a = np.array([0.0, 0.0, 0.0])
    v_b = np.array([1.0, 0.0, 0.0])
    panel_0 = np.stack([
        v_a, v_b,
        np.array([1.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ])
    panel_1 = np.stack([
        v_a, v_b,
        np.array([1.0, -1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    ])
    panel_2 = np.stack([
        v_a, v_b,
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
    ])
    panels = np.stack([panel_0, panel_1, panel_2])
    header = (
        "Synthetic T-junction",
        "  1.000000  9.810000",
        "  0  0",
        "  3",
    )
    mesh = GdfMesh(header_lines=header, panels=panels)
    with pytest.raises(ValueError, match=r"T-junctions") as excinfo:
        validate_panel_normals(mesh)
    assert "PANEL-NORMAL-NONCONVEX-BODIES" in str(excinfo.value)


def test_open_component_is_silent_false_negative_with_warning() -> None:
    """DOCUMENTED FALSE NEGATIVE (2026-07-03): per-panel ray-parity
    cannot detect a reversed panel on an open shell when the panel's
    ray exits through the opening. This is a topological limitation of
    the algorithm, pinned here so future refactors don't accidentally
    hide it.

    Setup: topless cube (5 outward panels; z=1 top removed) has 4 open
    boundary edges (the top perimeter). Flipping the bottom face makes
    its normal point +z into the (now-open) interior; the ray from the
    bottom face travels +z through the opening and hits zero triangles
    → parity even → reported as OUTWARD. The open-boundary UserWarning
    fires from validate_panel_normals, alerting the caller that
    ray-parity results are unreliable on open meshes.

    See: PANEL-NORMAL-NONCONVEX-BODIES / BEM-MESH-THIN-SURFACE-ORIENTATION.
    """
    cube = _unit_cube_outward()
    open_panels = cube.panels[1:].copy()
    open_mesh = GdfMesh(
        header_lines=(
            cube.header_lines[0],
            cube.header_lines[1],
            cube.header_lines[2],
            "  5",
        ),
        panels=open_panels,
    )
    # Pre-corruption: warning fires; parity reports 0 inward (clean).
    with pytest.warns(UserWarning, match="open boundary"):
        report = validate_panel_normals(open_mesh, return_report=True)
    assert report.n_open_edges == 4, (
        f"Topless cube: expected 4 open edges (top perimeter); got "
        f"{report.n_open_edges}."
    )
    assert report.inward_indices.size == 0
    assert report.indeterminate_indices.size == 0

    # Corruption: flip the bottom face.
    corrupted = open_panels.copy()
    corrupted[0] = corrupted[0, ::-1, :]
    corrupted_mesh = GdfMesh(header_lines=open_mesh.header_lines, panels=corrupted)
    with pytest.warns(UserWarning, match="open boundary"):
        c_report = validate_panel_normals(corrupted_mesh, return_report=True)
    # SILENT FALSE NEGATIVE: the flipped bottom-face ray exits through
    # the missing top → 0 crossings → parity says outward.
    assert c_report.inward_indices.size == 0, (
        "Documented false-negative failed: ray-parity should NOT detect "
        "the flipped bottom face on the topless cube (ray exits through "
        "open top). If this ever detects the flip, revisit "
        "BEM-MESH-THIN-SURFACE-ORIENTATION -- the algorithm changed."
    )
    assert c_report.n_open_edges == 4


def test_two_shell_isolated_plate_is_silent_false_negative_with_warning() -> None:
    """DOCUMENTED FALSE NEGATIVE (2026-07-03): a well-separated
    isolated plate beside a closed shell has 4 open edges (the plate
    perimeter) and no closed volume for the plate's ray to re-enter.
    Flipping the plate is topologically undetectable by ray-parity.
    This is the "genuinely two-sided sheet" case excluded from
    mesh_hygiene scope by the plan's narrowed §Q5 punt.

    See: PANEL-NORMAL-NONCONVEX-BODIES / BEM-MESH-THIN-SURFACE-ORIENTATION.
    """
    cube = _unit_cube_outward()
    plate_panel = np.stack([
        np.array([3.0, 0.0, 0.5]),
        np.array([4.0, 0.0, 0.5]),
        np.array([4.0, 1.0, 0.5]),
        np.array([3.0, 1.0, 0.5]),
    ])
    two_shell_panels = np.concatenate(
        [cube.panels, plate_panel[None, ...]], axis=0
    )
    header = (
        "Synthetic two-shell miniature (cube + isolated plate)",
        "  1.000000  9.810000",
        "  0  0",
        "  7",
    )
    two_shell = GdfMesh(header_lines=header, panels=two_shell_panels)
    with pytest.warns(UserWarning, match="open boundary"):
        report = validate_panel_normals(two_shell, return_report=True)
    assert report.n_open_edges == 4, (
        f"Isolated plate: expected 4 open edges (plate perimeter); got "
        f"{report.n_open_edges}."
    )
    assert report.inward_indices.size == 0
    assert report.indeterminate_indices.size == 0

    corrupted = two_shell_panels.copy()
    corrupted[6] = corrupted[6, ::-1, :]
    mesh = GdfMesh(header_lines=header, panels=corrupted)
    with pytest.warns(UserWarning, match="open boundary"):
        c_report = validate_panel_normals(mesh, return_report=True)
    # SILENT FALSE NEGATIVE: the plate's flipped ray points -z away
    # from the cube; the cube is at x∈[0,1] and the plate at x∈[3,4]
    # so the ray never encounters cube geometry. 0 crossings → outward.
    assert c_report.inward_indices.size == 0, (
        "Documented false-negative failed: ray-parity should NOT detect "
        "an isolated-plate flip when the plate has no closing geometry "
        "along its normal. If this ever detects the flip, revisit "
        "BEM-MESH-THIN-SURFACE-ORIENTATION -- the algorithm changed."
    )
    assert c_report.n_open_edges == 4


# ---------------------------------------------------------------------------
# Error-message content coverage
# ---------------------------------------------------------------------------


def test_reversed_normal_message_content() -> None:
    """The §I2 error message on the ORIGINAL fixture contains:
    - the total-panel count (1488),
    - the reversed-panel count (216, per the recalibrated red gate),
    - the first-10 reversed-panel indices,
    - a pointer to fix_panel_normals,
    - a pointer to conventions doc Item 5.
    """
    original = load_gdf_panels(_FIXTURE_ORIGINAL)
    with pytest.warns(UserWarning, match="open boundary"):
        with pytest.raises(ValueError) as excinfo:
            validate_panel_normals(original)
    msg = str(excinfo.value)
    assert "1488" in msg, "message missing total panel count"
    assert "216" in msg, "message missing reversed count (216 post-Q2-final)"
    assert "Reversed-panel indices (first 10):" in msg
    assert "floatsim.hydro.mesh_hygiene.fix_panel_normals" in msg
    assert "docs/multibody-conventions.md" in msg
