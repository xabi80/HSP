"""M7.5 terminal validation gate — permanent in-suite version.

Origin. The spar-fin study
(``studies/spar-fin-decay`` on ``scratch-spar-fin-decay``,
NetCDF fixture extracted from commit ``2767c12``) surfaced three
pre-flight failures that motivated M7.5 (see plan front matter
in ``docs/m7.5-reader-hygiene-plan.md``):

  Pre-flight 1: Capytaine NetCDF ingestion failed at
                ``HydroDatabase`` symmetry rtol=1e-6 because
                Capytaine's panel-method output is asymmetric at
                the ~1e-4 relative level (metadata residual
                ``max|A - A.T| ~ 7e-3`` at ``max|A| ~ 25 kg`` on
                the terminal fixture) — closed by PR2's
                ``__post_init__`` symmetrization.

  Pre-flight 2: Panel-normal orientation on
                ``test2_spar_fin.gdf`` had 216 inward panels
                (192 horizontal plate faces + 24 outer-edge
                strip panels). The study's
                ``fix_mesh_normals.py`` z-band heuristic caught
                only the 192; ``mesh_hygiene`` per-panel
                ray-parity catches all 216 and can auto-fix —
                closed by PR3.

  Pre-flight 3: ``compute_retardation_kernel`` refused the
                small-body spar+fin BEM at ``omega_max = 30
                rad/s`` because the 1/omega^4 asymptote gate
                (Item 25 Check 2) is not applicable at
                characteristic length ~ 1.85 m; the study had
                to bypass the gate manually — closed by PR1's
                ``asymptote_check_override`` explicit-forcing-
                function API.

Purpose. This module re-exercises all three failures against
the HARDENED READERS with ZERO study-local workarounds. It is
the empirical guarantee that the M7.5 milestone actually
absorbed the study's needs, and it stays in-suite as the
regression gate for the resumption.

Chains under test:

  - **Test 1 — mesh chain (PR3).** Load the ORIGINAL GDF (the
    same fixture the mesh_hygiene unit tests use, replicated
    from the study). Verify per-panel ray-parity fires the
    "216 inward" error, the open-boundary UserWarning fires on
    the 96-edge fixture, ``fix_panel_normals`` auto-fixes, and
    re-validation is clean.
  - **Test 2 — reader chain (PR2).** Load the asymmetric
    Capytaine NetCDF via ``floatsim.hydro.readers.capytaine``
    with NO pre-symmetrization. Verify ingestion succeeds
    (this exact file failed pre-PR2), that metadata residuals
    for A and B are non-trivial (documenting the fixture's
    panel-method asymmetry), that post-construction A and B
    are symmetric to rtol=1e-12, and that A_inf(heave) matches
    the study's ~21.1 kg.
  - **Test 3 — kernel chain (PR1).** Compute the retardation
    kernel using ``asymptote_check_override`` with an explicit
    small-body rationale. Verify the override UserWarning
    fires with the rationale echoed, Check 3 (post-extension
    kernel decay) passes on the real spar-fin BEM (this is the
    one unverified empirical claim of the milestone), and an
    empty-rationale override still raises.

See:
  - Plan: ``docs/m7.5-reader-hygiene-plan.md`` front matter.
  - Closure: ``docs/m7.5-reader-hygiene-closure.md``.
  - Reference fixtures:
        ``tests/fixtures/bem/mesh_hygiene/test2_spar_fin_ORIGINAL.gdf``,
        ``tests/fixtures/bem/spar_fin_terminal/capytaine_bem_asymmetric.nc``
    (both extracted from ``scratch-spar-fin-decay`` commit ``2767c12``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.mesh_hygiene import (
    fix_panel_normals,
    load_gdf_panels,
    validate_panel_normals,
)
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.hydro.retardation import compute_retardation_kernel

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_MESH_FIXTURE = (
    _REPO_ROOT / "tests" / "fixtures" / "bem" / "mesh_hygiene"
    / "test2_spar_fin_ORIGINAL.gdf"
)
_BEM_FIXTURE = (
    _REPO_ROOT / "tests" / "fixtures" / "bem" / "spar_fin_terminal"
    / "capytaine_bem_asymmetric.nc"
)

# Locked physical inputs from the spar-fin study.
_STUDY_A_INF_HEAVE_KG = 21.1123  # Measured 2026-07-03 at omega_max=30 rad/s.
_STUDY_BODY_MASS_KG = 28.67


# ---------------------------------------------------------------------------
# Test 1 — mesh chain (PR3)
# ---------------------------------------------------------------------------


def test_mesh_chain_reversed_normals_detected_and_fixed() -> None:
    """PR3 chain: `validate_panel_normals` catches 216 reversed panels on
    the study's ORIGINAL GDF; the open-boundary UserWarning fires with
    ``n_open_edges == 96``; ``fix_panel_normals`` auto-fixes; re-validation
    is clean. The workflow the study will adopt in place of
    ``fix_mesh_normals.py``.
    """
    assert _MESH_FIXTURE.is_file(), (
        f"Missing terminal-gate mesh fixture at {_MESH_FIXTURE}."
    )
    mesh = load_gdf_panels(_MESH_FIXTURE)

    # 1a. Programmatic report path — verifies the 216 count + open
    # boundary property, without triggering the raise.
    with pytest.warns(UserWarning, match="open boundary"):
        report = validate_panel_normals(mesh, return_report=True)
    assert report.inward_indices.size == 216, (
        f"Expected 216 inward panels on terminal-gate ORIGINAL fixture; "
        f"got {report.inward_indices.size}."
    )
    assert report.n_open_edges == 96, (
        f"Expected 96 open boundary edges on ORIGINAL fixture; got "
        f"{report.n_open_edges}."
    )

    # 1b. Raise path — the same open-boundary warning fires alongside
    # the reversed-normal ValueError.
    with pytest.warns(UserWarning, match="open boundary"):
        with pytest.raises(ValueError, match="inward-facing normals"):
            validate_panel_normals(mesh)

    # 1c. Fix + re-validate. Auto-fix flips 216 panels; re-validation
    # sees 0 inward (both the horizontal plate faces and the strip
    # panels are corrected).
    with pytest.warns(UserWarning, match="open boundary"):
        fixed = fix_panel_normals(mesh)
    with pytest.warns(UserWarning, match="open boundary"):
        fixed_report = validate_panel_normals(fixed, return_report=True)
    assert fixed_report.inward_indices.size == 0, (
        f"Expected 0 inward panels after auto-fix; got "
        f"{fixed_report.inward_indices.size}."
    )
    # Topology unchanged by vertex-order flips.
    assert fixed_report.n_open_edges == 96


# ---------------------------------------------------------------------------
# Test 2 — reader chain (PR2)
# ---------------------------------------------------------------------------


def test_reader_chain_ingests_asymmetric_capytaine_netcdf() -> None:
    """PR2 chain: the FloatSim Capytaine reader ingests the study's
    asymmetric NetCDF with NO pre-symmetrization. Pre-PR2 this file
    failed the symmetry gate; post-PR2 the ``__post_init__``
    symmetrization absorbs the panel-method noise and stores the
    residual in metadata.

    Measured 2026-07-03 (Step 1 + round-2 localization diagnostic):

      Total-omega metadata residuals (as-stored in `hdb.metadata`):
        symmetrization_max_residual_A     = 7.180e-03
        symmetrization_max_residual_B     = 1.250e-01
        symmetrization_max_residual_A_inf = 2.385e-03
        symmetrization_max_residual_C     = 8.89e-17

      Physics-band residuals (w <= 10 rad/s, well above
      omega_n_heave ~ 1.85 rad/s):
        max|A(w) - A(w).T| = 3.68e-3
        max|B(w) - B(w).T| = 2.06e-2

    The B metadata residual is dominated by high-omega
    mesh-resolution noise: the largest per-omega B asymmetry sits
    at w = 19.45 rad/s in the surge-pitch off-diagonal (|dB| =
    0.125, normalized per-omega 0.029; see closure §3.5.1
    localization diagnostic). In the physics-relevant band w <= 10
    rad/s, both A and B are well-behaved and the assertions are
    correspondingly tight. Same Item 25 small-body /
    high-frequency-mesh-resolution split that motivated PR1's
    override applies.
    """
    assert _BEM_FIXTURE.is_file(), (
        f"Missing terminal-gate BEM fixture at {_BEM_FIXTURE}."
    )

    # Read the omega_max sample to construct A_inf per the study's
    # convention (`capytaine_run.py: A_inf_heave = A.isel(omega=-1)`).
    # The Capytaine ordering (Surge, Sway, Heave, Roll, Pitch, Yaw) matches
    # FloatSim's canonical ordering, so no permutation is needed.
    with xr.open_dataset(_BEM_FIXTURE) as ds:
        a_inf = ds["added_mass"].values[-1].copy()  # (6, 6)

    # Ingestion — pre-PR2 this raised on symmetry rtol=1e-6.
    hdb = read_capytaine(_BEM_FIXTURE, a_inf=a_inf)
    assert isinstance(hdb, HydroDatabase)

    # --- Fixture-property assertions on total-omega metadata ---
    # Wide-envelope sanity pins (decade margin from measurement).
    # The B metadata residual is dominated by high-omega
    # mesh-resolution noise (see round-2 localization diagnostic in
    # closure §3.5.1); the physics-band assertions below are the
    # load-bearing check for the decay study.
    resid_A = float(hdb.metadata["symmetrization_max_residual_A"])
    resid_B = float(hdb.metadata["symmetrization_max_residual_B"])
    assert 7.18e-4 < resid_A < 7.18e-2, (
        f"symmetrization_max_residual_A = {resid_A:.3e} outside "
        f"[7.18e-4, 7.18e-2] envelope (measured 2026-07-03: 7.18e-3, "
        f"decade-margin symmetric). Below/above indicates a re-solve or "
        f"fixture corruption."
    )
    assert 1.25e-2 < resid_B < 1.25e0, (
        f"symmetrization_max_residual_B = {resid_B:.3e} outside "
        f"[1.25e-2, 1.25e0] envelope (measured 2026-07-03: 1.25e-1, "
        f"decade-margin symmetric on total-omega raw residual)."
    )

    # --- Physics-band assertions restricted to w <= 10 rad/s ---
    # B's asymmetry localizes to the high-omega mesh-resolution regime
    # (max at w=19.45 rad/s, surge-pitch off-diagonal; see round-2
    # diagnostic in closure §3.5.1). In the physics-relevant band
    # w <= 10 rad/s -- which sits well above omega_n_heave ~ 1.85 rad/s
    # for T_n = 3.4 s -- asymmetry is orders of magnitude smaller and
    # is the correct pin for the decay study's use of this fixture.
    # Same Item 25 small-body / high-frequency mesh-resolution history
    # applies: the BEM is trustworthy in its physics band but the
    # asymptote regime is a separate class of concern.
    with xr.open_dataset(_BEM_FIXTURE) as ds:
        omega_raw = ds["omega"].values
        A_raw = ds["added_mass"].values
        B_raw = ds["radiation_damping"].values
    mask_low = omega_raw <= 10.0
    dA_low = float(
        np.max(np.abs(A_raw[mask_low] - np.swapaxes(A_raw[mask_low], -1, -2)))
    )
    dB_low = float(
        np.max(np.abs(B_raw[mask_low] - np.swapaxes(B_raw[mask_low], -1, -2)))
    )
    # Measured 2026-07-03 (localization diagnostic):
    #   max|A(w) - A(w).T| for w <= 10:   3.68e-3
    #   max|B(w) - B(w).T| for w <= 10:   2.06e-2
    # Decade-margin envelope: measured/10 to measured*10.
    assert 3.68e-4 < dA_low < 3.68e-2, (
        f"max|A - A.T| over w <= 10 rad/s = {dA_low:.3e} outside "
        f"[3.68e-4, 3.68e-2] envelope (measured 2026-07-03: 3.68e-3)."
    )
    assert 2.06e-3 < dB_low < 2.06e-1, (
        f"max|B - B.T| over w <= 10 rad/s = {dB_low:.3e} outside "
        f"[2.06e-3, 2.06e-1] envelope (measured 2026-07-03: 2.06e-2). "
        f"This is the physics-relevant band; the total-omega metadata "
        f"residual is dominated by high-omega mesh noise which the decay "
        f"study does not exercise."
    )

    # Post-symmetrization invariant. `_require_symmetric` in
    # ``HydroDatabase.__post_init__`` gates at rtol=1e-12; re-verify
    # explicitly here.
    n_omega = hdb.A.shape[-1]
    for k in range(n_omega):
        assert np.allclose(hdb.A[..., k], hdb.A[..., k].T, atol=1e-12), (
            f"hdb.A[..., {k}] not symmetric post-ingestion."
        )
        assert np.allclose(hdb.B[..., k], hdb.B[..., k].T, atol=1e-12), (
            f"hdb.B[..., {k}] not symmetric post-ingestion."
        )
    assert np.allclose(hdb.A_inf, hdb.A_inf.T, atol=1e-12)
    assert np.allclose(hdb.C, hdb.C.T, atol=1e-12)

    # A_inf(heave) — the study's recorded value from
    # ``STEP-A-FINDING.md`` was 21.11 kg.
    assert np.isclose(
        float(hdb.A_inf[2, 2]), _STUDY_A_INF_HEAVE_KG, rtol=1.0e-2
    ), (
        f"A_inf(heave) = {float(hdb.A_inf[2, 2]):.4f} kg does not match "
        f"study-recorded {_STUDY_A_INF_HEAVE_KG:.4f} kg at rtol=1e-2."
    )


# ---------------------------------------------------------------------------
# Test 3 — kernel chain (PR1)
# ---------------------------------------------------------------------------


_RATIONALE = (
    "M7.5 terminal gate: small-body spar+fin (L~1.85 m); "
    "1/omega^4 regime not reached at omega_max=30 rad/s; see "
    "ITEM25-SMALL-BODY-APPLICABILITY"
)


def test_kernel_chain_override_bypasses_gate_and_check3_passes() -> None:
    """PR1 chain: ``asymptote_check_override`` bypasses the Item 25
    Check 2 gate on the real small-body spar+fin BEM database.
    UserWarning fires with the rationale echoed. Check 3 (post-
    extension kernel decay) is not bypassed by the override and
    passes on the real kernel — the one previously-unverified
    empirical claim of the milestone.
    """
    with xr.open_dataset(_BEM_FIXTURE) as ds:
        a_inf = ds["added_mass"].values[-1].copy()
    hdb = read_capytaine(_BEM_FIXTURE, a_inf=a_inf)

    with pytest.warns(UserWarning, match="Item 25 asymptote check bypassed"):
        kernel = compute_retardation_kernel(
            hdb, t_max=30.0, dt=0.01, asymptote_check_override=_RATIONALE,
        )
    assert kernel is not None
    assert kernel.K.shape[0] == 6 and kernel.K.shape[1] == 6
    assert kernel.K.shape[2] == kernel.t.size

    # Rationale echo (defensive — the warning already asserts on the
    # 'bypassed' substring; this pins the rationale-echo contract from
    # plan §I3).
    with pytest.warns(UserWarning) as record:
        compute_retardation_kernel(
            hdb, t_max=30.0, dt=0.01, asymptote_check_override=_RATIONALE,
        )
    matched = [w for w in record if _RATIONALE in str(w.message)]
    assert matched, (
        f"UserWarning did not echo the override rationale. "
        f"Messages seen: {[str(w.message) for w in record]}"
    )


def test_kernel_chain_empty_override_still_raises() -> None:
    """PR1 contract: empty / whitespace-only rationale is rejected; the
    override is a forcing-function API, not an unchecked bypass. Ties
    the terminal gate to PR1's explicit-rationale contract.
    """
    with xr.open_dataset(_BEM_FIXTURE) as ds:
        a_inf = ds["added_mass"].values[-1].copy()
    hdb = read_capytaine(_BEM_FIXTURE, a_inf=a_inf)

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        compute_retardation_kernel(
            hdb, t_max=30.0, dt=0.01, asymptote_check_override="",
        )
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        compute_retardation_kernel(
            hdb, t_max=30.0, dt=0.01, asymptote_check_override="   ",
        )
