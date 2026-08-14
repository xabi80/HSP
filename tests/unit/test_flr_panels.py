"""Panel pressure extraction must integrate to Capytaine's own resultants.

This is guard **(b)** in the schema's §7.2 language: a *panel-extraction*
diagnostic. It tests that the field we pull off the hull, integrated over that
hull, reproduces the force Capytaine itself reports.

It is deliberately **not** the G1.6 gate. G1.6 compares the **sum** of the FK and
diffraction fields against *FloatSim's combined applied excitation*, and lives in
FloatFEA's reader. An extraction that matches Capytaine perfectly could still
disagree with what the simulator applied — which is exactly the failure G1.6
exists to catch and this test cannot.

Run on a small sphere rather than the 1488-panel hull: the property under test is
the extraction and its integration, and a coarse mesh exercises both while
keeping the BEM solve fast.
"""

from __future__ import annotations

import numpy as np
import pytest

capytaine = pytest.importorskip("capytaine", reason="BEM extraction needs capytaine")

from floatsim.io.flr_panels import (  # noqa: E402
    N_HYDRO_DOF,
    PANEL_SOURCES,
    PanelGeometry,
    displacement_from_reference,
    extract_froude_krylov,
    extract_scattered,
    integrate_panel_pressure,
)

_RHO = 1025.0
_OMEGA = 2.0


@pytest.fixture(scope="module")
def solved():
    """A small floating sphere, solved with details kept."""
    import capytaine as cpt

    mesh = cpt.mesh_sphere(radius=1.0, center=(0, 0, -1.5), resolution=(6, 12))
    body = cpt.FloatingBody(mesh=mesh, name="s")
    body.add_translation_dof(name="Heave")
    body = body.immersed_part()
    solver = cpt.BEMSolver()

    diff = cpt.DiffractionProblem(body=body, omega=_OMEGA, wave_direction=0.0, rho=_RHO)
    rad = cpt.RadiationProblem(body=body, omega=_OMEGA, radiating_dof="Heave", rho=_RHO)
    res_d = solver.solve(diff, keep_details=True)
    res_r = solver.solve(rad, keep_details=True)

    faces = body.mesh.faces_centers
    geom = PanelGeometry(
        centroid=np.asarray(faces, dtype=np.float64),
        area=np.asarray(body.mesh.faces_areas, dtype=np.float64),
        normal=np.asarray(body.mesh.faces_normals, dtype=np.float64),
    )
    return solver, diff, res_d, res_r, geom


def test_froude_krylov_field_integrates_to_capytaine_fk_force(solved) -> None:
    """FK is the incident field on the hull; integrating it must match Capytaine."""
    _, diff, res_d, _, geom = solved
    p_fk = extract_froude_krylov(diff, geom)
    force = integrate_panel_pressure(p_fk, geom)

    from capytaine.bem.airy_waves import froude_krylov_force

    reported = complex(froude_krylov_force(diff)["Heave"])
    # Heave is the z component of the integrated field.
    assert force[2] == pytest.approx(reported, rel=2e-2), (
        f"FK extraction integrates to {force[2]:.6g} but Capytaine reports "
        f"{reported:.6g}"
    )


def test_diffraction_field_integrates_to_capytaine_diffraction_force(solved) -> None:
    """The scattered field, likewise, against Capytaine's own diffraction force."""
    solver, _, res_d, _, geom = solved
    p_diff = extract_scattered(solver, res_d, geom)
    force = integrate_panel_pressure(p_diff, geom)

    # DiffractionResult.forces is the SCATTERED force alone in capytaine 2.3.1,
    # not the total excitation -- verified against froude_krylov_force, which is
    # reported separately. Subtracting would have double-counted.
    reported = complex(res_d.forces["Heave"])
    assert force[2] == pytest.approx(reported, rel=2e-2), (
        f"diffraction extraction integrates to {force[2]:.6g} but Capytaine "
        f"reports {reported:.6g}"
    )


def test_radiation_field_integrates_to_added_mass_and_damping(solved) -> None:
    """The radiation field must reproduce the A and B Capytaine reports.

    **The sign of the damping term was MEASURED, not assumed.** The relation that
    holds in capytaine 2.3.1 is ``F_rad = omega^2 A + i omega B`` -- the
    conjugate of the form several references write -- which fixes the
    time-dependence convention as ``e^{-i omega t}``. FK and diffraction match
    with no sign issue, so the difference is in the A/B relation, not in the
    pressure extraction.

    This matters downstream: FloatFEA reconstructing radiation from A and B with
    the other sign would put a **180 degree phase error on the damping term**
    while leaving added mass correct -- large, coherent at the fundamental, and
    exactly the signature the schema's G1.6 failure-reading table attributes to a
    phase-convention error.
    """
    solver, _, _, res_r, geom = solved
    p_rad = extract_scattered(solver, res_r, geom)
    force = integrate_panel_pressure(p_rad, geom)

    a = float(res_r.added_mass["Heave"])
    b = float(res_r.radiation_damping["Heave"])
    expected = _OMEGA**2 * a + 1j * _OMEGA * b
    assert force[2] == pytest.approx(expected, rel=5e-2), (
        f"radiation extraction integrates to {force[2]:.6g}, expected "
        f"{expected:.6g} from A={a:.6g}, B={b:.6g}"
    )


def test_the_sign_convention_is_load_bearing(solved) -> None:
    """Dropping the outward-normal minus sign must break the match.

    Panel normals point outward into the fluid, so a positive pressure pushes
    the hull inward. A sign error here would flip every exported field while
    leaving its magnitude perfect — the kind of defect that survives a
    magnitude-only check.
    """
    from capytaine.bem.airy_waves import froude_krylov_force

    _, diff, _, _, geom = solved
    p_fk = extract_froude_krylov(diff, geom)
    correct = integrate_panel_pressure(p_fk, geom)[2]
    flipped = -correct
    reported = complex(froude_krylov_force(diff)["Heave"])
    assert abs(correct - reported) < abs(flipped - reported)


def test_displacement_from_reference_drives_the_validity_bound() -> None:
    """The quantity the validator applies its bound to."""
    ref = np.array([0.0, 0.0, -1.5, 1.0, 0.0, 0.0, 0.0])
    poses = np.array(
        [
            [0.0, 0.0, -1.5, 1.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, -1.5, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    d = displacement_from_reference(poses, ref)
    np.testing.assert_allclose(d, [0.0, 5.0])


def test_hydro_dof_count_is_72_not_102() -> None:
    """Only the twelve buoys carry hydrodynamics.

    The global state vector is 102 DOF across 17 bodies, but the hubs and the
    platform are structural. Summing radiation over 102 would index past the BEM
    data or pick up DOF with no radiation field at all.
    """
    assert N_HYDRO_DOF == 72
    assert N_HYDRO_DOF == 12 * 6


def test_froude_krylov_and_diffraction_are_separate_sources() -> None:
    """The v1.1 split, asserted so a later merge breaks a test."""
    assert "froude_krylov" in PANEL_SOURCES
    assert "diffraction" in PANEL_SOURCES
    assert "excitation" not in PANEL_SOURCES


def test_panel_count_mismatch_is_rejected() -> None:
    geom = PanelGeometry(
        centroid=np.zeros((4, 3)), area=np.ones(4), normal=np.tile([0, 0, 1.0], (4, 1))
    )
    with pytest.raises(ValueError, match="same mesh"):
        integrate_panel_pressure(np.zeros(5, dtype=complex), geom)
