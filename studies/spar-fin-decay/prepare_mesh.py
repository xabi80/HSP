"""Prepare the spar-fin BEM mesh via floatsim.hydro.mesh_hygiene (workflow alpha).

Runs the M7.5 PR3 tier-1 (per-panel ray-parity) + tier-2
(hydrostatic-volume physics screen) on the study's ORIGINAL GDF and
writes a fully corrected mesh for capytaine_run.py.

Workflow alpha (M7.5 resumption 2026-07-03) supersedes fix_mesh_normals.py:

    ORIGINAL GDF (as exported from OrcaWave)
        216 inward panels (192 horizontal plate + 24 outer-edge strip)
        96 open boundary edges (plate-spar junction + disk rim)
        |
        v
    floatsim.hydro.mesh_hygiene.fix_panel_normals(default)
        flips all 216 inward panels via per-panel ray-parity
        (strict superset of fix_mesh_normals.py's 192-panel z-band fix)
        |
        v
    test2_spar_fin_fullfix.gdf         <- consumed by capytaine_run.py

See:
    docs/m7.5-reader-hygiene-closure.md §7 (resumption plan);
    docs/multibody-conventions.md Item 5 (panel-normal convention);
    tests/validation/test_m7_5_terminal_gate.py::test_mesh_chain_...
        (in-suite terminal gate on the same fixture).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from floatsim.hydro.mesh_hygiene import (
    GdfMesh,
    check_hydrostatic_volume,
    fix_panel_normals,
    load_gdf_panels,
    validate_panel_normals,
    write_gdf_panels,
)

from waterline_balance import displaced_volume_below_waterline

_HERE = Path(__file__).resolve().parent
_ORIGINAL_MESH = _HERE / "mesh" / "test2_spar_fin_ORIGINAL.gdf"
_FULLFIX_MESH = _HERE / "mesh" / "test2_spar_fin_fullfix.gdf"
_EQDRAFT_MESH = _HERE / "mesh" / "test2_spar_fin_fullfix_eqdraft.gdf"

_RHO_KG_M3 = 1025.0
_BODY_MASS_KG = 28.67
_R_SPAR = 0.0841
_A_WP = np.pi * _R_SPAR**2

# Equilibrium sink measured 2026-07-04 by waterline_balance.py:
# at the design waterline the fullfix mesh displaces 24.467 kg
# (< 28.67 kg body), so the free-floating buoy sinks
# dz = (M - m_disp)/(rho * A_wp) = 0.1846 m to balance.
_DZ_EQ_M = 0.1846


def main() -> None:
    print("=" * 70)
    print("spar-fin mesh prep -- workflow alpha (M7.5 mesh_hygiene)")
    print("=" * 70)
    print(f"Input mesh:  {_ORIGINAL_MESH}")
    print(f"Output mesh: {_FULLFIX_MESH}")
    print()

    if not _ORIGINAL_MESH.is_file():
        raise SystemExit(f"FATAL: ORIGINAL mesh not found at {_ORIGINAL_MESH}")

    mesh = load_gdf_panels(_ORIGINAL_MESH)
    print(f"Loaded {mesh.n_panels} panels.")

    # --- Tier 1a: validate as-stored (expect ValueError with 216 inward) ---
    print()
    print("Tier 1 (report path) -- expecting 216 inward, 96 open edges ...")
    report = validate_panel_normals(mesh, return_report=True)
    print(f"  inward panels:        {report.inward_indices.size}")
    print(f"  indeterminate panels: {report.indeterminate_indices.size}")
    print(f"  n_open_edges:         {report.n_open_edges}")
    if report.inward_indices.size != 216 or report.n_open_edges != 96:
        raise SystemExit(
            f"UNEXPECTED: report ({report.inward_indices.size} inward, "
            f"{report.n_open_edges} open edges) does not match the terminal-"
            f"gate ground truth (216 / 96). Investigate mesh source before "
            f"proceeding."
        )
    print("  MATCH: terminal-gate ground truth (216 inward, 96 open edges).")

    # --- Tier 1b: auto-fix (default per-panel ray-parity path) ---
    print()
    print("Auto-fix via fix_panel_normals (default ray-parity path) ...")
    fixed = fix_panel_normals(mesh)
    write_gdf_panels(fixed, _FULLFIX_MESH)
    print(f"  Wrote fullfix mesh to {_FULLFIX_MESH}.")

    print()
    print("Re-validating fullfix mesh -- expecting 0 inward ...")
    fixed_report = validate_panel_normals(fixed, return_report=True)
    print(f"  inward panels:        {fixed_report.inward_indices.size}")
    print(f"  indeterminate panels: {fixed_report.indeterminate_indices.size}")
    print(f"  n_open_edges:         {fixed_report.n_open_edges}")
    if fixed_report.inward_indices.size != 0:
        raise SystemExit(
            f"UNEXPECTED: {fixed_report.inward_indices.size} inward panels "
            f"remain after auto-fix. Investigate before running BEM."
        )
    print("  MATCH: fullfix mesh is orientation-clean.")

    # --- Tier 2: hydrostatic-volume physics screen ---
    print()
    print("Tier 2 (check_hydrostatic_volume) ...")
    vr = check_hydrostatic_volume(fixed, rho=_RHO_KG_M3, mass=_BODY_MASS_KG)
    print(f"  signed_volume:     {vr.signed_volume:+.4e} m^3")
    print(f"  displaced_mass:    {vr.displaced_mass:+.4f} kg")
    print(f"  body mass:         {vr.mass:.4f} kg")
    print(f"  residual_fraction: {vr.residual_fraction:+.4f}   (reserve buoyancy)")
    print()
    print(
        "Note: residual_fraction ~ +0.43 is a documented reserve-buoyancy\n"
        "  property of the spar-buoy geometry (fully-submerged displaced\n"
        "  mass ~40 kg vs 28.67 kg body). See closure §3.4 + tracker\n"
        "  BEM-MESH-STRIP-PANELS-STUDY-FIXTURE."
    )
    # --- Translate fullfix mesh to the true equilibrium draft ---
    print()
    print("=" * 70)
    print(f"Translating fullfix mesh DOWN by dz = {_DZ_EQ_M} m -> eqdraft mesh")
    print("=" * 70)
    eq_panels = fixed.panels.copy()
    eq_panels[..., 2] -= _DZ_EQ_M  # z_new = z_old - dz
    eqdraft = GdfMesh(header_lines=fixed.header_lines, panels=eq_panels)
    write_gdf_panels(eqdraft, _EQDRAFT_MESH)
    print(f"  Wrote eqdraft mesh to {_EQDRAFT_MESH}.")

    # Verify displaced mass at the new (z=0) waterline == body mass.
    v_disp_eq = displaced_volume_below_waterline(eqdraft.panels)
    m_disp_eq = _RHO_KG_M3 * v_disp_eq
    rel = (m_disp_eq - _BODY_MASS_KG) / _BODY_MASS_KG
    print(f"  displaced volume at eqdraft waterline = {v_disp_eq:.6e} m^3")
    print(f"  displaced mass  at eqdraft waterline  = {m_disp_eq:.4f} kg "
          f"(body {_BODY_MASS_KG} kg, rel {rel:+.4%})")
    if abs(rel) > 0.01:
        print(f"  WARNING: displaced mass off by {rel:+.2%} (> 1%); "
              f"dz may need refinement.")
    else:
        print("  OK: equilibrium displaced mass within 1% of body mass.")
    print(f"  New CoG z = -0.8317 - {_DZ_EQ_M} = {-0.8317 - _DZ_EQ_M:.4f} m")
    print()
    print("Mesh prep complete. capytaine_run.py consumes the eqdraft mesh.")


if __name__ == "__main__":
    main()
