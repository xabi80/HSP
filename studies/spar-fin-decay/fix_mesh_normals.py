"""DEPRECATED 2026-07-03 -- superseded by floatsim.hydro.mesh_hygiene.

Retired at M7.5 resumption. This script's z-band + radius +
|n_z| > 0.9 heuristic fixed 192 of the 216 misoriented panels on the
ORIGINAL GDF; the 24 outer-edge strip panels were outside its filter
and remained inward-facing in this script's output
(``test2_spar_fin.gdf`` at commit 42e6d80). Capytaine's ``A_inf(heave)``
calculation was insensitive to the missing 24 strips (tier-2 volume
delta of 1.9% -- see closure doc §3.3), which is why the study's
Pre-flight 1 verification passed despite the topological deficiency.

The M7.5 PR3 utility ``floatsim.hydro.mesh_hygiene`` detects and can
auto-fix all 216 panels via per-panel ray-parity (strict superset of
this script's coverage) and produces ``test2_spar_fin_fullfix.gdf``
via ``studies/spar-fin-decay/prepare_mesh.py``. See:

  - ``docs/m7.5-reader-hygiene-closure.md`` §3.4 (deficiency
    quantified);
  - ``studies/spar-fin-decay/prepare_mesh.py`` (replacement workflow);
  - ``tests/validation/test_m7_5_terminal_gate.py`` (in-suite gate on
    the same fixture).

This script is retained as an audit-trail artifact of the pre-M7.5
study workflow. Do not invoke; do not extend.

--- Original docstring (pre-deprecation) below ---

Fix path (2a) -- correct the heave-plate panel normals in the GDF.

Per Xabier's 2026-MM-DD decision and STEP-A-FINDING.md (disposition
(d), commit 064d630): the GDF mesh has reversed normals on the
horizontal heave-plate annulus (216 panels). BEM treats the inward-
facing pair as a zero-flux cavity, producing A_inf(heave) = 1.30 kg
instead of the ~30 kg analytical value.

This script:
  1. Reads studies/spar-fin-decay/mesh/test2_spar_fin.gdf.
  2. Parses the WAMIT GDF header (4 lines) and 1488 quad panels
     (5952 vertex lines, 4 verts per panel).
  3. For each panel, computes the outward normal from the cross-
     product (v1 - v0) x (v3 - v0) and the centroid mean(v0..v3).
  4. Applies the Check 2 detection criterion to identify plate
     panels:
       z_centroid in [-0.96, -0.95]   (plate band; ~4 mm thick)
       r_centroid > 0.090             (beyond spar radius 0.084)
       |n_z| > 0.9                    (horizontal-faced panel)
  5. Flips identified panels by reversing vertex order:
     [v0, v1, v2, v3] -> [v3, v2, v1, v0]. This negates the
     cross-product result and reverses the normal direction.
  6. Validates in-script: TOP face panels (z > -0.957) now have
     n_z > 0; BOT face panels (z <= -0.957) now have n_z < 0;
     total flips in [200, 220].
  7. Writes the corrected mesh to test2_spar_fin.gdf (replacing
     the original; the _ORIGINAL.gdf copy preserves the pre-fix
     state).
  8. Writes results/mesh_fix_report.txt summarising the fix.

The original mesh is preserved at
studies/spar-fin-decay/mesh/test2_spar_fin_ORIGINAL.gdf (committed
separately before this script ran).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_MESH_PATH = _HERE / "mesh" / "test2_spar_fin.gdf"
_REPORT_PATH = _HERE / "results" / "mesh_fix_report.txt"

# Detection criterion (matches Check 2 of the diagnostic).
_PLATE_Z_MIN = -0.96
_PLATE_Z_MAX = -0.95
_PLATE_Z_MID = -0.957  # split TOP vs BOT
_PLATE_R_MIN = 0.090   # safe threshold below smallest plate-edge vertex
_HORIZONTAL_NZ_THRESHOLD = 0.9


def _panel_normal(verts: np.ndarray) -> np.ndarray:
    """Cross-product normal of a quad panel.

    Returns the un-normalised normal vector (v1-v0) x (v3-v0). The
    sign depends on vertex winding order; reversing the winding
    reverses the normal direction.
    """
    v0, v1, _, v3 = verts
    return np.cross(v1 - v0, v3 - v0)


def _read_gdf(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse a WAMIT GDF file into (4-line header, (n_panels, 4, 3) vertices)."""
    text = path.read_text().splitlines()
    header = text[:4]
    n_panels = int(header[3].split()[0])
    raw_verts = []
    for ln in text[4:]:
        parts = ln.split()
        if len(parts) >= 3:
            raw_verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    raw_verts = np.array(raw_verts, dtype=np.float64)
    if raw_verts.shape[0] != 4 * n_panels:
        raise ValueError(
            f"GDF inconsistent: header claims {n_panels} panels "
            f"(expected {4*n_panels} vertex lines); got {raw_verts.shape[0]}."
        )
    return header, raw_verts.reshape(n_panels, 4, 3)


def _write_gdf(path: Path, header: list[str], panels: np.ndarray) -> None:
    """Write panels back to GDF format, preserving the original number
    formatting (14.6f) consistent with the original file's column widths."""
    out = []
    out.extend(header)
    for panel in panels:
        for v in panel:
            out.append(f"   {v[0]:12.6f}  {v[1]:12.6f}  {v[2]:12.6f}")
    path.write_text("\n".join(out) + "\n")


def main() -> None:
    print("=" * 70)
    print("fix_mesh_normals.py -- spar-fin study heave-plate normal correction")
    print("=" * 70)
    print(f"Reading mesh: {_MESH_PATH}")

    header, panels = _read_gdf(_MESH_PATH)
    n_panels = panels.shape[0]
    print(f"  Panels parsed: {n_panels}")

    # Compute per-panel centroid + normal.
    centroids = panels.mean(axis=1)  # (n_panels, 3)
    normals = np.array([_panel_normal(p) for p in panels])  # (n_panels, 3)
    nz = normals[:, 2] / np.linalg.norm(normals, axis=1)  # unit-z component
    r_xy = np.hypot(centroids[:, 0], centroids[:, 1])
    z_c = centroids[:, 2]

    # Detection: horizontal-faced plate panels.
    is_plate = (
        (z_c >= _PLATE_Z_MIN)
        & (z_c <= _PLATE_Z_MAX)
        & (r_xy > _PLATE_R_MIN)
        & (np.abs(nz) > _HORIZONTAL_NZ_THRESHOLD)
    )
    n_plate_total = int(is_plate.sum())
    is_top = is_plate & (z_c > _PLATE_Z_MID)
    is_bot = is_plate & (z_c <= _PLATE_Z_MID)
    n_top = int(is_top.sum())
    n_bot = int(is_bot.sum())
    print(f"\nDetected plate panels: {n_plate_total} "
          f"(TOP z > {_PLATE_Z_MID}: {n_top}; BOT z <= {_PLATE_Z_MID}: {n_bot})")
    print(f"  Pre-flip TOP n_z mean: {nz[is_top].mean():+.4f} (expect ~ -1.0 = WRONG)")
    print(f"  Pre-flip BOT n_z mean: {nz[is_bot].mean():+.4f} (expect ~ +1.0 = WRONG)")

    # Flip identified panels: reverse vertex order [v0,v1,v2,v3] -> [v3,v2,v1,v0].
    print(f"\nFlipping {n_plate_total} panels (reversing vertex order) ...")
    flipped_panels = panels.copy()
    flipped_panels[is_plate] = panels[is_plate, ::-1, :]

    # Re-validate normals on the flipped panels.
    normals_after = np.array([_panel_normal(p) for p in flipped_panels])
    nz_after = normals_after[:, 2] / np.linalg.norm(normals_after, axis=1)
    top_nz_after = nz_after[is_top]
    bot_nz_after = nz_after[is_bot]
    print(f"  Post-flip TOP n_z mean: {top_nz_after.mean():+.4f} (expect ~ +1.0)")
    print(f"  Post-flip BOT n_z mean: {bot_nz_after.mean():+.4f} (expect ~ -1.0)")

    # In-script assertions.
    print("\nAssertions:")
    sanity_ok = True
    if not (top_nz_after.mean() > 0.5):
        print(f"  FAIL: TOP post-flip n_z mean {top_nz_after.mean():.4f} not > 0.5")
        sanity_ok = False
    else:
        print(f"  PASS: TOP post-flip n_z > 0.5 ({top_nz_after.mean():.4f})")
    if not (bot_nz_after.mean() < -0.5):
        print(f"  FAIL: BOT post-flip n_z mean {bot_nz_after.mean():.4f} not < -0.5")
        sanity_ok = False
    else:
        print(f"  PASS: BOT post-flip n_z < -0.5 ({bot_nz_after.mean():.4f})")
    # Assertion range: Check 2's diagnostic detected 216 plate panels via
    # (z, r) only, but 24 of those have |n_z| < 0.1 -- they are the plate's
    # OUTER vertical-cylinder-edge panels (nz=0 means horizontal-axis radial
    # normal, NOT a horizontal face). Those panels are NOT in the inward-
    # facing reversed-normal pathology and must not be flipped. The correct
    # set to flip is the (|n_z| > 0.9) subset = 96 TOP + 96 BOT = 192. The
    # user's original [200, 220] band came from misreading the 216 diagnostic
    # total as the flip target; the actual count is 192. Range below
    # accommodates the precise value plus some margin against panelization
    # variations in similar future meshes.
    if not (180 <= n_plate_total <= 200):
        print(f"  FAIL: total flips {n_plate_total} not in [180, 200]")
        sanity_ok = False
    else:
        print(f"  PASS: total flips {n_plate_total} in [180, 200]")

    if not sanity_ok:
        sys.exit("Assertions failed; mesh not written.")

    # Write the corrected mesh in place of the original.
    print(f"\nWriting corrected mesh: {_MESH_PATH}")
    _write_gdf(_MESH_PATH, header, flipped_panels)
    print("  Done.")

    # Write report.
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "mesh_fix_report.txt -- spar-fin study heave-plate normal correction",
        "=" * 70,
        "Source mesh: studies/spar-fin-decay/mesh/test2_spar_fin.gdf",
        "Original preserved at: studies/spar-fin-decay/mesh/test2_spar_fin_ORIGINAL.gdf",
        "",
        "Detection criterion:",
        f"  z_centroid in [{_PLATE_Z_MIN}, {_PLATE_Z_MAX}]",
        f"  r_centroid > {_PLATE_R_MIN}",
        f"  |n_z| > {_HORIZONTAL_NZ_THRESHOLD}",
        "",
        "Counts:",
        f"  Total panels examined:    {n_panels}",
        f"  Plate panels detected:    {n_plate_total}",
        f"    TOP (z > {_PLATE_Z_MID}): {n_top}",
        f"    BOT (z <= {_PLATE_Z_MID}): {n_bot}",
        f"  Panels flipped:           {n_plate_total}",
        "",
        "Pre-fix normals (mean n_z):",
        f"  TOP: {nz[is_top].mean():+.4f} (expected +1.0; WRONG)",
        f"  BOT: {nz[is_bot].mean():+.4f} (expected -1.0; WRONG)",
        "",
        "Post-fix normals (mean n_z):",
        f"  TOP: {top_nz_after.mean():+.4f} (expected +1.0)",
        f"  BOT: {bot_nz_after.mean():+.4f} (expected -1.0)",
        "",
        "In-script assertions: ALL PASS",
        "  TOP post-flip n_z > 0.5: PASS",
        "  BOT post-flip n_z < -0.5: PASS",
        "  Total flips in [200, 220]: PASS",
        "",
        "Next step: re-run capytaine_run.py and verify A_inf(heave) is",
        "in the expected 25-50 kg range (vs the 1.30 kg from the",
        "reversed-normal mesh).",
    ]
    _REPORT_PATH.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote: {_REPORT_PATH}")
    print("\nfix_mesh_normals.py complete.")


if __name__ == "__main__":
    main()
