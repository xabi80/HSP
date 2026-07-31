# 12-buoy platform study (M11b)

The terminal FloatSim configuration: **4 clusters × 3 spar-fin buoys = 12
buoys**, articulated (`yaw_locked`) buoy→hub and hub→platform, on a rigid
central cross (dry, above water). 17 bodies, 102 DOF, 16 joints (64
constraints), 38 free DOF. Geometry: [`docs/platform-geometry.md`](../../docs/platform-geometry.md);
plan: [`docs/m11-platform-plan.md`](../../docs/m11-platform-plan.md) (M11b).

## Files

- **`platform_common.py`** — geometry, mass balance (402.04 kg, per-buoy
  33.5033), the platform **draft** (`PLATFORM_DZ = 0.21638 m`, re-derived on
  the mesh by `derive_draft()`), `buoy_centers()`, and **`build_platform_deck()`**
  (the 17-body / 16-joint deck, reused by the tests and PR7).
- **`build_platform_mesh.py`** — builds + `mesh_hygiene`-screens the
  17,856-panel mesh at the platform draft; `python build_platform_mesh.py`
  writes `mesh/platform12_fullfix.gdf`.

## The mesh is regenerated, not committed

`mesh/platform12_fullfix.gdf` (17,856 panels, ~3.2 MB) is a **deterministic
output** of `build_platform_mesh.py` from the committed single-buoy hull, so it
is git-ignored (M11b PR6 decision). **Regenerate it before the BEM solve:**

```bash
python build_platform_mesh.py
```

## Status

- **PR6 (this):** mesh generator + assembly. Gated by
  `tests/validation/test_m11b_pr6_platform_assembly.py` (draft/geometry, mesh
  hygiene, and the n=102 assembly preconditions — all promoted from M11b
  Phase-1 measurements).
- **PR7 (next):** BEM at scale on the regenerated mesh. The contaminated-slice
  **conditioning detector is embedded in PR7** (per-frequency conditioning
  number gate); wetted-count cost model and C4v-symmetry lever per M11b
  Phase-1 Findings G1/G2. **RHS-bound** (72 DOF), ≈ 42 min, 12.71 GB peak
  (not memory-bound).
