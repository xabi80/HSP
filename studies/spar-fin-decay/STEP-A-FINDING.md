# Step A finding — BEM geometry vs Morison heave-plate spec mismatch

**Date:** 2026-06-27.
**Status:** STOP per the locked plan (Step B Check 1 fail). Awaiting
Xabier's geometry-interpretation decision before proceeding to Step C
(decks).

## What ran cleanly

- **Capytaine 2.3.1 installed** as a study-level dep (not added to
  FloatSim runtime baseline; same pattern as `openfast_io` per
  CLAUDE.md §9).
- **BEM solve:** 560 problems (80 omegas × 6 radiation DOFs + 80
  diffraction problems) solved in ~1m35s. No solver errors.
- **NetCDF output:** `studies/spar-fin-decay/capytaine_bem.nc` saved
  with the FloatSim-Capytaine-reader-compatible schema
  (`added_mass`, `radiation_damping`, `excitation_force`,
  `Froude_Krylov_force`, `diffraction_force`, `hydrostatic_stiffness`,
  with the `complex=['re','im']` split via Capytaine's
  `separate_complex_values`).
- **Hydrostatic stiffness via `immersed_part()` + `rotation_center`**:
  Capytaine 2.x requires both explicitly (the BEM solver auto-clips
  but the hydrostatic computation does not). The
  `capytaine_run.py` script captures the API for any future runs.

## What the BEM says

| metric | Capytaine | locked-plan prediction | rel-err |
|---|---|---|---|
| C33 (heave hydrostatic) | 221.08 N/m | 223.43 N/m | -1.05 % ✅ |
| A_inf(heave) | 1.30 kg | 30-70 kg | -96 % to -98 % ❌ |
| Predicted T_n | 2.31 s | 3.2-3.8 s | -28 % to -39 % ❌ |
| Predicted ζ_rad | 0.05 % | 5-15 % | -99 % ❌ |
| B(omega) ≥ 0 everywhere | yes | yes | ✅ |

## What the mesh actually represents

GDF vertex inspection (5952 vertices for 1488 panels):

- **z range**: -1.094 m to +0.757 m. The mesh extends both above
  and below the assumed waterline z = 0.
- **Waterline cross-section** (96 vertices at |z| < 0.001 m):
  x range [-0.0841, +0.0841] m, y range [-0.0841, +0.0841] m
  → circular spar of **radius 0.0841 m** at the waterline.
- **Mid-immersed cross-section** (z ∈ [-0.15, -0.05]): same
  [-0.0841, +0.0841] m in x and y. The spar maintains constant
  radius below the waterline.
- **Near-bottom cross-section** (z < -1.0 m): same
  [-0.0841, +0.0841] m. **No horizontal disk feature at the
  bottom.**

The mesh represents a **slender vertical spar of radius
0.0841 m**, length ~1.85 m, with the immersed portion ~1.094 m.
The GDF header "spar+fin 5mm offset" suggests the "fin" is a thin
vertical surface offset from the spar axis by 5 mm — NOT a
horizontal heave plate.

The waterplane-area-derived C33 confirms r_spar matches the
expected 0.0841 m: A_wp = π · 0.0841² = 0.0222 m²;
ρ · g · A_wp = 1025 · 9.81 · 0.0222 = 223.4 N/m vs Capytaine's
221.08 N/m (1 % discretization gap).

The slender-spar A_inf_heave matches the expected end-effect
formula for a vertical cylinder with no bottom disk:
A_inf ≈ (8/3) · ρ · r³ = (8/3) · 1025 · 0.0841³ = 1.63 kg
(close to the 1.30 kg Capytaine result; the slight gap is the
fin's contribution + mesh-end shape).

## The geometry-vs-Morison-spec mismatch

The locked-plan Morison heave-plate spec:

- Projected area = π · (0.215)² = 0.1452 m²
- C_D = 5.0 (Tao & Cai 2004; valid for *horizontal disks*)
- Located at (0, 0, -1.022) m, vertical normal

**This describes a horizontal disk of radius 0.215 m** —
fundamentally different from what the BEM mesh contains. The
disk would contribute:

- A_inf_heave ≈ (8/3) · ρ · R³ = (8/3) · 1025 · 0.215³ = 27.2 kg
  → would shift A_inf into the user-expected 30-70 kg band.
- Substantial heave radiation damping (B at peak ~5-10 kg/s) at
  the relevant frequencies.

The user's locked-plan predictions (T_n 3.2-3.8 s; ζ_rad
5-15 %) are arithmetically consistent with a body having
A_inf ≈ 30 kg and B(omega_n) ≈ 5-10 kg/s — exactly what a
0.215 m heave plate adds. So the predictions ARE internally
consistent **with the disk-included geometry**, but the BEM
mesh does NOT contain that disk.

## The decision point

Three possible dispositions:

**(a) The Morison heave plate is intentionally separate from the
BEM** — the BEM captures the spar's slender-body hydrodynamics
(radiation, FK + diffraction excitation) while the Morison
element ADDS the heave-plate physics that the mesh deliberately
omits. In this case:
- Proceed with both decks.
- Reset predictions: BEM-only ζ_rad ≈ 0.05 % (essentially
  undamped, decays slowly), T_n ≈ 2.3 s.
- BEM+Morison ζ_total dominated by quadratic drag from the
  heave plate; envelope hyperbolic (not exponential) per the
  M5 PR4 Faltinsen Ch.4 framework.
- This is a legitimate physical model: spar hydrodynamics in
  BEM, heave-plate drag in Morison.

**(b) The mesh is missing the heave-plate geometry** — the BEM
should have included a horizontal disk at the fin location, and
running with the slender-spar mesh produces a hydrodynamically
incomplete model. In this case:
- A new mesh with the heave-plate disk is needed before Step C.
- The 5mm-offset fin in the current mesh is either irrelevant
  or also indicates a different physical feature.

**(c) The Morison spec is wrong** — there's no heave plate; the
fin is a vertical surface (per the GDF header). The 0.215-m
projected area was a misinterpretation, and the Morison element
should be something else (or omitted entirely). In this case:
- BEM-only decay IS the full prediction.
- The "with Morison" run becomes either trivial (no Morison
  needed) or describes a different drag feature than originally
  specified.

The mesh evidence (slender spar, vertical fin at 5mm offset)
points toward (a) or (c). The Morison spec's specific values
(R = 0.215 m, C_D = 5.0 from Tao & Cai for horizontal disks)
strongly suggest the user intended (a).

## What's needed from Xabier

Pick the disposition and confirm or revise the Morison element:

- **If (a)**: confirm. Proceed with both decks; reset expected
  predictions in the README per the BEM data.
- **If (b)**: request new mesh with the heave-plate disk
  included. Re-run Capytaine.
- **If (c)**: confirm. Proceed with BEM-only deck only; second
  deck becomes optional.

## Capytaine 2.x API notes captured for the future

Two pieces of API drift documented in `capytaine_run.py`:

1. `body.add_all_rigid_body_dofs()` works as the user's plan
   expected.
2. `compute_hydrostatic_stiffness` requires:
   - explicit `body.immersed_part()` (BEM solver auto-clips but
     hydrostatic path doesn't), AND
   - `body.rotation_center` attribute (defaults to nothing;
     needs to be set, typically to the CoG for rigid-body
     dynamics).
3. Complex-valued variables (e.g. `excitation_force`) must be
   passed through `capytaine.io.xarray.separate_complex_values`
   before `to_netcdf` — otherwise netCDF4 raises on
   complex128 dtype. This produces the `complex=['re','im']`
   split that the FloatSim Capytaine reader expects.

## Files in this commit

- `studies/spar-fin-decay/README.md` — study scope + locked inputs.
- `studies/spar-fin-decay/mesh/test2_spar_fin.gdf` — the mesh
  (copied from `Orca/orcawave/HSFP_Test_SparFin/`).
- `studies/spar-fin-decay/capytaine_run.py` — Step A + B script
  (Capytaine BEM + sanity checks).
- `studies/spar-fin-decay/capytaine_bem.nc` — BEM output, valid
  for FloatSim Capytaine reader. ~5 MB.
- `studies/spar-fin-decay/STEP-A-FINDING.md` — this file.

Steps C onward are blocked on the geometry-interpretation
decision above.
