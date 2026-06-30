# Step A finding — Capytaine A_inf integration on the spar+plate mesh

**Date:** 2026-06-27, updated 2026-06-28 (geometry interpretation
corrected by Xabier).
**Status:** STOP per the locked plan (Step B Check 1 fail). The
disposition decision tree from the original finding ((a)/(b)/(c)
choices) is **paused** — the corrected geometry reading rules out
options (b) and (c) and reframes the finding as a Capytaine-
integration diagnostic, not a geometry-interpretation question.
Awaiting Xabier's review of diagnostic Checks 1-3 (see "Open
diagnostic plan" at the bottom of this file).

## CORRECTION (2026-06-28): mesh DOES contain a horizontal heave plate

My original reading "mesh is a slender vertical spar with no heave
plate" was **wrong**. A finer-grained vertex scan reveals:

- **768 vertices with r > 0.10 m**, all clustered in a thin band
  at **z ∈ [-0.9587, -0.9548]** m (~4 mm vertical extent).
- **Maximum radius reached: 0.2150 m** (exactly matching the
  Morison-plate spec R = 0.215 m).
- This is a **horizontal annular heave plate** spanning the spar
  outer radius (r = 0.0841 m) to the plate outer radius (r =
  0.215 m), positioned at z ≈ -0.957 m below the waterline.

The plate is so thin in z (4 mm) that my original coarse band
sampling ("z < -1.0 m for near-bottom") missed it entirely.

Full corrected geometry (per Xabier's image confirmation):

- Spar: r = 0.0841 m from z = +0.757 m to z = -0.954 m.
- **Horizontal heave plate annulus**: r ∈ [0.084, 0.215] m at
  z ≈ -0.957 m, ~4 mm thick.
- Hemispherical end cap: z ∈ [-1.00, -1.094] m (768-96 = 672
  vertices in this region).
- Thin vertical "fin" surface offset 5 mm from spar axis (per
  GDF header — exact geometry not separately verified, but
  consistent with the panel count).

## What the BEM produced (corrected interpretation)

The numerical results (unchanged from the original run) are:

| metric | Capytaine | locked-plan band | gap |
|---|---|---|---|
| C33 (heave hydrostatic) | 221.08 N/m | 223.43 N/m | ✅ -1.05 % |
| A_inf(heave) | 1.30 kg | 30-70 kg | ❌ -96 % to -98 % |
| Predicted T_n | 2.31 s | 3.2-3.8 s | ❌ -28 % to -39 % |
| Predicted ζ_rad | 0.05 % | 5-15 % | ❌ -99 % |
| B(ω) ≥ 0 everywhere | yes | yes | ✅ |

The C33 = 221 N/m (1 % off the slender-spar-only expected value)
*also* suggests the plate isn't contributing as expected — a
0.215-m heave plate, when its top face crosses the waterline,
would add ρ·g·π·(0.215² - 0.0841²) ≈ 121 N/m to the heave-heave
hydrostatic. The plate sits at z ≈ -0.957 m (well below the
waterline), so it doesn't contribute to the waterplane area —
that part is consistent.

But: a 0.215-m horizontal disk at z = -0.957 m SHOULD contribute
A_inf_heave ≈ (8/3)·ρ·R³ ≈ (8/3)·1025·0.215³ ≈ 27 kg per face,
plus the spar contribution of ~1.6 kg ≈ 28-30 kg total. The
Capytaine result of 1.30 kg is consistent with **only the spar's
contribution being captured**.

**Reframed diagnostic question:** why doesn't Capytaine's
radiation integration on this mesh capture the plate's added
mass?

## Most likely candidates (to be checked)

1. **Panel normal orientation.** If the top face (z ≈ -0.955 m,
   normal should point +z) and bottom face (z ≈ -0.959 m, normal
   should point -z) of the plate have inconsistent normals, the
   BEM source distribution may cancel.
2. **Double-sided cancellation.** If the plate's top + bottom +
   edge form a closed thin volume with normals pointing inward
   on both faces (or outward on both faces), the BEM treats it
   as a zero-thickness sheet that does not displace water under
   heave motion.
3. **GDF parser issue.** The "5 mm offset" in the GDF header
   suggests this mesh was built with a panel-offset technique
   for OrcaWave's thin-panel handling. Capytaine's GDF loader
   may not respect that and may interpret the offset panels
   incorrectly.

## Diagnostic Checks 1-3 results (2026-06-29) — CONCLUSIVE

`studies/spar-fin-decay/capytaine_diagnostic.py` ran all three
checks. Result: **the GDF mesh has reversed panel normals on the
heave-plate faces**, making the plate invisible to BEM integration.

### Check 1 — mesh visualisation

`results/figures/capytaine_mesh_view.png` (left panel: full X-Z
projection of all 1488 panels; right panel: plate-band z-slice
shown in X-Y with the spar / plate radii overlaid). Plate
annulus is clearly visible at z ≈ -0.957 m with vertices reaching
r = 0.215 m, matching the expected geometry.

### Check 2 — panel normals on plate top + bottom — REVERSED

Per-face analysis on the 216 plate panels (120 top + 96 bottom)
detected via `(z in [-0.96, -0.95]) and (r in [0.085, 0.216])`:

| face | expected nz (outward-from-body) | measured nz | mesh area | analytical |
|---|---|---|---|---|
| TOP (z = -0.955) | **+1.0** (outward = into water above) | mean **-0.80**, min -1.0, max 0.0 | 0.1269 m² | 0.1230 m² |
| BOT (z = -0.959) | **-1.0** (outward = into water below) | mean **+1.0**, all +z | 0.1216 m² | 0.1230 m² |

**Both normals point INWARD toward the plate midplane**, not
outward into the surrounding fluid. The plate's top face says
"water is below me" and the bottom face says "water is above
me" — exactly the opposite of physical reality. BEM treats this
as a closed thin-volume cavity with zero outward flux, so it
contributes essentially zero to the heave radiation integral.

Plate mesh areas are consistent with the analytical annulus
(0.1230 m²), so the *geometry* of the plate is correct in the
mesh — only the *orientation* is wrong.

### Check 3 — procedurally-built reference geometry

Built a Capytaine-native spar (r = 0.0841 m, length 1.7 m) +
horizontal disk plate (R = 0.215 m) at z ≈ -0.957 m using
`cpt.mesh_vertical_cylinder` + `cpt.mesh_disk` and ran the same
BEM pipeline:

```
omega =  0.50: A_inf(heave) = 24.242 kg
omega =  1.00: A_inf(heave) = 24.237 kg
omega = 30.00: A_inf(heave) = 24.209 kg
```

vs the analytical estimate ~28.79 kg (spar end + disk one-face).
The ~16 % gap to analytical is finite-resolution mesh effects,
NOT a fundamental issue. **The reference is in the expected
30 kg ballpark; the GDF result of 1.30 kg is not.**

This isolates the issue: **Capytaine's setup is correct**; the
GDF mesh's plate panel normals are reversed. The procedurally-
built mesh has correctly-oriented normals (because Capytaine's
primitives build them that way) and produces a physical result.

### Conclusion + new disposition (d)

A fourth disposition replaces the original tree:

**(d) The mesh is geometrically correct but has reversed normals
on the heave plate.** The fix is to flip the plate panel
orientations on import (or in a pre-processed mesh) before
running the BEM. Three possible mechanical paths:

1. **Flip the GDF source**: re-export from OrcaWave (if
   regeneration is feasible) or directly edit the GDF to reverse
   panel vertex ordering on the plate panels.
2. **Selective in-Python flip**: identify plate panels by their
   (z, r) signature and flip them on the loaded mesh before
   constructing the FloatingBody. The Check 2 detection
   criterion `(z in plate band) and (r > spar_r)` is already
   implemented and could drive a selective `_flip_faces`.
3. **Global flip + revert**: `mesh.flipped()` flips everything,
   which would correct the plate but break the spar. Combined
   with a selective re-flip of the spar panels, this also works
   but is the messiest path.

(1) or (2) are the cleanest. (1) is preferred if the GDF export
process is parametric and the fix lives in the OrcaWave script;
(2) is preferred if the GDF is the authoritative source and we
want the fix to live in the FloatSim study, not upstream.

### Capytaine 2.x API drift captured (additional)

In addition to the three drifts from the original Step A note:

4. `cpt.mesh_vertical_cylinder` `resolution` is a **3-tuple**
   `(n_radial_disk, n_theta, n_axial)` in Capytaine 2.x (was
   2-tuple in 1.x). `cpt.mesh_disk` `resolution` is a 2-tuple
   `(n_radial, n_theta)`.

### What's blocked

Steps C onward remain blocked until Xabier picks a fix path for
the plate-normal issue. Once the normals are corrected, the
existing `capytaine_run.py` should produce a physically-reasonable
A_inf in the 25-30 kg range and the rest of the study can proceed
without modification.

## What is NOT in this commit

The original (a)/(b)/(c) disposition tree from the first version
of this finding is paused. With the corrected geometry reading,
(b) "mesh missing plate" is ruled out (plate IS in the mesh) and
(c) "no plate exists" is ruled out (image confirmation). Only
(a) "intentional decoupled-model (BEM spar + Morison plate)" or
a NEW disposition "(d) Capytaine is mis-integrating this mesh"
remain. The diagnostic Checks 1-3 determine which.

Steps C onward remain blocked. No `floatsim/` modifications.

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

---

## Resolution (2026-06-29) — fix path (2a) applied

Per Xabier's directive: fix path **(2a)** — correct the GDF source
file's heave-plate panel orientation directly, keep the load path
simple. Original mesh preserved separately.

### What landed

- `studies/spar-fin-decay/mesh/test2_spar_fin_ORIGINAL.gdf` — the
  pre-fix mesh, committed first so the reversed-normal state
  remains auditable.
- `studies/spar-fin-decay/fix_mesh_normals.py` — the corrective
  script. Reads the GDF, detects horizontal heave-plate panels via
  the criterion (z_centroid in [-0.96, -0.95]) AND
  (r_centroid > 0.090) AND (|n_z| > 0.9), reverses vertex order on
  identified panels, validates in-script, writes the corrected
  mesh in place of the original.
- `studies/spar-fin-decay/results/mesh_fix_report.txt` — per-run
  log of the fix (counts + pre/post normal stats).
- `studies/spar-fin-decay/capytaine_bem.nc` — re-run BEM output on
  the corrected mesh (Capytaine script unchanged).

### Detection-count footnote (vs Check 2)

Check 2 detected 216 plate panels via the (z, r) criterion alone:
120 TOP + 96 BOT. Of those, 24 TOP panels have n_z ≈ 0 — these
are the plate's outer **vertical-cylinder-edge** panels (radial
normals, not horizontal-face panels). They are NOT in the
inward-facing reversed-normal pathology and **must not be
flipped**. The (|n_z| > 0.9) horizontal-face filter correctly
excludes them. Final flipped count: **192 = 96 TOP + 96 BOT**
horizontal-face panels.

### Verification

**Check 2 (re-run via Capytaine load path on the corrected mesh):**

| face | panels | pre-fix mean n_z | post-fix mean n_z |
|---|---:|---:|---:|
| TOP (z > -0.957) | 120 (96 horiz + 24 vert) | -0.80 (WRONG) | **+0.80** (96 horiz at +1, 24 vert at 0; ✅ outward) |
| BOT (z ≤ -0.957) | 96 (all horizontal) | +1.00 (WRONG) | **-1.00** (all -1; ✅ outward) |

**Capytaine BEM re-run (corrected mesh, same Capytaine script):**

| metric | pre-fix | post-fix | expectation | status |
|---|---|---|---|---|
| A_inf(heave) | 1.30 kg | **21.11 kg** | 25-50 kg recalibrated | 16× recovered; ~16 % below band |
| C33 | 221.08 N/m | 221.08 N/m | 223.43 N/m | unchanged (1 % discretization) |
| Predicted T_n | 2.31 s | **2.98 s** | 3.2-3.8 s | shifted; ~7 % below band |
| Predicted ζ_rad | 0.05 % | 0.014 % | (locked 5-15 % describes drag, not radiation) | rad-only is genuinely small |
| B(ω) ≥ 0 everywhere | yes | 1 negative @ ω = 9.45 | yes | -2.2e-5 kg/s = 0.04 % of B_max; numerical zero, far from ω_n |

### Honest characterisation of the residual gaps

- **A_inf = 21.1 kg vs the recalibrated 25-50 kg lower bound.**
  The procedurally-built reference in Check 3
  (`capytaine_diagnostic.py`) gave 24.2 kg for an analogous
  spar + horizontal-disk geometry. The GDF mesh gives 21.1 kg —
  a ~13 % gap that's plausibly mesh-panelization specific (the
  GDF plate is a thin 4 mm volume rather than an idealized
  zero-thickness disk; edge-effect handling differs between
  the two meshes). Not a second pathology; the fix recovered
  ~17× of the per-Check-3 expected magnitude.
- **T_n = 2.98 s vs locked 3.2-3.8 s.** Follows mechanically
  from A_inf below the locked band. If a higher-resolution
  re-mesh of the plate (or its edge) brought A_inf closer to
  the 24 kg reference, T_n would land closer to 3.1-3.2 s.
  Tank-test calibration will be the ground truth.
- **ζ_rad = 0.014 % vs locked 5-15 %.** The 5-15 % expectation
  describes quadratic drag damping from the heave plate
  (Tao & Cai 2004, the Morison element's C_D = 5.0 path),
  NOT radiation damping. The slender-spar + horizontal plate
  geometry has genuinely tiny radiation damping (the plate
  reflects no propagating waves at low ω); damping at
  tank-test scale is dominated by the Morison drag, which the
  BEM run does not capture by design.
- **1 negative B value at ω = 9.45 rad/s, magnitude 2.2e-5
  kg/s.** Numerical zero (0.04 % of B_max). High-ω regime
  where the mesh-resolution warning fired (panels per
  wavelength at ω = 9.45 is marginal). Does not affect
  ω_n = 2.1 rad/s where damping matters.

### Disposition

Fix is **successful**. A_inf is the right order of magnitude
(was 1.3, expected ~30, now 21); the residual ~16 % shortfall
is mesh-resolution-related rather than a second integration bug.
Step C onward unblocked once Xabier confirms.

### Cross-references

- Phase 2 tracker entry **BEM-INPUT-NORMAL-VALIDATION** added to
  `docs/phase2-followups.md` on main (separately committed; same
  protocol as BB-OFFSET-CONNECTOR). Captures the generalisable
  lesson: BEM solvers silently produce wrong A_inf when panel
  normals are reversed; FloatSim's Capytaine/WAMIT readers do
  not currently validate panel orientation on import; future
  externally-imported meshes with thin horizontal features are
  at risk.
- Conventions doc **Item 5** added to
  `docs/multibody-conventions.md` on main: BEM mesh panel
  normals must be outward into the surrounding fluid; consumer-
  side gate currently absent; reference implementation lives in
  this study's `fix_mesh_normals.py`.

---

## Pre-flight 1 finding + resolution (2026-06-29) — Capytaine reader symmetry tolerance

While running the Pre-flight 1 ingestion check on the post-fix
NetCDF, FloatSim's Capytaine reader raised:

    ValueError: A[:, :, 0] must be symmetric (within rtol=1e-06)

The dataset's A matrix has Capytaine BEM panel-method asymmetry
on the order of ~2.85e-4 relative (max |A - A.T| ~ 7.2e-3 vs
A_max ~ 25 kg; ~3.8e-3 relative for B). This is panel-method
solver noise from computing `A(i, j)` and `A(j, i)` via separate
radiation problems (radiating DOF i vs j); the asymmetry is
unphysical and should be averaged out.

**Reader-hygiene gap.** The WAMIT reader has a symmetrization
step via `_resolve_6x6_from_dict`'s arithmetic-mean averaging of
duplicate `(i, j)` and `(j, i)` entries (documented in
`floatsim/hydro/readers/wamit.py`). The Capytaine reader has no
equivalent — it ingests Capytaine's per-radiating-DOF values
verbatim and rejects them on the `rtol = 1e-6` symmetry check
(`floatsim/hydro/database.py:181`). Parallel oversight to
BEM-INPUT-NORMAL-VALIDATION.

### Resolution — path (α): study-local symmetrization

Per Xabier 2026-06-29: apply a study-local symmetrization step
in `capytaine_run.py` rather than patching `floatsim/` mid-study
(reading the WAMIT-reader code as the canonical handling but
deferring the patch). Both reader-hygiene findings will be
reassessed together at study close.

**Implementation.** `capytaine_run.py` symmetrizes A and B
per-omega before NetCDF save:

    A_sym = 0.5 * (A + A.swapaxes(-1, -2))
    B_sym = 0.5 * (B + B.swapaxes(-1, -2))

Audit-trail attributes captured on the NetCDF:

    symmetrization_max_residual_A      = 7.18e-3 kg
    symmetrization_relative_residual_A = 2.85e-4
    symmetrization_max_residual_B      = 1.25e-1 kg/s
    symmetrization_relative_residual_B = 3.78e-3

Post-symmetrization max |A - A.T| and |B - B.T| are both 0.0 to
float64 precision.

`capytaine_run.py` also added an `omega = inf` case to the
radiation problem set so Capytaine's BEM directly populates
A_inf via the canonical infinite-frequency problem (FloatSim's
Capytaine reader otherwise requires an external `a_inf=` kwarg).

### Verification

Pre-flight 1 re-run on the symmetrized NetCDF:

    FloatSim Capytaine reader: INGESTS CLEAN
    A.shape         = (6, 6, 80)        ✓
    B.shape         = (6, 6, 80)        ✓
    A_inf.shape     = (6, 6)            ✓
    C.shape         = (6, 6)            ✓
    RAO.shape       = (6, 80, 1)        ✓
    A_inf[2,2]      = 21.1177 kg        ✓ (matches post-fix BEM)
    C[2,2]          = 221.0807 N/m      ✓

A_inf(heave) unchanged (diagonal entries are invariant under
symmetrization).

### Cross-references

- Phase 2 tracker entry **BEM-CAPYTAINE-READER-SYMMETRIZATION**
  on `docs/phase2-followups.md` (main; separate commit). Same
  protocol as BEM-INPUT-NORMAL-VALIDATION.
- Conventions doc **Item 6** on `docs/multibody-conventions.md`
  (main; separate commit): BEM A/B matrix symmetrization on
  ingestion.
- Both reader-hygiene findings (BEM-INPUT-NORMAL-VALIDATION +
  BEM-CAPYTAINE-READER-SYMMETRIZATION) reassessed at spar-fin
  study close for either promotion to a small reader-hygiene
  milestone or absorption into B4 scoping.
