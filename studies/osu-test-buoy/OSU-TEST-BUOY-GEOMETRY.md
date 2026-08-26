# OSU Test Buoy — geometry measurements (for FloatSim adaptation)

Measurements of the **physical test buoy** ("6-inch model") that will be tank-tested,
taken from the CAD model to re-point the FloatSim single spar-buoy to the real geometry.

- **Source:** `OSU Test Buoy.stp` (CATIA V5R19, STEP AP203, **units = mm**). Kept in the
  buoy OneDrive folder (`Documents/buoy/`), not in this repo (9.3 MB).
- **Tooling:** read with **gmsh** (OpenCASCADE kernel) — `pip install gmsh` on the
  contributor machine; it is NOT a FloatSim runtime dependency (like `openfast_io`,
  scripting-only). Reproduce with `step_measurements.py` / `mesh_from_step.py`.

## Overall
- Envelope **328 × 339 × 2274 mm**; total solid (steel/parts) volume **6.41 L**.
- 30 solids. Layout bottom→top: ballast/heave-plate frame · lower cap · 6″ pipe spar
  (with internal ballast carriage on a track) · upper cap + gimbal ring + gimbal mount +
  12 V motor + lanyard spool.

## Key parts (gmsh: bbox, volume, centroid z)
| part | size (mm) | volume | z_cog (mm) |
|---|---|---|---|
| **Pipe 6 sch 40 (spar)** | **Ø159 OD / ≈Ø149 ID** (wall ~5 mm), **L 1683** | 4.00 L (steel wall) | +841 |
| **Ballast base plate** | **328 × 198 × 50** (rectangular) | 0.310 L | −416 |
| Ballast webs (several) | 0.7–25 mm thick × ~250 mm plates | ~0.06–0.11 L each | −130…−400 |
| Lower cap 6 inch | 159 × 186 × 127 | 0.290 L | −29 |
| Ballast carriage | small, at z = 746 / 996 / 1246 / 1496 (track positions) | 0.007 L each | — |
| Upper cap assembly | 159 × 169 × 103 | 0.302 L | +1712 |
| 6″ gimbal ring | Ø≈220 × 45 | 0.211 L | +1696 |
| gimbal mount | 240 × 60 × 158 | 0.128 L | +1783 |

## ⚠ The heave plate is NOT a disc
The bottom ballast doubles as the heave plate and is an **open, webbed, perforated
frame** — a **rectangular ~328 × 198 mm base plate** with a **cage of thin webs (with
lightening holes)** flaring up to the spar. It is **non-axisymmetric** and thin in one
horizontal axis. (An earlier read as a "Ø396 flat disc" — from parsing circular *edges*
— was wrong; see `OSU_ballast_shape.png`, `OSU_Test_Buoy_mesh.png`.)

**Hydrodynamic implication:** the disc heave-plate model (Cd_n = 5, area πa²) does **not**
transfer. Also a potential-flow BEM of the CAD surface would **over-predict the added
mass** (it blocks the flow the real perforations pass through) and under-predict damping.
So:
- **Spar (pipe):** a clean cylinder → Capytaine BEM is reliable (added mass / radiation /
  excitation).
- **Heave-plate/ballast frame:** its added mass **and** drag are best obtained from the
  **tank test** (the "option b" rotational/heave forced-oscillation or free-decay); for an
  open frame that measurement is essential, not optional. Empirical porous-plate
  corrections are the fallback.

## FloatSim mapping (single spar-buoy)
| FloatSim param | current | → test buoy |
|---|---|---|
| spar radius `R_SPAR` | 0.0841 m | **0.0795 m** (Ø159) |
| spar length | ~1.85 m | **1.683 m** |
| spar waterplane stiffness ρg·A_wp | 221 N/m | **≈200 N/m** |
| heave plate | Ø0.43 disc, Cd_n=5 | **rectangular 0.328×0.198 webbed frame** (hydro from tank/BEM, not the disc model) |

The spar physics carries over almost unchanged (heave period still ~2.5 s); the plate is
the piece that needs the real hydro.

## Mass / floating condition (from `OSU Spar Buoy Platform Metric.xlsx` + measured waterline)
Anchors: **structure = 8.16 kg** (all parts minus the ballast box), **unloaded waterline =
967 mm** from the cylinder bottom (= 0.574·L). Reconciled (`hydro_from_measurements.py`):

| quantity | value |
|---|---|
| total floating mass | **≈21.5 kg (fresh) / 22.1 kg (salt)** — geometry check (cyl+frame+lead=21.8 kg) agrees ~1.5% |
| ballast (lead), to float at 967 mm | ~13.4 kg |
| CoG | **z = −0.907 m** (waterline frame; 0.059 m above cyl bottom) — deep, very stable |
| cylinder | z ∈ **[−0.967, +0.717]** (submerged 0.967 m, freeboard 0.717 m) |
| heave-plate/ballast frame | z ∈ [−1.10, −1.43] (base plate ~−1.38 m); draft 1.425 m |

(OSU Hinsdale lab is **fresh** water; period is density-independent for a free-floating body.)

## Spar BEM (capytaine)
Wetted spar cylinder (Ø0.1593, immersed 0.967 m), `hydro_from_measurements.py`:
- **C33 = 194 N/m** (matches the spreadsheet ~195 ✓); spar heave added mass only ~1.1 kg →
  **spar-only heave period 2.14 s**.
- **Heave-plate bracket:** a *solid* equal-area disc (Ø0.29) adds ~7.9 kg → 2.49 s (upper
  bound); the real perforated/webbed plate adds **less** → heave period likely **~2.2–2.4 s**.
  (Our current FloatSim buoy is ~3.0 s — its fin is larger; the OSU plate is smaller/
  perforated, so a **shorter** period. Re-point the model accordingly.)

## Adapted FloatSim model (placeholder plate) — built & decay-checked
Full 6-DOF Capytaine database `capytaine_osu_buoy.nc` (spar + a **placeholder** solid
equal-area disc for the heave plate), read by FloatSim's `read_capytaine`. Build:
`bem_database.py`; buoy constants + drag: `osu_buoy_common.py`; decay: `osu_decay.py`;
inertia: `inertia_from_step.py`.
- Hydrostatics **C33 = 194.5 N/m** ✓; A∞(heave) = 9.7 kg (spar ~1.1 + placeholder plate ~8.6).
- BEM on the **validated fine ω-grid** (reused from the single-buoy BEM; fine at low ω, out
  to 30 rad/s) → the radiation kernel **passes FloatSim's Check-3 decay gate with no
  override** (a coarse grid did not; the slender spar's surge/roll kernels ring long).
- **Heave free-decay T = 2.52 s** (matches the FD) — the placeholder value. The real
  perforated/webbed plate adds *less* added mass → **shorter, ~2.3–2.4 s**.
- **Pitch decay 2.11 s**, with **I_xx=I_yy=10.2, I_zz=0.063 kg·m²** — inertia about the CoG
  from gmsh's exact per-part tensors + the mass model (structure 8.16 kg over the solids at a
  uniform effective density; lead 13.36 kg at the ballast). Uniform-density is the one
  assumption (no per-part material list); replace with CATIA mass properties if available.

## Heave decay prediction (pre-test)
Before the tank test, the decay splits cleanly into a part we can predict and a part
the test exists to measure (`predict_decay.py` → `OSU_heave_decay_prediction.png`):

- **Period is predictable.** `T = 2π·√((M + A₃₃)/C₃₃)`. `M = 21.52 kg` and
  `C₃₃ = 194.5 N/m` are pinned (mass/waterline and the 6″ pipe diameter). The only
  unknown is the heave-plate added mass `A₃₃`, and it is *bounded*: spar-only ≈ 1.5 kg
  (T = 2.16 s) → near-solid equal-area disc ≈ 8.5 kg (T = 2.47 s). Best estimate
  `A₃₃ ≈ 4–5 kg` → **T ≈ 2.3–2.4 s** (band 2.2–2.5 s).
- **Damping is the measurement.** Quadratic (Morison) drag on the perforated/webbed
  plate → amplitude-dependent log-decrement (curved envelope, largest on the first
  swing). The open frame's `Cd` is exactly what potential-flow BEM cannot give, so the
  prediction is a band: **ζ₁ ≈ 8–15 %** of critical on a 100 mm release, decaying over
  ~10–20 cycles.

| the test measures… | …which pins |
|---|---|
| oscillation **period** | plate **added mass** `A₃₃` (closes the 2.2–2.5 s bracket) |
| **decay rate** vs amplitude | plate **drag** `Cd` (the perforated-frame unknown) |

A measured period well below ~2.2 s is a flag to recheck mass/waterline before trusting
the damping fit.

## Capytaine diffraction & radiation analysis — how to explain it
(`capytaine_explainer.py` → `Capytaine_explained.png`; the BEM mesh itself is in
`OSU_buoy_mesh_capytaine.png` via `mesh_view.py`; the database is built by
`bem_database.py`.) Capytaine is a **boundary-element (panel) solver for linear
potential-flow hydrodynamics**: give it the wetted hull as panels, it solves the flow
(velocity potential φ) around it in regular waves at each frequency ω. "Potential flow"
means **inviscid, irrotational, small (linear)** waves and motions — that single
assumption drives both its outputs and its limits.

Because the problem is linear, the wave-body interaction **splits by superposition into
two sub-problems** solved separately and added:

1. **Radiation — "body as a wavemaker."** Turn incident waves off; force the body to
   oscillate in still water, one DOF at a time, at each ω. The reaction force gives the
   part in phase with acceleration → **added mass A(ω)** and the part in phase with
   velocity → **radiation damping B(ω)** (6×6 matrices per ω, with cross-DOF coupling).
2. **Diffraction — "body as an obstacle."** Hold the body *fixed*; let waves scatter off
   it. The net wave-pressure force is the **excitation force F_exc(ω, β)** =
   **Froude–Krylov** (undisturbed incident pressure) + **diffraction** (scattering
   correction).

Outputs per frequency: **A(ω), B(ω), F_exc(ω)**, plus the hydrostatic stiffness **C**
(geometry). These are written to `capytaine_osu_buoy.nc`.

**Bridge to the time domain (Cummins).** Frequency-domain A(ω)/B(ω) can't be dropped into
a time-stepper directly (a transient contains all frequencies), so:

> `(M + A∞) ẍ + ∫₀ᵗ K(t−τ) ẋ(τ) dτ + C x = F_exc(t) + F_drag(ẋ)`

with **A∞** = infinite-frequency added mass and **K(t) = (2/π)∫B(ω)cos(ωt)dω** the
*retardation kernel* (fluid memory of previously radiated waves). Radiation → `A∞ + K(t)`;
diffraction → `F_exc`; hydrostatics → `C`. FloatSim integrates this (`compute_retardation_kernel`,
`assemble_cummins_lhs`).

### Who computes what — Capytaine vs FloatSim
Capytaine is run **once, offline**, and produces only the *linear, inviscid hydrodynamic
coefficients* (frequency domain). FloatSim then builds and solves the *actual equation of
motion in time* — and supplies everything potential flow can't see (viscous drag, the
body's own mass/inertia, mooring, equilibrium). Explicit split:

| Step | Tool | Domain | Produces |
|---|---|---|---|
| Panel mesh of the wetted hull | **Capytaine** | geometry | 3408 wetted panels (`mesh_view.py`) |
| Radiation problem (per ω) | **Capytaine** | frequency | added mass `A(ω)`, `A∞`; radiation damping `B(ω)` |
| Diffraction problem (per ω, heading) | **Capytaine** | frequency | excitation `F_exc(ω,β)` = Froude-Krylov + diffraction |
| Hydrostatics | **Capytaine** | geometry | buoyancy stiffness `C` (gravity term added by FloatSim) |
| *→ written to `capytaine_osu_buoy.nc`* | | | *the hand-off* |
| Retardation kernel `K(t)` from `B(ω)` | **FloatSim** | time | fluid-memory convolution kernel |
| Rigid-body mass & inertia `M` | **FloatSim** | — | from the structure (gmsh + spreadsheet) |
| Assemble LHS `(M+A∞)`, `C` + `m·g·z_G` | **FloatSim** | — | Cummins left-hand side (incl. gravity restoring) |
| Viscous / quadratic **Morison drag** `F_drag` | **FloatSim** | time | spar + heave-plate drag — *not* in the BEM |
| Static equilibrium | **FloatSim** | — | trim / draft |
| Time integration of the Cummins EoM | **FloatSim** | time | motion `x(t)` → decay period `T` & damping `ζ` |

In one line: **Capytaine gives the hydrodynamic coefficients; FloatSim assembles and
time-integrates the equation of motion** (adding mass, viscous drag, mooring, equilibrium,
and the `B(ω)→K(t)` conversion). The heave-decay period and damping are a FloatSim output,
computed *from* the Capytaine coefficients — not something Capytaine reports.

**Limits worth stating (and why the heave plate needs the tank):** potential flow is
**inviscid**, so (i) all viscous/quadratic damping is added separately as **Morison drag**
`F_drag`, *not* from the BEM; and (ii) a panel model treats every surface as solid, so it
**over-predicts added mass for the open/perforated heave-plate frame** (it blocks flow the
real perforations pass). The spar BEM is reliable; the plate's A and B are BEM
*stand-ins* until the tank test measures them — the same `A₃₃`/`Cd` bracket as the
prediction above.

## Assumptions & their basis
The model inputs and the justification for each. The two ⚑ rows are placeholders the
**tank measures** (period → plate added mass; decay rate → plate `Cd`). Coefficient ranges
follow standard practice (DNV-RP-C205; Sarpkaya); the perforated-plate added-mass reduction
follows porous-disk theory (Molin). Set in `osu_buoy_common.py` / `bem_database.py`.

| Assumption | Value | Basis / support |
|---|---|---|
| Spar drag `Cd` (transverse) | 1.2 | Smooth circular cylinder (Morison); standard 1.0–1.2 (DNV-RP-C205, Sarpkaya). Sets surge & pitch drag; negligible in heave. |
| Heave-plate normal `Cd_n` | **5.0** ⚑ | Flat disc at low KC (≈ 2–3 at a 100 mm release, `KC = 2πa/D`): flow separates, published `Cd ≈ 4–8`. Dominates heave damping. Placeholder → tank pins it. |
| Heave-plate tangential `Cd_t` | 1.5 | Edge / skin friction on the rim; small contribution. |
| Plate added mass | **solid equal-area disc, Ø0.287 m** ⚑ | Potential-flow UPPER bound; the perforated/webbed frame passes flow → adds less (porous-disk theory, Molin). Bracket → tank pins `A₃₃`. |
| Mass / CoG | 21.52 kg / −0.907 m | Spreadsheet structure 8.16 kg + measured unloaded waterline 967 mm; independent geometry check (cyl+frame+lead) agrees ~1.5%. |
| Pitch / roll inertia | 10.2 kg·m² | gmsh per-part inertia tensors + uniform effective density; the deep lead ballast dominates → robust to the internal mass split. CATIA would refine. |
| Water & BEM | fresh (ρ=998); deep; linear | OSU Hinsdale lab is fresh (period density-independent for a free body); potential flow is inviscid → viscous drag added via Morison. Confirm tank depth vs 1.42 m draft. |

`KC = 2πa/D` with release amplitude `a ≈ 0.1 m` and plate scale `D ≈ 0.20–0.29 m` → `KC ≈ 2–3`
at the first swing (dropping as the motion decays) — the low-KC, high-`Cd` regime.

## Heave-plate depth vs pitch performance (design study)
`plate_depth_study.py` → `plate_depth_pitch_study.png` (data cached in
`plate_depth_results.json`). Question: with the **ballast kept deep** (mass, CoG −0.907 m,
inertia fixed → pitch restoring `C55` unchanged), does moving *only* the heave drag device
in z change pitch? Swept the BEM disc + Morison element over z ∈ [−1.10, −1.70] m (clean gap
below the spar bottom), rebuilding the BEM and running heave + pitch free-decay at each.

**Result — plate depth barely moves pitch.** Over a 4× change in lever arm (L = 0.19 → 0.79 m
below CoG): pitch period 2.10 → 2.14 s (~2%), pitch damping ζ₁ 1.5 → 1.7% (0.2 pt); heave
2.48–2.53 s / ~6.1% (flat); `C55` 261–269 N·m/rad (flat). Deeper is marginally better for pitch.

| plate z (m) | L below CoG (m) | pitch T (s) | pitch ζ₁ (%) | heave T (s) | heave ζ₁ (%) | C55 |
|---|---|---|---|---|---|---|
| −1.10 (shallow) | 0.19 | 2.10 | 1.5 | 2.48 | 6.2 | 269 |
| −1.383 (current) | 0.48 | 2.12 | 1.5 | 2.53 | 6.1 | 265 |
| −1.70 (deep) | 0.79 | 2.14 | 1.7 | 2.53 | 6.1 | 261 |

**Why.** An axial heave plate meets pitch **edge-on**: pitch about the CoG translates the
on-axis plate horizontally (velocity θ̇·L), sliding it edgewise, so the broadside added mass
and `Cd_n ≈ 5` damping that dominate *heave* are almost unengaged by pitch. The only
pitch-relevant plate effect is its **tilt** (rotation about its own diameter), which scales
with plate **size**, not depth. So plate depth is a heave knob, not a pitch knob; stability
is a separate ballast/CoG knob.

**Design implication.** Place the drag device where draft / structure / packaging want it —
pitch, heave and stability are all nearly indifferent over this range; co-locating it with
the deep ballast (current) costs nothing and is marginally best for pitch. To actually raise
pitch damping the levers are plate **size**, plates **offset from the axis** (they move
broadside in pitch), or **spar strakes / roughness** — not the axial plate's depth.

**Caveats.** Placeholder solid disc (heave carries the usual "perforated adds less" caveat;
the pitch conclusion is *more* robust to it, since the plate barely participates in pitch
either way). Couldn't cleanly mesh the plate at the spar bottom (disc-kissing-cap BEM
artifact) or up the spar (needs an annular collar); too-shallow also risks broaching. Coarse
mesh for speed — the current-depth point reproduces the production values (2.12 s / 1.5%;
2.53 s / 6.1%).

## Status / next
- **DONE:** geometry + mesh; mass/CoG/draft; spar BEM; full placeholder database + adapted
  buoy model + decay check (heave 2.52 s, pitch 2.11 s); **production-grade kernel** (fine
  grid, no override); **real inertia** (gmsh).
- **OPEN (needs the tank):** the perforated heave-plate's **added mass AND damping** — a
  potential-flow BEM over-predicts them for an open frame, so the disc is a bracket; a
  heave/pitch free-decay or forced-oscillation at the design KC pins both (same campaign as
  the pitch-damping option-b test). Optional: CATIA mass properties to replace the
  uniform-density inertia assumption.
