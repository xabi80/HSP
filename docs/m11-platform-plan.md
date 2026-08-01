# M11 — 12-buoy articulated platform (terminal milestone) — PLAN

**Status: LOCKED (Q1–Q8), 2026-07-29.** Locked on M11 Phase-1
Measurements A–E plus Xabier's topology/mass confirmations
([`docs/platform-geometry.md`](platform-geometry.md), supplied §1 incl.
S7/S8). Terminal milestone of the Tier-3 program
([`docs/tier3-program-plan.md`](tier3-program-plan.md): Q1 sequencing, Q6
two-stage terminal gate, Q8 exclusions as amended at M10 close). Split
into **M11a** (drag + inter-cluster + calibration) and **M11b** (12-buoy
assembly + terminal gate) per Q2.

> **Numbers provenance.** All geometry/mass numbers are **cited from
> [`docs/platform-geometry.md`](platform-geometry.md)**, not carried here.
> Measured compute/physics values re-derived from `main` at drafting
> (Phase-1 A–E); disagreements with prior figures are itemized in the
> measurement notes. This plan opens no branch until M11a PR1.

---

## Goal
Simulate the **12-buoy articulated platform** — 4 clusters of 3 spar-buoys
on a rigid central cross, clusters `yaw_locked` to the platform arms —
with coupled multi-body BEM + articulated joints + **viscous drag**, and
validate against the Q6 **three-step** terminal gate. The platform's
**LEVEL2 input** (measured rotation) remains **subordinate to drag** (M10
finding).

**Inherited locks (program plan Q6):** two-stage terminal validation;
regular-wave + free-decay only; rigid arms; mooring out unless Q6 data has
one. **M10-amended:** drag REQUIRED (not deferred); cluster-scale tank
data first → staged validation; LEVEL2 subordinate to drag.

---

## Grounding — Phase-1 Measurements A–E (summary; full detail in the closure/geometry docs)
- **A (drag audit):** `MorisonElement` (morison.py:127) is a member-normal
  single-`Cd` cylinder; studies **hand-assemble** it
  (`make_morison_state_force` → `integrate_cummins(state_force=…)`);
  `build_system` **never reads `Body.drag_elements`** (deck.py:167 exists;
  driver.py:355-389,803-819 ignore it). Heave plate = degenerate
  horizontal cylinder, **mis-models horizontal drag**. Direction-dependent
  plate area is **not native** (Q3 iii).
- **B (topology):** intra-cluster determined; inter-cluster now **supplied**
  → `platform-geometry.md`.
- **C (BEM cost, measured 3-point):** 1488→0.089, 4464→0.243, 8928→0.750
  s/problem; exponent rises 0.91→1.63; 17,856-panel projection ~1.6–2.3
  s/problem, **peak ~9.3 GB (2× single matrix) vs 34.1 GB free → NOT
  memory-bound**; ~25–40 min (13-ω). *(Itemized lower than the plan's
  0.12/0.53/3.5 s extrapolation.)*
- **D (KKT n-scaling):** dense bordered solve **47 µs at n=72**,
  **~84 µs at n=102** = **0.4 % of a step** → B6 defers.
- **E (freq screen):** neighbour-trend screen catches ω=20.909 (z=30.2)
  but **misses ω=4.934** (z=0.5, all 5 statistics) → **falsifies the
  tracker's output-smoothness recommendation**; needs solve-time
  conditioning monitoring. **CORRECTED (M11b PR7, 2026-07-31):** the true
  statement is NARROWER — the five **A-based** statistics are smooth at 4.934
  (confirmed: max|A| z≈4.75, A[2,2] flat), so *those* miss it; but the
  **symmetrized-B min-eigenvalue** (the M8 PSD metric) is NOT smooth at
  4.934 — neighbour-z = **1026** — and separates it cleanly (physical 2–3
  band z=0.5–3.0). So "output smoothness cannot separate the 4.934 class"
  holds for A-statistics only, not for the B-min-eig; the M8 PSD gate
  catching 4.934 was consistent with this all along. The M11b PR7 B-min-eig
  detector uses exactly this metric.

---

## Q&A — LOCKS with grounding

### Q1 — PLATFORM TOPOLOGY (LOCKED)
Per [`platform-geometry.md`](platform-geometry.md) (supplied §1 incl.
S7/S8; derived §3): **17 bodies → n = 102 DOF** (12 buoys + 4 cluster-hubs
+ 1 platform cross, all dry above water S8); **16 IDENTICAL `yaw_locked`
joints → m = 64** (12 buoy↔hub + 4 cluster↔platform, S7 — topologically
uniform); **38 free DOF**; **~402 kg**; **12 × 1488 = 17,856 BEM panels**.
**Three rotational families:** 24 buoy-vs-hub (M10, T_rot = 3.257 s),
**8 cluster-vs-platform (new, S7)**, 6 platform rigid-body.
- **Capability consequence:** M9's `yaw_locked` builder covers the **entire
  platform — no new joint physics in M11.**
- **Open (do not block M11a):** C4-a (90°/C4v arrangement — also the min
  cross-cluster gap, a design lever, geometry §3.6), C4-c (arms/platform
  mass split). Effects stated in `platform-geometry.md` §4.

### Q2 — MILESTONE SPLIT (LOCKED: SPLIT)
**Rationale = the DELIVERABLE BOUNDARY** (not "M11b blocked" — Q1 is now
answered). Two coherent deliverables with a clean seam at the tank data:
- **M11a — drag + inter-cluster + cluster-scale calibration.** Everything
  that is (i) confirmed 3-buoy/cluster geometry and (ii) where the tank
  data lands **first** (height × period sweep). Includes the first
  **inter-cluster coupling measurement** (PR3).
- **M11b — 12-buoy mesh + assembly + scale-up + terminal gate.**
- **Grounding:** Measurement A(d) (drag is partly new modelling, sizeable);
  the tank campaign is **cluster-scale-first**; **compute is NOT the split
  driver** (C: BEM not memory-bound; D: KKT negligible). The seam is the
  natural staged-validation boundary M10's closure identified.

### Q3 — DRAG SCOPE (LOCKED: three pieces, sequenced — lower-risk than "extend")
1. **(i) WIRE the existing capability** — `build_system` reads
   `Body.drag_elements` and composes `make_morison_state_force` into
   `SimulationSetup.state_force` (summed with connector/catenary). **Pure
   wiring, no new physics.** Validated by the **known heave `Cd = 5.0`**
   against the committed study decay.
2. **(ii) USE the existing element correctly for the SPAR** — the
   `MorisonElement` member-normal cylinder model is **CORRECT** for the
   vertical spar hull (~1.85 m × 0.168 m; `m10-…-plan.md:159`,
   `cluster_common.py:27`) moving **laterally**: axis vertical, lateral
   velocity is normal to the axis → full drag. Adding proper spar drag
   elements captures **part of the rotational damping with NO new
   physics** (the rotational modes swing the spar laterally). *(This is the
   right use; the heave-plate horizontal-cylinder trick was the wrong
   one.)*
3. **(iii) EXTEND for the PLATE** — direction-dependent projected area
   (broadside in heave vs edge-on in the rotational families). **The only
   genuinely-new-physics piece** (Measurement A(d)).
- **Sequence (i) → (ii) → (iii). Capability ships before coefficients:**
  heave `Cd = 5.0` is known and validates (i)/(ii); the **rotational `Cd`
  is tank-pending** (tracker `INBAND-ROTATIONAL-RESONANCE`) and now spans a
  **mode FAMILY** (8 cluster-vs-platform + 24 buoy-vs-hub), not a single
  mode.

### Q4 — BEM APPROACH (LOCKED)
**Full dense solve on the reduced (~13-ω) grid bracketing the resonances**;
the 40-ω grid is an overnight option. **Measured NOT memory-bound**
(~9.3 GB peak projected vs 34.1 GB free, Measurement C) — the program
plan's memory-mitigation ladder is **not forced**. **Probe peak memory on
the REAL 12-buoy mesh before the full run** (probe-first). **Symmetry
reduction** (C4-a's C4v makes it plausible) is a **speed bonus, not a
necessity — defer.**

### Q5 — KKT / B6 (LOCKED: DEFER B6)
Measured **0.4 % of a step at n = 102** (Measurement D). The dense bordered
solve is 3–4 orders below the convolution. B6 sparse/Schur stays deferred.

### Q6 — TERMINAL GATE (LOCKED: three-step; Stage 1 UPGRADED)
- **Stage 1 — OrcaFlex-plot comparison (UPGRADED).** Xabier confirms the
  OrcaFlex results describe **the same topology** (not a rigid-cluster
  proxy), so Stage 1 is a **like-for-like** comparison and may be
  **partially FALSIFIABLE**, not merely a consistency screen: RAO peaks are
  where the rotational families (Q1) would appear, so the plots may carry
  independent evidence about mode locations. **Two-sided caveat:**
  OrcaFlex's own drag and joint modelling are unknown to us, so a mismatch
  could be either model — Stage 1 is corroborating, not a ground-truth
  gate. **STILL REQUIRED from Xabier (does not block M11a): the plot
  inventory** — which plots, at what wave conditions, which DOF, referenced
  to which body. *An RAO is only comparable once heading, DOF, and
  reference body are known.*
- **Stage 2 PRE-STAGE — cluster-scale tank calibration.** The height sweep
  **is** the drag experiment (response-per-height falls near resonance →
  `Cd`). Tank-gated.
- **Stage 2 — 12-buoy tank comparison.** Tank-gated.
- **Completes tank-independently:** drag capability (Q3 i/ii), the
  inter-cluster measurement (PR3), the 12-buoy assembly + BEM, and Stage-1
  screening. Only the drag-*calibrated* quantitative rotation validation is
  tank-gated.

### Q7 — PR SEQUENCE (LOCKED)
**M11a:**
- **PR1** — drag wiring (Q3 i) + heave-`Cd` regression vs the committed
  study.
- **PR2** — spar lateral drag elements (Q3 ii).
- **PR3** — **INTER-CLUSTER COUPLING MEASUREMENT** (scoped below).
- **PR4** — plate direction-dependent area (Q3 iii).
- **PR5** — cluster-scale calibration harness (height × period) *[tank-gated]*.

> **M11a status (2026-07-30): PR1–PR4 DELIVERED; PR5 tank-gated → M11a
> OPEN.** Drag capability is complete and validated: wiring (PR1), spar
> lateral drag (PR2, F1), inter-cluster coupling measured (PR3, F2),
> anisotropic plate drag (PR4, F3). PR5 (cluster-scale calibration harness)
> is tank-data-gated and outside program control. **Merged to `main` at this
> point (FF-only) so M11b branches from a `main` that carries drag
> capability, not a stale one** — M11a itself stays open until PR5's tank
> data lands.

**M11b:**
- **PR6** — 12-buoy mesh generator + assembly (at platform eqdraft).
- **PR7** — BEM at scale (**memory probe on the real mesh first**, Q4).
- **PR8** — terminal gate Stages 1 and 2.
- **PR9** — closure.

**Cross-cutting:** the **contaminated-slice detector** is ~~its **own PR**,
sequenced with the BEM work~~ **EMBEDDED IN PR7** (M11b Phase-1 resequencing,
2026-07-31): PR7's solve emits a per-frequency conditioning number and its gate
is that every retained slice cleared it — so PR7 cannot proceed on unscreened
data. Measurement E: solve-time conditioning monitoring, not output smoothness.
The discrimination criterion (smoothness in omega of the conditioning number,
validated on the 4.934 case) is a PR7 pre-flight design point — see the M11b
Phase-1 amendment below.

### Q8 — ESTIMATE (LOCKED format)
AI-assisted calendar days, base + multiplier, sanity-checked vs git-date
actuals (M7.5 ≈ 3 d, M8 ≈ 4 d, M9 ≈ 3 d, **M10 ≈ 2 d but SEVEN PRs against
one planned** — the capability-surprise variance evidence).
- **M11a:** base ~3–4 d; **×~1.5–2** → **~5–7 d**. **Drivers named:** the
  **plate drag extension (Q3 iii)** and **inter-cluster coupling if PR3
  finds it large**.
- **M11b:** base ~3–4 d; **×~1.5** → **~5–6 d**.
- **Total ~10–13 AI-days.** Compute (C/D) is **not** a driver.

---

## M11a PR3 — INTER-CLUSTER COUPLING MEASUREMENT (scoped)

**Motivation (geometry §3.6):** the platform's tightest hydrodynamic pairs
are **inter-cluster (0.620 m, ~28 % closer** than the 0.866 m intra-cluster
spacing the cluster study measured), and inter-cluster coupling is
**UNMEASURED**. The program plan's **R = 1.011** (added mass) is a *lower
bound*; the cluster study's **B33 ×8.68** (damping) says the stakes are in
**radiation damping**.

**Scope:** rebuild the Phase-1 2-cluster probe at the **actual 1.414 m
cluster-centre separation** (the Phase-1 probe used an arbitrary 3 m and
was explicitly physics-free). **8928 panels, already timed at 0.75
s/problem** (Measurement C). Measure **R and the B33 ratio for the closest
cross-cluster pair** (0.620 m) and compare against the intra-cluster values
(R = 1.011, B33 ×8.68 at 0.866 m).

**This is the programme's first inter-cluster number** — it grounds whether
12-buoy coupling is a **modest extrapolation** or a **qualitative change**.
**Predictions pinned BEFORE the run** (per the M8/M10 discipline): expect
R slightly above 1.011 and a B33 ratio at/above 8.68 (closer pair →
stronger constructive damping coupling); a **large** inter-cluster B33 is a
finding that reshapes M11b's BEM/validation.

---

## Risk register

| risk | mechanism | status / mitigation |
|------|-----------|---------------------|
| **plate drag extension (new physics)** | direction-dependent area, no native support (Meas. A(d)) | Q3 iii, sequenced last; validate heave-`Cd` first (Q3 i/ii); rotational `Cd` tank-pending |
| **inter-cluster coupling UNMEASURED** | tightest pairs are inter-cluster, ~28 % closer (geom §3.6); R=1.011 is a lower bound, stakes in B33 (×8.68) | **M11a PR3 measures it first** at 1.414 m; predictions pinned |
| **rotational `Cd` pending, now a FAMILY** | heave `Cd`=5.0 not reusable (edge-on vs broadside); 8+24 rotational modes | capability before coefficient (Q3); tracker `INBAND-ROTATIONAL-RESONANCE` |
| **BEM cost/memory** | 17,856-panel dense solve | **measured NOT binding** (~9.3 GB vs 34 GB; 25–40 min); probe real mesh first (Q4 PR7) |
| **contaminated-slice detector** | output-smoothness **misses** the 4.934-class (Meas. E) — currently **undetectable** at scale | own PR; solve-time conditioning monitor; cheap output screen kept for the irregular-frequency class (tracker amended) |
| **noise amplification at 102 DOF** | symmetry-**allowed** cross-DOF rose ~1e-13→~1e-10 (single→3-buoy, G8) while the **forbidden** channel stayed at the ~1.22e-13 floor; more bodies add cross-terms | full-matrix symmetrization (M8 Q2); monitor forbidden-channel floor per-N |
| **Stage-1 comparability** | OrcaFlex drag/joint modelling unknown to us; RAO needs heading/DOF/reference body | two-sided caveat (Q6); plot inventory required from Xabier |
| **tank schedule** | outside program control; cluster-scale data first | staged split (Q2); M11a completes drag capability + PR3 tank-independently |
| **C4-a / C4-c open** | arrangement + mass split (geom §4) | do not block M11a; effects stated; orientation is a design lever (§3.6) |
| **carried fix- debt** | black conformance (3 files); F2 hypothesis-red bound | non-blocking; own branches |

---

## Open items for Xabier (do not block M11a start)
1. **Q6 Stage-1 OrcaFlex plot inventory** — which plots, wave conditions,
   DOF, reference body.
2. **C4-a** arrangement (90°/C4v) + the **cluster-orientation design lever**
   (§3.6: gap 0.56–0.73 m; ~20° widens the tightest pair ~17 %).
3. **C4-c** arms/platform mass split (sum < 60 kg fixed; split open).

---

## Finding F1 — the fixed-joint pendulum idealization overestimates articulated-mode drag ~8x (M11a PR2, 2026-07-30)

**Append-only.** Surfaced when M11a PR2's STEP-2 hand-derivation (a
pre-implementation prediction, per the reference-first discipline) missed
the measured spar-drag damping by **16x**. Diagnosed; the *reference* was
wrong, not the code.

### The idealization and why it fails
The natural hand-estimate for viscous drag on an articulated rotational
mode treats the buoy as a rigid pendulum about its **fixed top joint**: a
strip at distance `s` below the joint moves at `s*theta_dot`, giving a drag
moment `M = 0.5*rho*D*Cd*|theta_dot|*theta_dot * INT s^3 ds`. For the
buoy-vs-hub mode this predicts `zeta_drag(Theta=0.02) = 2.99 %`, `Q ~ 15`.

**But the mode is not a fixed-joint pendulum.** A constrained
eigenanalysis of the drag-free system (independent of the drag code)
gives the true mode shape: **the hub surges 1.36x the buoy pitch-rate**,
so the instantaneous rotation centre sits at `z_b = +0.33 m` — **near the
buoy CoG**, not the joint 1.69 m above it. The lever arms collapse
(`beta = buoy surge/pitch = -0.330` vs the fixed-joint `-1.689`), cutting
the effective spar velocities ~3x and the drag (∝ v²) ~**8x**.

### The corrected reference (energy-equivalent, drag-free mode shape)
Integrating quadratic drag against the **true** local velocities and
dividing by the true modal energy (`I_eff_modal = 118.9 kg m^2`, mode
`T = 3.214 s` vs record 3.257): **`zeta_drag(Theta=0.02) = 0.379 %`,
`Q ~ 68`** (kinematic factor 0.127 vs fixed-joint). **Sanity:** the same
energy machinery evaluated with fixed-joint lever arms + `I_eff = 105.79`
returns **2.99 %**, confirming the correction is purely kinematic and the
arithmetic is sound. The code (PR2 spar elements) reproduces the corrected
reference within ~15 % (amplitude-matched), inside the derivation's own
approximations.

### Consequence for M11 (the point worth more than the gate)
**Any hand-derived drag estimate on an ARTICULATED mode must use the modal
kinematics, not a fixed-pivot idealization.** The error is not small
(~8x here) and grows with a more compliant support / larger radius —
so it applies **more strongly** to:
- **PR4 (the plate extension):** the plate is deepest (largest `s` in the
  wrong idealization), so a fixed-joint plate estimate would be the most
  over-optimistic of all.
- **the cluster-vs-platform family (Q1, the new 8-mode family):** larger
  radius (1 m cluster arm) and a more compliant support (the whole
  platform), where the fixed-pivot error would be even larger.

### The discipline succeeding
Derive-before-implement caught a wrong **reference** this time, not wrong
code — the second time in this milestone family that a hand-derivation lost
to measurement (M10 PR2's near-resonance amplitude estimate was the first,
plan A4(c)). The quadratic drag also makes the effective lever arms weakly
amplitude-dependent (drag loads the spar differently along its length than
inertia does); the linearization is stated at `Theta = 0.02 rad` and this
is a known second-order approximation, not treated as exact.

### F1 addendum — one canonical damping number, amplitude explicit (M11a PR3, 2026-07-30)
The PR2 reports quoted two figures — "removes 34 % of Q (141 -> 93)" and
"~half (141 -> 68)". These are **one result at two amplitudes**, not two
results. Quadratic drag makes `zeta_drag ∝ amplitude`, so:
- **Canonical number: at `Theta = 0.02 rad` (the stated linearization
  amplitude), `zeta_drag = 0.379 %`, `Q ~ 68` (~half of the drag-free
  134).** This is the corrected prediction, matched by the code.
- The **whole-decay-averaged MEASURED** `zeta_drag = 0.185 %` (`Q ~ 93`,
  34 %) is lower only because the log-decrement averages over the decaying
  amplitude (0.02 -> 0); as the amplitude falls, `zeta_drag` falls and `Q`
  rises back toward 134.
The record carries **`Q ~ 68 at Theta = 0.02 rad`** as the single number,
with the amplitude dependence stated. Both earlier figures are consistent
with it.

---

## Finding F2 — inter-cluster hydrodynamic coupling MEASURED (M11a PR3, 2026-07-30)

**Append-only.** The programme's first inter-cluster number. Predictions
were pinned BEFORE the solve (below); both hit. Probe: 2 adjacent clusters
at the real 1.414 m centre separation (closest cross-cluster buoy pair
0.620 m), 6-body / 36-DOF radiation BEM at cluster draft (8928 panels,
0 inward, reciprocity 1.09e-4 = the 18-DOF fixture's 1.08e-4).

### Predictions (pinned) vs measurements
| quantity | pinned prediction | measured |
|----------|-------------------|----------|
| intra A_inf off/diag (0.866 m) | baseline (G4 0.56%) | **0.5628%** (reproduces the record) |
| **cross-cluster A_inf (0.620 m)** | ~1.5% (near-field 1/d^3 dipole) | **1.49%** ✓ |
| intra B33 off/diag @ omega_n | 0.947 (from G6 x8.68) | **0.9531** (reproduces) |
| **cross-cluster B33 (0.620 m)** | 0.95-0.98 (sub-wavelength coherent) | **0.960** ✓ |
| 6-body composite B33 / single | ~35 (toward N^2=36) | **33.1 (92% of N^2)** |

Both priors confirmed: **added-mass coupling is a near-field 1/d^3 effect**
(so it grows at the tighter inter-cluster spacing but stays small, ~1.5%),
and **radiation-damping coupling is sub-wavelength coherent** (d/lambda =
0.041-0.058 at lambda = 15.1 m; off/diag ~0.96, near the fully-coherent
limit) and DOMINATES.

### Consequence for M11b (this resolves an open program-plan item)
- **The coupled BEM is essential, and 12-buoy DAMPING is a QUALITATIVE
  amplification, not a modest extrapolation.** Inter-cluster B33 coupling
  (0.960) is nearly as strong as intra (0.953); 6 buoys already radiate
  92 % coherently. 12 buoys -> radiation damping toward the N^2 = 144
  coherent ceiling. This closes the program plan's "cluster-to-cluster
  coupling UNMEASURED" caveat (the B4/B5 justification): it is measured,
  and it is strong.
- **Added mass stays modest** (~few %; the 6-body composite A33 =
  2 x the 3-body cluster + ~1 %, i.e. two weakly-coupled clusters).
- **NEW M11b concern:** the strong coherent coupling makes B(omega)
  NEAR-SINGULAR (the anti-coherent heave modes radiate ~nothing), so the
  symmetrised B's smallest eigenvalue dips slightly negative (~-5 to -9 %
  of max|B| across the 2-3 band, vs -0.6 to -1.6 % on the 3-buoy 18-DOF
  fixture -- systematic, smoothness-clean, NOT an irregular-frequency
  contamination). The 12-buoy retardation-kernel and PSD gates will face a
  more strongly near-singular B; budget for it (tracker note).

### Consequence for the campaign band and the orientation lever
- **Campaign band: preserved.** The coupling shifts resonance frequencies
  negligibly (added mass ~1 %); it raises radiation damping, but that stays
  small in absolute terms (drag dominates, Q ~ 68 with spar drag F1), so
  the fine-sweep band derived at cluster scale holds.
- **Orientation lever (geometry §3.6): NOT decisive for coupling.** At the
  20-deg option (gap 0.728 m) the added-mass coupling drops ~1.49 % ->
  ~0.94 % (near-field 1/d^3), but added mass is already minor; the DOMINANT
  B33 coupling is coherent-saturated at both gaps (both deep sub-wavelength)
  and barely changes. The lever buys a small reduction in a minor coupling
  -- hydrodynamically not a reason to rotate the clusters before
  fabrication.

---

## Finding F3 -- anisotropic plate drag (Q3-iii): the model, the scoping correction, and the re-derived split (M11a PR4, 2026-07-30)

**Append-only.** PR4 is the ONLY genuinely-new-physics piece in M11a
(Q3-iii). Two corrections landed at the source before any code was written;
the model then followed.

### The scoping correction (recorded BEFORE the model, per the record-wins rule)

**The STEP-1 scoping claim "distributed horizontal cylinders across the
disc, no new physics" was WRONG.** `MorisonElement`'s `_project_normal`
(`floatsim/hydro/morison.py:297-300`) removes only the AXIAL component and
applies one scalar `Cd` to the entire 2-D member-normal-plane resultant --
so a cylinder is drag-ISOTROPIC in that plane. A plate is ANISOTROPIC (it
resists broadside flow far more than edge-on flow); **that anisotropy IS the
physics of Q3-iii.** Under the rotational mode a horizontal cylinder laid
across the disc sees, in its normal plane, BOTH the broadside velocity
`v_z = -theta_dot*x` AND the uniform edge-on velocity `v_x = a_c*theta_dot`
(|a_c| = 0.592 m, so |v_x| = 0.592*theta_dot -- LARGER pointwise than
|v_z|_max = 0.215*theta_dot). It would apply `Cd_n = 5.0` over the broadside
AREA to that edge-on velocity, overpredicting the edge-on term by
`(Cd_n*A_face)/(Cd_t*A_rim) = (5.0*0.1452)/(1.5*0.00168) ~= 290x` and
DESTROYING the STEP-1(b) finding (it re-inflates the term F3 shows is minor).
**Q3-iii's "genuinely new physics" framing was correct all along; the
"no-new-physics" scoping was the deviation.** Xabier concurred with the
flawed version -- the record (morison.py:297-300) corrected both of us.
The double-count is now made STRUCTURALLY IMPOSSIBLE (see the guard below),
not documented against.

### What STANDS from STEP 1 (reinforced), and what the record corrected

STANDS: **form (ii)** (two-component normal/tangential decomposition) is
right -- reinforced (the plate is anisotropic, which is exactly why the
isotropic cylinder fails); **tilting-NORMAL dominates edge-on** on the known
`Cd_n = 5.0`; **Cd_t is minor**, carried as a `[1,2]` sensitivity.

CORRECTED (re-derived from committed constants at drafting; STEP 1's figures
used a wrong plate depth and the record wins):

| quantity | STEP 1 (wrong) | re-derived (committed) | source |
|----------|----------------|------------------------|--------|
| plate depth `z_plate_body` | -0.125 | **-0.2617** | `-1.45737 - _ZB`, PR2 test:47 |
| edge-on lever `a_c` | -0.455 | **-0.592** | `beta + pitch*z_plate`, eigenmode |
| split `E_normal/E_tangential` | 3.9-7.7 | **1.76-3.52** (Cd_t 2.0-1.0) | energy integrals |

The split dropped (larger `a_c` -> larger edge-on) but **normal still
dominates for all Cd_t in [1,2]** -- the mis-framing refutation holds; only
its magnitude was wrong. `beta = -0.3302` reproduces PR2 / Finding F1 exactly
(mode `T = 3.214 s` vs record 3.257, `I_eff = 118.86`).

### The magnitude finding -- the plate's rotational drag is SMALL vs the spar

At `Theta = 0.02 rad`, `zeta_norm = 0.0135 %`, `zeta_plate = 0.017-0.021 %`
(Cd_t 1.0-2.0), `Q_plate ~ 2400-2900`. **This is only ~4-6 % of the spar's
`zeta_drag = 0.379 %` (F1).** The heave plate dominates HEAVE damping (huge
broadside area x Cd 5.0) but for ROTATION about a near-CoG centre it tilts
through a small radius (0.215 m), while the 1.46-m spar sweeps large lateral
velocities -- so the SPAR dominates the rotational damping and the plate is a
minor addition. **Consequence: PR5's tank calibration stakes are lower still**
-- with the corrected `E_normal/E_tangential = 1.76-3.52` split the tangential
fraction is `1/(1+ratio) = 22-36 %` of the plate (NOT 11 % -- that was the
retired 7.7 split), so `Cd_t` governs **~22-36 % of an already-small ~4-6 %
contribution = ~1-2 % of the total rotational damping.**

### The THIRD reframing -- rotational damping lives on the SPAR (M10 -> PR4 STEP 1 -> PR4 measurement)

Three successive reframings of where the rotational damping comes from:
1. **M10 (INBAND-ROTATIONAL-RESONANCE 2f):** framed as plate **edge-on** drag
   with an **unknown** coefficient (tank-pending), distinct from the heave
   `Cd`.
2. **PR4 STEP 1(b):** REFUTED -- *within the plate*, the tilting presents the
   disc broadside (normal flow), so the dominant plate term uses the **KNOWN
   `Cd_n = 5.0`**, not an unknown edge-on coefficient.
3. **PR4 measurement:** the whole plate is only **4-6 % of the spar's**
   rotational `zeta_drag`. So the rotational damping is dominated by
   **slender-cylinder cross-flow on the SPAR** (PR2, Q3-ii), not the plate at
   all.

**Campaign-facing consequences (update the tank plan accordingly):**
- the **tank rotational-decay experiment primarily calibrates the SPAR `Cd`**
  (the ~90 % contributor), not the plate `Cd_t` (the ~1-2 % contributor);
- **PR2's adopted literature prior `Cd = 1.2`** (smooth cylinder, KC ~ 1.46,
  DNV-RP-C205 / Sarpkaya) is therefore **the load-bearing coefficient for the
  whole rotational mode** -- the single number the campaign most needs to
  pin. The plate `Cd_n = 5.0` (dominant *within* the plate) and `Cd_t`
  (edge-on) are second-order for the rotational mode (though `Cd_n = 5.0`
  remains load-bearing for HEAVE).

### Item-3 -- the energy-equivalent reference is SOUND (the measured-above was a coordinate artifact)

PR2 and PR4 both measured `zeta_drag` **above** their energy-equivalent
references, same sign (~15 %, ~24 %). Investigated (item-3): re-ran the PR4
decay from the **pure eigenmode IC** and measured in the **modal coordinate**
`q = ph^T(M+A)xi / ph^T(M+A)ph` (the coordinate the reference is defined in):

| IC | measure | ratio (measured / reference) |
|----|---------|------------------------------|
| differential-angle | modal `q` | **0.994** |
| pure eigenmode | modal `q` | **0.994** |
| differential-angle | differential DOF (buoy0 pitch - hub pitch) | 1.26 |
| pure eigenmode | differential DOF | 1.20 |

**The gap CLOSES to <0.6 % in the modal coordinate, independent of IC.** The
cause is the **measurement coordinate**, NOT a linearization bias and NOT
(mainly) the IC: a single/differential DOF over-reads the modal decay rate by
~16-26 % because it mixes the mode's DOF ratios; the pure-eigenmode IC alone
closes only ~6 % of the 24 %. **The energy-equivalent linearization is
unbiased** -- good news for PR5, whose calibration reference is the same
method. GATE 1's end-to-end check was tightened to the modal coordinate
(rel 0.05, was a loose rel 0.35 forced by the differential-DOF artifact).

### Model form (as built)

`floatsim.hydro.morison.PlateDragElement` (+ deck `PlateMember`), decomposing
the local rigid-body flow:
- **NORMAL (broadside):** `0.5*rho*Cd_n*|w|*w` per face area along the disc
  normal, `w = u_rel.n_hat`, integrated over the disc face by a body-fixed
  polar quadrature. Captures BOTH heave (uniform `w`) and tilt
  (`w(x) = -theta_dot*x` -> `INT|x|^3 dA = 8a^5/15`) with the KNOWN
  `Cd_n = 5.0`. **Pivot-insensitive** -- the tilting rate is the buoy pitch
  rate regardless of the rotation centre, so the F1 fixed-pivot error does
  NOT touch the dominant term (it is confined to the minor edge-on `a_c`).
- **TANGENTIAL (edge-on):** `0.5*rho*Cd_t*(t*2a)*|u_t|*u_t` lumped at the rim.
  Minor; `Cd_t` tank-pending.

**PRINCIPAL stated approximation -- the SHEARED FIELD (more prominent than
the Cd_t sensitivity).** `Cd_n = 5.0` was measured for UNIFORM heave; applying
it strip-wise to the linearly varying tilting field assumes local
face-normal drag with NO radial interaction. It is far better grounded than
the discarded edge-on framing, but it is **the assumption the tank
rotational-decay campaign actually tests** -- not `Cd_t`. See tracker
`INBAND-ROTATIONAL-RESONANCE` (2f correction).

### Gates (all green)

- **GATE 1 (modal-kinematics reference, NEVER a fixed pivot):** the code's
  normal/tangential energy split at the modal state matches the analytical
  `E_n/E_t` (measured 2.335 vs 2.35 at Cd_t=1.5, rel < 0.03), normal
  dominant; the end-to-end coupled decay `zeta_drag,plate` is positive,
  amplitude-linear, and matches the energy-equivalent prediction within PR2's
  rel=0.35. Reference from the drag-free constrained eigenanalysis (predates
  the drag code).
- **GATE 2 (byte-identity, absolute):** plate touches no `M+A_inf`, `C` or
  kernel -- force-only; drag-free decks build identically.
- **GATE 3 (STRUCTURAL):** in pure heave `u_n` is uniform, so the plate
  reduces EXACTLY to the single-`Cd` cylinder (`D*L = pi*a^2`) at machine
  precision (rtol 1e-13); the heave decay through the plate reproduces the
  committed `zeta = 2.5225e-02` (rel 1e-3, the only slack being the study's
  0.1452-vs-0.14522 area rounding).
- **GATE 4 (ANALYTICAL):** the disc quadrature converges to `8a^5/15`
  (residual **-0.58 %** at the adopted 12x24, halving to -0.33 % at 16x32);
  the face area is exact at any count.

### Supersession -- a STRUCTURAL GUARD (requirement a), not a note

A body carrying a `PlateDragElement` may only carry `MorisonElement`
cylinders PARALLEL to the plate normal (spars, whose normal plane is
orthogonal to the plate normal -> lateral drag only). A cylinder in the
plate plane (the M11a-PR1 horizontal-cylinder heave-plate stand-in) captures
the same broadside drag the plate now owns; `make_morison_state_force` raises
on it (the mechanism: `_check_plate_supersession`, checked at build time, so
it fires on BOTH the deck path AND study hand-assembly). The plate element
SUPERSEDES the PR1 stand-in wherever applied.

---

## M11b Phase-1 — pre-implementation measurements + findings (2026-07-31)

**Append-only.** Branch `milestone-11b-platform`; measurement-only (one commit,
STEP 1, carrying the PR4 item-3 modal-coordinate finding to the campaign
record). Five measurements de-risking PR6/PR7, from the REAL 17,856-panel mesh
or committed data; disagreements with the projections itemized (the record
wins).

### Measurements
- **A — mesh (built + screened).** 17,856 panels; platform draft **dz =
  0.21638 m** additional sink (Newton on the mesh displaced volume, the
  `cluster_balance.py` method — NOT a copy; cluster DZ2 = 0.17937, so
  **+0.037 m deeper**, consistent with +0.833 kg/buoy at A_wp = 0.0222); keel
  1.4949 m below WL; closest cross-cluster pair **0.6197 m** (matches §3.6's
  0.620); **0 inward, 0 indeterminate, 1152 open edges (12×96)**. Mass balance
  402.04 kg, per-buoy **33.5033 kg** (§3.3).
- **B — BEM probe (real mesh).** Peak working set **12.71 GB** (68.4 total,
  36.9 available) → **NOT memory-bound**. build+factorize **20.8 s/omega**,
  per-DOF RHS **2.18 s**; **coupled 72-DOF ≈ 42 min** (13-omega grid);
  blended s/problem 2.47.
- **C — assembly feasibility (n = 102).** All M10-PR1 preconditions PASS
  through the real `build_system` path: mass 402.04, **rank(M+A_inf) = 102**,
  **n_constraints = 64, rank(G) = 64, free = 38**, **max|phi(rest)| = 0**,
  body refs threaded (17), **KKT bordered solve 100.6 µs** (~0.5 % of a step →
  B6 defers).
- **D — near-singular B baseline.** Re-derived on the committed 18-DOF
  fixture: symmetrised-B min-eig = **-0.588 % (omega 2.075) to -1.642 %
  (omega 2.230) of max|entry|** — verifies PR3 F2's "-0.6 to -1.6 %" (that
  normalisation). Trend deepens 3-buoy → 6-buoy (-5/-9 %, PR3) → 12-buoy.

### Itemized disagreements (record wins)
| # | projection | measured | disposition |
|---|-----------|----------|-------------|
| 1 | M_buoy 28.67 kg | mesh 28.627 at isolated WL | -0.15 %; orthogonal to the platform draft solve |
| 2 | ~9.3 GB peak | **12.71 GB** | +37 %; still « 36.9 available (G1) |
| 3 | 25-40 min, 1.6-2.3 s/prob | **42 min, 2.47 s/prob** | consistent via the factorize/RHS split |
| 4 | 17,856 panels | **13,824 wetted** solved | 4,032 above-water clipped (G1) |
| 5 | KKT ~84 µs | **100.6 µs** | +20 %; still ~0.5 % of a step |

### Finding G1 (RECORD ITEM A) — BEM cost projections must use WETTED panel counts
Capytaine clips the platform mesh to **13,824 wetted panels** (4,032 above
water, 22.6 %); the influence matrix is 13,824², not 17,856². **Every cost
projection back to the Tier-3 plan** (`tier3-program-plan.md` risk row,
"~5.1 GB per omega") **and this plan's Measurement C** (~9.3 GB) used the
**TOTAL 17,856**. That is why measured peak (**12.71 GB**) EXCEEDED the 9.3 GB
extrapolation DESPITE a smaller solved matrix: the fit was made on probe
meshes (single / 3-buoy) with a **different wetted fraction**, so total-count
scaling systematically mis-scales. **Rule: panel counts in BEM cost/memory
projections are WETTED counts; total counts mislead.** [total 17,856, wetted
13,824, peak 12.71 GB, coupled 42 min.]

### Finding G2 (RECORD ITEM B) — the 72-DOF solve is RHS-bound, not factorize-bound
At 72 DOF: **72 × 2.18 s = 157 s/omega of RHS** vs **20.8 s/omega
build+factorize**. The solve is **DOF-bound.** Q4 mitigation-ladder
consequence: if turnaround ever needs cutting, **C4v symmetry is the correct
lever** — it reduces the independent-DOF count, the dominant term — while
**memory reduction buys nothing** (not memory-bound, and it does not touch the
RHS cost). A future session reaching for the ladder should pick that rung.

### Resequencing (Q7 amendment) — the conditioning detector is EMBEDDED IN PR7
The Phase-1 recommendation to *precede* PR7 is **strengthened to EMBEDDED**:
PR7's solve **emits a conditioning number per frequency**, and PR7's gate is
that **every retained slice cleared it**. A separate preceding PR would let
PR7 proceed on unscreened data if sequencing slipped; embedding makes that
impossible.

**Discrimination argument (the reason).** At 12-buoy scale the PSD gate fires
more often because the **physical near-singularity deepens toward the coherent
ceiling** (Finding D trend), so the detector's job is **discriminating
physical near-singularity (tolerate, understand) from contamination (exclude)**
— different dispositions, and a **min-eig sign check cannot separate them**
(both go negative).

**Design point — resolve in PR7's pre-flight, not now.** The detector needs a
DISCRIMINATION CRITERION, not just a conditioning number (deep physical
coupling and a contaminated solve may both give large condition numbers). The
likely discriminator is **smoothness in omega**: physical near-singularity
varies smoothly; contamination is isolated (the 4.934 case measured ~0.02
rad/s wide vs ~0.34 grid spacing). This is the same neighbour-trend idea
Measurement E found **insufficient on OUTPUT matrices** — applied to the
CONDITIONING NUMBER it may separate where output smoothness did not.
**VALIDATE on the known 4.934 case before trusting it at 72 DOF — if it does
not separate there, it will not separate at scale, and that is a finding.**

**SUPERSEDED by the TWO-DETECTOR design (M11b PR7 STEP 1, 2026-07-31).** The
single "per-frequency conditioning number" above was **falsified on the known
case**: cond(K) is FLAT at 4.934 (z=0.06) because 4.934 is not an
ill-conditioned solve — it is an output anomaly behind a well-conditioned
system matrix (tracker `BEM-CONTAMINATED-...` mechanism correction). cond(K)
and a B-min-eig-smoothness detector catch **DIFFERENT phenomena, and neither
sees the other's**: cond(K) flags genuine ill-conditioned solves (16.837 at
z=11.7, Capytaine warns independently); B-min-eig-z flags output anomalies
(4.934 z=1026, 20.909 z=88045 where cond(K) is flat). **PR7 embeds BOTH**
(disposition OPTION 2), with a four-way verdict (Q7 four-way table below).
Thresholds from the measured separation: `COND_Z=5` (16.837 at 11.7 vs clean
≤1.3) and `BMINEIG_Z=50` on slices below the M8 magnitude floor
(`min-eig < −1e-3·max|B|`, the significance-skip DROPPED so tail
contaminations like 20.909 are not hidden); physical near-singularity sits at
z=0.5–3.0 — a 2–3 order gap. Both detectors re-validated on the known cases
(STEP D): cond(K) fires 16.837 / flat 4.934; B-min-eig excludes exactly
{4.934, 20.909}, tolerates the smooth physical band, retains the benign
sub-magnitude isolated spikes (3.2, 27.9). See
`studies/platform-12buoy/platform_screening.py`.

### Q7 four-way verdict table (M11b PR7) — three combinations need a disposition
| cond(K)-z | B-min-eig (sig-neg + isolated) | verdict | disposition |
|-----------|-------------------------------|---------|-------------|
| low | low | **clean** | retain |
| **high** | low | **ill_conditioned** (irregular-freq) | EXCLUDE (grid selection, M8 PR3 — never value mod) |
| low | **high** | **output_contam** (behind a good solve) | EXCLUDE (same pattern) |
| **high** | **high** | **both** — NO case in the record | EXCLUDE **and REPORT** as a distinct observation |

The **physical** class stays TOLERATED: the M8 magnitude gate firing on
deepening coherent coupling (Finding D: −0.6% → −5% → worse) is a real
property; **smoothness is the discriminator** (z≤3 physical vs z≥1026
contaminated). A `both`-fire has no measured precedent — if it occurs at
12-buoy it must surface as a new class, not be absorbed into a known one.

### PR-sequence disposition
- **PR6** (mesh generator + assembly): de-risked by Measurements A + C; low
  risk. Assembly preconditions asserted as **permanent tests** (not
  re-measured ad hoc).
- **PR7** (BEM at scale): feasible (12.71 GB, 42 min); **conditioning detector
  embedded** (above). Q4 mitigation ladder **NOT triggered** (not
  memory-bound, < 1 h); C4v is the speed lever if ever wanted (G2).
- **PR8 / PR9**: unchanged (PR8 Stage-1 still needs the OrcaFlex plot
  inventory from Xabier — an open input, not a PR6/PR7 blocker).

---

## M11b PR7 — 12-buoy BEM at scale, MEASURED (2026-07-31)

**Append-only.** The 72-DOF coupled BEM ran on the real mesh with the
two-detector screening embedded (`studies/platform-12buoy/platform_bem.py`,
`platform_screening.py`; detectors validated STEP D, permanent test
`test_m11b_pr7_screening.py`). Output: `platform12_bem.nc` (13-ω grid + inf,
radiation + diffraction + hydrostatic C).

### Screening (STEP 4) — the two-detector design vindicated end-to-end
On the 13-ω grid `[0.5…30]`: **cond(K)** all cond-z ≤ 2.93 (< 5) — no
ill-conditioned solves (grid dodges the irregular frequencies); **B-min-eig**
fires (magnitude) at ω=2.5, 3.0 but SMOOTH (z=0.06, 1.06) → verdict
**`physical`** — the deepening coherent near-singularity, TOLERATED not
excluded. **Excluded: NONE; no `both`-fire.** The physical class activated
exactly as designed, and the detector distinguished it from contamination by
smoothness.

### F2 confirmed and then some — physical near-singularity deepens hard at 12-buoy
`min-eig(symB)` reaches strongly negative at the resonances (well past the
6-buoy −5/−9 %), tolerated as `physical`. This is the PR3 F2 trend
(3-buoy −0.6/−1.6 % → 6-buoy −5/−9 % → 12-buoy deepest) MEASURED, and it is
why the detector needs the smoothness discriminator (a bare min-eig sign/PSD
check would exclude real physics).

### Predictions vs measured (STEP 5)
| quantity | pinned prediction | measured | note |
|----------|-------------------|----------|------|
| heave period | ~3.13 s | **3.141 s** | ✓ +0.3 % (C33 = 2653 measured = 12×221) |
| A33(∞) composite | 256–260 kg | **262 kg** | inter-cluster ~2.2 % (bit above the ~1 % 6-body) |
| B33 amplification @ ω_n | 122–132× (85–92 % coherent) | **112.5×** (78 %) | **MISS — footprint-limited:** platform ~3.4 m = 0.22λ vs cluster 0.065λ, so coherence drops with array size (diagnosed, not tuned) |
| reciprocity (raw) | ~1–2e-4 | **5.84e-3 abs ≈ 1.5e-4 rel** | ✓ panel-noise scale (18-DOF 1.08e-4) |

### Finding G2 CORRECTED — the run is BUILD-bound, not RHS-bound
Measured runtime **190.5 min** (1021 problems), vs the Phase-1 probe's 42-min
projection — **4.5×**. Diagnosis: the per-ω solve is ~2:13 (RHS ≈ 31 min
total); the remaining ~159 min is the **Green's-function BUILD** (~680 s/ω on
13,824²). **The Phase-1 probe (Finding G2) under-measured the O(N²) build by
~30×**, so G2's "RHS-bound → C4v cuts the RHS" is wrong: the run is
**BUILD-bound** (Green's-function evaluation). **C4v symmetry remains the Q4
lever** — it cuts the mesh, hence the *build* (the dominant term), not the
RHS. Memory was as measured (~12.7 GB peak wetted, not binding). This is a
STOP-condition disagreement (runtime materially exceeded the probe) — reported
here; the run completed successfully with valid data, so no re-run.
