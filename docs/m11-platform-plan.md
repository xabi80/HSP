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
  conditioning monitoring.

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

**M11b:**
- **PR6** — 12-buoy mesh generator + assembly (at platform eqdraft).
- **PR7** — BEM at scale (**memory probe on the real mesh first**, Q4).
- **PR8** — terminal gate Stages 1 and 2.
- **PR9** — closure.

**Cross-cutting:** the **contaminated-slice detector** is its **own PR**,
sequenced with the BEM work (Measurement E: solve-time conditioning
monitoring, not output smoothness — see tracker amendment).

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
