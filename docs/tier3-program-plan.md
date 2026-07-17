# Tier 3 Program Plan — 12-buoy articulated platform (M8-M11)

**Status: LOCKED (Q1-Q8), 2026-07-05.** Program-level plan; sits
above the per-milestone Q&A-lock plans (house style, e.g.
[`m7.5-reader-hygiene-plan.md`](m7.5-reader-hygiene-plan.md)). Each of
M8-M11 gets its own plan before its branch opens (see Milestone-plan
protocol).

**Program goal.** Solve a **12-buoy articulated floating platform** in
FloatSim — coupled multi-body BEM hydrodynamics (added mass, radiation
damping, wave excitation) + articulated rigid joints between hulls —
and validate it against the two-stage terminal gate of Q6.

> **Numbers provenance.** Every numerical grounding value below was
> **re-derived from the committed record on `main` at drafting time**
> (2026-07-05): the spar-fin and 3-buoy cluster studies
> (`studies/`), the cluster closeout `cbc0dc1`, the 18-DOF multi-body
> diagnostic, and the M7-Foundation solver/timing baselines. It is NOT
> carried from conversational context. An earlier draft instruction
> carried stale figures (R = 1.086, T_n = 3.196 s, cross-terms 3-4 %,
> "B33 +4 %", "sway 4 orders above the forbidden floor"); those were
> caught at the Phase-2 stale-numbers stop (2026-07-05) and superseded
> by the measured values — two of them ("B33 +4 %", the sway/floor
> inversion) had already been corrected on `main` at `cbc0dc1`. See
> the Milestone-plan protocol for the re-derivation rule this event
> established.

---

## Empirical preamble — grounding table (measured inputs)

Each row is cited by the locks that rest on it (Gn labels).

| id | measured input | value (from `main`) | source |
|---|---|---|---|
| **G1** | A_inf(heave) chain | 1.30 kg (reversed-normal GDF bug) → 21.11 kg (mesh_hygiene-fixed, design draft) → 21.12 kg (isolated eqdraft) → 21.12 kg (single at cluster draft) → 21.1209 kg (18-DOF diagonal A33_ii) | spar-fin `STEP-A-FINDING.md`; reference BEM; 18-DOF diagnostic |
| **G2** | interaction ratio R = A33_composite / (3·A33_single) | **1.011** (+1.1 %); diagonal/cross decomposition: A33 heave-block = 21.12 on diagonal, 0.1185 on every off-diagonal | cluster `interaction.json`; 18-DOF heave block |
| **G3** | 9-block consistency (rigid-mode sum vs composite) | 18-DOF heave-block sum 64.0738 kg vs single-body composite A33 64.0738 kg → **+0.000 %** | 18-DOF diagnostic (b) |
| **G4** | heave-heave cross-term magnitude | **0.56 %** of diagonal (0.1185 / 21.12); 6 off-diagonal terms → the +1.1 % of R | 18-DOF diagnostic (c) |
| **G5** | 18×18 A_inf symmetry (panel-noise scale) | max\|A_ij − A_ji\| = 3.31e-3; relative to max\|A\| (30.69) = **1.08e-4** | 18-DOF diagnostic (d) |
| **G6** | radiation-damping coupling (the large one) | B33_composite / B33_single at omega_n = **8.68×** (constructive; toward the N²=9 coherent ceiling). Ratio composite/(3·single) = 2.894 | cluster closeout `cbc0dc1` |
| **G7** | zeta_rad scaling decomposition | single 1.13e-4 (0.0112 % crit) → cluster 2.89e-4 (0.0288 % crit), **rose 2.58×**: B33 up 8.68× partly offset by sqrt((M+A)·C) up ~3.1× | both studies |
| **G8** | cross-DOF direction (noise floor) | sway = 1.22e-13 (symmetry-**forbidden** channel, at floor); surge/pitch = 4.2e-11 / 2.64e-10 (symmetry-**allowed**, elevated); allowed/forbidden = 1546× | cluster `crossdof_diagnostic.py` (`cbc0dc1`) |
| **G9** | per-step solve timing baseline | F4 N=4 (n_dof=24): 21 ms/step; per-step is one dense `np.linalg.solve(A_eff, rhs)` re-factorized each step | M7-Foundation F4; newmark.py:324 |
| **G10** | solver DAE insertion points | `A_eff` assembled once (newmark.py:264); per-step solve (newmark.py:324); block-diagonal global LHS (state.py:142-144); equilibrium root-find (equilibrium.py:89); explicit-lag floor `dt < 2/omega_max` (connector.py:26,521) | Phase-1 Step 2 audit |

**The one-paragraph read.** Intra-cluster **added-mass** coupling is
small (G2/G4: R = 1.011, cross-terms 0.56 %) — but this does **not**
make the coupled model optional, because (i) the radiation **damping**
coupling is enormous (G6: 8.68×), (ii) the 0.56 % is intra-cluster at
0.44 m hull gaps and says nothing about cluster-to-cluster coupling at
12-buoy scale (UNMEASURED), and (iii) per-body wave excitation with
inter-body scattering is structurally impossible without the coupled
model regardless of cross-term size. See Q1/Q2 rationale.

---

## Scope + exclusions (Q8 LOCKED)

**In scope:** coupled multi-body BEM ingestion (added mass + radiation
damping + **excitation**); coupled retardation kernels; articulated
rigid-joint dynamics (DAE/KKT); 12-buoy free-decay **and regular-wave
response** terminal validation.

**Excluded (Q8):**
- **Irregular-sea statistics / spectral response** — OUT (decay +
  regular wave only, matching the studies).
- **Platform station-keeping mooring design** — OUT (a moored
  terminal case may enter only if Q6 reference data has one).
- **Structural / elastic arms** — OUT; **rigid arms assumed**. This is
  also a *modeling assumption that feeds M9's joint definitions* — the
  arms are rigid links, the articulation is at the joints, not in arm
  flex.
- **Morison Cd calibration** — OUT; **consumed** from the tank
  campaign (Q6 Stage 2), not produced. Decay-rate fidelity lives here
  (G7: radiation is ~86× below the viscous term), so this is the
  program's most important *external* input.
- **Regular-wave response** — **IN** (Q2 excitation addition; required
  for the Q6 terminal wave-response stage).

---

## Q&A — locks with grounding

### Q1 — Milestone sequencing (LOCKED)

**LOCK:**
`M8` (B4/B5: coupled BEM ingestion + coupled kernels + **excitation**)
→ `M9` (B2 DAE/KKT integrator; B1 joint types as constraint-Jacobian
builders; **BB-OFFSET-CONNECTOR** closure *verified at M9 planning*)
→ `M10` (articulated-3 validation study)
→ **LEVEL2 decision gate** (input: M10's measured rotation amplitudes)
→ `M11` (12-buoy terminal).

**Supersession (honest record).** An earlier advisory floated
**LEVEL2-first** (wire the quaternion integrator before the joint
layer). **Withdrawn.** The intermediate target (M10 articulated-3) is
heave-dominated, and joint rotation amplitudes are **UNMEASURED** —
committing LEVEL2 as an early milestone would be building for an
unquantified need. B4/B5 is **force-side** (it changes the RHS: coupled
A(ω), B(ω), F_exc), which is *orthogonal* to the state representation
(LEVEL2 changes how `xi[3:6]` is integrated). So B4/B5 leads and LEVEL2
is a **gate** fed by M10's measurement, not an early milestone.
**Rider carried to M9's plan:** design the constraint-Jacobian
`G` rotation-parameterization-agnostic where feasible, so a later
LEVEL2 swap (Euler→quaternion) does not force a joint-layer rewrite.

**GROUNDING:** G2 + G4 (added-mass coupling small but real) and G6
(damping coupling large) make B4/B5 load-bearing; LEVEL2 necessity is
MEASUREMENT PENDING at M10 (pitch-perturbation rotation amplitudes).

### Q2 — B4 multi-body data model (LOCKED)

**LOCK:** a **block-structured 6N×6N `HydroDatabase`** with **body
labels** carried through, matching the measured Capytaine dataset
structure. The 18-DOF diagnostic repr (verbatim):

```
dims: {'omega': 13, 'radiating_dof': 18, 'influenced_dof': 18}
radiating_dof coords: ['buoy1__Surge','buoy1__Sway','buoy1__Heave',
  'buoy1__Roll','buoy1__Pitch','buoy1__Yaw','buoy2__Surge', ...,
  'buoy3__Yaw']   (18 labels; pattern 'buoyK__DOF')
added_mass dims: ('omega','radiating_dof','influenced_dof'), shape (13,18,18)
```

**Full-matrix symmetrization at `__post_init__`** (generalizing PR2's
6×6-blockwise pass to one 6N×6N pass), grounded on G5 (18×18 asymmetry
= 1.08e-4, the same panel-noise scale as the single-body 6×6, so one
tolerance covers the full matrix).

**SCOPE ADDITION — multi-body excitation block.** Per-body
Froude-Krylov + diffraction force **with inter-body scattering**
(`F_exc,i(ω)` includes waves scattered off the other hulls). Without
it the Q6 terminal wave-response stage has no forcing. **Diffraction
problems ride the M8 Capytaine runs** (same solve, added to the
radiation problem set — the studies already assemble diffraction
alongside radiation).

**GROUNDING:** G-preamble dataset structure; G5 symmetrization scale.
Fixture ready: `studies/cluster-3buoy-rigid/capytaine_multibody_18dof.nc`.

### Q3 — B5 coupled retardation kernel (LOCKED structure; one sub-item open)

**LOCK (structure):** full-matrix **Filon** kernel over the 6N×6N
`B(ω)`, single `t_max` default. The integrator's `RadiationConvolution`
already operates per-DOF-pair, so a 6N×6N kernel drops in.

**OPEN sub-item → M8 Phase-A diagnostic.** Reframed as primarily a
**damping-kernel** question: the added-mass cross-terms are tiny (G4:
0.56 %), but the **radiation cross-terms are large relative to the
diagonal** (G6: the constructive 8.68× lives in the off-diagonal
`B_ij`). So the question is whether the **cross `B_ij(ω)`** decays on
the same time scale as the diagonal — i.e. whether cross-kernels
warrant a separate `t_max`. Measured on the existing fixture
`capytaine_multibody_18dof.nc` (extended to full `B_ij(ω)` on the fine
grid) at M8 Phase A, before the kernel assembly is locked.

**GROUNDING:** G4 (A-cross small) vs G6 (B-cross large) — the contrast
that makes this a damping-side question.

### Q4 — Joint formulation (LOCKED)

**LOCK:** **DAE / KKT** — constraint Jacobian `G`, Lagrange
multipliers `λ`, joints as constraint-Jacobian builders (B1). **Penalty
joints rejected.**

**GROUNDING:** G9/G10. The per-step solve is a single dense
`np.linalg.solve(A_eff, rhs)` (newmark.py:324) on a **constant** `A_eff`
(newmark.py:264) — a KKT border `[[A_eff, Gᵀ],[G, 0]]` extends that one
solve naturally. A penalty joint instead inflates `omega_max` and, via
the explicit-lag floor `dt < 2/omega_max` (connector.py:26,521),
collapses `dt` — multiplying step count without bound as joint
stiffness rises. **Sub-Q (verify at M9 planning, not assumed here):**
does **BB-OFFSET-CONNECTOR** (the M7-Foundation body-body attachment-
offset limitation) close *for free* inside the DAE, as the multibody-
conventions doc expects?

### Q5 — Per-milestone validation gates (LOCKED)

- **M8 (coupled BEM) — kinematic-condensation gate.** A validation
  script projects the 18×18 assembly onto the **rigid mode**: build
  the 18×6 rigid-body kinematic map `T` (each hull's 6-DOF motion as
  the rigid cluster translates/rotates), form `T^T M T`, `T^T A(ω) T`,
  `T^T B(ω) T`, `T^T C T`, run the **condensed 6×6 decay**, and require
  it to reproduce the rigid-cluster study at **rtol 1e-2** (heave
  `T_n = 3.106 s`; R). This is pure linear algebra on the M8 coupled
  outputs — **no joint machinery**. **Static pre-gate already PASSED
  at +0.000 %** (G3: 18-DOF 9-block heave sum vs composite).
- **M9 (joints/DAE).** Gate = analytical **2-body hinge** (closed-form
  natural frequency) + energy conservation over a free swing
  (conservation tier `rtol=1e-10, atol=1e-12`).
- **M10 (articulated-3).** Gate = articulated-3 heave decay reproduces
  the rigid-cluster study **when the joints lock heave** (built-in
  cross-check: the rigid study *is* the joints-locked limit) +
  **pitch-perturbation runs that measure joint rotation amplitudes**
  (the LEVEL2 gate's input).
- **LEVEL2 gate (post-M10).** If measured rotation amplitudes exceed
  the F2 small-angle bound (`|θ| < 0.1 rad ≈ 5.7°`, conventions
  Item 2), LEVEL2 (tracker LEVEL2-INTEGRATOR-UNWIRED) becomes a
  required M10.5; else deferred.
- **M11 (terminal).** = Q6.

**GROUNDING:** G3 (static pre-gate passed); G2/T_n (dynamic target).

### Q6 — Terminal gate (LOCKED, two-stage)

- **Stage 1 — screening.** Digitized OrcaFlex 12-buoy RAO +
  acceleration plots; compare peak frequencies and magnitudes at
  **5-10 % tolerance**. Documented explicitly as **consistency
  screening against a prior model, NOT ground truth** (the
  OrcaFlex/OrcaWave licenses are gone — CLAUDE.md §12 — so only
  pre-existing exported artifacts exist, and only as plots).
  **Requires:** Xabier supplies the plot inventory at **M11 planning**;
  the digitization procedure is documented then.
- **Stage 2 — validation.** Tank-campaign data: **Cd calibrated from
  tank decays first** (consumed per Q8), then quantitative response
  comparison. Gate tolerances set at **M11 planning** when data
  quality is known.
- **The program is NOT complete at Stage 1.** Stage-2 completion **may
  postdate M11 code-complete** if the test campaign does — stated
  explicitly so "M11 done" is not mistaken for "program validated."

### Q7 — Estimates (LOCKED)

Base = optimistic engineering estimate; **×3-4** applies the M5/M6
historical "took 3-4× longer than planned" multiplier. **Units caveat:**
the multiplier was calibrated on **human-effort** milestones; M7.5 was
*planned* 2-3 weeks but *closed in ~2 calendar days* (`a14a1ef`
2026-07-02 → `3a6d00f` 2026-07-04, AI-assisted) — the multiplier `< 1`
in AI-assisted calendar time. Estimate **unit** is a live question for
each milestone's own plan.

| milestone | base (human-effort) | ×3-4 | note |
|---|---|---|---|
| M8 B4/B5 + excitation | 3-5 wk | 9-20 wk | coupled reader + kernel + diffraction; 18-DOF fixture ready |
| M9 B2/B1 (DAE) | 4-6 wk | 12-24 wk | integrator rework — largest unknown |
| M10 articulated-3 | 2-3 wk | 6-12 wk | validation study (`studies/` pattern) |
| LEVEL2 (if gated in) | 3-6 wk | 9-24 wk | tracker LEVEL2-INTEGRATOR-UNWIRED |
| M11 12-buoy terminal | 3-4 wk | 9-16 wk | + BEM cost (risk register) |
| **newest multiplier datum** | M7.5 planned 2-3 wk | **actual ~2 days** | AI-assisted; calendar multiplier < 1 |

### Q8 — Exclusions (LOCKED)

See Scope + exclusions above: irregular-sea statistics OUT; platform
mooring OUT; elastic arms OUT (rigid assumed — also an M9 joint-model
input); Cd calibration OUT (consumed from tank campaign); regular-wave
response IN.

---

## Sequencing-risk register

| risk | evidence | mitigation |
|---|---|---|
| **LEVEL2 discovered necessary early** | rotation amplitudes UNMEASURED until M10 | gate after M10; rotation-agnostic `G` rider (Q1) so a late swap is cheap |
| **B2 solve-structure rework larger than the audit suggests** | per-step solve is one dense `np.linalg.solve` on a constant `A_eff` (G9/G10) — a KKT border is structurally small, but re-factorizing per step + losing block-diagonality may be a real cost | prototype the 2-body KKT at M9 planning before committing the integrator rework |
| **12-buoy BEM cost** | 12 × 1488 = **17,856 panels**. Per-problem cost measured: 1488 p → 0.12 s, 4464 p → 0.53 s ⇒ scaling power log(0.53/0.12)/log(4464/1488) = **1.35**. So 17,856 p (4× of 4464) ⇒ 0.53 × 4^1.35 = **~3.5 s/problem**. 72 DOF × 13-omega reduced grid = 936 problems ⇒ 936 × 3.5 ≈ **55 min**; 40-omega grid ⇒ ~2952 problems ⇒ **~2.9 hr**. **Binding constraint is likely memory:** a dense 17,856² complex128 influence matrix = 17,856² × 16 B = **~5.1 GB per omega**. | **mitigation ladder:** (1) frequency-grid economy (reduced omega set, bracket omega_n); (2) per-cluster symmetry exploitation (the 12-buoy platform has a point-group symmetry — Capytaine `xarray`/reflection planes reduce the unique solves); (3) batch / overnight runs; (4) Capytaine solver options (iterative solver / hierarchical matrices if memory-bound). Probe memory before the full run. |
| **noise amplification at 72 DOF** | cluster's symmetry-allowed cross-DOF rose ~1e-13 (single) → ~1e-10 (3-buoy) as panel noise propagated through more cross-terms (G8); 72 DOF adds another factor | full-matrix symmetrization (Q2); monitor the forbidden-channel noise floor as a per-N diagnostic |
| **Stage-2 gate schedule risk** | the tank campaign timing is **outside program control**; Stage-2 validation may postdate M11 code-complete | decouple M11 code-complete from Stage-2 validation in the program status (Q6); Stage-1 screening is the interim confidence signal |

---

## Milestone-plan protocol

- **Each of M8-M11 gets its own Q&A-lock plan** in the house style
  (`m*-plan.md`) **before its branch opens.** This program plan is the
  parent; milestone plans inherit its locks and add milestone-specific
  ones.
- **Append-only amendments.** When a milestone surfaces a lock-grade
  finding, this program plan is amended append-only with a timestamped
  entry (never a silent in-place edit). Precedent: the **M7.5 Q2
  amendment chain** (six delete-and-replace amendments + a test-driven
  correction, recorded in `m7.5-reader-hygiene-closure.md` §4) — that
  is how lock revisions are recorded honestly.
- **Numerical-grounding re-derivation rule (established this session).**
  Numerical grounding values in any plan or instruction are
  **re-derived from the committed record at drafting time, never
  carried from conversational context; discrepancy is a stop
  condition.** Precedent for the rule: the **Phase-2 stale-numbers
  stop (2026-07-05)** — a program-plan instruction carried stale
  figures (R=1.086, T_n=3.196 s, "B33 +4 %", "sway 4 orders above
  floor"), two of which had already been corrected on `main` at
  `cbc0dc1`; drafting halted and itemized the discrepancy rather than
  committing self-contradictory grounding. This sits alongside the
  M7.5 Q2 chain as the two worked precedents for "the record wins."

---

## Session-continuity notes

- **Empirical preamble is the two committed studies** (spar-fin merge
  `cf9eda3`, cluster FF-merge `cbc0dc1`) + the 18-DOF diagnostic
  (fixture `capytaine_multibody_18dof.nc`). All grounding values
  re-derived from `main` on 2026-07-05.
- **Locks Q1-Q8 are set;** open items are explicitly scoped to their
  milestone (Q3 cross-kernel `t_max` → M8 Phase A; BB-OFFSET-CONNECTOR
  closure → M9 planning; LEVEL2 necessity → M10; Q6 reference inventory
  → M11 planning).
- **Next session = M8's own Q&A-lock plan.** Do not open M8 planning,
  branches, or B4 scoping from this document.
