# M10 — Articulated-3 validation study — PLAN

**Status: LOCKED (Q1–Q8), 2026-07-28.** Locked on Phase-1
Measurements A–E (below, all re-derived from the committed record;
zero disagreements). Third milestone of the Tier-3 program
([`docs/tier3-program-plan.md`](tier3-program-plan.md) Q1: M10 → LEVEL2
decision gate → M11; Q5: the M10 gate). Implementation is the next
session; this plan opens no branch and no PR.

---

## Goal

Produce the **first articulated multi-body simulation** on the FloatSim
stack — three buoys on joints replacing the rigid 3-buoy cluster's
welded arm — and from it the **LEVEL2 decision input**: the measured
joint rotation amplitudes that decide whether the quaternion integrator
(`LEVEL2-INTEGRATOR-UNWIRED`) becomes a required M10.5 or is deferred
(program plan Q5: threshold `|θ| < 0.1 rad ≈ 5.7°`, conventions Item 2).

### Scope exclusions
- **No 12-buoy** — that is M11. M10 is the 3-buoy validation study.
- **No LEVEL2 implementation** — M10 produces the DECISION INPUT only;
  the Euler→quaternion swap is a post-M10 gate.
- **No new physics** — reuse the committed 18-DOF coupled BEM fixture
  and the M9 joint layer.

---

## Phase-1 measurements (re-derived from the committed record; zero disagreements)

### Measurement A — articulated-vs-rigid mass consistency (the heave gate's precondition)

Committed rigid cluster (`studies/cluster-3buoy-rigid/`):

| quantity | committed value | source |
|----------|-----------------|--------|
| M_cluster | 98.01 kg (= 3×28.67 hulls + 12.0 arms) | `cluster_common.py:32,39,49` |
| CoG z | −0.98886768 m | `mass_properties.json:cog_m` |
| I_about_cog | diag(113.29389, 113.29389, 22.84450) kg·m² | `mass_properties.json:inertia_about_cog` |
| dz2 | 0.17936743 m | `cluster_common.py:54` |
| per-hull mass | 28.67 kg | `cluster_common.py:32` |
| per-hull inertia (CoM) | diag(24.0, 24.0, 0.114) kg·m² | `cluster_common.py:34-35` (M8 `_I_HULL`) |
| hull CoG z | −1.19566743 m (= CoG_Z_SINGLE − dz2 = −1.0163 − 0.17937) | `mass_properties.json:z_buoy_cog` |
| arm CoG z | +0.49336957 m | `mass_properties.json:z_arm` |

`dz2` is the cluster's extra draft below the isolated single-buoy
waterline, floating the 12 kg arm by buoyancy. The rigid model lumps
the 12 kg as **3 uniform radial rods** (4 kg each, hub→buoy, L=0.5,
CoM at r=0.25, z=z_arm), `I_rod = (mL²/12)(I − u⊗u)`, parallel-axis
summed with the hulls about the composite CoG
(`cluster_balance.py:88-97`).

The per-buoy properties are FIXED by the record; the heave gate needs
only the total mass + A33 (trivially matched). The **arm mass/inertia
split** in the articulated model is a modelling choice — locked in Q2,
flagged there against the physical build.

### Measurement B — joint topology and constraint count (numerically verified)

- **Hub topology:** 3 buoys + 1 hub = **24 DOF**; 3 `yaw_locked`
  joints × (3 translations + yaw) = **m = 12**; free = 24 − 12 = **12**
  (hub 6 + 3 buoys × 2 rotations). Measured at the study geometry
  (joints at the arm-buoy interface, offset from both CoMs):
  `phi(rest) = 0`, `G` is (12, 24), **rank(G) = 12 (full)**,
  cond(KKT) ≈ 1.9e2.
- **Hub-free ring rejected:** 3 pairwise joints form a closed loop →
  **redundant constraints** (rank(G) < 12) → the M9 λ-blow-up mode;
  the naive 18 − 12 = 6 is also physically wrong (12 free DOF needed).
- **KKT scaling point:** (n_dof=24, m=12). M9 measured (24, 5). Same
  n=24 → informs **m-scaling** only, NOT the **n-scaling** question
  (n=72 at M11).

### Measurement C — hub-body singularity (tested; reconciled with M9 PR4)

The predictive quantity is **`rank(M_plus_Ainf)`** — the integrator
precomputes `w_inv = np.linalg.inv(M_plus_Ainf)` for the position
projection (`newmark.py:352`); a singular mass matrix throws there,
before any step. (The bordered KKT solve is a direct solve,
`newmark.py:159`, and is full-rank even at I=0 — the reconciled M9 PR4
mechanism.) Dry hub (A_inf = 0) measurements:

| hub inertia | `rank(M_plus_Ainf)` | integrator |
|-------------|---------------------|-----------|
| rod-derived diag≈(0.5, 0.5, 1.0) | **24/24** | OK |
| bare point mass (I=0) | 21/24 | LinAlgError in `w_inv` |
| regularized 12·1e-6·I | 24/24 | OK |

Physical rod inertia → no singularity, no regularization (Q3).

### Measurement D — capability audit (file:line)

| capability | status | evidence |
|-----------|--------|----------|
| (a) Item-25 override in the coupled path | **GAP** | override at `retardation.py:210`; coupled path calls without it (`driver.py:497`) |
| (b) deck `yaw_locked` offset joint | OK | `YawLockedJoint` (deck.py, M9 PR3) |
| (c) mixed shared-DB buoys + dry hub | **GAP** | dry body rejected by `Body._exactly_one_hydro_source` (`deck.py:161`); coupled path requires every body labelled |
| (d) Morison/drag on buoys in the coupled assembly | **GAP** | `build_system` assembles no `drag_elements` |

Closed by Q4: (a) + (c) become PR0; (d) deferred with rationale.

### Measurement E — heave-gate reference (derived before the model exists)

```
T_n = 2π √( (M_cluster + A33_composite_inf) / C33_composite )
    = 2π √( (98.01 + 64.0738043) / 663.2420101 )
    = 3.106087 s      (ω_n = 2.0228618 rad/s)
```
Committed `interaction.json:T_n_with_interaction = 3.1060873561 s` —
agreement **rel 8.4e-12**. Symmetry: 3-fold + y-mirror → a symmetric
heave IC excites only symmetric modes → **zero pitch**; nonzero pitch
signals an assembly/constraint asymmetry bug.

---

## Locks (Q1–Q8)

### Q1 — Body/joint topology (LOCKED)
**4-body topology:** 3 buoys + 1 hub, **3 `yaw_locked` joints** at
offset attachments (buoy↔hub), **m = 12**, **12 free DOF**. Grounding:
Measurement B — rank(G) = 12 (full) at the study geometry,
cond(KKT) ≈ 1.9e2; the hub-free ring is rejected (redundant
constraints). The hub is a dry structural body.

### Q2 — Arm-mass distribution and hub properties (LOCKED)
The **hub carries all 12 kg** of arm structure at the arm CoG
(0, 0, +0.49337), with the **rod-derived inertia** diag≈(0.5, 0.5, 1.0)
kg·m²; per-buoy properties unchanged (28.67 kg, diag(24, 24, 0.114)).

**Physical rationale (on the record):** the joints articulate at the
**BUOY end**, so the hub + the three arms form **one rigid spider**,
and each buoy hangs from an arm tip, free in roll/pitch (yaw locked).
The hub-body's mass and inertia are therefore exactly the welded
arm-structure's, and the composite (hub-spider + 3 buoys) reproduces
the rigid cluster's total M, CoG, and I **by construction** — which is
precisely what makes the heave gate a true cross-check rather than a
re-fit.

**FLAG (in-plan, pending confirmation against the physical build):**
this is a modelling choice. If the real joints sit at the **hub end**
instead, the split inverts — the arms rotate with the buoys, the hub
becomes a bare central mass, and the rotational dynamics (and the hub
singularity picture, Q3) change. Confirm the joint location against the
build before trusting the rotation measurement (Gate 2); the heave gate
is insensitive to the split.

### Q3 — Hub regularization (LOCKED)
**None.** The physical rod inertia gives `rank(M_plus_Ainf) = 24/24`
(Measurement C). **Stated fallback** (only if a topology variant ever
drives the hub to zero/near-zero rotational inertia): the M9 PR4
precedent — regularize `I_hub = M_arm·ε`, then verify
period-insensitivity across **two magnitudes** — measured at PR-time,
not assumed.

### Q4 — Execution path (LOCKED): a scoped PR0
A deliberate, scoped capability PR **PR0** delivers:
- **(a) Item-25 override exposure** through the coupled `build_system`
  path: thread `asymptote_check_override` to the
  `compute_retardation_kernel` call at `driver.py:497` (and through
  `build_system`'s signature). The cluster hulls (L~1.85 m) need it.
- **(c) Dry-body support:** relax `Body._exactly_one_hydro_source`
  (`deck.py:161`) to permit a **structural body with no hydro source**,
  and make the coupled assembly handle a deck that **MIXES** a
  shared-N-body-database group (the 3 buoys) with a hydro-free body
  (the hub). **That mixed case is PR0's gate** (build the mixed deck →
  assemble → the hub contributes rigid mass only, no BEM block).

**Gap (d) drag/Morison — DEFERRED**, rationale on the record: the heave
gate measures a **period** (damping-insensitive), and a **BEM-only**
model **overestimates** rotation amplitude (no drag dissipation), so a
**sub-threshold BEM-only rotation is conservative** for the LEVEL2
decision. If measured rotations land **near 0.1 rad**, that is itself a
reportable finding and drag gets revisited (in M11 planning).

PR0 is **deliberate scoped capability on M11's critical path — not
incidental cleanup** (CLAUDE.md §9): the coupled deck path must
eventually carry the 12-buoy platform, which likewise has structural
(non-hydro) members and needs the small-body override; M10 is the first
place that capability is exercised end-to-end.

### Q5 — Hydrodynamic model (LOCKED)
Reuse the committed **18-DOF coupled fixture**
(`studies/cluster-3buoy-rigid/capytaine_multibody_18dof.nc`) with the
**contaminated frequencies excluded** (ω ≈ 4.934, 20.909 — the M8 PR3
grid-selection pattern, not a value edit). **Linear-hydro assumption,
stated explicitly:** the BEM is computed at **one fixed configuration**
(the nominal cluster) while the buoys rotate relative to one another —
exact at zero relative rotation, degrading as they rotate. **Q6
interaction:** large measured rotations would undermine the very
assumption that produced them — that is a **finding, not a failure**,
and it is reported as such.

### Q6 — The rotation measurement (LOCKED: option iii, conditional)
Decay is the **heave gate**; the **LEVEL2 input wants regular-wave
excitation**, because a decay amplitude is a function of the chosen IC
and cannot answer the service question. **PR2 first checks whether the
driver supports time-domain wave forcing and reports file:line either
way.** (Known context: `integrate_cummins` accepts a time-domain
`external_force(t)` hook at `newmark.py:218`; the definitive
"is a regular-wave force builder wired and usable with the coupled
18-DOF model?" check is PR2's.)
- **If supported:** run a regular-wave case at a **stated amplitude
  with its rationale**; report max |θ| per joint.
- **If not:** the LEVEL2 answer ships **PROVISIONAL** (decay-based,
  IC-dependent), and wave analysis moves to **M11 planning**.

Threshold fixed: **`|θ| < 0.1 rad`** (conventions Item 2).

### Q7 — PR/step sequence (LOCKED)
- **PR0** — capability: Item-25 override exposure + dry-body/mixed-deck
  support in the coupled path; gate = the mixed deck assembles;
  byte-identity on existing committed decks (M8/M9 N=1 pattern).
- **PR1** — assemble the articulated-3 model; equilibrium + the **heave
  cross-check gate** (Measurement E) + the **zero-pitch symmetry
  check**.
- **PR2** — the **rotation measurement** (Q6) + the LEVEL2 write-up.
- **PR3** — closure doc (measured rotations, LEVEL2 recommendation,
  the (24, 12) KKT timing point).

### Q8 — Estimate (LOCKED format)
AI-assisted calendar days. Milestone actuals from git dates: **M7.5 ≈
3 d** (Jun 30 → Jul 3), **M8 ≈ 4 d** (Jul 18 → 22), **M9 ≈ 3 d**
(Jul 25 → 28).
- **Base:** ~3 AI-days — PR0 (~1 d, small scoped capability + its
  gate), PR1 (~1 d, assembly + two gates), PR3 (~0.5 d).
- **× multiplier ~1.5** on PR2 for the Q6 fork → **~4–5 AI-days** total.
**Variance drivers, named:** **Q6 scope** (decay-only vs a wave case —
the driver-support check gates it), then **PR0** (whether the mixed
shared-DB + dry-body assembly surfaces a shape issue in the coupled
path).

---

## Gates (references derived before implementation — the M9 pattern)

### Heave gate (correctness, pass/fail)
`T_n = 2π√((98.01 + 64.0738)/663.242) = 3.106087 s`, reproducing the
committed `T_n_with_interaction` to **rel 8.4e-12**; **gate rtol 1e-2**
(M8 cross-check band). With all joint translations locked the
articulated cluster's pure heave is rigid-body-identical, so this is a
true cross-check. **Symmetry rider:** 3-fold + y-mirror ⇒ a symmetric
heave IC excites only symmetric modes ⇒ **zero pitch**; nonzero pitch
indicates an **assembly/constraint asymmetry bug**, not physics.

### Rotation gate (measurement, NOT pass/fail)
Report **max |θ| per joint** with the **IC (or wave amplitude) stated**.
The LEVEL2 decision follows: `max|θ| < 0.1 rad` (Item 2) → LEVEL2
deferred; `≥ 0.1 rad` → LEVEL2 becomes a required M10.5 (program plan
Q5). A near-threshold result also triggers the drag revisit (Q4).

---

## Risk register

| risk | mechanism | mitigation / status |
|------|-----------|---------------------|
| hub singularity | dry point-mass hub → singular `M_plus_Ainf` → `LinAlgError` in the projection `w_inv` (`newmark.py:352`) — **predictive check is `rank(M_plus_Ainf)`, not KKT conditioning** | **resolved** (Measurement C): physical rod inertia → rank 24; PR4 regularization fallback stated |
| G rank / redundancy | hub-free closed loop over-constrains (rank(G)<12) → λ blow-up | **measured** full-rank (12) for the hub topology; re-checked at PR1 on the assembled model |
| linear-hydro vs measured rotations | BEM fixed at one config while buoys rotate; assumption degrades at large rotation | Q5/Q6 interaction; a large rotation is a **finding** (it undermines its own premise), reported not hidden |
| IC-dependence of the LEVEL2 answer | a decay amplitude depends on the chosen IC; service rotations differ | Q6 — wave case if the driver supports it, else the answer ships PROVISIONAL |
| KKT scaling | (n=24, m=12) repeats M9's **n=24** with a larger m | informs **m-scaling** only; the **n-scaling** question (n=72) stays **open for M11 planning** — stated honestly |
| carried fix- debt | black-conformance (3 files); the F2 magnitude-scaled hypothesis-red bound | **non-blocking**; tracked for their own branches (M9 closure S6) |
