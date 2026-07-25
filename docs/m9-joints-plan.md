# M9 — Articulated joints (velocity-level KKT + joint types + Q5 closure) — PLAN

**Status: LOCKED (Q1–Q8), 2026-07-25.** Second milestone of the Tier 3
program ([`tier3-program-plan.md`](tier3-program-plan.md), `a623bda`);
inherits its **Q4 lock: DAE/KKT, penalty joints rejected**, and its
**Q1 rider: constraint Jacobian `G` rotation-parameterization-agnostic
where feasible**. Grounded in the Phase-1 measurements below — every
number re-derived from the committed record or measured fresh at
planning time, per the program's standing rule.

**Milestone goal.**
- **B2** — a KKT constraint layer on the generalized-alpha integrator:
  constraint Jacobian `G`, Lagrange multipliers `λ`, one bordered
  per-step solve.
- **B1** — joint types as constraint-Jacobian builders: the **hinge**
  (gate case) and the **12-buoy joint** (3 translations + yaw locked,
  roll/pitch free — the required production case).
- **Q5 (inherited from M8 closure S7)** — `build_system` coupled path:
  shared N-body database + per-body `hydro_body_label` + a deck joints
  section.
- **BB-OFFSET-CONNECTOR** — closure verified by test, not asserted.

**Scope exclusions.** NO articulated-3 validation *study* (M10); NO
LEVEL2 quaternion wiring (`LEVEL2-INTEGRATOR-UNWIRED`, gated after
M10); NO new hydro. Small-angle regime only (see the nonlinearity
scope statement in the gates section).

---

## Phase-1 measurements (grounding — all measured, none carried)

| id | measurement | result | source |
|---|---|---|---|
| **MA** | Newmark solve structure | `A_eff = (1-α_m)(M+A_inf) + (1-α_f)h²β C` assembled once (`newmark.py:264`), constant; per step **one** `np.linalg.solve(A_eff, rhs)` (`:324`), **re-factorized every step** (no cached factor, docstring `:216`); non-iterative predictor → acceleration solve → position/velocity update. The corrector is **algebraic in end-of-step state** — acceleration is the solved variable, velocity/position are affine in it. | `newmark.py` end-to-end |
| **MA-timing** | time-per-step, `n_dof=24` | **10.49 ms/step** (kernel `N_K=20001`, dt=0.01), convolution-dominated; isolated `np.linalg.solve(A_eff)` **0.0059 ms/solve**. Supersedes the F4-era 21 ms/step (G9) — environment drift; this is the current baseline. | standalone, N4 fixture |
| **MA-force** | connector force path | explicit lagged RHS at previous-step state (`newmark.py:310`, `connector.py:414`); stability floor `dt < 2/ω_max` (`connector.py:26,521`). Constraint forces take a NEW path (implicit, in the border) — no lag, no floor. | `connector.py`, `newmark.py` |
| **MB** | hinge gate references | Compound pendulum (uniform bar, hinge→earth): `ω_n = √(3g/2L)`, **`T_n = 1.637947 s`** (m=1 kg, L=1 m); scipy `solve_ivp` independent check `1.637945 s` (rel **6.9e-7**), energy drift **2.4e-10** over 20 s. Double pendulum (two point masses, massless rods l=1): modes **`T_+ = 2.621052 s`**, **`T_- = 1.085675 s`** (`ω² = (g/l)(2∓√2)`); scipy **1.8e-6 / 2.1e-6**. Finite-amplitude correction at gate amplitude θ₀=0.02: **2.7e-5** (=θ₀²/16), far below gate rtol. | standalone + `solve_ivp` |
| **MC** | energy-drift baseline | Unconstrained conservative oscillator, implicit restoring (the KKT-comparable path), dt=0.01, 100 cycles: `rho_inf=1.0` → **(max-min)/mean = 3.8e-14** (secular +2.2e-17/cycle, machine precision); `rho_inf=0.9` → **6.7e-8** (designed dissipation). **Finding:** the explicit-lag connector-spring path is *unstable* on the same problem (+30 %/cycle) — re-confirming why penalty joints were rejected; only the implicit path conserves. | standalone, integrator |
| **MD** | Q5 inheritance | `build_system` per-body block-diagonal (`driver.py:454-463`); `Body.hydro_database` per-body; no shared-database declaration; no joints in the deck (`Connection` union, `deck.py:215`). | `driver.py`, `deck.py` |
| **ME** | geometric-stiffness closure (planning finding — see below) | The velocity-form constraint `G = [I, −(R(θ)r)~]` with configuration-dependent arm reproduces the compound pendulum exactly: reduced-DAE integration gives `1.637991 s` at θ₀=0.02 vs reference `1.637947 s` — residual 2.71e-5 **is** the finite-amplitude term. | standalone, this plan |

### Planning finding ME — first-order kinematics loses the pendulum

Derived while checking the gate physics: with strictly first-order
rotation kinematics (`φ = x + θ×r`, constant `G = [I, −r̃]`) and
constant gravity at the CoM, the constrained system has **no restoring
at all** — the pendulum's stiffness is *geometric*: it is the product
of the constraint force (≈ mg) with the first-order rotation of the
moment arm, i.e. it lives in **`G`'s configuration dependence**
(2-D: `G = [[1,0,d cosθ],[0,1,−d sinθ]]`; the `−d sinθ ≈ −dθ` entry ×
λ_z ≈ mg gives the `−mgdθ` torque). Eliminating λ analytically
recovers `(I_c + md²)θ̈ = −mgd sinθ` exactly (verified numerically,
ME). **Consequence, binding on PR1/PR2:** `G` is evaluated at the
current configuration each step with the rotated arm `R(θ)r`, and the
constraint residual `φ` carries second-order Rodrigues
`R(θ) = I + θ̃ + ½θ̃²`. A constant-`G` implementation produces smooth,
plausible, dead-wrong motion (a pinned body that never swings) — and
the hinge gate, derived independently before implementation, is
exactly what catches it. This is the plausible-wrong-dynamics risk row
realized and disarmed at planning time.

---

## λ-units derivation (mandatory; pinned by the PR2 static test)

**Formulation (Q1/Q2 locks below).** The constraint force enters the
discrete generalized-alpha balance **wholly implicitly** (lumped, no
lagged history):

```
(1-α_m)M_eff a_{n+1} + α_m M_eff a_n + (1-α_f)C x_{n+1} + α_f C x_n + μ_n
    = (1-α_f)F_{n+1} + α_f F_n + Gᵀ Λ_{n+1}
```

which rearranges onto the existing rhs (`newmark.py:316-323`) as the
bordered system

```
[ A_eff   −Gᵀ ] [ a_{n+1} ]   [ rhs                                  ]
[ G        0  ] [ Λ_{n+1} ] = [ −(1/hγ) G (u_n + h(1−γ) a_n)         ]
```

with the second row equivalent to the velocity-level constraint
`G u_{n+1} = 0` under the Newmark velocity update
`u_{n+1} = u_n + h(1−γ)a_n + hγ a_{n+1}`. `G` is evaluated at the
predictor configuration `x_pred` (known before the solve — no
iteration).

**Units.** Every term of the first row is a generalized force. The
translational rows of `G` are dimensionless (∂position/∂position ≈
identity blocks and dimensionless arm-skews scaled by metres on the
rotational *columns*... concretely: `G_trans = [I₃, −(Rr)~]`, so
`Gᵀ Λ`'s force block is `Λ_trans` and its moment block is
`(Rr) × Λ_trans`). Therefore:

> **`Λ` IS the physical constraint force: N on translational rows,
> N·m on rotational rows. There is NO dt scaling.** The `1/(hγ)`
> factor lives on the constraint RHS (row 2), where it converts a
> velocity residual into the acceleration budget — it never touches
> the multiplier column.

**Statics — exact.** At rest (`a=0, u=0, x` const, `μ=0`) the α-weights
sum to one, so the first row reads `F + GᵀΛ = Cx`: `Λ` equals the
physical constraint force **exactly**, independent of dt, α_f, γ.

**Dynamics — O(h)-blended.** In motion, the lumped `Λ_{n+1}`
approximates the α_f-blended multiplier `λ(t_{n+1-α_f}) + O(h)`. The
alternative bookkeeping (α_f-weighted history carry, giving `Λ =
λ(t_{n+1})` exactly) is documented and **rejected for M9**: it adds a
λ-history state for an O(h) interpretation refinement invisible to
every M9 gate; statics — which the joint-load extraction cares about —
is already exact.

**Stop-condition check:** the derivation closes with no dt ambiguity.
**Pin:** PR2 static recovery test — the hanging compound pendulum's
hinge force is `(0, 0, +mg) = (0, 0, 9.81) N` closed-form; the solved
`Λ` must recover it. Measured value reported verbatim in PR2.

---

## Locks

### Q1 — Constraint formulation (LOCKED): velocity-level KKT + position projection

**Enforce `G u_{n+1} = 0` in the bordered solve** (index-1 reduced,
velocity level), plus **position projection** onto `φ(x) = 0`.
Grounding: MA — the corrector is algebraic in end-of-step state and
acceleration is the solved variable; the velocity constraint is an
*affine* function of `a_{n+1}` through the Newmark update, so
velocity-level enforcement is exact at the solve with no
differentiation-induced `Ġu` term (scleronomic joints ⇒ RHS is just
the known predictor part). Position drift is O(h²)/step and controlled
by projection: mass-metric least-squares
`Δx = −W Gᵀ (G W Gᵀ)⁻¹ φ(x)`, `W = (M+A_inf)⁻¹` (the kinetic-energy
norm — the physically-derived weight, not a tuning knob), one Newton
iteration (φ is near-linear at small angle), with matching velocity
re-projection. **Projection FREQUENCY is a design parameter MEASURED
at PR2** (every step vs every N: drift ‖φ‖ vs cost), not assumed.

**Baumgarte REJECTED on the record:** its stabilization parameters are
tuning knobs without physical derivation — the pattern this project
refuses (same reasoning as the M8 no-interpolation rule). Index-3
direct enforcement rejected: couples the constraint into the position
update, which is not the solved variable (MA); nonlinear per-step
solves for machinery the velocity-level form gets exactly.

### Q2 — KKT solve (LOCKED): bordered dense, assembled per step

`[[A_eff, −Gᵀ],[G, 0]]`, built and solved with `np.linalg.solve` each
step. Grounding: the unconstrained path **already re-factorizes A_eff
every step** (MA; `newmark.py:324` — the "assembled once" is the
matrix, not the factorization), so the border is **marginal cost, not
structural change**. MA-timing: physical solve 0.0059 ms vs 10.49
ms/step (convolution-dominated); the border at `m ≤ 15` stays
microseconds. **Overhead budget: < 50 % of the unconstrained 10.49
ms/step at n_dof=24, MEASURED at PR2; exceeding the budget is a
finding to report, not a gate failure.** Redundant-constraint guard:
`rank(G) < m` raises at setup (KKT conditioning risk row).

### Q3 — Joint builders (LOCKED): angular-velocity form

`G` maps **generalized velocities `(v, ω)` → constraint rates**, never
parameterization derivatives. This is HOW the program's
rotation-parameterization-agnostic rider is honored structurally: a
LEVEL2 parameterization change touches kinematics only; the constraint
definitions survive. Concretely: translational rows
`v_ref + ω×(R(θ)r) = 0` ⇒ `G = [±I₃, ∓(R(θ)r)~]`; rotational rows are
locked-direction selectors on the relative `ω`. `φ` (for projection)
uses second-order Rodrigues (finding ME). Two required builders:
1. **`yaw_locked_joint`** — the 12-buoy joint: 3 translations + yaw
   locked, roll/pitch free (4 rows).
2. **`hinge_joint`** — the gate case: 3 translations + 2 rotations ⊥
   axis locked, 1 free (5 rows).

Bodies referenced by **index** at the `floatsim.bodies` layer (the
`LinearConnector` convention, earth = −1); by **name/label** at the
deck layer with the M8 label contract
(`tests/support/condensation.py` the reference implementation).
**PR1 unit gate: finite-difference verification of every Jacobian** —
directional derivatives along *group-consistent* flows (rotations
composed via scipy `Rotation`, not added), so the test itself is
parameterization-honest; plus exact analytic equality at θ=0.

### Q4 — BB-OFFSET-CONNECTOR: closed by construction, TESTED before struck

Constraint points carry body-frame offsets natively (`G`'s `∓(Rr)~`
blocks deliver `F = GᵀΛ` with automatically-consistent moments to both
bodies — the reference-point asymmetry that broke the penalty
connector, `connector.py:172`, never arises because the force is
solved, not prescribed). **The closure claim is tested, not
asserted:** PR4 runs the two-body hinge (the double pendulum — its
inter-body constraint point is offset from body CoGs, exactly the
tracker's failure topology) against the MB mode references, and
cross-checks the original failure mode (the penalty path still raises
`NotImplementedError`; the joint path passes) BEFORE the tracker entry
is struck.

### Q5 — Deck schema + `build_system` coupled path (LOCKED)

- **Schema:** deck-level `shared_hydro_database` declaration (single
  shared N-body database, M9 scope); `Body.hydro_body_label:
  str | None = None` selecting its block by label;
  `Deck.joints: list[...]` — discriminated union like `Connection`
  (`deck.py:215`), entries `hinge` / `yaw_locked` with
  `body_a/body_b` (names or `earth`), body-frame attach points, axis.
- **Assembly:** when the shared database is declared, build the
  coupled `6N×6N` LHS + kernel **directly** from it (the M8 reader
  yields a labelled 6N `HydroDatabase`; the M8 kernel is 6N-generic),
  bypassing block-diagonal stacking; per-body gravity restoring added
  per block honoring `C_source` (the M5 hydrostatic-gravity lesson).
  Label→block mapping by label, hard raises on mismatch / missing /
  duplicate.
- **HARD GATE (the M8 N=1 pattern applied to `build_system`): every
  existing committed deck parses and builds BYTE-IDENTICALLY** —
  sha256 of assembled `M_plus_Ainf` / `C` / kernel `K` captured on
  pre-change code, asserted post-change; the no-labels single-database
  path is untouched code.

### Q6 — PR sequence (LOCKED)

- **PR1** — `floatsim/bodies/joints.py`: builders + `JointSet`
  (φ, ω-form G). Step A: FD-gate design (group-consistent flows).
  Step B: implement. Step C: FD gates pass at randomized
  configurations; θ=0 analytic equality; raise paths.
- **PR2** — KKT integrator: bordered solve + projection in
  `integrate_cummins` (optional `constraints`), λ recorded. Step C
  gates: hinge period vs MB (dev-run of the terminal gate), λ static
  recovery = mg, energy vs MC baseline, **measurements**: overhead vs
  10.49 ms/step, projection-frequency sweep.
- **PR3** — deck schema + `build_system` coupled path. Step C: the
  byte-identity hard gate + coupled-path label-contract raises.
- **PR4** — terminal gates as permanent tests (hinge, energy, λ
  recovery, two-body BB-OFFSET double pendulum), tracker strike,
  closure doc, full suite.

### Q7 — Pre-M9 audit doc (LOCKED)

`docs/audits/m9-joints-audit.md` — MA + MD formalized, the ME
geometric-stiffness finding as institutional knowledge, the
connector-vs-constraint force-path distinction, λ-units cross-ref.
Committed with this plan.

### Q8 — Estimate (LOCKED)

| milestone | window (git dates) | days | PRs |
|---|---|---|---|
| M7.5 | `6d457c2` 07-03 → `3a6d00f` 07-04 | ~2 | 4 |
| M8 | plan `11b2d8a` 07-18 → close `1e5b297` 07-22 | ~4–5 (incl. contaminated-slice surprise; base ~3) | 4 |
| **M9** | base ~4 × **1.5** multiplier | **~6** | 4 |

**Variance driver, named with evidence:** the constraint formulation
surviving contact with the integrator. Phase 1 already shows the
contact surface is subtle — finding ME (geometric stiffness lost by
first-order kinematics) would have produced a silently-dead pendulum
had the gate not been derived first; the projection-frequency and
energy behaviors are MEASUREMENT-PENDING to PR2 by design.

---

## Terminal gates

### Gate 1 — hinge analytical (with full derivation; never calibrated on FloatSim)

Uniform bar, m = 1 kg, L = 1 m, hinged to earth at one end, gravity
g = 9.81, released from θ₀ = 0.02 rad at rest.
Derivation: `I_pivot = I_c + md² = mL²/12 + m(L/2)² = mL²/3`;
`I_pivot θ̈ = −mg(L/2) sinθ` ⇒ small-angle
`ω_n = √(mg(L/2)/I_pivot) = √(3g/2L) = 3.836014 rad/s`,
**`T_n = 1.637947 s`**. Independent verification (Phase 1, scipy
`solve_ivp` on the reduced ODE): period rel-diff **6.9e-7**, energy
drift **2.4e-10** over 20 s. Finite-amplitude correction at θ₀ = 0.02:
θ₀²/16 = **2.5e-5** (measured 2.7e-5, ME) — negligible against the
gate tolerance. **Gate: FloatSim constrained run reproduces T_n at
rtol 1e-3.** (Expected discretization error O((ω_n h)²)/12 ≈ 1.2e-4 at
dt = 0.01 — ~8× margin.)
Two-body form (PR4, BB-OFFSET): double pendulum, point masses,
modes **T_+ = 2.621052 s / T_- = 1.085675 s**, same rtol.

**NONLINEARITY SCOPE STATEMENT.** The hinge gate validates the
constraint machinery in the small-angle regime (θ₀ = 0.02 rad).
Large-angle validity is LEVEL2's question, measured at M10, and is NOT
certified by this gate. A passing hinge gate means the constraint
formulation is correct, not that the small-angle kinematics are
adequate for the articulated platform.

### Gate 2 — energy conservation (tolerance from the measured baseline, not convention)

Energy functional `E = ½ uᵀ(M+A_inf)u + PE` (no radiation, no drag) on
the constrained conservative hinge run, dt = 0.01, `rho_inf = 1.0`,
100 cycles. Measured unconstrained baseline (MC): **(max-min)/mean =
3.8e-14** at matched settings; the integrator's own designed
dissipation at production `rho_inf = 0.9` is **6.7e-8**. **Gate
ceiling: (max-min)/mean < 1e-6 over 100 cycles** (≈ 15× the production
dissipation scale — the constraint machinery must not add more than
one order above the integrator's own designed loss). PR2 measures the
actual constrained value, reported verbatim; **PR4 locks the permanent
tolerance at 10× the measured value or 1e-6, whichever is tighter** —
baseline-derived, never loosened past the ceiling.

### Gate 3 — λ static recovery (pins the units derivation)

Hanging compound pendulum at rest: closed-form hinge force
`(0, 0, +mg) = (0, 0, 9.81) N`. The solved `Λ` (which the derivation
above states IS the physical force, dt-free) must recover it.
Measured value reported verbatim at PR2; permanent test at PR4.

---

## Risk register

| risk | mechanism | mitigation |
|---|---|---|
| plausible-but-wrong dynamics | a constrained integrator produces smooth wrong motion; **realized at planning as finding ME** (constant-G kills the pendulum silently) | both gates derived + independently verified BEFORE implementation (MB in the plan with derivation); the hinge gate catches exactly the ME failure |
| KKT conditioning at 6N+m | saddle-point border; near-redundant constraints blow up λ | `rank(G) < m` hard raise at setup; KKT condition number reported at PR2 |
| constraint drift over long runs | velocity-level enforcement drifts at position O(h²)/step | position projection (Q1), frequency measured at PR2; `‖φ‖` residual asserted in the PR4 gates |
| dt–stability interaction with the explicit-lag connector floor | decks can mix implicit joints with explicit penalty connectors; the tightest connector floor still governs dt | `check_connector_stability` stays in pre-flight; plan states joints do NOT relax a connector's floor |
| Q5 schema churn breaking legacy decks | new shared-db + joints fields shift deck validation | byte-identity HARD gate (Q5): sha256 of assembled outputs pre/post on committed decks; no-labels path untouched |
| KKT overhead scaling with n | dense bordered LU is O((n+m)³); benign at n=24 (MA-timing) but grows faster than the convolution's O(n²·N_K) | measured at PR2 (n=24); M10's articulated-3 provides the second scaling point at no extra cost; two points + O(n³) behavior decide BEFORE M11 planning whether sparse/Schur (program B6) is pulled forward — measurement scheduled, not assumed |
