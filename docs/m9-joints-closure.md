# M9 — Articulated joints (velocity-level KKT): closure

**Status: CLOSED, 2026-07-27.** Second milestone of the Tier-3 program
toward the 12-buoy articulated platform. Delivered the joint-constraint
layer: index-1 velocity-level KKT enforcement with position projection,
the `hinge` and `yaw_locked` joint types, and the Q5 `build_system`
coupled path for a shared N-body BEM database. Plan:
[`docs/m9-joints-plan.md`](m9-joints-plan.md); pre-M9 audit:
[`docs/audits/m9-joints-audit.md`](audits/m9-joints-audit.md).

---

## S1 — Scope and deliverables

| PR | commit | deliverable |
|----|--------|-------------|
| PR1 | `cac095b` | `floatsim/bodies/joints.py`: `Joint`/`JointSet`, `hinge_joint`, `yaw_locked_joint`; φ residual + angular-velocity-form Jacobian `G`; finite-difference gate (group-consistent flows) |
| PR2 | `5b7eb9c` | velocity-level KKT integrator in `newmark.py`: bordered dense solve + midpoint-`G` iteration + position projection; `IntegrationResult.lam`; single-body terminal gates; plan **Amendment A1** |
| PR3 | `31c9624` | deck schema (`shared_hydro_database`, `hydro_body_label`, `joints`); `build_system` coupled path (by-label 6N assembly); byte-identity hard gate + label-contract raises |
| (adjacent) | `22af89f` | `fix-lint-debt`: cleared pre-existing ruff (E501, N806 in `driver.py`) + mypy (`mesh_hygiene.py` no-any-return) debt — see S6 |
| PR4 | (this) | two-body double-pendulum terminal gate (Q4 BB-OFFSET closure); tracker strike; this closure doc |

---

## S2 — Terminal gates (all PASSED, measured)

All references are **independently derived, closed-form** — never
FloatSim's own output (plan Terminal Gates §1–3, Measurements MB/ME).

### Gate 1a — hinge period (single compound pendulum)
Uniform bar, hinged to earth: `T_n = √(3g/2L) = 1.637947 s`. FloatSim
constrained run reproduces it at **rtol 1e-3** (measured discretization
~1.2e-4 at dt=0.01). `test_m9_kkt_integrator.py::test_hinge_gate_reproduces_pendulum_period`.

### Gate 1b — double pendulum modes (two-body, BB-OFFSET topology) — NEW at PR4
Two point masses on massless rods `l`, absolute-angle normal modes
`ω² = (g/l)(2∓√2)` ⇒ `T_+ = 2.621052 s`, `T_- = 1.085675 s`. Each mode
excited by its analytic eigenvector (θ₂/θ₁ = ±√2), measured by
zero-crossings over 40 cycles at θ₀ = 0.01, dt = 0.01:

| mode | reference | measured rel-err | drift |
|------|-----------|------------------|-------|
| `T_+` | 2.621052 s | **8.8e-5** | 2e-16 |
| `T_-` | 1.085675 s | **4.4e-4** | 2e-15 |

Both within rtol 1e-3. `test_m9_double_pendulum.py`. (θ₀ = 0.01 rather
than the single hinge's 0.02 — the second bob swings √2× larger, so the
faster mode's finite-amplitude term sits at ~9e-4 at θ₀ = 0.02, right
at the gate; 0.01 gives a 2× margin. Point masses need a tiny
`I_c = m l²·1e-6` KKT regularization — see S3.)

### Gate 2 — energy conservation (three clauses, re-derived from the measured floor — Amendment A1)
Constrained conservative hinge, dt = 0.01, `ρ∞ = 1.0`, 100 cycles:
1. **magnitude:** `(max−min)/mean = 2.7e-3` (< 5e-3);
2. **numerical damping:** `ζ_num ≈ 2e-6` (< 1e-5), from the −0.13 %
   amplitude change / 100 cycles;
3. **O(h) scaling:** energy variation `5.4e-3 / 2.7e-3 / 1.4e-3` at
   dt = 0.02 / 0.01 / 0.005 — the truncation signature that a broken
   midpoint iteration would kill.

`test_m9_kkt_integrator.py::{test_energy_gate_magnitude_and_numerical_damping, test_energy_gate_o_h_scaling}`.

### Gate 3 — λ static recovery (pins the units derivation)
Hanging compound pendulum at rest: the solved multiplier IS the
physical hinge reaction `(0, 0, +mg) = (0, 0, 9.81) N`, dt-free
(rel 1e-4). `test_m9_kkt_integrator.py::test_lambda_recovers_static_hinge_force`.

### Drift
Position projection every step holds `‖φ‖` at machine precision
(measured 1.1e-16 single-body; ≤ 2e-15 double pendulum).

### Full-suite counts at M9 close (both scopes, measured)

Two scopes, one selection, stated explicitly so the headline KPI is
not scope-ambiguous:

| scope | invocation | passed | skipped | xfailed | failed | collected |
|-------|-----------|--------|---------|---------|--------|-----------|
| **full (incl. slow)** | `pytest -q` | **728** | 50 | 20 | 1 | 799 |
| non-slow (PR gate) | `pytest -q -m "not slow"` | 638 | 46 | 16 | 1 | 799 (98 deselected) |

The two differ only by the **98 slow tests** deselected under the
marker (90 passed + 4 skipped + 4 xfailed); the single `failed` is the
same pre-existing hypothesis red in both (S6). Runtime: full 1:30:05,
non-slow ~0:15–0:23 on this machine.

**Reconciliation to M8-close** (`docs/m8-coupled-bem-closure.md` §3.5:
`688 / 50 / 20 / 0`, full scope): full-suite passed `688 → 728 = +40`;
`+41` M9 tests added (collection `758 → 799`, monotonic — **no tests
lost**) minus the one existing property test that flipped pass→fail
(the hypothesis red). The M9-internal "708" figure quoted mid-milestone
was the **full-suite** count after PR1+PR2 (`688 + 21 − 1`); PR3 (+17)
and PR4 (+3) carry it to 728. The PR3/PR4 "635 → 638" figures were the
**non-slow** scope — not a regression from 708, a different selection.
For any external KPI, the correct number is **728 automated checks
passing at M9 close, full suite including slow tests** (50 skipped, 20
xfailed, 1 tracked-failing).

---

## S3 — Empirical findings

### Finding ME (planning) — geometric stiffness lives in `G`
With strictly first-order kinematics (constant `G`), the pendulum
restoring vanishes and the constrained integrator produces smooth,
silently-wrong motion. The velocity-form `G = [I, −(R(θ)r)~]` with a
configuration-dependent arm reproduces the compound pendulum exactly.
**Binding consequence:** `G` is evaluated at the swung configuration
each step. Derived before implementation; the hinge gate catches
exactly this failure.

### Amendment A1 (PR2) — `G`'s evaluation point governs energy (factor ~800)
Enforcing the velocity constraint needs `G` at some configuration in
`[xₙ, xₙ₊₁]`. Measured amplitude over 100 cycles (ρ∞ = 1.0):

| `G` evaluated at | amplitude / 100 cyc | verdict |
|---|---|---|
| predictor `x_pred` (plan-implied) | **−99.7 %** | catastrophic |
| endpoint `xₙ₊₁` | −99.7 % | catastrophic |
| **midpoint `(xₙ+xₙ₊₁)/2`, 2 iters** | **−0.13 %** | energy-consistent |

Isolation: **unconstrained** stepping conserves to 3.842e-14 (bit-
matching the MC baseline), proving the dissipation is entirely `G`'s
evaluation point, not the stepper. Fixed-point iteration converged at 2.
The plan's Q1 stop-condition ("if the first choice injects energy past
the MC baseline, PR2 re-does the formulation") fired as designed — and
the re-do happened **before `newmark.py` was touched**, via
prototype-before-editing.

### PR4 — point masses need a KKT regularization (period-insensitive)
A point mass has a singular rotational-inertia block; the bordered KKT
solve is `LinAlgError: Singular matrix` at `I_c = 0`. A tiny isotropic
`I_c = m l²·1e-6` restores well-posedness with **no measurable effect
on the periods** (5-sig-fig-identical at `I_c = 1e-6` and `1e-5` — the
mode inertia is carried by CoM translation, not the body's own spin).

### KKT cost (measured, PR2)
Unconstrained baseline `10.49 ms/step` at n_dof = 24 (kernel
`N_K = 20001`, convolution-dominated); the isolated `np.linalg.solve`
is `0.0059 ms/solve`. Adding the hinge border (m = 5): `10.79 ms/step`,
**+2.2 %** — well under the plan's 50 % overhead budget. The border is
microseconds; the convolution dominates.

---

## S4 — Q5 coupled path + byte-identity (PR3)

`build_system` gained a `shared_hydro_database` branch: a labelled 6N
`HydroDatabase` (the M8 reader output) is assembled **directly** into a
coupled 6N LHS + kernel, bypassing block-diagonal stacking; body→block
mapping is **by label** (the M8 `tests/support/condensation.py`
contract), with hard raises on single-body-db, per-body/shared mixing,
missing, and unused labels. Per-block gravity restoring honours
`C_source` (the M5 hydrostatic-gravity lesson).

**Byte-identity HARD gate (M8 N=1 pattern):** the per-body path
statements are unchanged — only re-indented under the
`shared_hydro_database is None` branch. Held by the independent
hand-wired oracle `test_driver.py::test_step_C_kernel_K_matches_hand_wired`
(build_system vs hand-wired at rtol = atol = 1e-12 on `M_plus_Ainf`,
`C`, and `kernel.K`), green post-PR3; the committed
`two_body_semisub_barge.yml` parses unchanged. Coverage:
`test_m9_coupled_build.py` (16 fast validator/label-raise tests + 1
slow end-to-end permutation).

---

## S5 — Tracker dispositions (`docs/phase2-followups.md`)

- **BB-OFFSET-CONNECTOR** — **struck** (Closed 2026-07-27 by M9 PR4).
  The joint/multiplier path realizes path 2 ("Free emergence from B2")
  named in the entry's own Scope: different geometric arms per endpoint
  are just different rows of `G`; the multipliers enforce Newton-III at
  the attachment point in the inertial frame. **Tested, not asserted**
  (plan Q4): `test_m9_double_pendulum.py::test_bb_offset_penalty_raises_but_joint_path_holds`
  asserts both halves — the penalty `LinearSpring` still raises
  `NotImplementedError`; the joint path holds the same offset constraint
  at machine precision. The penalty `LinearConnector` limit itself is
  untouched — offset body-body couplings are now expressed as joints.

---

## S6 — Deviations from plan / process

- **Pre-existing red carried through M9 (M9 did NOT cause it).**
  `test_connector_attachment_transform.py::test_property_F_ref_equals_T_pullback_of_F_attach`
  fails on a hypothesis counterexample: `rel = 1.00000002e-8` vs
  `rtol = 1e-8` — one float64 ULP over threshold at K ~ 5e7, in the
  connector module M9 never touches. **Pre-existence evidence:**
  `git diff main` on both the test file and
  `floatsim/bodies/connector.py` is **empty** (byte-identical to
  `main`), so the property fails identically on `main` — it is not
  reachable by any M9 change; and the rtol was **already loosened once**,
  1e-9 → 1e-8, at commit `bbb5b9b` ("tests: loosen F_ref pullback
  identity tolerance 1e-9 -> 1e-8", whose own message documents the same
  hypothesis-FP-floor mode with counterexamples that "just barely
  exceed" the gate). This is therefore the next iteration of a known
  float64-floor issue, not an M9 regression. Tracked as
  **F2-HYPOTHESIS-TOLERANCE-EMPIRICAL**; the fix is a magnitude-scaled
  bound (explicitly NOT a third rtol bump), deferred to its own `fix-`
  branch. It is the **sole** failure in every suite scope (non-slow:
  638 passed / 1 failed; full suite: 728 passed / 1 failed — see the
  S2 "Full-suite counts" table).
- **Double-pendulum amplitude θ₀ = 0.01** (vs the single hinge's 0.02),
  and the **`I_c = m l²·1e-6` point-mass regularization** — both
  documented in S2/S3; neither perturbs the validated periods.
- **Adjacent `fix-lint-debt` commit (`22af89f`).** M9 PR3 surfaced 4
  ruff + 2 mypy errors that predate all M9 work (the earlier
  `fix-lint-debt 29c9b64` never covered the top-level `driver.py`).
  Fixed on a dedicated branch, FF-merged — kept out of the feature PRs
  per CLAUDE.md §9.
- **Deferred black-conformance debt (a recorded decision, not an
  omission).** `black --check floatsim/` (black 24.10.0) would reformat
  three files — `floatsim/bodies/connector.py`,
  `floatsim/mooring/catenary_analytic.py`,
  `floatsim/hydro/mesh_hygiene.py` — which carry manually over-wrapped
  lines black collapses. This is a separate, broader question (possibly
  a black 24.x point-version drift rather than true debt). **Decision
  (Xabier, 2026-07-27): leave it for now** — handled in its own pass,
  not folded into M9. The M9 `fix-lint-debt` commit deliberately touched
  only the ruff + mypy debt and did not reformat these files.

---

## S7 — What M9 hands forward

- **LEVEL2 quaternion wiring** (`LEVEL2-INTEGRATOR-UNWIRED`). The joint
  gates certify the constraint formulation in the small-angle regime
  (θ₀ ≤ 0.02); large-angle validity is LEVEL2's question, measured at
  **M10**, and is explicitly NOT certified by these gates (plan
  NONLINEARITY SCOPE STATEMENT).
- **KKT scaling second point.** Cost measured at n_dof = 24 (PR2);
  M10's articulated-3 supplies the second scaling point at no extra
  cost. Two points + the O((n+m)³) border behaviour decide **before
  M11 planning** whether sparse/Schur (program B6) is pulled forward.
- **The 12-buoy `yaw_locked` joint** is validated structurally (the
  4-row joint assembles and solves in the KKT path; constraint holds,
  energy conserved — `test_m9_kkt_integrator.py::test_yaw_locked_joint_holds_and_conserves`).
  The full articulated 12-body platform is M10+.
