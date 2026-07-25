# M9 pre-milestone audit — articulated joints (KKT)

**Per plan Q7 lock** ([`m9-joints-plan.md`](../m9-joints-plan.md)),
formalised before PR1 following the M8 precedent. Content is Phase-1
Measurements A + D, re-derived from the repo at drafting time
(2026-07-25), plus the ME geometric-stiffness finding and the
force-path / λ-units notes the KKT layer sits on.

**Purpose.** Establish exactly where the constraint layer inserts into
the existing integrator, so B2 is scoped as a bordered-solve extension
rather than an integrator rewrite — and so the one non-obvious physics
trap (finding ME) is on the record before code is written.

---

## 1. Solver structure (Measurement A) — the KKT insertion points

`integrate_cummins` (`floatsim/solver/newmark.py`) is a
generalized-alpha (Chung–Hulbert) integrator. Per-step structure:

| element | file:line | fact |
|---|---|---|
| `A_eff` assembly | `newmark.py:264` | `A_eff = (1-α_m)(M+A_inf) + (1-α_f)h²β C` built **once**, before the loop; constant across the run |
| per-step solve | `newmark.py:324` | one `np.linalg.solve(A_eff, rhs)` for `xi_ddot_{n+1}` — **re-factorized every step** (numpy LU, no cache; docstring `:216-219` notes the reuse opportunity is unclaimed) |
| predictor | `newmark.py:314` | `xi_pred` depends only on step-n state (known before the solve) |
| corrector | `newmark.py:326-327` | `xi_{n+1} = xi_pred + h²β a_{n+1}`; `u_{n+1} = u_n + h((1-γ)a_n + γ a_{n+1})` — **affine in the solved acceleration** |
| state-force | `newmark.py:310` | connector/mooring/drag enter as an **explicit lagged RHS** at the previous step's state |

**Consequence for B2.** The acceleration solve at `:324` is the single
insertion point. The KKT border
`[[A_eff, −Gᵀ],[G, 0]] [a; Λ] = [rhs; c]` extends exactly that solve;
the predictor/corrector updates are unchanged. Because the corrector is
algebraic in end-of-step state and `A_eff` is **already re-factorized
per step** (not cached), assembling the `(n+m)²` border each step is
**marginal cost, not a structural change** — the "assembled once"
property is the matrix, never the factorization. This is the Q2
grounding.

**State layout.** `xi/xi_dot/xi_ddot` stay `(n_dof,)`; the multipliers
`Λ` are `(m,)` **parallel storage** — solved each step, retained for
constraint-load diagnostics (gate 3, joint-load extraction), not part
of the integrated physical state.

**Force-path distinction (binds Q1/Q4).** Connector forces are
explicit and lagged (`:310`), which imposes the stability floor
`dt < 2/ω_max` (`connector.py:26,521`) and — measured (MC) — makes a
stiff undamped connector *pump energy* (+30 %/cycle). Constraint forces
take the **new implicit path** (in the border, solved at the current
step): no lag, no floor, no energy pumping. This is why the program
rejected penalty joints and why BB-OFFSET closes inside the DAE
(Q4) — the constraint force is *solved*, delivering consistent moments
to both bodies through `G`'s arm blocks, where the penalty connector's
reference-point Newton-III (`connector.py:172`) could not.

**Timing baseline (MA-timing).** `n_dof=24`: **10.49 ms/step**
(convolution-dominated, `N_K=20001`); isolated `A_eff` solve **0.0059
ms**. The KKT border is free against the convolution at Phase-1 scales;
the O(n³) scaling caveat is the risk register's business (measured at
PR2/M10).

---

## 2. Finding ME — geometric stiffness lives in G's configuration dependence

Derived while checking that the locked formulation reproduces the gate
reference (plan MB). **With strictly first-order rotation kinematics**
(`φ = x + θ×r`, hence constant `G = [I, −r̃]`) and gravity applied at
the CoM, the hinged body has **no restoring** — a pinned body that
never swings. The compound pendulum's stiffness is **geometric**: the
product of the constraint reaction (≈ mg) with the first-order rotation
of the moment arm, which lives entirely in **`G`'s dependence on
configuration**:

- 2-D reduction: `G = [[1, 0, d cosθ], [0, 1, −d sinθ]]`; the `−d sinθ ≈
  −dθ` entry times `Λ_z ≈ mg` yields the `−mgdθ` restoring torque.
- Eliminating `Λ` analytically recovers `(I_c + md²)θ̈ = −mgd sinθ`
  **exactly**; numerically (reduced DAE, θ₀=0.02) the period is
  `1.637991 s` vs the reference `1.637947 s` — the residual **2.71e-5**
  *is* the finite-amplitude term θ₀²/16, not an error.

**Binding consequence (PR1/PR2).** `G` must be evaluated at the current
configuration each step with the **rotated arm `R(θ)r`**, and the
position residual `φ` must carry second-order Rodrigues
`R(θ) = I + θ̃ + ½θ̃²`. A constant-`G` implementation produces smooth,
plausible, **dead-wrong** motion. The hinge gate — derived independently
before implementation (plan MB) — is exactly what catches it. This is
the plausible-wrong-dynamics risk, realized and disarmed at planning
time; it is the reason the gate is non-negotiable and the reference is
closed-form.

---

## 3. λ-units (cross-reference)

Full derivation in the plan (§ "λ-units derivation"). Summary for the
audit trail: the multiplier enters the balance wholly implicitly on the
first bordered row; `Gᵀ` has dimensionless translational blocks, so
**`Λ` is the physical constraint force (N / N·m), dt-free**, and at
statics equals it **exactly** (α-weights sum to one). The O(h) blended
interpretation in motion is documented and the λ-history refinement
rejected for M9 (invisible to every gate; statics is what joint-load
extraction needs, and it is exact). Pinned by the gate-3 static
recovery test (`Λ` must recover `mg`).

---

## 4. Q5 inheritance (Measurement D) — the coupled build_system census

| component | current | M9 change |
|---|---|---|
| `build_system` | per-body: `bem_databases: dict[str, HydroDatabase]`, `_per_body_lhs` + `compute_retardation_kernel` per body, then `assemble_global_lhs`/`assemble_global_kernel` block-diagonal (`driver.py:454-463`) | add a coupled branch: when a shared N-body database is declared, build the `6N×6N` LHS+kernel directly from it (M8 reader + M8 6N-generic kernel), bypassing block-diagonal stacking |
| `Body.hydro_database` | per-body `HydroDatabaseRef` (`deck.py:149`) | add `hydro_body_label: str \| None`; deck-level shared-database declaration |
| joints | none — `Connection = LinearSpring \| Catenary \| RigidLink` (`deck.py:215`) | new `Deck.joints` discriminated union (hinge, yaw_locked) |
| label→block map | — | by label (M8 contract, `tests/support/condensation.py` reference); hard raise on mismatch/missing/duplicate |

**Back-compat.** The coupled branch is taken **only** when a shared
database is declared; the no-labels per-body path is untouched code —
the M8 `N=1`-with-labels pattern applied at `build_system` level. The
PR3 byte-identity hard gate (sha256 of assembled `M_plus_Ainf`/`C`/`K`
on every committed deck, pre vs post) is the confirmation.

---

## 5. Standing finding carried from M8 — inherited lint debt

The M8 closure S6 deferred `fix-`-branch items (4 pre-existing
`mypy --strict` reader errors were **fixed** in `fix-lint-debt-post-m8`
`ed04cf1`; `studies/` ruff gated in `29c9b64`). No open lint debt
remains at M9 start — recorded here so the M9 audit trail is complete.
