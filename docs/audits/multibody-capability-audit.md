# Multi-body capability audit

**Date:** 2026-05-11
**Scope:** Inventory of FloatSim's current multi-body capability against
two target scenarios — (A) 4 floating bodies connected by structural
members and (B) 12 floating bodies connected by selective-DOF joints.
**Status:** Audit only. No code changes. Authoritative source on file
paths cited in each section.

The reference for the original M4 design intent is
[`docs/milestone-4-plan.md`](../milestone-4-plan.md) and ARCHITECTURE.md
§2.2 / §4. The reference for what's actually landed is the source under
`floatsim/` plus the four M4 validation tests in `tests/validation/`.

---

## 1. Constraint / connection primitives

### What exists

**`floatsim/bodies/connector.py`** is the single home for inter-body
linear couplings.

- `LinearConnector(body_a, body_b, K, B, rest_offset)` — a generalised
  6-DOF linear spring-damper between two bodies, with body-to-earth
  attachment via the sentinel `body_index = -1` (`_EARTH` constant,
  [`connector.py:53`](../../floatsim/bodies/connector.py)). `K` and `B`
  are arbitrary symmetric 6x6 matrices over the DOF order
  `(surge, sway, heave, roll, pitch, yaw)`. `rest_offset` is a length-6
  generalised displacement at which the spring is unstretched.
- `heave_rigid_link(body_a, body_b, penalty_stiffness, penalty_damping)`
  — convenience builder for the heave-axis-only penalty rigid link
  (K = diag(0, 0, k, 0, 0, 0)).
- `make_connector_state_force(connectors, n_dof)` — builds the
  `(t, xi, xi_dot) -> F[6N]` closure consumed by `integrate_cummins`.
  Validates body indices fail-fast against `n_dof // 6`.
- `check_connector_stability(lhs, connectors, dt, safety_factor)` —
  diagnostic that computes the explicit-stability bound
  `dt < safety * 2 / omega` per connector DOF, where
  `omega = sqrt(K_ii / mu_eff)` and `mu_eff` is the two-body reduced
  mass (or single-body mass for body-earth). Returns one message per
  violating mode.
- `connector_drift(xi_hist, connector)` — diagnostic returning peak
  `|xi_a - xi_b - rest_offset|` per DOF over a run.

### What does NOT exist

- **No true constraint primitives.** There is no Lagrange-multiplier
  joint (hinge, ball, sliding, prismatic, universal). Every "rigid"
  behaviour is realised via a high-stiffness penalty spring. The M4
  plan explicitly defers the DAE / Lagrange-multiplier path to Phase 2
  ([`milestone-4-plan.md` Q1](../milestone-4-plan.md)).
- **No general N-DOF rigid link.** The `RigidLink` deck schema entry
  ([`io/deck.py:191`](../../floatsim/io/deck.py)) is locked to the
  heave-only constraint at M4 PR3; the docstring says
  *"a general N-DOF rigid link is deferred to Phase 2."*
- **No selective-DOF joint helper.** Selective-DOF locking IS
  expressible at the `LinearConnector` primitive level — set the
  diagonal `K` entries to penalty for locked DOFs and 0 for free DOFs
  — but no convenience builder exists for common joint types (ball,
  hinge, etc.). The caller writes the 6x6 K matrix by hand.
- **No anchor-offset transforms.** Connectors couple body reference
  points to body reference points. The `LinearConnector` docstring is
  explicit that *"attachment offsets, rotational frame transformations,
  and nonlinear stiffness curves are deferred"*
  ([`connector.py:33-39`](../../floatsim/bodies/connector.py)). The
  deck schema accepts `anchor_a_body` / `anchor_b_body` /
  `anchor_b_global` on `LinearSpring` and `attach_a_body` /
  `attach_b_body` on `Catenary`, but no module consumes those fields
  into a physical attachment-arm transform — see §3 on composition.
- **No nonlinear spring / bumper / elastomer support.** Only linear
  K, B.

### Inventory summary

| primitive | implemented | scope |
|---|---|---|
| 6-DOF linear spring-damper | yes | body↔body or body↔earth, body-reference-point only |
| Heave-only penalty rigid link | yes | convenience over `LinearConnector` |
| General 6-DOF penalty rigid link | **partial** — caller can hand-build K, no helper | deck schema rejects it (`RigidLink.type = heave_only`) |
| Selective-DOF joint (lock subset, free subset) | **partial** — caller can hand-build K | no helper, no validation test |
| Lagrange-multiplier joint (hinge, ball, sliding) | **no** | deferred to Phase 2 per M4 plan |
| Anchor offset (attachment arm) | **no** | deck schema has fields; no module consumes them |
| Nonlinear spring | **no** | deferred |

---

## 2. BEM database and reader multi-body support

### What exists

**`floatsim/hydro/database.py`** is single-body by construction:

- `HydroDatabase` shapes are hard-coded to single body
  ([`database.py:147-156`](../../floatsim/hydro/database.py)):
  - `A`, `B` — `(6, 6, n_w)`
  - `A_inf`, `C` — `(6, 6)`
  - `RAO` — `(6, n_w, n_h)`
  - `reference_point` — `(3,)`
- All validation in `__post_init__` enforces the 6x6 / 6x6xn_w shapes.
  A multi-body BEM run cannot be stored in a single `HydroDatabase`.

**Readers** all produce single-body databases:

- `floatsim/hydro/readers/wamit.py` — module docstring at
  [`wamit.py:55-57`](../../floatsim/hydro/readers/wamit.py) is
  explicit: *"M5 PR1: single body, single draught. Multi-body output
  blocks, mean-drift `.2`, QTFs (`.8`, `.9`, `.12`), and pressure
  `.pat` files are out of scope."*
- `floatsim/hydro/readers/capytaine.py` — single-body translation from
  Capytaine's NetCDF output.
- `floatsim/hydro/readers/orcaflex_vessel_yaml.py` — module docstring
  at [`orcaflex_vessel_yaml.py:40-41`](../../floatsim/hydro/readers/orcaflex_vessel_yaml.py)
  says *"Multi-body and multi-draught extensions wait for Milestone 4."*
  M4 came and went without these landing.
- `floatsim/hydro/readers/__init__.py:53` — dispatch docstring confirms
  *"Validated single-body BEM database."*

### What does NOT exist

- **No multi-body HydroDatabase shape.** No `(6N, 6N, n_w)` carrier
  for cross-body added mass and damping. The
  [`radiation.py:18-24`](../../floatsim/hydro/radiation.py) docstring
  hints at multi-body intent ("off-block-diagonal entries populated
  when the BEM run captured hydrodynamic interaction"), but the shape
  to carry such entries does not exist anywhere in the codebase.
- **No reader for multi-body BEM output.** WAMIT, Capytaine and OrcaWave
  all support multi-body runs natively (the file formats carry
  per-body and cross-body blocks). FloatSim's readers do not parse
  them. For a 4- or 12-body close-packed configuration where
  hydrodynamic interaction matters, there is currently no ingestion
  path.

### Inventory summary

| capability | implemented | scope |
|---|---|---|
| Single-body BEM ingestion (WAMIT, Capytaine, OrcaFlex YAML) | yes | three readers; OC4 marin_semi validated end-to-end through M6 |
| Multi-body BEM ingestion | **no** | three reader files (or one new aggregator) would need a multi-body code path |
| `HydroDatabase` cross-body A_ij, B_ij storage | **no** | shape would need to expand from `(6, 6, n_w)` to `(6N_b, 6N_b, n_w)` |
| Cross-body RAO (one excitation per pair) | **no** | RAO is `(6, n_w, n_h)` — per-body |

For the 4-body structural-member case **and** the 12-body
selective-DOF-joint case, this is the dominant missing piece if the
bodies are close-packed: each body would need its own single-body BEM
and the inter-body hydrodynamic coupling would be ignored. This is
acceptable only when the bodies are far-spaced relative to a typical
wavelength (>1-2 λ_peak separation).

---

## 3. Assembly: building the global LHS

### What exists

**`floatsim/hydro/radiation.py`** assembles the single-body LHS:

- `assemble_cummins_lhs(rigid_body_mass, hdb, mass, cog_offset_from_bem_origin, gravity)`
  ([`radiation.py:95`](../../floatsim/hydro/radiation.py)) returns a
  `CumminsLHS(M_plus_Ainf, C)` carrying the 6x6 matrices.
- `CumminsLHS` validates the matrix is square `6N x 6N` for some
  `N >= 1` ([`radiation.py:87-92`](../../floatsim/hydro/radiation.py))
  and exposes `n_dof` and `n_bodies` accessors. The dataclass itself
  is dimensionally agnostic — single-body or multi-body — but the
  `assemble_cummins_lhs` factory is single-body only.

**`floatsim/solver/state.py`** does the multi-body stacking:

- `pack_state(per_body)` / `unpack_state(xi)` —
  ([`state.py:44`, `state.py:74`](../../floatsim/solver/state.py))
  pack/unpack helpers between per-body 6-vectors and the global 6N
  vector.
- `assemble_global_lhs(per_body)`
  ([`state.py:117`](../../floatsim/solver/state.py)) — stacks N
  single-body `CumminsLHS` instances **block-diagonally** into a 6N x 6N
  global LHS via `_block_diagonal`. The module docstring
  ([`state.py:18-31`](../../floatsim/solver/state.py)) is explicit:
  *"Hydrodynamic cross-coupling between bodies (off-block-diagonal
  entries from a multi-body BEM run) will be plugged in later by
  assembling the global matrices directly rather than via these
  helpers."*
- `assemble_global_kernel(per_body)`
  ([`state.py:147`](../../floatsim/solver/state.py)) — does the same
  block-diagonal stacking for the retardation kernel (`K` of shape
  `(6N, 6N, N_t)`). Requires all inputs to share `dt` and `n_lags`;
  raises otherwise.

### What does NOT exist

- **No off-block-diagonal hydrodynamic coupling.** Both
  `assemble_global_lhs` and `assemble_global_kernel` are pure
  block-diagonal stackers. Cross-body added mass and damping cannot be
  populated through this code path. There is no alternative path that
  assembles a coupled multi-body LHS directly.
- **No coupled retardation kernel transform.** `compute_retardation_kernel`
  ([`retardation.py:184`](../../floatsim/hydro/retardation.py)) takes
  a single-body `HydroDatabase` (shape `(6, 6, n_w)`) and returns
  `K` of shape `(6, 6, N_t)`. There is no multi-body variant.
- **No deck-driven assembly driver.** `floatsim/io/deck.py` validates a
  full multi-body deck (1+ `Body` entries, list of `Connection`
  entries with discriminated union over `LinearSpring | Catenary | RigidLink`),
  but no module in `floatsim/` consumes a loaded `Deck` and produces
  the global LHS + state-force closure. `grep -rn "load_deck\|from_deck"
  floatsim/` returns only the `load_deck` definition itself. The
  validation tests assemble systems by hand.

### Validated scope (M4)

| test file | bodies | coupling shape | notes |
|---|---|---|---|
| `tests/validation/test_m4_two_body_assembly.py` | N = 2 | block-diagonal (uncoupled) | each body reproduces the M2 analytical heave-decay period independently |
| `tests/validation/test_m4_rigid_link_heave.py` | N = 2 | block-diagonal + 1 heave penalty | symmetric heave mode preserves single-body T_n |
| `tests/validation/test_m4_two_body_moored.py` | N = 2 | block-diagonal + heave link + 2 catenaries on body 0 | glue test — equilibrium + 5 s dynamic, hand-wired state-force closure |
| `tests/validation/test_m4_catenary_irvine.py` | N = 0 (geometry) | n/a | the static-catenary closed-form verification, no body coupling |

`grep -n "N >= 2\|N=3\|N=4" tests/validation/test_m4_*.py` returns no
match for N≥3. **N = 2 is the largest validated configuration.**

---

## 4. Solver and time integrator

### What exists

**`floatsim/solver/equilibrium.py`** — `solve_static_equilibrium`:

- Reads `n_dof` from the supplied `CumminsLHS` and works at any size
  `6N` ([`equilibrium.py:139`](../../floatsim/solver/equilibrium.py)).
- Uses `scipy.optimize.root(method="hybr")` — finite-difference
  Jacobian. No multi-body-specific code path.
- Carries a diagonal regularisation `lambda_reg * I` added to `C` to
  keep `hybr`'s Jacobian full-rank on DOFs without hydrostatic
  restoring ([`equilibrium.py:32-46`](../../floatsim/solver/equilibrium.py)).
- Used successfully at `n_dof = 6` (M2-M6 cross-checks) and
  `n_dof = 12` (`test_m4_two_body_moored.py`). No upper-N limit
  beyond `scipy.optimize.root`'s own scaling.

**`floatsim/solver/newmark.py`** — `integrate_cummins`:

- *"The integrator is agnostic to the number of bodies: `n_dof = 6`
  for a single body (M2 path) or `n_dof = 6N` for `N`-body runs."*
  ([`newmark.py:14-18`](../../floatsim/solver/newmark.py))
- Reads `n_dof` from the supplied `CumminsLHS` and `RetardationKernel`,
  validates they agree ([`newmark.py:225-229`](../../floatsim/solver/newmark.py)).
- The Chung-Hulbert generalized-alpha update is linear-algebra-only —
  `numpy.linalg.solve(A_eff, rhs)` ([`newmark.py:324`](../../floatsim/solver/newmark.py))
  — and works at any `n_dof = 6N`.
- State-dependent forces (connectors, catenaries, Morison) treated
  **explicitly** at the previous step's state, identical treatment to
  the convolution `mu_n` ([`newmark.py:55-66`, `newmark.py:307-311`](../../floatsim/solver/newmark.py)).
  This imposes the explicit-stability floor `dt < 2 / omega_max` on
  stiff penalty modes.

### What does NOT exist

- **No quaternion / Newton-Euler kinematics in the integrator.**
  `floatsim/bodies/rigid_body.py` ships
  `quaternion_identity`, `quaternion_multiply`, `rotation_matrix`,
  `quaternion_from_euler_zyx`, `integrate_quaternion`, and
  `rigid_body_accelerations`. They are unit-tested
  (`tests/unit/test_rigid_body.py` — torque-free symmetric-top
  precession, `‖q‖` preservation, `R(q)·R(q)^T = I`). **None of them
  is called by `integrate_cummins`.** The integrator treats `xi[3:6]`
  per body slot as small-angle linear Euler. ARCHITECTURE.md §9.2
  describes a "Level 2" fidelity with body-frame BEM coefficients
  rotated to inertial per step; FloatSim runs **Level 1** in practice
  (constant body-frame `M + A_inf` and `C`). This is invisible at the
  small-angle regimes M2-M6 exercise (max pitch ~5°), but becomes a
  fidelity gap at 10°+ rotations.
- **No DAE / implicit treatment for stiff penalty modes.** Explicit
  treatment is hardcoded in the integrator step loop. M4 plan Q1
  records the intentional decision: *"Accept (don't fight) the
  `dt < 2/ω_penalty` stability floor; document it, emit a startup
  diagnostic. DAE path deferred to Phase 2."*
- **No factorisation reuse.** `np.linalg.solve(A_eff, rhs)`
  re-factorises `A_eff` every step. For `n_dof = 12` this is cheap;
  for `n_dof = 72` (12-body) the dense LU is O((6N)^3) per step ≈ 100x
  the single-body cost. Documented as a deferred opportunity
  ([`newmark.py:217-219`](../../floatsim/solver/newmark.py)).
- **No multi-body solver tests beyond N = 2.** The integrator is
  exercised at `n_dof = 12` in the three M4 tests; no validation at
  `n_dof = 18`, `24`, ..., `72`.

### Inventory summary

| component | works at N = 1 | works at N = 2 | works at N >= 3 |
|---|---|---|---|
| `solve_static_equilibrium` | yes (M2-M6) | yes (M4 PR6) | dimensionally; untested |
| `integrate_cummins` | yes (M2-M6) | yes (M4 PR6) | dimensionally; untested |
| `assemble_global_lhs` | yes | yes (M4 PR1) | dimensionally; untested |
| `assemble_global_kernel` | yes | yes (M4 PR1) | dimensionally; untested |
| `make_connector_state_force` | yes (body↔earth) | yes (body↔body) | dimensionally; untested |
| Quaternion integration wired in | **no — small-angle Euler in xi[3:6]** | **no** | **no** |
| DAE / Lagrange-multiplier path | **no — penalty only** | **no** | **no** |

---

## 5. Gaps for the two target scenarios

### Scenario A — 4 bodies + structural members (L03)

**Concept.** Four floating bodies (e.g., the four corner columns of a
semi-submersible-like structure) rigidly tied together by steel braces
that transmit relative motion in all 6 DOFs but with very high
stiffness.

**What works today, in principle:**

- `assemble_global_lhs([hdb_1, hdb_2, hdb_3, hdb_4])` produces a
  `24 x 24` block-diagonal LHS.
- Multiple `LinearConnector` instances with hand-built 6-DOF K matrices
  (penalty stiffness on all six DOFs) express each strut.
- `make_connector_state_force(connectors, n_dof=24)` composes them.
- `solve_static_equilibrium` + `integrate_cummins` should run at
  `n_dof = 24`.

**What blocks it operationally:**

1. **Stability floor.** For a steel strut with `EA / L ~ 1e9 N/m`
   axial stiffness, `omega = sqrt(K_axial / mu_eff)` with
   `mu_eff = m_a m_b / (m_a + m_b) ~ 1e7 kg` gives `omega ~ 316 rad/s`
   and `dt_stable < 6 ms`. Rotational penalty (`K_θθ ~ EA L`) gives
   even faster modes. Real-time integration becomes impractical
   without DAE / implicit treatment — a Phase 2 commitment
   ([M4 plan Q1](../milestone-4-plan.md)).
2. **No `make_morison_state_force` + `make_catenary_state_force` +
   `make_connector_state_force` composer.** The three state-force
   makers do not auto-compose. The caller writes a top-level closure
   that calls each and sums (see `_build_mooring_state_force` in
   [`test_m4_two_body_moored.py:178`](../../tests/validation/test_m4_two_body_moored.py)).
   For 4 bodies + many struts + per-body Morison this is tractable but
   has zero test coverage.
3. **No deck-driven assembly.** The deck schema accepts an N-body deck
   with `LinearSpring` connections, but no module in `floatsim/`
   reads the loaded `Deck` and produces a global LHS + state-force
   closure. Every existing multi-body run is hand-wired in a test.
4. **No N>=3 validation.** Cross-body coupling correctness, drift
   diagnostics on rotational penalty modes, equilibrium convergence
   on degenerate connection topologies (e.g., a closed-loop strut
   layout) are unverified.
5. **Anchor-offset transforms.** A real strut connects to body A at
   point `r_a_body` and to body B at point `r_b_body`, not the
   reference points. The deck schema captures `anchor_a_body` and
   `anchor_b_body` ([`io/deck.py:165-167`](../../floatsim/io/deck.py)),
   but `LinearConnector` couples reference-point displacements
   directly. The transform from attachment-point relative-motion to
   reference-point generalised force needs to be built; the M4 plan
   defers it explicitly
   ([`connector.py:33-39`](../../floatsim/bodies/connector.py)).
6. **No multi-body BEM cross-coupling.** Four close-packed columns
   would have non-negligible inter-column added mass and radiation
   damping at typical floater scales. Phase 1 cannot ingest this.
   Mitigation: separate the bodies far enough that interaction is
   negligible — usable for first-cut studies; not usable when the
   bodies are within ~1 λ of each other.

**Verdict for A.** Expressible at the primitive level for 4 bodies
spaced widely enough to ignore inter-body BEM coupling, willing to
accept a small `dt` (penalty stability floor), and willing to write
the deck-to-system glue by hand. Productionisable as a research
configuration with substantial additional engineering. No physical
fidelity check at N >= 3 exists in the test suite.

### Scenario B — 12 bodies + selective-DOF joints

**Concept.** Twelve floating bodies connected by a network of joints
that each lock some DOFs (e.g., ball joint locks 3 translations, frees
3 rotations) and free the others. Eventual goal scale.

**What works today, in principle:**

- `assemble_global_lhs([hdb_1, ..., hdb_12])` → `72 x 72` block-diagonal
  LHS.
- `LinearConnector` with `K = diag(k, k, k, 0, 0, 0)` expresses
  "lock translations, free rotations" via penalty (and analogous
  diagonal patterns for hinge, sliding, prismatic etc.).
- `integrate_cummins` accepts `n_dof = 72`.

**What blocks it operationally — all the Scenario-A gaps, plus:**

7. **Cost.** `np.linalg.solve(A_eff, rhs)` per step at `n_dof = 72`
   is ~1700x the single-body cost. Plus a `(72, 72, N_t)` retardation
   kernel evaluation per step. Without factorisation reuse and
   probably without a sparser representation of A_eff, even short
   research runs become expensive.
8. **Joint variety.** Selective-DOF locks expressible as penalty
   diagonals are limited: a hinge (1 free rotation) or a ball joint
   (3 free rotations) work as diagonal patterns, but a sliding joint
   along a non-axis direction (e.g., a 45° rail) requires either an
   off-diagonal K or — cleanly — a true constraint formulation. The
   diagonal-K-only approach is a partial selective-DOF; a full
   selective-DOF library needs rotated K matrices per joint axis or a
   Lagrange-multiplier path.
9. **Moderate rotations.** A 12-body articulated assembly is much more
   likely to see >10° relative rotations than a tightly-braced
   semi-submersible. The small-angle linear-Cummins integrator will
   lose fidelity. The quaternion / Newton-Euler infrastructure
   exists in `floatsim/bodies/rigid_body.py` but is not wired into the
   integrator — closing this gap is non-trivial (it changes the
   integrator's state representation from a flat 6N vector to per-body
   `(quaternion, position)` plus `(omega, velocity)`, with downstream
   ripples through every state-force consumer).
10. **Multi-body BEM coupling becomes more important.** Twelve
    close-packed floaters interact hydrodynamically; ignoring it is
    likely unphysical even for first-cut studies. The 3 readers and
    `HydroDatabase` shape both need to extend, AND a coupled
    retardation kernel transform needs implementation.
11. **Sparsity / structure exploitation.** A 12-body system with
    sparse joint connectivity has block-sparse `A_eff` and state-force
    Jacobian. The current dense `np.linalg.solve` and per-step
    closure loop ignore that. Without sparse handling, the engineering
    effort to scale to 12 bodies is dominated by linear-algebra cost.

**Verdict for B.** Not realistic in Phase 1 without substantial
infrastructure work in at least four areas: DAE / implicit
constraint formulation, quaternion integrator, multi-body BEM,
sparsity-aware linear algebra. The penalty-only N >= 12 path is
expressible primitively but would be slow, unphysical, and untested.

---

## 6. Summary table: gaps by component

| component | scenario A (4 bodies + struts) | scenario B (12 bodies + joints) |
|---|---|---|
| 6-DOF connector primitive | usable today | usable today |
| Selective-DOF joint primitive | usable for diagonal patterns | partial; off-axis patterns need new code |
| Lagrange-multiplier constraints | **not available** (Phase 2) | **required** for stable integration |
| Anchor-offset transforms | needed; **deferred** in M4 PR3 | needed; **deferred** |
| Single-body BEM ingestion | yes (3 readers) | yes (3 readers) |
| Multi-body BEM ingestion (cross-coupling) | needed if bodies close-packed; **not implemented** | **required**; not implemented |
| Block-diagonal LHS assembly | yes | yes |
| Off-block-diagonal LHS assembly | **not implemented** | **required**; not implemented |
| Static equilibrium solver | yes (size-agnostic) | yes (size-agnostic); scaling untested |
| Newmark integrator | yes (size-agnostic, linear) | yes (size-agnostic); scaling expensive, fidelity questionable |
| Quaternion integration wired in | not needed for small-angle | needed; **not wired** |
| Explicit penalty stability floor | acceptable with small `dt` | impractical — needs DAE or implicit |
| Deck-driven composition | **no driver exists** | **no driver exists** |
| Validation at this N | **no N >= 3 test** | **no N >= 3 test** |

---

## 7. Recommended next steps (ordering, not commitments)

These would be the natural pieces of work; sequencing is a separate
decision.

**Foundation (needed by both scenarios)**

- F1. Build a deck-driven composition driver: `floatsim.driver.build_system(deck)`
  returning `(lhs, kernel, state_force, initial_state)` ready for the
  integrator. ~300-500 LOC. Unblocks every end-to-end multi-body run.
- F2. Add an `assemble_attachment_transformed_connector` helper that
  consumes `(body_a, anchor_a_body, body_b, anchor_b_body, K, B)`
  and emits a `LinearConnector` whose reference-point K, B, and
  rest_offset encode the attachment-arm cross-couplings (small-angle
  linear).
- F3. Add a `make_catenary_state_force(catenary_specs, n_dof)` helper
  paralleling `make_connector_state_force` so catenaries compose
  uniformly with springs.
- F4. Add a structural validation test at N = 4 (block-diagonal,
  no coupling — equivalent to F1's first integration test).

**Scenario A enablers (4-body structural)**

- A1. General 6-DOF rigid-link helper (K = penalty_diag(k, k, k,
  kθ, kθ, kθ)) + deck schema entry beyond the heave-only case.
- A2. Connector drift diagnostic at all 6 DOFs (currently exists per
  DOF, but no aggregated check).
- A3. Validation case: 4 columns + 6 struts (one closed-loop topology)
  reproducing combined-rigid-body free-decay modes.

**Scenario B enablers (12-body articulated)**

- B1. Selective-DOF joint helpers (`make_hinge`, `make_ball_joint`,
  `make_prismatic_joint`) with rotated K matrices for off-axis cases.
- B2. Constraint formulation: DAE path via Lagrange multipliers
  (requires integrator surgery), or alternative implicit penalty
  treatment. Phase 2 commitment per M4 plan Q1.
- B3. Quaternion integrator: wire `integrate_quaternion` +
  `rigid_body_accelerations` into the time-domain solver. This is a
  major rework — changes the integrator's state representation and
  every state-force consumer. Likely a Phase 2 milestone in its own
  right.
- B4. Multi-body BEM ingestion: extend one reader (probably WAMIT
  first; it has the most explicit multi-body schema) to produce a
  multi-body `HydroDatabase`. Requires the database shape to expand
  to `(6N_b, 6N_b, n_w)`.
- B5. Coupled retardation kernel transform on the multi-body BEM.
- B6. Sparsity-aware linear algebra in the integrator and state-force
  assembly.

The Foundation block is the cheapest win and unblocks scenario A
research-level use. Scenario B is structurally a Phase 2 milestone.

---

*Audit close. No code changes accompany this document.*
