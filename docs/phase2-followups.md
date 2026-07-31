# Phase 2 follow-ups — tracker

**Purpose.** Single registry of deferred work that has a documented
mechanism, an identified scope, and is explicitly out of the current
milestone. Each entry carries enough context to act on cold — file
paths, audit references, and dependency notes — so items do not rot
the way the WAMIT dimensionalisation bug did (CLAUDE.md §13
Example 5; conventions doc Item 23).

This document is **append-only during a milestone**. Items are
struck through (`~~`) and dated when closed by a successor PR.
Entries are not deleted — the audit-trail value is in the record of
deferral and resolution.

**Source documents:** items are drawn from
[`docs/audits/multibody-capability-audit.md`](audits/multibody-capability-audit.md)
§7 (Scenario A / B enablers + the audit's named follow-up
LEVEL2-INTEGRATOR-UNWIRED) and from any future audit / cross-check
that surfaces a deferred item.

---

## Active entries

### LEVEL2-INTEGRATOR-UNWIRED — Quaternion integrator wiring

**Mechanism.** ARCHITECTURE.md §9.2 specifies "Level 2" fidelity
for the time-domain solver: nonlinear restoring, linear hydro
coefficients in body frame rotated to inertial per step. The
quaternion / Newton-Euler kinematics in
[`floatsim/bodies/rigid_body.py`](../floatsim/bodies/rigid_body.py)
(`quaternion_identity`, `quaternion_multiply`, `rotation_matrix`,
`quaternion_from_euler_zyx`, `integrate_quaternion`,
`rigid_body_accelerations`) are unit-tested in
`tests/unit/test_rigid_body.py` (torque-free symmetric-top
precession, `‖q‖` preservation over 10⁴ steps, `R(q)·R(q)^T = I`).
**None of them are called by `integrate_cummins`.** The
integrator treats `xi[3:6]` per body slot as small-angle linear
Euler angles. The Morison module rotates per-step for its drag
computation, but the BEM-derived `M + A_inf` and `C` are NOT
rotated per step.

**Audit reference.**
[`docs/audits/multibody-capability-audit.md`](audits/multibody-capability-audit.md)
§4 "What does NOT exist" + §7 item B3.

**Why latent / visibility.** Invisible at the small-angle regimes
M2-M6 exercised (max pitch ~5°). Becomes a real fidelity gap at
10°+ rotations. A future scenario with moderate-to-large rotations
(survival sea state, mooring failure transient, 12-body articulated
assembly per Scenario B) would surface the gap as a quantitative
disagreement with reference tools that DO carry full kinematics
(OpenFAST's ElastoDyn with `PtfmYDOF = True`, OrcaFlex's
6-DOF time-domain mode).

**Scope.** A Phase 2 milestone in its own right. Wiring it requires:
- Changing the integrator's per-body state representation from a
  flat 6-vector `(translation, small-angle-Euler)` to a 7-vector
  `(translation, quaternion)` + 6-vector `(velocity, angular_velocity)`.
- Updating every state-force consumer that reads `xi[3:6]` as Euler
  angles (`floatsim/hydro/morison.py` already does the
  Euler→quaternion translation per-step; this rework removes the
  per-step translation step but ripples through every other
  consumer that assumes the flat-6N convention).
- Updating `assemble_global_lhs` / `assemble_cummins_lhs` to
  separate the BEM coefficients (body-frame, constant) from the
  per-step rotation to inertial.
- Validation: a moderate-rotation scenario where the Level-1 vs
  Level-2 disagreement is measurable and the Level-2 answer
  matches a reference tool.

**Estimated effort.** ~3-6 weeks of focused work. Comparable in
scope to M4 + part of M6.

**Blocks.** Scenario B (12-body articulated). Any moderate-rotation
research case.

**Status.** Open. Not currently scheduled. Audit-surfaced
2026-05-11.

---

### A1 — General 6-DOF rigid-link helper

**Mechanism.** Only the heave-axis penalty rigid link
(`heave_rigid_link`) has a convenience helper in
[`floatsim/bodies/connector.py`](../floatsim/bodies/connector.py).
The general N-DOF rigid link (penalty diagonals on a chosen subset of
DOFs, or all 6) is expressible at the `LinearConnector` level (the
caller hand-builds K) but has no helper. The deck schema's
`RigidLink` connection type is also locked to heave-only at M4 PR3
([`io/deck.py:191`](../floatsim/io/deck.py)).

**Audit reference.** Audit §7 item A1.

**Scope.** Two pieces:
1. `general_rigid_link(body_a, body_b, locked_dofs, penalty_stiffness, penalty_damping)`
   helper in `floatsim/bodies/connector.py`. Returns a
   `LinearConnector` with `K = diag` over the locked DOFs.
2. Deck schema extension: `RigidLink.locked_dofs` field (list of
   DOF names) with backward-compatibility default (`["heave"]`).
3. Drift diagnostic update: the existing `connector_drift` works
   per DOF; the report should aggregate to the locked-DOF subset.

**Estimated effort.** ~1 week. Small surface, well-bounded.

**Blocks.** Scenario A (4-body structural-member systems). Any
deck where bodies are coupled in DOFs other than heave.

**Status.** Open. Audit-surfaced 2026-05-11. Candidate for promotion
to a successor milestone after M7-Foundation closes.

---

### A2 — Connector drift diagnostic aggregation

**Mechanism.** `connector_drift(xi_hist, connector)` returns peak
absolute drift per DOF over a run
([`connector.py:212`](../floatsim/bodies/connector.py)). For a
multi-connector system there is no aggregator. The M4 plan Q1
specified a drift diagnostic with `> 0.1 %` warning at startup;
that's a per-connector / per-DOF check, but the multi-connector
aggregate ("worst drift across all locked DOFs of all connectors")
is not implemented.

**Audit reference.** Audit §7 item A2.

**Scope.** Small addition to
`floatsim/bodies/connector.py` returning a summary dataclass
listing worst-case drift per connector + per DOF + per body, with a
configurable warning threshold.

**Estimated effort.** ~2 days.

**Blocks.** Production-confidence runs at Scenario A scale.

**Status.** Open. Audit-surfaced 2026-05-11.

---

### A3 — N = 4 structural-member (connector-coupled) validation

**Mechanism.** F4 in M7-Foundation validates N = 4 in the
**block-diagonal** case. Connector-coupled N >= 3 (the structural-
member case — e.g., 4 columns + 6 braces) has no validation test.
At least one validation case is needed to confirm the integrator,
state-force composition, and stability gate all work correctly with
non-trivial coupling at N >= 3.

**Audit reference.** Audit §7 item A3.

**Scope.** A new validation test ~300 lines. Realistic candidate:
4 identical M2 bodies arranged in a square, connected by 4 edge
braces with `heave_rigid_link` penalty. Predicted symmetric mode
preserves single-body T_n (analytical eigenvalue problem
solvable). Drift diagnostic + explicit-stability gate must pass.

**Estimated effort.** ~1-2 weeks. Requires A1 (general rigid link
for non-heave struts) if the validation case uses anything beyond
heave coupling.

**Blocks.** Confidence in Scenario A use cases (L03-style).

**Status.** Open. Audit-surfaced 2026-05-11. Sequencing decision
deferred until M7-Foundation F4 reports.

---

### B1 — Selective-DOF joint helpers (hinge, ball, prismatic)

**Mechanism.** Selective-DOF locks expressible as diagonal K
patterns are limited: hinge (1 free rotation) and ball joint (3 free
rotations) work as diagonal K with zeros on the free DOFs. Sliding
joints along non-axis directions need a rotated K matrix; cleanly
modelled, they need either off-diagonal K or a true constraint
formulation. The diagonal-K-only approach is a partial selective-DOF;
a full library needs rotated K matrices per joint axis or a
Lagrange-multiplier path.

**Audit reference.** Audit §7 item B1.

**Scope.** Helpers parallel to `heave_rigid_link`:
- `hinge_joint(body_a, body_b, hinge_axis_body, ...)` — locks
  3 translations + 2 rotations, frees 1 rotation about the hinge
  axis.
- `ball_joint(body_a, body_b, attach_a_body, attach_b_body, ...)`
  — locks 3 translations at the attachment point, frees all 3
  rotations.
- `prismatic_joint(body_a, body_b, axis_body, ...)` — frees 1
  translation along the axis, locks the other 5 DOFs.

Each helper builds a 6x6 K with rotated penalty entries. Plus
deck schema entries (discriminated union over joint types).

**Estimated effort.** ~2-3 weeks for the three helpers, deck schema,
unit tests, and one integration validation case.

**Blocks.** Scenario B (12-body articulated). Specialised research
cases requiring articulated body trains.

**Status.** Open. Audit-surfaced 2026-05-11.

---

### B2 — Lagrange-multiplier DAE constraint formulation

**Mechanism.** Stiff penalty springs impose an explicit-stability
floor `dt < 2 / omega_max`. For steel-stiffness penalties at typical
floater scales, `omega_max` can be ~300-1000 rad/s, giving `dt < 2-6
ms`. Real-time integration of large multi-body systems with stiff
joints requires either implicit treatment of the penalty force or a
true constraint formulation via Lagrange multipliers. The M4 plan
Q1 records the intentional deferral: *"Accept (don't fight) the
`dt < 2/ω_penalty` stability floor; document it, emit a startup
diagnostic. DAE path deferred to Phase 2."*

**Audit reference.** Audit §7 item B2 + [`docs/milestone-4-plan.md`](milestone-4-plan.md)
Q1.

**Scope.** Major integrator surgery:
- Augment the state vector with Lagrange multipliers per
  constrained DOF.
- Index-2 or index-3 DAE formulation (typically index-3 for rigid
  constraints with stabilisation per Baumgarte or projection).
- Replace `np.linalg.solve(A_eff, rhs)` with a saddle-point solve
  of the augmented KKT system.
- Constraint-violation diagnostics (analog of `connector_drift`
  but with multiplier-history checks).

**Estimated effort.** ~6-8 weeks of focused work, with substantial
literature review. Comparable in scope to M2 (Cummins time-domain
solver build-up).

**Blocks.** Scenario B (12-body articulated). Real-time multi-body
research at scale.

**Status.** Open. Phase 2 commitment per M4 plan. Audit-surfaced
2026-05-11 (re-confirmation).

---

### B3 — see LEVEL2-INTEGRATOR-UNWIRED above

Cross-reference. The "wire the quaternion integrator into
`integrate_cummins`" entry is the same item. Kept named at
LEVEL2-INTEGRATOR-UNWIRED for the audit-trail discoverability
(matches the user's tracking-language choice on 2026-05-11).

---

### ~~B4~~ — Multi-body BEM cross-coupling ingestion

**Closed 2026-07-21 by M8 PR1 (`a1399b2`) + PR2 (`14a447e`).**
`HydroDatabase` gained `body_labels: tuple[str, ...] | None` (None =
legacy 6-DOF branch, pre-M8 code verbatim; labels = `6N` shapes;
`n_bodies` from labels, never shape arithmetic). `read_capytaine`
gained the multi-body path keyed on the number of DISTINCT body-label
prefixes (`buoyK__DOF`), producing a labelled `6N x 6N` database with
full cross-coupling blocks, radiation + excitation; the lags→leads
conjugation is shared with the single-body path (one copy — cannot
diverge). Exercised end-to-end by the 18-DOF cluster fixture
(`studies/cluster-3buoy-rigid/capytaine_multibody_18dof.nc`,
production grid) and the M8 PR4 condensation gates
(`tests/validation/test_m8_condensation_gates.py`). See
`docs/m8-coupled-bem-closure.md`.

---

### ~~B4~~ (original entry, retained for audit trail)

**Mechanism.** `HydroDatabase` shapes are hard-coded single-body
([`database.py:147-156`](../floatsim/hydro/database.py)). All three
readers (WAMIT, Capytaine, OrcaFlex YAML) produce single-body
output. For close-packed multi-body floaters (e.g., the four
columns of an OC4-like semi at typical column spacing of ~50 m
versus a peak wavelength of ~150 m), inter-body added mass and
radiation damping become non-negligible. The magnitude is
deck-dependent (rough estimate: 5-20% of single-body diagonal
values at typical floater column spacings; not sourced, expect a
WAMIT/OrcaWave cross-coupled run on the actual L03 geometry before
treating the magnitude as anything other than a qualitative
expectation). Phase 1 cannot represent this regardless of the
magnitude.

**Audit reference.** Audit §2 + §7 item B4.

**Scope.**
- Extend `HydroDatabase` shape from `(6, 6, n_w)` to
  `(6N_b, 6N_b, n_w)` for `A`, `B`; from `(6, 6)` to `(6N_b, 6N_b)`
  for `A_inf`, `C`; from `(6, n_w, n_h)` to `(6N_b, n_w, n_h)` for
  `RAO`. Per-body `reference_point` array.
- Extend at least one reader to parse multi-body BEM output.
  WAMIT first (the most explicit multi-body schema). The WAMIT
  format ships per-body and cross-body blocks via the `.4` motion
  RAO file convention + `.frc` body-mass file; the existing
  `.1` / `.3` / `.hst` readers need shape promotion.
- Validation: a 2-body close-packed WAMIT fixture (e.g., the
  IEA Wind 15 MW UMaine VolturnUS-S 3-column case) cross-checking
  per-body diagonal blocks against the existing single-body
  reader on the same geometry-isolated case.

**Estimated effort.** ~3-4 weeks for the WAMIT reader path,
shape extension, and one validation case. Comparable in scope
to M5 PR1.

**Blocks.** **B4 is the gate for L03 (Scenario A) being a
meaningful validation at all.** Without B4, an L03 cross-check
against OrcaFlex would compare FloatSim's
uncoupled-block-diagonal-BEM model to OrcaFlex's
coupled-multi-body-BEM model — any disagreement is **missing
physics, not a bug**, and the cross-check signal is contaminated
by the absent off-diagonal terms. L03 should not be attempted as a
validation case until B4 lands. (Pure exploratory runs at L03
geometry with block-diagonal BEM are useful for solver / driver
shakedown but cannot be reported as a validation result.)
Also blocks Scenario B (12-body articulated) where bodies are
physically near each other.

**Status.** Open. Audit-surfaced 2026-05-11.

---

### ~~B5~~ — Coupled retardation kernel transform on multi-body BEM

**Closed 2026-07-21 by M8 PR3 (`2d59907`).**
`compute_retardation_kernel` generalized to the full `6N x 6N`
(four 6-hardcodes → `n_dof = B.shape[0]`; the Filon quadrature was
already size-generic), with the per-entry `i == j` gate branch and the
NEW multi-body PSD gate on `B(omega)` (plan Q3 iii) — which caught a
real whole-matrix contaminated frequency slice on first contact with a
production-grid fixture (tracker
BEM-CONTAMINATED-FREQUENCY-SLICE-CLUSTER-DRAFT, still OPEN — an M11
blocker). Positive gate on the contaminated-omega-excluded grid
(Check 3 worst 0.025 % at t_max = 60 s); permanent negative gate
asserts PSD fires on the unmodified fixture. See
`docs/m8-coupled-bem-closure.md` §S3.3–S4.

---

### ~~B5~~ (original entry, retained for audit trail)

**Mechanism.** `compute_retardation_kernel` takes a single-body
`HydroDatabase` (shape `(6, 6, n_w)`) and returns `K` of shape
`(6, 6, N_t)`. There is no multi-body variant. Required as soon as
B4 multi-body BEM input is available.

**Audit reference.** Audit §7 item B5.

**Scope.** Extend `compute_retardation_kernel` to accept a
multi-body `HydroDatabase` (shape `(6N_b, 6N_b, n_w)`) and return
`K` of shape `(6N_b, 6N_b, N_t)`. The Filon cosine transform code
([`floatsim/hydro/_filon.py`](../floatsim/hydro/_filon.py)) is
already shape-agnostic in its block dimension; the wrapper needs
shape adjustment + the three-check kernel gate (Item 25) needs to
be run per-block-pair (diagonal + cross). Cross-block decay must be
audited.

**Estimated effort.** ~2 weeks. Depends on B4 landing.

**Blocks.** Same as B4.

**Status.** Open. Audit-surfaced 2026-05-11. Sequencing: B4 → B5.

---

### B6 — Sparsity-aware linear algebra

**Mechanism.** `np.linalg.solve(A_eff, rhs)` in the integrator step
loop re-factorises `A_eff` every step. At `n_dof = 72` (12-body) the
dense LU is O((6N)³) per step ≈ 1700x the single-body cost. Documented
as a deferred opportunity at
[`newmark.py:217-219`](../floatsim/solver/newmark.py). For
block-sparse `A_eff` (no inter-body BEM coupling + few connector
non-zeros), a block-sparse representation + factor reuse would
recover most of the cost.

**Audit reference.** Audit §7 item B6.

**Scope.**
- Profile the integrator step at `n_dof = 24, 48, 72` to confirm
  `np.linalg.solve` is the bottleneck (likely yes, but verify).
- Replace dense `solve` with `scipy.sparse.linalg.splu` (factor
  once, solve many) or with a manually-coded block factorisation
  exploiting the connector topology.
- Validation: identical results at `rtol = 1e-12` on the M2 / M4 /
  M7-Foundation test suite. Performance metric: 10x speedup on
  the largest case targets a 12-body scenario; smaller is fine for
  a Scenario A case.

**Estimated effort.** ~2-3 weeks. Performance gain is the
deliverable; correctness is the gate.

**Blocks.** Operational use of Scenario B (12-body). Improves but
does not block Scenario A.

**Empirical baseline (M7-Foundation PR1, commit `b34295e`).** F4
at `n_dof = 24` measured ~21 ms/step with the current dense
`np.linalg.solve(A_eff, rhs)` per-step factorisation
(`210.77 s / 10,000 steps`). Linear extrapolation to `n_dof = 72`
projects ~600+ ms/step, ~100 min for a comparable 10k-step run.
(Cubic in `n_dof` for the LU dominates; the einsum convolution
sum scales `O(n_dof^2 * N_t)` and adds to the same direction.)
Confirms **B6 is required, not optional, at the 12-buoy scale**.
Speedup target: 10× at `n_dof = 72` would bring 12-buoy runs into
the 10-minute range, comparable to current single-body
production work. This data point turns the "estimated 1700×" in
the audit reference above into a measured-projects-to baseline.

**Status.** Open. Audit-surfaced 2026-05-11. Baseline numbers
updated post-F4 (M7-Foundation PR1, 2026-05-11).

---

### ~~BB-OFFSET-CONNECTOR~~ — Body-body LinearConnector with non-zero attachment offset

**Closed 2026-07-27 by M9 PR4 (velocity-level KKT joints).** The joint
(Lagrange-multiplier) path is exactly **path 2 ("Free emergence from
B2")** named in this entry's own Scope below: a body-body constraint
with different geometric arms on each side simply contributes different
rows to the constraint Jacobian `G` (`[I, -(R r)~]` per endpoint), and
the multipliers enforce Newton-III at the attachment point in the
inertial frame — where it actually holds — not at the reference points.
No per-endpoint `K` factors, no `LinearConnector` surgery. Closure is
**tested, not asserted** (plan Q4): the double-pendulum gate's
inter-body hinge attaches at an offset from body 1's CoG (this entry's
failure topology), and `tests/validation/test_m9_double_pendulum.py::
test_bb_offset_penalty_raises_but_joint_path_holds` asserts both halves
in one place — the penalty `LinearSpring` path still raises
`NotImplementedError`, while the joint path holds the same offset
constraint at machine precision (drift < 1e-10) and returns the
physical multiplier. The penalty `LinearConnector` limit itself
(`connector.py:172`, `driver.py` `_materialise_linear_spring`) is
untouched by M9; body-body offset couplings are now expressed as joints
rather than penalty springs. See `docs/m9-joints-closure.md`.

---

### ~~BB-OFFSET-CONNECTOR~~ (original entry, retained for audit trail)

**Mechanism.** `LinearConnector`
([`floatsim/bodies/connector.py`](../floatsim/bodies/connector.py))
assumes symmetric Newton-III at reference points (`F_b = -F_a`
exactly). When one body has a non-zero attachment arm, the
moment-arm cross product gives `F_a_ref` a moment block that
`F_b_ref` (at its reference, no arm) lacks — the pair is
asymmetric. Body-body connections with any non-zero offset cannot
be represented in the current framework without per-endpoint K
factors.

**Audit reference.** Surfaced during M7-Foundation PR2
(commit `54703b7`) at the derivation of F2's attachment-offset
transform; see PR2 commit message + the diagnostic doc at
[`docs/diagnostics/m7-pr2-framework-limit.md`](diagnostics/m7-pr2-framework-limit.md).
F2's locked scope (body-earth single offset) and all existing
fixtures (M4 PR3 heave-rigid-link, M6 PR5 OC4 mooring) live in
the subset where this constraint does not bite — which is why
M6 did not surface it.

**Why latent.** Invisible at the regimes M2-M6 exercised
(everything either body-earth or body-body-at-reference-points).
Becomes a real limit the moment a body-body offset connection
is needed: fenders, hawsers, fairlead-to-fairlead lines, and
articulated kinematic links beyond the simplest cases all
require it.

**Scope.** Two paths:

1. **Direct.** Extend `LinearConnector` to carry per-endpoint
   K factors (and B / rest_offset), and modify
   `make_connector_state_force` to apply them asymmetrically.
   The 6x6 K becomes two 6x6 matrices `K_aa = T_a^T @ K @ T_a`
   and `K_bb = T_b^T @ K @ T_b` with cross-coupling
   `K_ab = T_a^T @ K @ T_b` (also Newton-III consistent at the
   attachment-point level but not at the reference-point level).
   ~1-2 weeks of framework-level surgery, ripples through
   `connector_drift` and any other code that reads the existing
   6x6 K shape.
2. **Free emergence from B2.** The Lagrange-multiplier DAE
   formulation handles the asymmetry naturally via the
   constraint Jacobian — different geometric arms on each side
   simply contribute different rows to the constraint Jacobian,
   and the multipliers ensure Newton-III at the attachment in
   the inertial frame (where it actually holds), not at the
   reference points.

The B2 path is cleaner **if B2 is going to happen anyway**.
The Direct path is the pragmatic choice if a real fixture
demands body-body offset before B2 is scheduled.

**Estimated effort.** Direct: ~1-2 weeks. Free-from-B2: zero
incremental beyond B2.

**Blocks.** General body-body offset connections; the deck
schema's `LinearSpring` full expressivity (the schema currently
accepts both `attach_a_body` and `attach_b_body` but the
framework can't represent both non-zero). At M7-Foundation PR4
(F1), `build_system` will raise `NotImplementedError` on
body-body `LinearSpring` entries with any non-zero offset,
citing this tracker entry — see
[`docs/m7-foundation-plan.md`](m7-foundation-plan.md) Q9 for
the pinned PR4 disposition.

**Status.** Open. Surfaced 2026-06-01 (M7-Foundation PR2).
Sequencing: pre-empt with the Direct path if a fixture demands
body-body offset before B2 is scheduled; otherwise emerges
free from B2.

---

### F2-HYPOTHESIS-TOLERANCE-EMPIRICAL — F2 cached-K_ref precision bound is empirical, not analytical

**Mechanism.** F2's `assemble_attachment_transformed_connector`
([`floatsim/bodies/connector.py`](../floatsim/bodies/connector.py))
builds a cached `K_ref = T^T @ K_attach @ T` at construction
time (one 6x6 matrix, applied once per integrator step per
body). This is an intentional caching optimisation for the
integrator hot path. Its precision at very small attachment
offset ``r`` is bounded by a cancellation-like effect: at
`|r| ~ 1e-8 m`, the `r`-dependent corrections
(`r_tilde @ K_attach @ (-r_tilde)`, `r_tilde @ K_attach_upper`,
`K_attach_lower @ (-r_tilde)`) are all of order `1e-8 * ||K||`,
which is 8 orders below the O(1) diagonal entries they fold
into. Storing this sum as a float64 matrix loses the tiny
corrections at their relative precision level. The equivalent
on-the-fly path (`T @ xi` first, then `K_attach @ result`,
then `T^T @ result`) applies the same corrections as small
perturbations to O(1)-magnitude vectors throughout the chain,
preserving them. Same algebra, two different numerical
trajectories; the cached-K_ref path is genuinely less precise
in the very-small-r regime.

**Audit reference.** Surfaced during the M7.5 pre-milestone
audit baseline pytest run at main head `dd7dc9f` (see
[`docs/audits/m7.5-reader-audit.md`](audits/m7.5-reader-audit.md)
§Item 5). The failing hypothesis property test was
`tests/unit/test_connector_attachment_transform.py::test_property_F_ref_equals_T_pullback_of_F_attach`
(M7-Foundation PR2 code, commit `0d5f82f`), whose docstring
explicitly anticipated this class of empirical-tolerance
failure ("rtol = 1e-9 (looser than the identity-test 1e-12 to
accommodate hypothesis-induced floating-point noise in
K @ T @ ... compositions)").

**Why latent / visibility.** Hypothesis property tests explore
input regimes that fixed-input tests never exercise. The
failing shrunk example (`r = [1e-8, 1e-8, 1e-8]`, small pure
rotation `xi[3:6] = [3.125e-2]*3`, triangular SPD K with
entries 1..6) produced a 1.87e-9 absolute discrepancy against
a `rtol=atol=1e-9` gate — just barely exceeding. The original
M7-Foundation PR2 tolerance choice at `0d5f82f` was itself
empirical (docstring-documented); the 1e-9 gate passed all
seeds Hypothesis explored during PR2 landing. The M7.5 audit's
first pytest run explored different seeds and surfaced the
edge case.

**Regime — the failing regime is unphysical.** The 1.87e-9
discrepancy occurs at `r = 1e-8 m = 10 nm`, five or more
orders of magnitude below any physical body attachment offset
(real Phase-1 fixtures use offsets in the `1e-3` to `1e0` m
range). At `r = 1e-3` the discrepancy shrinks proportionally
to `~1e-19`, well below any tolerance the test suite gates
against. F2's precision at physical `r` is unaffected.

**Current disposition.** Empirical tolerance loosened from
`rtol=atol=1e-9` to `rtol=atol=1e-8` at commit `bbb5b9b`
(2026-07-01), giving one decade of margin over the observed
shrunk example. Test passes; audit's baseline-all-green claim
holds. This is a REFINEMENT of the M7-Foundation PR2
empirical-tolerance choice, not a fix for a bug.

**Scope — deferred analytical work.**

1. **Derive the actual precision bound analytically.** The
   observed discrepancy is `~5 orders larger` than naive
   per-matmul FP-roundoff bounds predict for these matrix
   sizes and value scales. The excess comes from
   cancellation-like effects in K_ref construction at small
   `r`. An analytical bound of the form
   `O(cancellation_factor · eps · ||K|| · ||r|| · ||xi||)`
   would let us set the tolerance from theory rather than
   from one shrunk hypothesis example. A different
   hypothesis seed or `max_examples=1000` could surface a
   worse case (e.g., `5e-8` or `1e-7`) that requires
   loosening again.
2. **Investigate cache-skip for very small `r`.** F2 could
   fall back to the on-the-fly path (which preserves
   precision) when `||r|| < r_threshold`, and only cache
   when caching is precision-safe. Trade-off: the
   integrator hot path gains a branch per body per step.
   Determine whether the branch cost is acceptable in
   the current single-body-focused regime (Phase 2 might
   need it for the 12-body case).
3. **Consider hypothesis-test regime constraints.** The
   `_bounded_arm` strategy at
   [`test_connector_attachment_transform.py`](../tests/unit/test_connector_attachment_transform.py)
   `:_bounded_arm` allows `r` in `[-5, +5]` m without a
   lower bound. Adding `min_value = 1e-6` or similar would
   restrict hypothesis to physical regimes and prevent
   future edge cases from surfacing spurious failures.
   Alternatively, the property test could ASSERT the
   bound analytically ("if `||r|| < 1e-6`, expected
   precision is ...").

**Estimated effort.** (1) `~1 week` (analytical derivation
+ verification against expanded hypothesis budget). (2)
`~2-3 days` (branch + integrator perf measurement +
regression test). (3) `~1 day` (test-strategy tweak). Not
sequenced; any of the three could be done independently.

**Blocks.** Nothing immediately. Matters for future work
where F2's tolerance would need to be tightened
(e.g., higher-precision cross-checks against known-stable
references), where high-precision regimes are exercised
(multi-body rigid-link chains at very tight offsets), or
where the empirical-tolerance chain gets stress-tested by
a hypothesis-strategy change.

**M9 PR2 addendum (2026-07-26) — SECOND tightening event, now
with a reproducing counterexample.** The M9 PR2 full-suite run
surfaced a failing hypothesis example on the SAME assertion
(`test_property_F_ref_equals_T_pullback_of_F_attach`):

- **Counterexample:** `K_attach ~ 5e7` (a stiff penalty
  spring), values of magnitude `~1e8`; `F_ref = Tᵀ F_attach`
  vs the hand computation differ by **rel 1.00000002e-8**
  against `rtol = 1e-8` — the **float64 relative floor at
  magnitude ~1e8, touched exactly** (a chain of ops on ~1e8
  values carries ~1e-8 relative rounding).
- **Why it appeared now, not at PR1 / M8-close:** the
  full-suite collection order differs from the unit tier,
  shifting hypothesis's RNG; it explored a stiffer example
  this run. The counterexample is now cached in
  `.hypothesis/examples/` and replays deterministically.
- **Pre-existence verified:** `git diff main` on both
  `connector.py` and the test is **empty**; the failure
  reproduces on `main`; M9 touched neither. M9 caused zero
  regressions (its +21 full-suite delta is exactly the new
  PR1+PR2 tests).
- **This is the SECOND tightening event** on this assertion
  (`bbb5b9b` was the first, 1e-9 → 1e-8). **A third bare rtol
  bump (→ 1e-7) is explicitly the wrong move** (CLAUDE.md §9):
  it is another turn of a ratchet that never resolves, and
  `bbb5b9b` is the recorded precedent for why this entry exists.
- **Sharpened deferred work:** the fix is a **magnitude-scaled
  bound** — `atol` scaled by `||K_attach||` (or the transform's
  condition number `||T||·||Tᵀ||`), giving an analytically
  defensible floor — NOT another decade on `rtol`. Option (1)
  above ("derive the real bound") is now the priority; the
  reproducing counterexample makes it directly verifiable.

**Status.** Open. Surfaced 2026-07-01 (M7.5 pre-milestone
audit). Empirical tolerance in place at `bbb5b9b`;
analytical bound derivation and cache-skip investigation
deferred. Not blocking M7.5. **Re-surfaced with a cached
reproducing counterexample at M9 PR2 (2026-07-26); the M9
branch carries this one pre-existing red, tracked here.**

---

### ~~BEM-INPUT-NORMAL-VALIDATION~~ — BEM mesh panel-normal orientation validation

**Closed 2026-07-03 by M7.5 PR3 (`6d457c2`).**
`floatsim.hydro.mesh_hygiene` delivers per-panel ray-parity
validation + auto-fix + tier-2 hydrostatic-volume physics
screen; terminal-gate test 1
(`tests/validation/test_m7_5_terminal_gate.py::test_mesh_chain_reversed_normals_detected_and_fixed`)
exercises the workflow end-to-end on the study's ORIGINAL GDF
(216 inward → 0 inward after auto-fix; 96 open edges warned
in-suite). The FloatSim-level reader-hygiene resolution (path 2
below) is delivered as a standalone utility rather than wired
into the readers per plan §Q2 — users invoke
`validate_panel_normals` / `fix_panel_normals` in their study
pre-BEM-solve scripts.

---

### ~~BEM-INPUT-NORMAL-VALIDATION~~ (original entry, retained for audit trail)

**Mechanism.** BEM-import paths from external mesh sources
(GDF, STL, NEMOH, etc.) can carry reversed panel normals. BEM
solvers do not crash on incorrectly-oriented normals; they
silently produce wrong added-mass and damping results. In the
worst observed case
([`studies/spar-fin-decay`](../studies/spar-fin-decay/), mesh
`test2_spar_fin.gdf` at commit `064d630`), a horizontal heave
plate annulus with both faces oriented inward produced
`A_inf(heave) = 1.30 kg` vs analytical ~30 kg — a factor ~25
error with no warning. The plate was effectively invisible to
the BEM integral.

**Audit reference.**
[`studies/spar-fin-decay/STEP-A-FINDING.md`](../studies/spar-fin-decay/STEP-A-FINDING.md);
Check 2 of the diagnostic at commit `064d630`; resolution
documented at the corrected-mesh commit on
`scratch-spar-fin-decay`.

**Why latent.** M2-M6 fixture meshes (OC4 marin_semi etc.) have
no thin horizontal features in regions where the normal
direction is ambiguous; the OC4 cylindrical columns are
forgiving. The first mesh with a thin (~4 mm) horizontal
heave-plate feature surfaced the issue. Any future
externally-imported mesh with thin horizontal features (heave
plates, fins, sharp hull transitions) is at risk.

**Scope.** Two resolution paths:

1. **Upstream discipline:** documented mesh-prep step that
   validates panel normals before any BEM run, with the
   centroid-outward test as the validation criterion. Fix at
   the source mesh.
2. **FloatSim-level reader hygiene:** a pre-ingestion
   normal-validation step in FloatSim's Capytaine / WAMIT
   readers (and any future BEM reader) that runs the
   centroid-outward test and either auto-corrects or raises
   with diagnostic info. The detection logic from the spar-fin
   study (`fix_mesh_normals.py`) is the seed; generalising it
   for arbitrary closed meshes requires care around concave
   regions where "outward" is ambiguous.

**Estimated effort.** Path (1): operational discipline, not
code. Path (2): ~1 week if scoped to convex-meshes-only with
the centroid test; longer if generalised. Natural fit for B4
(multi-body BEM ingestion) Tier 3 work — same code surface,
same audit layer.

**Blocks.** Any future BEM-based study importing external
meshes, particularly anything with thin horizontal surfaces.
The spar-fin study's load-time fix is study-specific; future
studies on different geometries will need to re-validate
normals each time until the FloatSim-level hygiene check
exists.

**Status.** Open. Surfaced 2026-06-29 via the spar-fin study.
Sequencing: revisit during B4 scoping if normal-validation is
naturally co-located with multi-body BEM ingestion; otherwise
standalone milestone candidate.

---

### ~~BEM-CAPYTAINE-READER-SYMMETRIZATION~~ — Capytaine reader missing A/B symmetry tolerance

**Closed 2026-07-03 by M7.5 PR2 (`dafed8c`).**
`HydroDatabase.__post_init__` symmetrizes A, B, A_inf, and C
via `0.5 · (M + M.T)` (per-omega for A/B); the
pre-symmetrization asymmetry residual is stored on
`metadata["symmetrization_max_residual_{A,B,A_inf,C}"]`.
Terminal-gate test 2
(`tests/validation/test_m7_5_terminal_gate.py::test_reader_chain_ingests_asymmetric_capytaine_netcdf`)
ingests the study's own asymmetric NetCDF fixture
(`tests/fixtures/bem/spar_fin_terminal/capytaine_bem_asymmetric.nc`
extracted from `2767c12`) and verifies residual_A ≈ 7.18e-3,
residual_B ≈ 1.25e-1, and post-symmetrization symmetry to
rtol=1e-12. Delivered at HydroDatabase per plan §Q1 lock
rather than at the reader — every reader inherits the
symmetrization for free.

---

### ~~BEM-CAPYTAINE-READER-SYMMETRIZATION~~ (original entry, retained for audit trail)

**Mechanism.** Capytaine's BEM panel-method computes `A(i, j)`
and `A(j, i)` via independent radiation problems (radiating DOF
i vs j). Solver discretization produces ~1e-4 to 1e-3 relative
asymmetry between the two — unphysical (A should be symmetric
by reciprocity) and well below any tolerance that matters
physically, but enough to fail FloatSim's `rtol = 1e-6`
symmetry check at `floatsim/hydro/database.py:181`. The
FloatSim **WAMIT** reader already handles this via
`_resolve_6x6_from_dict`'s arithmetic-mean averaging of
duplicate `(i, j)` and `(j, i)` entries; the **Capytaine**
reader does not. On the spar+fin mesh, this manifested as
~2.85e-4 relative A asymmetry and ~3.78e-3 B asymmetry —
rejecting an otherwise-correct dataset.

**Audit reference.**
[`studies/spar-fin-decay/STEP-A-FINDING.md`](../studies/spar-fin-decay/STEP-A-FINDING.md)
Pre-flight 1 resolution section; commit `ef61d0e` on
`scratch-spar-fin-decay`. WAMIT reader's existing pattern at
[`floatsim/hydro/readers/wamit.py`](../floatsim/hydro/readers/wamit.py)
`_resolve_6x6_from_dict`.

**Why latent.** M5/M6 fixtures used the WAMIT reader (with
averaging) and a synthetic Capytaine fixture (carefully
constructed to be perfectly symmetric); real Capytaine BEM
output from non-trivial meshes carries panel-method noise that
no M5/M6 test exercised. The spar+fin study was the first
real-Capytaine-output ingestion attempt.

**Scope.** Symmetric to BEM-INPUT-NORMAL-VALIDATION above. Two
resolution paths:

1. **Upstream discipline:** documented mesh-prep / BEM-output
   step in the contributor's Capytaine workflow that
   symmetrizes A and B before saving the NetCDF. The spar+fin
   study's `capytaine_run.py` is the seed: a 5-line
   `0.5 * (M + M.swapaxes(-1, -2))` step.
2. **FloatSim-level reader hygiene:** add the symmetrization
   step to `floatsim/hydro/readers/capytaine.py`, matching the
   WAMIT reader's existing pattern. ~5-10 lines (the per-omega
   loop already exists for the complex-merge step at
   `_merge_split_complex`).

**Estimated effort.** Path (1): operational discipline. Path
(2): ~1 day if scoped to the existing reader's per-omega-loop
structure. Natural co-located fix with
BEM-INPUT-NORMAL-VALIDATION: both are reader-hygiene gaps
where the Capytaine reader lacks something the WAMIT reader
has.

**Blocks.** Any future Capytaine-BEM-based study with a real
(non-perfectly-symmetric) mesh. The spar-fin study's
study-local fix is workable but every future Capytaine study
will re-encounter the same issue until the FloatSim reader is
updated.

**Status.** Open. Surfaced 2026-06-29 via the spar-fin study
Pre-flight 1. Sequencing: reassess at study close together
with BEM-INPUT-NORMAL-VALIDATION for promotion to a small
reader-hygiene milestone or absorption into B4 scoping.

---

### ITEM25-SMALL-BODY-APPLICABILITY — Item 25 kernel-quality gate inapplicable to small-body BEM

**Addressed 2026-07-03 by M7.5 PR1 (`8c5da4a`), sub-scope
remains open per Q4 lock.** `compute_retardation_kernel`
gains the `asymptote_check_override: str | None` keyword;
non-empty rationale bypasses Check 1 and Check 2 while
keeping Check 3 authoritative. Terminal-gate test 3
(`tests/validation/test_m7_5_terminal_gate.py::test_kernel_chain_override_bypasses_gate_and_check3_passes`)
verifies the override on the real small-body spar+fin BEM:
warning fires with the rationale echoed, Check 3 passes on
the actual kernel, and empty-rationale still raises.
**The disposition-2 analytical-threshold quantification
(non-dimensional `omega_max · L / c` cutoff) remains
explicitly open per plan §Q4 lock: "Do not quantify
small-body threshold in M7.5"** — this is a research task
outside a hygiene-scoped milestone. Sub-scope status:
`Open` for the analytical threshold; `Delivered` for the
user-facing override.

**Mechanism.**
[`docs/openfast-cross-check-conventions.md`](openfast-cross-check-conventions.md)
Item 25 defines a three-check kernel-quality gate on
`compute_retardation_kernel` input:

1. **Asymptote consistency** — `std / |mean|` of
   `B(omega) * omega^4` over the last N tail samples must
   be below `_GATE_ASYMPTOTE_STD_OVER_MEAN = 0.10` per
   diagonal DOF (hard error on diagonal failure).
2. **Kramers-Kronig relation** — check the causality
   requirement linking A(omega) and B(omega).
3. **Kernel decay** — post-computation, the reconstructed
   K(t) must decay to below `_GATE_KERNEL_DECAY_RATIO`
   of its peak by `t_max`.

The asymptote check assumes the BEM omega grid extends into
the geometric 1/omega^4 far-field regime, characterised by
the non-dimensional group `omega_max · L / c` above some
threshold, where `L` is the body's characteristic length
and `c` is the wave celerity. For wave-tank-scale models
(`L ~ 1-2 m`) with `omega_max ~ 30 rad/s`, this group is
too small — the asymptotic regime is not reached anywhere
on the available BEM grid, and the asymptote check fires
spuriously regardless of BEM correctness.

**Audit reference.**
[`studies/spar-fin-decay/STEP-A-FINDING.md`](../studies/spar-fin-decay/STEP-A-FINDING.md)
Pre-flight 2 (2026-06-30): a diagnostic sweep of
`omega_max` in `[3, 15]` rad/s for the spar+fin geometry
(`L = 1.85 m`) found NO omega_max value passes the Item 25
asymptote check — the tail is nowhere clean because the
regime is nowhere asymptotic. The BEM output IS correct
by every other measure (analytical spar contributions
match, plate-face symmetries hold, K-K passes); Item 25's
asymptote check is simply the wrong gate for this scale.

**Why latent / visibility.** M6 Item 25 was calibrated
against OC4-scale floaters (`L ~ 50 m`, `omega_max ~ 5-15`
rad/s) where the 1/omega^4 asymptote IS visible because
the wavelength-to-size ratio reaches the far-field regime
within the BEM grid. Small-body BEM exits Item 25's
implicit validity envelope. M2-M6 fixtures used only
OC4-scale geometries; the first wave-tank-scale BEM (the
spar-fin study, 2026-06-30) surfaced the mismatch.

**Scope.** Two dispositions co-exist:

1. **M7.5 PR1 — user-facing override.**
   `compute_retardation_kernel` gains an
   `asymptote_check_override: str | None = None` keyword-
   only parameter (Q3 lock, see
   [`docs/m7.5-reader-hygiene-plan.md`](m7.5-reader-hygiene-plan.md)
   §Q3). Small-body users invoke the override with a
   non-empty rationale string; the kernel is computed via
   the zero-fill tail path documented at plan §I3. The
   override is by-user-judgment; the rationale string is
   the forcing-function acknowledgment that Item 25 does
   not apply.
2. **Deferred analytical work — quantify the threshold.**
   Compute a characteristic `omega_max · L / c` (or
   similar non-dimensional group) below which Item 25's
   asymptote check does not apply, so `compute_retardation_kernel`
   could either auto-skip the check or auto-select the
   zero-fill tail path. **Explicitly deferred per M7.5 Q4
   lock: "Do not quantify small-body threshold in M7.5"**
   (this belongs to a research-level milestone, not to a
   reader-hygiene bounded-scope milestone). May be
   absorbed into B4 (multi-body BEM ingestion) scoping if
   small-body multi-body geometries surface.

**Estimated effort.** The M7.5 PR1 override (disposition 1)
is `~180 lines code + ~200 lines test` per plan Phase 2 §PR1.
The analytical-threshold quantification (disposition 2) is a
research task with unclear bounds; a first-pass empirical
threshold sweep across geometries (spar+fin, spar-only,
disk, OC4-truncated, ...) is `~1-2 weeks` and would need
literature review for the theoretical `omega_max · L / c`
regime characterisation.

**Blocks.** Nothing immediately after M7.5 PR1 lands.
Small-body BEM users can proceed with the override.
Quantifying the threshold matters for future work where
automatic gate-bypass would be preferable to user
judgment — potentially Tier 3 workflows (multi-body
studies at wave-tank scale) if those materialise.

**Status.** Open. Surfaced 2026-06-30 (spar-fin study
Pre-flight 2). M7.5 PR1 provides the override mechanism
(empirical user judgment); analytical quantification
deferred.

---

### PANEL-NORMAL-NONCONVEX-BODIES — mesh_hygiene panel-normal validation punts on non-convex meshes

**Mechanism.**
`floatsim.hydro.mesh_hygiene.validate_panel_normals()`
(created in M7.5 PR3 per
[`docs/m7.5-reader-hygiene-plan.md`](m7.5-reader-hygiene-plan.md)
§Q2) uses the centroid-outward test: for each panel, the
dot product of the panel normal with the vector from
body-interior toward the panel centroid must be positive.
Body-interior is estimated as `mesh.vertices.mean(axis=0)`,
which is unambiguous for convex meshes — the arithmetic
centroid of a convex hull's vertices is guaranteed to lie
inside the hull.

For non-convex meshes (bodies with cavities, indentations,
moon pools, ducted geometry, or complex articulated
topology), `mesh.vertices.mean(axis=0)` may fall OUTSIDE
the body's interior region — e.g., in the empty volume
of a moon pool, or on the "wrong side" of a concave
surface. The centroid-outward test then produces either
false positives (correctly-oriented panels flagged as
reversed) or false negatives (actually-reversed panels
incorrectly passed), depending on the specific mesh
topology.

**Audit reference.**
[`docs/m7.5-reader-hygiene-plan.md`](m7.5-reader-hygiene-plan.md)
§Q5 lock (2026-06-30): "Non-convex body support — punt
entirely. Deferred with tracker entry
`PANEL-NORMAL-NONCONVEX-BODIES`. Scope creep would double
the audit surface without a red fixture."

**Why deferred.** M7.5's target validation scope
(spar-fin decay study; OC4-class fixtures; wave-tank models)
is all convex or near-convex geometry. Non-convex body
support requires a different interior-determination
strategy AND its own audit surface (fixtures, edge cases,
performance characterisation). Deferred per Q5 lock.

**Scope — candidate resolution strategies.** Two approaches
cover most non-convex cases:

1. **Ray-casting.** For each panel, cast a ray from the
   panel centroid along its normal direction; count
   intersections with the rest of the mesh. Even count
   means the normal points outward (the ray exits the body
   without re-entering); odd count means inward (the ray
   re-enters). Robust for arbitrary topology; runtime is
   O(n²) per validation (n panels × n candidate intersecting
   panels). Reusable off-the-shelf implementations exist
   (Trimesh, PyMesh) but pulls in a dependency.
2. **Signed-distance field (SDF).** Build an SDF for the
   mesh; sample points where the SDF is strongly negative
   (deep-interior points); use those as the reference
   "inside" set for the centroid-outward test on each
   panel. Overkill for validation alone but reusable for
   any other geometry work that needs "is this point
   inside." The SDF construction is its own algorithmic
   complexity.

**Estimated effort.** `~2 weeks` for ray-casting
integration + fixture set (moon-pool geometry;
articulated-body test case; existing convex regression
suite); `~3-4 weeks` for SDF-based validation with the
same audit surface. Both include the audit + testing +
docstring update; neither is a small piece of work.

**Blocks.** Nothing immediately. Any future study or
milestone involving non-convex geometry needs either a
different validation strategy or explicit opt-out. Not
exercised by any currently planned FloatSim work.
Candidate triggers for prioritising this work:

- A moon-pool floater cross-check case.
- A ship-shaped hull with internal cavities in a study.
- A future spar-fin-like study with a more complex
  attached-body configuration.
- A ducted-body Morison-element test case.

**Status.** Open. Deferred per M7.5 Q5 lock 2026-06-30.
Convex-only validation in M7.5 is sufficient for the
target scope; non-convex support deferred to a future
milestone if a triggering use case materialises.

**Amended 2026-07-02 (pre-PR3 lock correction):** the M7.5
PR3 algorithm changed from centroid-outward to
edge-consistency + signed volume, which handles non-convex
CLOSED bodies natively. The open scope of this entry
narrows to non-watertight / non-manifold meshes (open
shells, T-junctions, edges shared by != 2 panels,
disconnected multi-shell meshes needing per-shell
seeding). The algorithm detects and raises on these rather
than mis-validating. Ray-casting / SDF strategies remain
the candidate resolution paths if a future watertight-
but-topologically-complex fixture (multi-shell with
per-shell seeding, or a mesh with intentional
T-junctions from a specific solver export format) needs
handling.

**Amended 2026-07-03 (PR3 pre-flight, second narrowing):**
the M7.5 PR3 algorithm underwent a second amendment after
pre-flight surfaced that the terminal fixture is not a
closed manifold — 96 open edges at the plate-spar junction,
caused by the "spar+fin 5mm offset" thin-surface convention
documented in the mesh header. The second-amendment
algorithm accepts multi-shell polygon soup as its input
class: per-component flood-fill parity, signed volume for
closed components, ray-parity against the whole mesh for
open components (majority vote over sample-panel centroids,
graze-tolerant). Open shells and multi-shell meshes are now
SUPPORTED. The open scope of this entry narrows again to
only meshes with T-junctions (edges shared by more than two
panels) or otherwise ambiguous adjacency. T-junction
handling would need either explicit topology metadata from
the exporter, or a genuine geometric-ambiguity resolution
policy — neither is currently required by any FloatSim
fixture. The correction pattern (four Q2 amendments, this
one derived from measured fixture topology) suggests future
BEM-format expansions may surface similar topology
surprises; the algorithm's per-component structure gives us
headroom to add T-junction handling as a fourth branch if
needed.

**Amended 2026-07-03 (PR3 pre-flight, FINAL Q2 sixth
amendment).** Both prior algorithm amendments have been
DELETED. The M7.5 PR3 algorithm is now per-panel ray-parity,
assumption-free. The only remaining scope out of this entry
is T-junction / ambiguous adjacency (hard raise; no
algorithmic fix). Truly two-sided sheets are now the scope
of the new tracker entry
[**BEM-MESH-THIN-SURFACE-ORIENTATION**](#bem-mesh-thin-surface-orientation--orientation-convention-for-genuinely-two-sided-sheets)
below — the terminal fixture's strips were initially
suspected to be this class; measurement showed they have a
defined outward via ray-parity, so this entry no longer
covers them.

**Confirmed 2026-07-03 (PR4-A, milestone close).** The Q5 punt
held: PR3 detects T-junctions with a hard raise and warns on
the open-boundary false-negative mode (`n_open_edges > 0`);
final in-scope items for this entry are T-junction handling
and the open-shell false-negative documentation tests (both
pinned in-suite). No fixture in the M7.5 target scope
required resolving either class beyond detect-and-warn.

---

### BEM-MESH-THIN-SURFACE-ORIENTATION — Orientation convention for genuinely two-sided sheets

**Mechanism.** Some BEM meshes model genuinely two-sided
surfaces (flat membranes, theoretical zero-thickness
surfaces, sheets without an enclosing body). Such surfaces
have no defined "outward" direction — both sides are
equally valid. Per-panel ray-parity (M7.5 PR3's algorithm)
gives one answer or the other based on which side happens
to enclose more of the surrounding mesh, but there is no
physical correctness criterion.

**Audit reference.** Q5 punt (final form, 2026-07-03).
Surfaced during Q2 sixth-amendment discussion: the
terminal fixture's 24 strip panels were initially
suspected to be this class (radially inward-pointing
normals with no defined outward). Per-panel ray-parity
measurement (Step 0 diagnostic gate, 2026-07-03) showed
the strips DO have a defined outward — radially away
from the spar axis — and ray-parity detects and can fix
them. So the terminal fixture is NOT this class; it's a
mis-oriented thin surface that ray-parity handles
cleanly. This entry captures the genuine-two-sided case
in case a future fixture presents it.

**BEM-MESH-STRIP-PANELS-STUDY-FIXTURE (sub-item).** The
`test2_spar_fin_corrected.gdf` fixture from the spar-fin
study (committed at `scratch-spar-fin-decay` branch under
`studies/spar-fin-decay/mesh/` and pinned in-suite at
`tests/fixtures/bem/mesh_hygiene/`) carries 24 strip panels
that are objectively misoriented (radially inward normals).
`fix_mesh_normals.py` in the study never touched them
because its z-band + radius + `|n_z| > 0.9` filter
excluded them. `A_inf(heave) = 21.11 kg` computed via
Capytaine on this fixture matches analytical expectations
within 16% of the reference — tier-2
`check_hydrostatic_volume` (M7.5 PR3) quantifies the
insensitivity numerically.

**Tier-2 measurements** (2026-07-03, `rho = 1025 kg/m^3`,
body mass `28.67 kg`):

| variant   | `signed_volume`    | `displaced_mass` | `residual_fraction` |
|-----------|--------------------|-----------------:|--------------------:|
| ORIGINAL  | `+3.882e-02 m^3`   | `39.79 kg`       | `+0.388`            |
| CORRECTED | `+3.914e-02 m^3`   | `40.12 kg`       | `+0.399`            |
| full_fix  | `+3.989e-02 m^3`   | `40.89 kg`       | `+0.426`            |

Pairwise `|ΔV|/V(CORRECTED)`: ORIG-vs-CORR = `0.82%`,
FULLFIX-vs-CORR = `1.92%`. The `+40%` residual is
mesh-buoyancy-vs-body-mass **reserve buoyancy**, not a
mismatch to close.

**Correction (2026-07-04, study resumption).**
`check_hydrostatic_volume` integrates the **full closed
mesh** — it returns the displacement-*if-fully-submerged*
(~40.9 kg), the genuine reserve buoyancy. The earlier phrase
"the meshed waterline displaces ~40.1 kg against a 28.67 kg
body" was a **misinterpretation**: the tier-2 number is not
the waterline displacement. Measured at the design waterline
(`z = 0`), the fullfix mesh displaces only **24.47 kg**
(manual z-clip and Capytaine `immersed_part()` agree to 6
sig figs) — *less* than the 28.67 kg body — so the
free-floating buoy sinks **~0.185 m** to balance
(`dz = (M − m_disp)/(ρ·A_wp)`, `A_wp = π·0.0841²`); true
unmoored draft ≈ 1.28 m. The `+40%` figure is correct as a
reserve-buoyancy / fully-submerged number; it is simply not
a waterline balance. See
`studies/spar-fin-decay/waterline_balance.py` and the
STEP-A-FINDING.md resumption addendum. Documented so the
resumption does not open this as a bug: it is a
fixture property.

**Open-boundary false-negative.** The fixture has 96 open
boundary edges (plate-spar junction + disk rim). M7.5 PR3's
tier-1 `validate_panel_normals` / `fix_panel_normals` emit
a `UserWarning` on any mesh with open edges: a reversed
panel whose ray exits through an opening is silently
reported as outward. The 24 strip panels do *not* trigger
this false negative (their rays hit body geometry radially
inward), which is why ray-parity detects them; the warning
is pinned in-suite as a fixture property, not a defect to
suppress. See conventions doc Item 5 and mesh_hygiene test
`test_open_component_is_silent_false_negative_with_warning`
for the documented failure mode.

PR4 spar-fin re-verification obligations:

1. Record the 24 misoriented strip panels.
2. Record the `+40%` reserve-buoyancy residual (as a
   documented fixture property, not a mismatch).
3. Record the 96 open-boundary edges and the tier-1
   false-negative warning.

When `scratch-spar-fin-decay` is next touched (post-M7.5),
an addendum to `studies/spar-fin-decay/STEP-A-FINDING.md`
should note the strip misorientation and cite the tier-2
residual as evidence that the study's Capytaine A_inf
result stands despite the topological deficiency.

**Scope.** Adding an "orientation convention override" API
to `mesh_hygiene` that lets users declare per-panel
outward directions explicitly for two-sided sheets (via a
supplementary metadata file or per-panel keyword arg).
Only becomes necessary if a future fixture presents a
genuinely two-sided sheet.

**Estimated effort.** ~2 weeks, mostly research (what does
the community do; is there a WAMIT/OrcaWave/Capytaine
convention for this?) plus small implementation.

**Blocks.** Nothing immediately. Any FloatSim study
involving flat membranes / sail surfaces / thin planar
wave-energy converters would need this.

**Status.** Open. Surfaced 2026-07-03 during Q2 sixth
amendment. Parked (research-scale).

---

### BEM-CONTAMINATED-FREQUENCY-SLICE-CLUSTER-DRAFT — whole-matrix contaminated BEM solve at an isolated frequency

**Mechanism.** At ω≈4.934 the **entire 18×18 BEM solve is contaminated**
for the spar hull at **cluster draft** — a near-singular
boundary-integral operator (pole straddle). The large diagonals
(surge / roll / pitch) show a coherent ~5 % undershoot; the heave
diagonal **flips sign** (`B[2,2] → −8.56e-2`, physical peak +2.74e-2)
only because heave's physical magnitude (~0.01) is ~3 orders below
surge / roll / pitch (~40), so a shared perturbation of the same
absolute scale inverts it. Heave is the *most visible symptom*, not a
heave-specific artifact.

**Measured evidence** (`studies/cluster-3buoy-rigid/defect_diagnostic.py`,
Diagnostic A). Single hull, cluster draft, `B_heave(ω)` verbatim:

```
4.700 +1.0833e-02 | 4.800 +1.0237e-02 | 4.900 +9.6271e-03
4.920 +8.8509e-03 | 4.930 +7.1769e-02 | 4.934 −8.5841e-02
4.940 +8.3712e-03 | 4.950 +8.4116e-03 | 5.000 +1.0569e-02
```

The ± dispersive pair at 4.930 (+8× trend) / 4.934 (sign-flipped),
bracketed by clean values at 4.920 / 4.940, is a **pole-straddle
signature**, not a random solve failure. Whole-matrix deviation at
ω=4.934 (neighbour midpoint → value → deviation):

| DOF | neighbours | value | deviation |
|---|---|---|---|
| surge | +5.028 / +9.161 | +6.798 | **−4.18 %** |
| roll  | +3.242 / +6.499 | +4.614 | **−5.26 %** |
| pitch | +3.201 / +6.927 | +4.722 | **−6.76 %** |
| heave | +0.0107 / +0.0059 | −0.0856 | −1133 % |

Full 18×18 at ω=4.934: min eig **−0.1201**, max eig **+20.40**,
min-eig/max\|B\| = **−0.250 %**.

**Feature width.** ~0.02 rad/s (perturbs the fine-grid points 4.930 and
4.934; clean at 4.920 / 4.940) vs the production grid spacing ~0.34
rad/s near ω=5 → on any realistic grid it manifests as a **single
isolated bad ω**.

**Draft dependence** (strongest clue). Clean at the single-buoy
eqdraft (single-body study BEM: `B_heave(4.934) = +0.0211`), present at
cluster draft (`−0.0858`), **same hull geometry** — only the DZ2
sinkage differs.

**Two-phenomena proof** (Diagnostic B). The proper lid workflow
(`immersed_part()` → `generate_lid` → `lid_mesh=`, 21 faces) **clears
the genuinely Capytaine-flagged irregular frequency at ω≈20.909**
(surge −1.294 → +6.144, back on trend) but leaves **ω=4.934 unchanged**
(−0.0856 → −0.0856). The lid workflow is therefore sound; 4.934's
lid-immunity is a *measured* property. Capytaine does **not** flag
4.934 as irregular. A cylinder-internal-irregular-frequency explanation
is **falsified**: lid waterplane centroid radius 0.063 m → `ka ≈ 0.16`
at ω=4.934, two orders below `j₀₁ = 2.405`.

**DETECTION GAP (M11).** The PSD gate (M8 PR3, `retardation.py`
`_validate_psd`) catches **this** instance **only** because heave's
near-zero magnitude turns a coherent ~5 % perturbation into a **sign
flip** (which drives a negative eigenvalue). A grid point that straddles
the pole **less** severely would produce ~5 % undershoot with **no sign
change** and **pass PSD undetected**. **M11 needs frequency-slice
smoothness screening — neighbour-trend deviation across the grid — not
PSD alone.**

**Mechanism status.** UNKNOWN beyond the above measurements. No
speculation.

**Blocks / M11.** Production 12-buoy BEM (72 DOF) will encounter this
class at some frequency. Mitigation candidate: **detect-and-re-solve at
a perturbed ω** — the feature width (~0.02 rad/s) is far below the grid
spacing, so a ~0.1 rad/s nudge moves the grid point off the pole. The
mesh/lid path demonstrably does **not** touch it.

**M11 Phase-1 UPDATE (2026-07-29) — Measurement E FALSIFIED this entry's
own recommendation.** The "DETECTION GAP (M11)" paragraph above recommended
**"frequency-slice smoothness screening — neighbour-trend deviation across
the grid."** M11 Phase-1 Measurement E built and validated that screen on
the committed 18-DOF fixture (five statistics: A per-entry, A
significance-filtered, B Frobenius, B local-robust-z, coherent
signed-diagonal). Result:
- **Genuine irregular frequency ω=20.909 — cleanly detected** (B
  local-robust-z = 30.2 vs clean ≤ 11.6, ~30× separation).
- **This entry's own ω=4.934 near-singular class — NOT detected** (z=0.5;
  separation < 1× across all five). It is a modest (~3–7 %) coherent
  whole-matrix perturbation sitting where B legitimately ramps toward its
  9.5–10 rad/s peaks, so it does not separate from real curvature.
**Consequence, stated plainly:** the 4.934-class contamination is currently
**UNDETECTABLE by any proposed output-smoothness method** at 12-buoy scale.
PSD caught the 18-DOF instance only because heave's near-zero magnitude
turned ~5 % into a sign flip; at 72+ DOF **there is no guarantee of a
comparably near-zero channel**, so PSD may miss it too.
**Revised recommendation:** the robust detector is **solve-time
conditioning monitoring** (flag ω where the Capytaine influence-matrix
solve is ill-conditioned — upstream, a different signal), **not** post-hoc
output smoothness. Keep the cheap output-smoothness screen as a
**complementary** check for the irregular-frequency class (which it does
catch). This is its **own M11 PR** (M11 plan Q7), sequenced with the BEM
work. Measured in `M11 Phase-1` (scripts under the session scratchpad);
see `docs/m11-platform-plan.md` Measurement E.

**Status.** Open. Surfaced 2026-07-20 by the M8 PR3 PSD gate on first
contact with a production-grid multi-body fixture. The 18-DOF fixture
is retained **unmodified**; PR3 excludes the contaminated ω from its
positive gate and asserts PSD fires on the unmodified fixture (negative
gate). Recommendation revised 2026-07-29 (M11 Phase-1 Measurement E — see
above). See `docs/m8-coupled-bem-plan.md` (PR3 Step C, risk register) and
`docs/audits/m8-coupled-bem-audit.md` (PR3 Step-A finding).

---

### INBAND-ROTATIONAL-RESONANCE — Coupled-cluster buoy-pitch mode needs drag before its wave rotation can be measured

**Mechanism.** The articulated-3 cluster (M10) has a buoy-pitch-about-
the-joint rotational mode with `T_rot = 3.257 s`, `zeta = 0.373 %`
(radiation-only), `Q ~ 134`, **in-band and adjacent to the 3.106 s heave
resonance**. Under regular waves near this mode the **drag-free** BEM
rotation reaches the Item-2 `0.1 rad` threshold at a `~4.5 mm` wave
amplitude (vs `5.4 m` off resonance — a `~1200x` contrast). The free
decay is stable and bounded, so the integrator/KKT handling is sound;
the wave-case runaway is genuine resonant buildup, not numerical
instability (M10 plan Amendment A4). Therefore the wave rotation the
LEVEL2 gate consumes **cannot be measured near resonance without a drag
term** in the coupled assembly.

**Audit reference.** `docs/m10-articulated3-plan.md` Amendment A4
(measurement + Q4 lock inconsistency); `docs/tier3-program-plan.md`
append-only amendment 2026-07-29 (drag capability -> REQUIRED M11).
Measured by `tests/validation/test_m10_pr2_wave_rotation.py` (convention
gate + off-resonance sensitivity + in-band-mode characterisation).

**Why latent / visibility.** Invisible through PR1: the M2-PR1 gates are
free-decay **periods** (`M+A+C`-dominated, damping-insensitive) and a
**symmetric** heave IC (excites no rotation). The mode only surfaces on
first **directional wave** activation of the coupled model (PR2) — the
CLAUDE.md §13 "correct in isolation, wrong on first full-scenario
activation" shape.

**Scope.**
- Drag **capability** (Morison / quadratic drag `state_force`) assembled
  onto the coupled 18-DOF (and 12-buoy) model — `build_system` assembles
  no `drag_elements` today (M10 plan gap (d)).
- Rotational-drag **characterisation** for the buoy-pitch-about-joint
  mode (tank; distinct from the heave-plate heave `Cd`).
- Re-measure the near-resonance rotation, then decide LEVEL2.

**Q2 sensitivity -> CONFIRMED (M10 close, A5(b)).** The Q2 arm-mass split
is **confirmed by inspection** (joints at the buoy top end; arms are
hub-side structure), so `T_rot = 3.257 s` is **single-valued**. The
earlier `+5.3 %` alternative split (`T_rot = 3.431 s`, `zeta = 0.449 %`,
`Q ~ 111`) was a sensitivity bound and is **retired** — the rotational
finding was robust to the split and is now on confirmed footing.

**Working joints CONFIRMED (M10 close, A5(a)).** The physical model has
working joints, so the mode is experimentally excitable: `T_rot = 3.257 s`
is a **falsifiable prediction** vs tank data (the programme's first
external check). The tank campaign will sweep wave **height and period**
(not heave decay only), so:
- the wave-height sweep **is** the rotational-drag experiment
  (response-per-height falls near resonance -> `Cd`);
- drag capability is needed to **predict the cluster tests**, not only to
  resolve LEVEL2 (program-plan amendment 2b-2e);
- ~~**NEW open question (2f):** the heave-plate `Cd = 5.0` (disc broadside
  to vertical flow) cannot be reused for the rotational mode (plate
  edge-on, horizontal flow) — the rotational damping must be measured,
  not inferred; the drag-widened resonance bandwidth is unknown until
  then.~~ **REFUTED at M11a PR4 (2026-07-30) — see the correction below.**

**2f CORRECTION (M11a PR4, 2026-07-30) — the edge-on framing was wrong.**
The derivation (plan Finding F3) refutes 2f. Under the rotational mode the
plate does not present its edge to the flow; it **TILTS**, so a disc point
at radial `x` moves VERTICALLY at `w(x) = -theta_dot*x` — that is
plate-NORMAL (broadside) flow across the disc, the SAME regime as heave,
using the **KNOWN `Cd_n = 5.0`**. The edge-on (tangential) motion of the
disc centre is a MINOR term (`E_normal/E_tangential = 1.76-3.52` re-derived,
so edge-on is `1/(1+ratio) = 22-36 %` of the plate — NOT 11 %, which was the
retired 7.7 split) governed by the tank-pending `Cd_t`. So:
- the dominant plate rotational damping uses a coefficient we **already
  have** (`Cd_n = 5.0`), NOT a tank-pending one — 2f's "cannot be reused" is
  false;
- only the minor edge-on `Cd_t` is tank-pending, and it governs a small
  fraction of an already-small contribution: the plate's rotational
  `zeta_drag ~ 0.017-0.021 %` at `Theta = 0.02`, only **~4-6 % of the spar's
  0.379 %** (F1) — the SPAR dominates rotational damping, not the plate;
- the drag-widened bandwidth is now known from the spar (Q ~ 68, F1); the
  plate barely changes it.

**THIRD reframing (net) — rotational damping lives on the SPAR.** The
sequence is (i) M10: plate edge-on, unknown coefficient (this entry's
original 2f); (ii) PR4 STEP 1(b): refuted — within the plate, normal flow
dominates on the known `Cd_n = 5.0`; (iii) PR4 measurement: the plate is only
4-6 % of the spar's contribution. **Net: rotational damping is dominated by
slender-cylinder cross-flow on the SPAR (PR2, Q3-ii), not the plate.**
Campaign consequences: the **tank rotational decay primarily calibrates the
SPAR `Cd`** (the ~90 % contributor), and **PR2's adopted literature prior
`Cd = 1.2`** (smooth cylinder, KC ~ 1.46) is now **the load-bearing
coefficient for the whole rotational mode** — the number the campaign most
needs to pin; `Cd_t` (~1-2 % of the total) and even the plate `Cd_n` are
second-order for ROTATION (`Cd_n = 5.0` stays load-bearing for HEAVE). The
energy-equivalent reference the calibration will use is **unbiased** (plan F3
item-3: the PR2/PR4 "measured-above" was a differential-DOF coordinate
artifact; the modal-coordinate measurement matches the reference to <0.6 %).

**Campaign data-reduction requirement (M11a PR4 item-3, 2026-07-31).** The
differential-DOF coordinate artifact applies to the **tank data reduction**,
not only the simulation: a decay ratio computed directly on a measured
relative-angle channel (e.g. buoy-to-hub) **over-reads the modal decay rate
by 16-26 %**, so a `Cd` fitted from it is biased high by that factor. **Tank
rotational-decay data must be reduced in MODAL coordinates** — project the
measured channels onto the model's mode shape before the log-decrement, or
apply the model-derived differential-to-modal correction. This bites the
**SPAR `Cd`** (the load-bearing coefficient per the third reframing above).
Recorded program-side at `docs/tier3-program-plan.md` campaign amendment (2g).

**PRINCIPAL stated approximation (SHEARED FIELD) — this, not `Cd_t`, is what
the tank rotational decay tests.** `Cd_n = 5.0` was measured for UNIFORM
heave; applying it strip-wise to the linearly varying tilting field assumes
local face-normal drag with no radial interaction. It is far better grounded
than the discarded edge-on framing, but it is an approximation, and the tank
rotational-decay campaign is the experiment that tests it. (The isotropic
horizontal-cylinder heave-plate stand-in was ALSO refuted — it applies
`Cd_n` to the large edge-on velocity, ~290x overpredicting that term; the
anisotropic `PlateDragElement` supersedes it, structurally guarded.)

**Estimated effort.** Capability ~1-2 wk (coupled drag `state_force` +
gate); characterisation is tank-campaign-gated (outside program control).
**Capability DELIVERED (M11a PR1 wiring, PR2 spar, PR4 plate).**

**Blocks.** The `M10 -> LEVEL2 decision gate -> M11` sequencing (Q1): the
gate's rotation-amplitude input is undetermined near resonance until drag
lands. **LEVEL2 is subordinate to drag** (drag gates the measurement
LEVEL2 consumes). Drag is also required to predict the staged cluster
wave-response tests (program-plan 2a-2b).

**Status.** Open. Surfaced 2026-07-29 (M10 PR2, first directional-wave
activation of the coupled articulated model); updated 2026-07-29 (M10
close) with working-joints + Q2 resolutions and the campaign scope.
Depends on M11 drag capability. See `docs/m10-articulated3-closure.md`
S4/S7 and the program-plan campaign-scope amendment.

---

### PER-BODY-ITEM25-OVERRIDE-UNEXPOSED — build_system per-body path cannot build a small-body BEM deck

**Mechanism.** `build_system`'s **per-body** assembly path calls
`compute_retardation_kernel(bem_databases[body.name], t_max=..., dt=...)`
at **`floatsim/driver.py:844`** WITHOUT an `asymptote_check_override`
argument. A small-body BEM (whose `B(ω)` has not reached the `1/ω^4`
asymptote by `ω_max`, e.g. the spar-fin hull, `std/mean of B·ω^4 = 0.60 >
0.10` gate) therefore cannot be built end-to-end through the per-body
path -- `compute_retardation_kernel` raises on the asymptote gate. The
**coupled** path DID get the override (M10 PR0, threaded at
`driver.py:626,736`); the per-body path did not.

**History (third appearance).** M8's closure recorded this as an
ergonomic gap (`docs/m8-coupled-bem-closure.md:175`,
`ITEM25-SMALL-BODY-APPLICABILITY`). M11 Phase 2 (`365b344`, plan Q3)
threaded the override through the **coupled** path only. **M11a PR1**
(`9017b0d`) surfaced that the **per-body** path still lacks it: the
spar-fin heave+drag regression (GATE 1) had to use the study's own
hand-assembled lhs/kernel rather than a full `build_system` per-body
build, exactly as the study documents (`studies/spar-fin-decay/
study_common.py:3-7`).

**Scope.** Small (thread `asymptote_check_override` through the per-body
branch of `build_system`, mirroring the coupled branch) and **orthogonal
to drag**. **Not blocking M11a** -- the completed studies hand-assemble,
and the coupled path (M11's actual 12-buoy target) is already threaded.

**Estimated effort.** < 0.5 d (parameter threading + a small-body per-body
build gate).

**Blocks.** Nothing on M11's critical path; a convenience so future
small-body single-body decks build through the driver rather than
hand-assembling.

**Status.** Open. Surfaced 2026-07-30 (M11a PR1). Candidate for a `fix-`
branch alongside the carried black-conformance (3 files) and the F2
magnitude-scaled hypothesis-red bound.

---

## Resolved entries

*(none yet)*

---

## Entry template for future additions

```
### <ID> — <Short name>

**Mechanism.** <What the deferred thing is, in 2-4 sentences.>

**Audit reference.** <File path + section.>

**Why latent / visibility.** <Optional — when the deferred item is
known to be invisible under current test coverage, document the
visibility floor explicitly. Required for any item that would otherwise
match the WAMIT-dim bug pattern.>

**Scope.** <Bulleted scope of work to close the item.>

**Estimated effort.** <Weeks of focused work, comparable to a prior
milestone or PR.>

**Blocks.** <What use cases / scenarios are gated by this entry.>

**Status.** Open. <Date surfaced.> <Optional sequencing notes.>
```

---

## Process

- **Append-only during a milestone.** New entries land alongside the
  audit / PR that surfaces them.
- **Close at the resolving PR.** When a successor PR lands the work,
  strike through the entry (`~~`) and add a closing date + the
  resolving commit hash.
- **Reassess at milestone close.** Each milestone-closure document
  (e.g., `docs/m7-foundation-closure.md`) reviews this tracker and
  promotes / re-sequences entries as the project priorities
  evolve.
- **Estimates are order-of-magnitude planning aids, not
  commitments.** M5 / M6 actual delivery (including fix branches)
  ran roughly 3-4x the original per-PR estimates — the original
  plan sized PRs at 150-300 lines each; actual delivery including
  the fix branches that the cross-check surfaced was 3-4x that
  (see [`docs/m6-closure.md`](m6-closure.md) §6.5). Treat
  "estimated effort" numbers here as the line that would be
  written before the work surfaces what it surfaces, and budget for
  the same multiplier when scheduling.

The tracker is the institutional response to conventions doc Item
23 ("deferred-known-bugs must be tracked, not just commented") at
the milestone-scope level. The WAMIT dimensionalisation latency
(five PRs of dormancy) is the anti-pattern this file exists to
prevent.
