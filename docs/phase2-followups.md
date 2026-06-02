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

### B4 — Multi-body BEM cross-coupling ingestion

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

### B5 — Coupled retardation kernel transform on multi-body BEM

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

### BB-OFFSET-CONNECTOR — Body-body LinearConnector with non-zero attachment offset

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
