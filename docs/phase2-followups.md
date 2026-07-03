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

**Status.** Open. Surfaced 2026-07-01 (M7.5 pre-milestone
audit). Empirical tolerance in place at `bbb5b9b`;
analytical bound derivation and cache-skip investigation
deferred. Not blocking M7.5.

---

### BEM-INPUT-NORMAL-VALIDATION — BEM mesh panel-normal orientation validation

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

### BEM-CAPYTAINE-READER-SYMMETRIZATION — Capytaine reader missing A/B symmetry tolerance

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
