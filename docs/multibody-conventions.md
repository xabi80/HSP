# Multibody conventions — operational reference

**Purpose.** Numbered registry of multibody conventions that govern
how FloatSim composes connectors / catenaries / drivers across
bodies. Each item documents an invariant assumed by some piece of
code: shape, sign, validity range, lag treatment, framework
boundary. Items are mechanism-driven (the convention is rooted in
algebra or framework design) rather than parameter-tuned.

This file is the multibody analog of the M6 conventions doc
[`openfast-cross-check-conventions.md`](openfast-cross-check-conventions.md)
which carries 33 items collected across M6's 5-PR cross-check
sweep. M7-Foundation closes with **4 starting items**; the
compendium grows organically from successor work rather than
manufactured retroactively.

This document is **append-only**. Items are not renumbered; if a
convention is corrected, the new state goes in a follow-up item
cross-referencing the original.

---

## Item 1 — `LinearConnector` framework: symmetric Newton-III at reference points

**Convention.** [`floatsim/bodies/connector.py`](../floatsim/bodies/connector.py)'s
`LinearConnector` represents a connector by a single 6x6 `K` (plus
`B`, `rest_offset`) acting on `delta = xi_a - xi_b - rest_offset`,
with `F_on_a = -K @ delta` and `F_on_b = +K @ delta` by Newton III.
**The framework assumes symmetric Newton-III at the body reference
points exactly.**

**Validity range.** Holds whenever both attachment points coincide
with the respective body reference points — i.e. both arms are
zero. (The "both arms equal in some frame" case is a hypothetical
generalisation no current fixture exercises and is not validated;
treat it as out of scope until a real case demands it.) Otherwise
the moment-arm cross-product asymmetry breaks the `F_b = -F_a`
constraint.

**Failure mode.** Body-body connections with any non-zero attachment
offset cannot be represented in this framework without per-endpoint
K factors. See:

- [`docs/phase2-followups.md#bb-offset-connector`](phase2-followups.md) —
  tracker entry **BB-OFFSET-CONNECTOR** with two resolution paths
  (Direct framework surgery vs free emergence from B2's Lagrange-
  multiplier DAE).
- [`docs/diagnostics/m7-pr2-framework-limit.md`](diagnostics/m7-pr2-framework-limit.md) —
  full algebraic derivation (four matching constraints all reduce to
  `T_a = T_b`).

**Consumer-side gate.** Both
`assemble_attachment_transformed_connector` (F2, Item 3 below) and
the deck-driven `build_system` (F1) raise `NotImplementedError` on
the unsupported case rather than silently dropping the moment
asymmetry. The error message cites BB-OFFSET-CONNECTOR per the M7
plan Q9-pinned disposition.

**Why M6 didn't surface this.** Every M2-M6 fixture lived in the
supported subset (body-body at reference, or body-earth single-
offset). The constraint was always there, just unexercised. PR2's
F2 derivation surfaced it the first time `LinearConnector`'s scope
required choosing a precondition on the offset configuration.

---

## Item 2 — F2 attachment-offset transform: small-angle linear validity bound

**Convention.** `assemble_attachment_transformed_connector` in
[`floatsim/bodies/connector.py`](../floatsim/bodies/connector.py)
applies the pull-back

    K_ref = T^T @ K_attach @ T
    T = [I_3, -r_tilde; 0, I_3]
    rest_offset_LC = T^{-1} @ rest_offset_attach

where `r_tilde` is the skew-symmetric cross-product matrix of the
body-frame arm `r`, and `r_tilde^T = -r_tilde` (so `T^{-1} = [I_3,
+r_tilde; 0, I_3]`).

The transform is **exact** for the small-angle linear regime
(`theta = xi[3:6]`). It accumulates O(θ²) error away from
`theta = 0`.

**Validity range.** `|theta| < 0.1 rad` (~5.7°) keeps the error
below 0.5 %. Larger rotations need the full quaternion-driven
transform — Phase 2 work tracked as
[**LEVEL2-INTEGRATOR-UNWIRED**](phase2-followups.md#level2-integrator-unwired--quaternion-integrator-wiring).

**Consumer-side gate.** None at the helper level — the validity
range is documented in the docstring + this conventions file. The
integrator (`integrate_cummins`) treats `xi[3:6]` as small-angle
linear Euler globally; FloatSim runs Level-1 in practice, so F2's
small-angle assumption is consistent with the rest of the stack
(not a downstream upgrade waiting to happen).

**Locked at.** [`docs/m7-foundation-plan.md`](m7-foundation-plan.md)
Q3. Pinned in tests at
[`tests/unit/test_connector_attachment_transform.py`](../tests/unit/test_connector_attachment_transform.py)
(identity tests at rtol = 1e-12 for closed-form K_ref / F_ref + 2
hypothesis property tests).

---

## Item 3 — F3 catenary state-force composer: vertical-plane + explicit-lag

**Convention.** `make_catenary_state_force` in
[`floatsim/mooring/catenary_analytic.py`](../floatsim/mooring/catenary_analytic.py)
produces a `(t, xi, xi_dot) -> F[6N]` closure with TWO modelling
assumptions:

1. **Vertical-catenary-plane.** Each line lies in the vertical plane
   containing the inertial-frame fairlead and the inertial-frame
   anchor. No current; no lateral force on the line. Inherited from
   M6 PR5 ([`test_m6_openfast_moored_eq.py`](../tests/validation/test_m6_openfast_moored_eq.py)).
2. **Explicit one-step lag.** The integrator evaluates the closure
   at the **previous step's state** `(t_{n-1}, xi_{n-1},
   xi_dot_{n-1})`, identical to the explicit-μ convention of
   `make_connector_state_force` and the convolution sum in
   `integrate_cummins`. This is NOT a per-composer choice; it is the
   integrator's state_force contract, and `make_catenary_state_force`
   inherits it by virtue of being a state_force callable.

**Validity range.** Quasi-static analytic catenary. Dynamic mooring
(MoorDyn-style lumped-mass) is out of Phase 1 scope (M6
conventions doc Item 26: F-DAMP-MATCH). Body-to-body catenaries are
out of M7-Foundation PR3 scope.

**Consumer-side gate.** `CatenaryAttachment.__post_init__` raises on
`body_index < 0` (no body-to-body or earth-to-earth). The deck-
driven `build_system` raises `NotImplementedError` on body-body
deck-Catenary entries with a clear message.

**Locked at.** [`docs/m7-foundation-plan.md`](m7-foundation-plan.md)
Q4. Pinned in tests at
[`tests/unit/test_catenary_state_force.py`](../tests/unit/test_catenary_state_force.py)
(13 identity tests against scripted hand-wired prediction at two
body poses, rtol = 1e-12).

**Identity with M6 PR5 hand-wired path.** Byte-equivalent at the
M6 PR5 OC4 fixture (3 chains at 837 m radius / 120° spacing,
fairleads at 40.8 m, anchors at z = -200 m). The M7 PR3 refactor of
M6 PR5 produced identical test-runner diagnostic output pre/post
(same heave |delta|, same tension rel-errs), confirming the
composer integrates faithfully into the M6 PR5 quasi-static
pipeline.

---

## Item 4 — `build_system` driver: state_force composition order + dimension convention

**Convention.** `floatsim.driver.build_system` returns a
`SimulationSetup` carrying:

- `lhs` (n_dof x n_dof, **position-space** dimension; n_dof = 6N).
- `kernel` ((n_dof, n_dof, N_t)).
- `state_force` (composed closure: `make_connector_state_force` +
  `make_catenary_state_force` summed; returns the zero vector when
  the deck has no connections).
- `xi0`, `xi_dot0` (each `(n_dof,)`, returned separately matching
  `integrate_cummins(xi0=..., xi_dot0=...)` — the 2·n_dof state-
  space pack is never materialised).
- `body_name_to_index` (deck-order map).

**Composition rule.** When the deck has both connector-style and
catenary-style connections, the composed `state_force` is the
arithmetic sum of the two closures. Order does not matter (addition
commutes); the rule documents the sum so a future investigator does
not look for one closure subsuming the other.

**Wave forcing NOT in state_force.** Wave forcing is an
`external_force` to `integrate_cummins`, conceptually distinct from
the connector / catenary state-dependent coupling. `build_system`
intentionally does not compose `make_regular_wave_force` —
[`docs/m7-foundation-plan.md`](m7-foundation-plan.md) Q1 records
this as one of two declined scope-creep temptations (the other
being BEM-reader dispatch).

**Validity range.** Block-diagonal hydrodynamics only. Each body
backed by an independent `HydroDatabase`; multi-body BEM cross-
coupling is tracker entry **B4**.

**Consumer-side gate.** `build_system` validates body-name
uniqueness, the "earth" sentinel non-collision, bem_databases
completeness, and connection-endpoint resolution before
materialising any helper.

**Locked at.** [`docs/m7-foundation-plan.md`](m7-foundation-plan.md)
Q1, Q2, Q7. Pinned in tests at
[`tests/unit/test_driver.py`](../tests/unit/test_driver.py)
(19 tests: 8 round-trip identity at rtol = 1e-12 vs hand-wired
M4 PR6, 2 single-body sanity, 2 earth-sentinel both directions,
1 BB-OFFSET-CONNECTOR error-message content, 6 locked-scope error
paths).

---

*Status: starting set of 4 items locked at M7-Foundation close
(2026-06-05). Grows organically with successor work.*
