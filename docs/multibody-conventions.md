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

## Item 5 — BEM mesh panel normals: outward orientation required for BEM integration validity

**Convention.** Any mesh passed to a BEM solver (Capytaine,
WAMIT, OrcaWave, etc.) must have panel normals pointing
**outward into the surrounding fluid**. The validity
criterion (amended 2026-07-02): every shared edge is
traversed in opposite directions by its two adjacent
panels (edge-consistency), and the total signed volume of
the resulting orientation is positive (`V = (1/6) · Σ
signed tetrahedron volumes per panel > 0`). Panels
disagreeing with the majority orientation are flipped
relative to it; if the majority signed volume is negative
the whole set is inverted.

**Validity range.** Applies to any closed orientable mesh
in any BEM workflow, convex or non-convex. Non-watertight
or non-manifold input (edges shared by != 2 panels,
disconnected multi-shell adjacency) is DETECTED and
raised explicitly rather than silently mis-validated;
see tracker entry
[**PANEL-NORMAL-NONCONVEX-BODIES**](phase2-followups.md#panel-normal-nonconvex-bodies--mesh_hygiene-panel-normal-validation-punts-on-non-convex-meshes)
(amended 2026-07-02, scope narrowed to non-watertight /
non-manifold).

**Amended M7.5 pre-PR3 (2026-07-02):** the previously
documented centroid-outward test is invalid for
non-convex closed bodies — it inverts on the spar+fin
heave-plate top annulus (interior-estimate at spar-axis
z~-0.1 sees the plate TOP at z=-0.955 from above, so
correct +z normals produce negative dot products).
Edge-consistency + signed volume is the correct general
criterion for closed orientable meshes; it needs no
"interior" concept at all, only the shared-edge
adjacency and the vertex triples of each panel.

**Amended M7.5 PR3-pre-flight (2026-07-03, second
amendment):** the input class is **multi-shell polygon
soup**, not closed orientable mesh. PR3 pre-flight surfaced
that the terminal spar+fin fixture is not vertex-welded at
the heave-plate/spar-wall junction — 96 open edges,
consistent with the fixture header `"spar+fin 5mm offset"`
documenting the plate as a 5-mm-offset thin surface. This
is standard WAMIT GDF practice for thin plates. The
amended validity criterion has three parts:

1. Per-connected-component flood-fill orientation parity
   (XOR propagation on shared edges; degenerate edges
   skipped).
2. Per-component absolute orientation: signed volume for
   closed components; ray-parity against the whole mesh
   for open components (majority vote over sample panel
   centroids, cast along parity-0 normal, count even/odd
   crossings, graze-tolerant discard).
3. T-junctions (edges shared by more than two panels)
   remain a hard error — genuine ambiguity.

The 96-open-edge finding is the motivating evidence: the
validity criterion must handle real WAMIT GDF exports, not
just closed manifolds.

**Amended M7.5 PR3 (2026-07-03, FINAL Q2 amendment).** The
edge-consistency + signed-volume algorithm was correct on
synthetic geometry (verified twice) but computes RELATIVE
orientation within shells — a question the terminal fixture
proved is not the protective one. The final validity
criterion is **per-panel ray-parity**: cast from each
panel's centroid + epsilon along the as-stored normal,
count intersections against all other panels, odd = inward.
Assumption-free (no manifold, no components, no parity
classes). Edge machinery retained only for T-junction
detection (raise). Superseded predecessors:

1. Centroid-outward test (original Q2 lock): inverted on
   concave features like the plate top annulus.
2. Edge-consistency + signed volume (first amendment):
   required closed manifold; failed on the multi-shell
   fixture with plate/strip topology.
3. Per-component parity + ray-parity for open components
   (second amendment): assumed uniform orientation within a
   connected component; failed on thin plates where strips
   are traversal-inconsistent with faces even in the
   correct state.

**Load-bearing protection is tier 2:**
`check_hydrostatic_volume` measures displaced volume
directly via the divergence-theorem sum over all panels
as-stored. Reversed faces corrupt the volume; the check
sees it without any topology assumption. The tier-1
per-panel ray-parity check is diagnostic detail; tier 2 is
the primary protective screen. **Deciding measurement**:
Step 0 diagnostic run 2026-07-03 (per-panel ray-parity on
both terminal fixtures) showed 216 inward on ORIGINAL and
24 inward on CORRECTED — proving the study's z-band
heuristic left the strip panels misoriented, invisibly to
Capytaine's A_inf calculation. BEM solvers do not crash on incorrectly-
oriented normals; they silently produce wrong added-mass and
damping. In the worst observed case
([`studies/spar-fin-decay`](../studies/spar-fin-decay/)), a
single-side-reversed heave plate annulus produced
`A_inf(heave)` 25× lower than physical (1.30 kg vs analytical
~30 kg) with no warning.

**Consumer-side gate.** Currently absent from FloatSim's
Capytaine reader (and presumably from the WAMIT reader; not
audited as of 2026-06-29). Tracked as
[**BEM-INPUT-NORMAL-VALIDATION**](phase2-followups.md#bem-input-normal-validation--bem-mesh-panel-normal-orientation-validation)
in the Phase 2 tracker. Studies importing external meshes must
validate panel normals before BEM runs (the spar-fin study's
[`fix_mesh_normals.py`](../studies/spar-fin-decay/fix_mesh_normals.py)
is the reference implementation).

**Locked at.**
[`studies/spar-fin-decay/STEP-A-FINDING.md`](../studies/spar-fin-decay/STEP-A-FINDING.md)
"Resolution" section; mesh-fix commit on the
`scratch-spar-fin-decay` branch; tracker entry on main. Sourced
from standard panel-method literature (Newman 1977 Ch. 10; Lee &
Newman WAMIT manual). The spar-fin study surfaced it
operationally — the convention itself is BEM-foundational, not
study-specific.

---

## Item 6 — BEM A/B matrix symmetrization on ingestion

**Convention.** The added-mass matrix `A(omega)` and radiation-
damping matrix `B(omega)` are physically symmetric by
reciprocity. BEM solvers that compute `A(i, j)` and `A(j, i)`
via independent radiation problems (radiating DOF i vs j —
Capytaine's default behavior) produce ~1e-4 to 1e-3 relative
panel-method asymmetry that must be averaged out during
ingestion via `M_sym = 0.5 * (M + M.T)` per omega.

**Validity range.** Applies to any BEM reader that consumes
solver output where `M(i, j)` and `M(j, i)` are independently
computed. The WAMIT format writes both halves redundantly and
the FloatSim WAMIT reader already symmetrizes via
`_resolve_6x6_from_dict`'s arithmetic-mean handling. The
FloatSim Capytaine reader currently does NOT symmetrize and
rejects panel-method-noisy datasets on the `rtol = 1e-6`
symmetry check at
[`floatsim/hydro/database.py:181`](../floatsim/hydro/database.py).

**Failure mode.** The reader raises `ValueError: A[:, :, k] must
be symmetric (within rtol=1e-06)` on first-omega slice — a hard
ingestion failure on real Capytaine BEM output from non-trivial
meshes. The dataset is physically correct; the rejection is a
hygiene-step omission.

**Consumer-side gate.** The WAMIT reader has the step. The
Capytaine reader does not (as of 2026-06-29). Tracked as
[**BEM-CAPYTAINE-READER-SYMMETRIZATION**](phase2-followups.md#bem-capytaine-reader-symmetrization--capytaine-reader-missing-ab-symmetry-tolerance)
in the Phase 2 tracker. Studies importing Capytaine BEM output
must symmetrize at output before saving the NetCDF (the
spar-fin study's
[`capytaine_run.py`](../studies/spar-fin-decay/capytaine_run.py)
symmetrization block is the reference implementation; also
records audit-trail attributes
`symmetrization_max_residual_A` /
`symmetrization_relative_residual_A` (same for B) on the
NetCDF).

**Locked at.**
[`studies/spar-fin-decay/STEP-A-FINDING.md`](../studies/spar-fin-decay/STEP-A-FINDING.md)
"Pre-flight 1 finding + resolution" section; commit `ef61d0e`
on `scratch-spar-fin-decay`; tracker entry on main (commit
`a0821ac`). Sourced from BEM reciprocity (Newman 1977 Ch. 6 —
added-mass / damping coefficients in inviscid theory satisfy
`A_ij = A_ji`).

---

*Status: 6 items locked at M7-Foundation close (2026-06-05)
through to the spar-fin study (2026-06-29). Grows organically
with successor work.*
