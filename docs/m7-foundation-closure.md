# M7-Foundation Closure — Multi-body driver, helpers, and N = 4 validation

**Milestone:** M7-Foundation — F1 / F2 / F3 / F4 bounded scope per
[`docs/m7-foundation-plan.md`](m7-foundation-plan.md).
**Reference systems:** M4 PR6 fixture (2 bodies + heave rigid link + 2
earth-anchored catenaries), plus a new 4-body block-diagonal fixture
(F4) — the first N ≥ 3 system in the repo's validation history.
**Closed:** 2026-06-05.
**Scope owner:** Xabier.

This document is the audit-trail record of M7-Foundation. It cross-
references the per-PR commit messages on the `milestone-7-foundation`
branch, the multibody-capability audit
([`docs/audits/multibody-capability-audit.md`](audits/multibody-capability-audit.md)),
the diagnostic docs in [`docs/diagnostics/`](diagnostics/), and the
Phase 2 tracker ([`docs/phase2-followups.md`](phase2-followups.md)).

The target reader is someone joining the project who needs to
understand what M7-Foundation delivered, what it deliberately did
not, and what the empirical data says about the three reassessment
options for the next milestone.

> **Note on anchor links.** The markdown anchors in this document
> (e.g. `phase2-followups.md#bb-offset-connector`) assume the
> repository structure at the commit hash where
> `m7-foundation-closure.md` was authored. If the referenced doc
> structure changes, regenerate the anchor list via the
> closure-doc maintenance task rather than chasing them one at a
> time. Same prevention rule as `m6-closure.md`.

---

## 1. Executive summary

M7-Foundation delivered all four bounded items (F1 deck-driven
driver, F2 attachment-offset transform, F3 catenary state-force
composer, F4 N = 4 block-diagonal validation) across four PRs on
the `milestone-7-foundation` branch. **Zero fix branches. Zero test
surfacings. One framework constraint** (BB-OFFSET-CONNECTOR) caught
by the PR2 derivation **before any test ran**, disposed of via the
Phase 2 tracker per the Q9-pinned PR4 disposition.

| metric | value |
|---|---|
| PRs | 4 (F4 / F2 / F3 / F1, in order) |
| fix-* sub-branches opened | 0 |
| Item-19 hypothesis firings | 2 (PR1 F4 at n_dof = 24; PR4 round-trip identity) |
| Item-19 hypothesis surfacings | 0 |
| framework constraints surfaced | 1 (BB-OFFSET-CONNECTOR, PR2 derivation, pre-test) |
| Phase 2 tracker entries added | 1 (BB-OFFSET-CONNECTOR) |
| Phase 2 tracker entries updated with empirical data | 1 (B6 empirical baseline) |
| size-agnostic M4 PR1 design tested at N ≥ 3 | yes (N = 4 block-diagonal, bit-identical rtol = 1e-12) |
| round-trip identity at rtol = 1e-12 | bit-identical (driver vs hand-wired M4 PR6) |
| total tests newly added | 81 (15 F2 unit + 13 F3 unit + 19 F1 driver unit + 34 F4 validation) |
| total existing tests refactored to flow through M7 helpers | 15 (9 M4 PR6 + 6 M6 PR5) |

The audit-driven discipline (CLAUDE.md §15) cleared the four PRs
first-try. The Item-19 hypothesis (CLAUDE.md §13) fired twice and
surfaced nothing in the exercised regime — recorded explicitly so a
future investigator does not mistake silence for proof-of-correctness
across all regimes. The framework caught its only constraint at the
derivation stage rather than at assertion time.

---

## 2. Per-PR results

Each PR's commit message carries the detail; this table is the
navigation index. Diagnostic docs are linked where they exist.

| # | PR | Scope | Tests | Pass | Notes |
|---|---|---|---:|---:|---|
| **PR1** | F4 N = 4 block-diagonal validation | First N ≥ 3 test in repo history; size-agnostic solver / integrator / assembly exercised at n_dof = 24 | 34 | 34 | [diagnostic doc](diagnostics/m7-pr1-multibody-scaling.md); pre-foundation audit cleared before red gate fired; Item-19 hypothesis fired and surfaced nothing |
| **PR2** | F2 `assemble_attachment_transformed_connector` | Small-angle linear `T^T @ K @ T` pull-back for body-earth single-offset connectors | 15 | 15 | [diagnostic doc](diagnostics/m7-pr2-framework-limit.md); BB-OFFSET-CONNECTOR surfaced at the derivation, before any test ran |
| **PR3** | F3 `make_catenary_state_force` composer | 6-DOF generalised-force closure parallel to `make_connector_state_force`; body-to-earth scope | 13 + 6 | 13 + 6 | rtol=1e-12 against scripted hand-wired prediction at two body poses (xi_eq + 5 m surge discriminator); M6 PR5 byte-equivalent refactor |
| **PR4** | F1 `build_system` driver | Deck-driven composition; `SimulationSetup` dataclass; F2 / F3 / `heave_rigid_link` dispatch | 19 | 19 | Round-trip identity vs hand-wired M4 PR6 at rtol = 1e-12 bit-identical on lhs/kernel/state_force/xi0; 3 pre-flight items pinned (single-body sanity, earth sentinel both directions, BB-OFFSET-CONNECTOR error-message content) |

**Aggregate.** 81 newly written tests (47 unit across F2 / F3 / F1
+ 34 F4 validation) all pass. Independently, 15 existing assertions
in two refactored validation suites (M4 PR6 with 9 assertions; M6
PR5 with 6 assertions) preserved unchanged, now flowing through the
M7-Foundation helpers (F3 composer in M6 PR5; build_system in M4
PR6) with byte-equivalent diagnostic output where checkable.

---

## 3. Empirical findings

M7-Foundation delivered four findings worth pinning beyond the
per-PR commit detail.

### 3.1 N = 4 block-diagonal validation: 34 assertions cleared at their respective tolerances

The size-agnostic M4 PR1 design (`assemble_global_lhs`,
`assemble_global_kernel`, `integrate_cummins`'s n_dof-agnostic step
loop, `pack_state` / `unpack_state`) had no test at N ≥ 3 before
M7-Foundation. The audit
([`multibody-capability-audit.md`](audits/multibody-capability-audit.md)
§3) flagged this as an Item-19 hypothesis (CLAUDE.md §13): the
latent N ≥ 3 code paths would surface something.

**Result.** PR1 F4 ran 34 assertions at n_dof = 24 across five
classes:

| class | assertion | tolerance |
|---|---|---|
| (A) | per-body heave period matches M2 analytical | rtol = 1e-2 |
| (B) | per-body heave log-decrement damping matches M2 analytical | rtol = 5e-2 |
| (C) | cross-DOF silence per body (20 silent slots) | atol = 1e-10 m |
| (D) | IC-scaling ratio per body (3 body-pair ratios; the pack/unpack discriminator) | rtol = 5e-3 |
| (E) | `cond(A_eff)` at n_dof = 24 equals n_dof = 6 reference | rtol = 1e-12 |

All 34 passed first-try. The varying tolerances reflect the varying
physical content of each assertion: (E) is a structural identity
where rtol = 1e-12 is the right gate; (A)-(D) carry numerical
discretisation that the M2 fixture itself accumulates at the
rtol = 1e-2 to 5e-3 level.

**Boundary of the claim** — the N = 4 exercise was:

- Block-diagonal (no inter-body BEM coupling).
- Identical bodies (M2 heave-only synthetic fixture, four copies).
- Small-angle regime (xi[3:6] ≡ 0 throughout F4).
- No connectors in F4.

So the proven-clean claim is: **the size-agnostic design holds at
N = 4 in the block-diagonal / identical-body / small-angle /
no-connector regime**. It does not establish:

- Behaviour at N = 8, 12, 72 (Phase 2; the audit's B6 entry tracks
  the perf scaling separately).
- Behaviour at off-block-diagonal BEM coupling (tracker B4).
- Behaviour at heterogeneous bodies (deferred from Q5 lock).
- Behaviour at moderate rotations > 5° (tracker LEVEL2-INTEGRATOR-
  UNWIRED).
- Behaviour at general connector topologies (tracker A3).

This updates the audit's "untested at N ≥ 3" finding to "N = 4
tested clean in the block-diagonal far-spaced regime". The audit's
§7 "Recommended next steps" Foundation block is fully discharged.

### 3.2 Round-trip identity: bit-identical at rtol = 1e-12 (PR4)

PR4's driver-route round-trip identity is a SEPARATE empirical
finding from §3.1's F4 assertions. The PR4 unit suite asserts that
`build_system(M4_PR6_deck, ...)` produces bit-identical output
against the hand-wired M4 PR6 setup (which uses the existing F2,
F3, and `heave_rigid_link` helpers underneath) at rtol = 1e-12 on:

- `lhs.M_plus_Ainf` (12 x 12)
- `lhs.C` (12 x 12)
- `kernel.K` (12 x 12 x 60001)
- `state_force(0, xi=zeros, 0)` (12-vector)
- `state_force(0, xi=both-surge-0.5, 0)` (12-vector; the catenary-
  asymmetric discriminator)
- `xi0` post-equilibrium (12-vector)

All six identity tests pass at rtol = 1e-12. The driver produces
the same system bit-for-bit — not just a system that passes the
same downstream physics gates. This is a different kind of result
than §3.1: §3.1 tests physical correctness; §3.2 tests that the
deck-driven assembly path is identical to the hand-wired path it
replaces. Both findings matter for different reasons.

### 3.3 F4 perf baseline: 21 ms/step at n_dof = 24

PR1 F4 ran 100 s of simulated time at dt = 0.01 s (10,000 steps) in
210.77 s wall clock — **~21 ms/step at n_dof = 24** with the
existing dense `np.linalg.solve(A_eff, rhs)` per-step factorisation
and the `n_dof × n_dof × N_t` einsum convolution sum.

Linear extrapolation to n_dof = 72 projects ~600+ ms/step, or
~100 min for a 10k-step run. (The LU dominates at O(n_dof³); the
einsum convolution adds at O(n_dof² · N_t).) This is the
empirical evidence behind Phase 2 tracker entry **B6**
([`phase2-followups.md`](phase2-followups.md)) and updates its
prior "estimated 1700×" qualitative claim to a measured-projects-
to baseline.

Speedup target: 10× at n_dof = 72 would bring 12-buoy runs into
the 10-minute range, comparable to current single-body production
work.

### 3.4 BB-OFFSET-CONNECTOR — framework constraint caught at derivation

PR2 surfaced a `LinearConnector`-framework constraint: the
framework assumes symmetric Newton-III at reference points (`F_b =
-F_a` exactly), which fails for body-body connections with non-zero
attachment offset (the moment-arm cross-product asymmetry).
Detailed algebra in
[`diagnostics/m7-pr2-framework-limit.md`](diagnostics/m7-pr2-framework-limit.md).

**Why M6 didn't surface this.** Every M2-M6 fixture lived in the
supported subset (body-body at reference, or body-earth single-
offset). The M4 plan Q1's penalty-stiffness decision pre-empted
the body-body-offset use case; the M6 PR5 catenary path went body-
to-earth. The constraint was always there, just unexercised. PR2's
explicit derivation surfaced it the first time F2's scope required
choosing a precondition on the offset configuration.

**Disposition.** PR2 raised `NotImplementedError` on the
unsupported case. The constraint went into the Phase 2 tracker as
**BB-OFFSET-CONNECTOR** ([`phase2-followups.md`](phase2-followups.md)).
PR4's `build_system` raises with a message citing the tracker entry,
per the Xabier-pinned Q9 disposition. Two resolution paths
documented in the tracker:

- **Direct.** Extend `LinearConnector` to per-endpoint K factors —
  ~1-2 weeks framework surgery.
- **Free emergence from B2.** Lagrange-multiplier DAE handles the
  asymmetry naturally — zero incremental cost if B2 is scheduled
  anyway.

**Catch-before-test is the audit discipline working.** This is the
M6 §15 audit pattern operating at the framework level rather than
the convention level — same idea, different surface.

### 3.5 Item-19 hypothesis: fired twice, surfaced nothing this round

The plan framed F4 as an Item-19 code-path exerciser: the
hypothesis was that the size-agnostic N ≥ 3 code path would surface
something. PR4's round-trip identity was a second firing — the
hypothesis was that build_system would diverge from the hand-wired
path somewhere.

**Both fired clean.** No surfacings. Worth recording with the right
epistemics:

> Item 19 is a hypothesis-generating discipline, not an oracle. A
> hypothesis that doesn't fire is a hypothesis that doesn't fire —
> not proof of correctness in unexercised regimes. The N ≥ 3 code
> paths could still surface at N = 8 / 12 / 72, at non-trivial
> connector topology (A3), at multi-body BEM coupling (B4 / B5),
> at heterogeneous bodies, or at moderate rotations. The empirical
> baselines from §3.1 and §3.2 become the deviation gates for
> those future audits.

This is the same framing recorded in
[`diagnostics/m7-pr1-multibody-scaling.md`](diagnostics/m7-pr1-multibody-scaling.md)
post-fire section. Restated here at milestone scope.

**Next Item-19 firings worth scheduling** (so they are not forgotten
when the next milestone scope is decided):

- (a) **N = 8 / 12 scaling.** Same N ≥ 3 size-agnostic code paths
  at larger N. The B6 perf baseline (§3.3) is the deviation gate.
- (b) **Heterogeneous bodies.** Q5 lock used 4 identical bodies as
  the cleanest pack/unpack discriminator. The heterogeneous case
  exercises pack/unpack indexing more rigorously than identical
  bodies do.
- (c) **Off-block-diagonal hydrodynamic coupling.** Gated by
  tracker entry B4. Until B4 lands, the only meaningful test is the
  block-diagonal subset already exercised.
- (d) **Moderate-rotation regimes** (`|theta| > 5°`). Gated by
  tracker entry LEVEL2-INTEGRATOR-UNWIRED. The integrator runs
  Level-1 small-angle Euler today; moderate-rotation accuracy needs
  the quaternion wiring.

### 3.6 M6 closure tension-figure correction

During PR3's M6 PR5 refactor regression check, an A/B comparison
surfaced that the M6 closure doc's "sub-0.15 % tensions" claim was
the pre-flight Step-A-prediction-vs-OF figure, not the actual
test-runner FS-vs-OF result. The test-runner result was always in
the +1.7 %-3.7 % range inside the 5 % gate.

The M6 closure doc was corrected on main at commit `5c4fd62`,
distinguishing the two figures explicitly. The discipline
self-correcting across milestones — surfacing a doc-level
artifact-vs-summary drift two weeks after M6 closed — is itself
worth recording.

The general rule the correction added:
[`m6-closure.md`](m6-closure.md) §S4 records both figures
distinctly:

- Test-runner FS-vs-OF (assertion record).
- Step-A prediction vs OF (stricter pre-flight).

Future closure docs distinguish "the assertion this PR pins" from
"any tighter pre-flight comparison" so the same drift does not
recur.

---

## 4. Discipline retrospective — M6 playbook on a bug-free milestone

M6 caught six Phase-1 bugs and the M6 closure doc structured around
them. M7-Foundation caught zero bugs in code — the audit caught
BB-OFFSET-CONNECTOR before any test ran. The four PRs closed clean.

This is a different shape than M6, not a smaller version. The
discipline produced what it's designed to produce in each case:

- **Audit-driven validation (CLAUDE.md §15).** Cleared all four PRs
  first-try. The pre-foundation audit at PR1 verified the size-
  agnostic code paths at n_dof = 24 (3 items exercised + 5
  reasoned) before F4's red gate fired. The PR2 framework-limit
  surfacing IS the audit pattern: trace the consumer-side
  assumption, find the precondition mismatch, dispose via tracker
  rather than silent workaround.

- **Pre-flight diagnostic discipline.** Each PR ran a Step A hand
  prediction before the implementation. PR1's prediction was the
  audit checklist + Item-19 framing. PR2's was the closed-form
  `T^T @ K @ T` algebra. PR3's was per-line catenary 6-vectors at
  two body poses (the second pose being a deliberate discriminator
  for the moment block). PR4's was the hand-wired M4 PR6 setup.
  Each PR's Step B implementation then matched its Step A targets
  at rtol = 1e-12.

- **Decision B xfail markers.** Not exercised this milestone — no
  assertion failed for a known-mechanism reason. Recorded explicitly
  so the absence isn't read as "the discipline isn't applicable".

- **Diagnostic-during-implementation.** Exercised by PR2's framework-
  limit doc and PR1's pre-flight addenda (i)-(v), which surfaced
  the convolution-step structural check, the rank-deficient C
  breadcrumb, the baseline scaling numbers as future-deviation
  gates, and the branching strategy / failure-mode response menu
  — all decided before red gates fired.

**The honest framing.** M6 caught bugs because there were latent
bugs to catch. M7-Foundation didn't catch bugs because the size-
agnostic M4 PR1 design held up cleanly in the exercised regime
and because the framework constraints that DID exist were caught
at derivation. Both outcomes are the discipline working — the
first by debug-and-fix, the second by audit-and-defer. Forcing
M7-Foundation's closure into M6's six-bugs structure would
manufacture a parallel that doesn't fit the empirical record.

---

## 5. Phase 2 tracker state after M7-Foundation

The Phase 2 tracker
([`phase2-followups.md`](phase2-followups.md)) currently carries
**10 distinct entries**, all open. (The tracker file also contains
a B3 cross-reference pointing to LEVEL2-INTEGRATOR-UNWIRED; B3 is
not a separate entry but appears in the file header structure for
discoverability — the audit's §7 enumeration that introduced B3 as
an ID is preserved.) M7-Foundation contributed:

- One new entry (**BB-OFFSET-CONNECTOR**).
- One empirical-data update to an existing entry (**B6**).
- No closures.

| ID | Title | M7-Foundation status |
|---|---|---|
| LEVEL2-INTEGRATOR-UNWIRED | Quaternion integrator wiring | Open. Not exercised by M7-Foundation's small-angle scope. |
| A1 | General 6-DOF rigid-link helper | Open. F2 covers the deck-driven path for the body-earth case; A1 remains for body-body multi-DOF. |
| A2 | Connector drift diagnostic aggregation | Open. Untouched. |
| A3 | N = 4 connector-coupled validation | Open. F4 covers N = 4 block-diagonal; A3 remains for N ≥ 3 with non-trivial coupling. |
| B1 | Selective-DOF joint helpers (hinge, ball, prismatic) | Open. Untouched. |
| B2 | Lagrange-multiplier DAE constraint formulation | Open. Untouched. BB-OFFSET-CONNECTOR's "free-from-B2" resolution path cross-references this entry. |
| B4 | Multi-body BEM cross-coupling ingestion | Open. Untouched. |
| B5 | Coupled retardation kernel transform | Open. Untouched. Depends on B4. |
| B6 | Sparsity-aware linear algebra | Open. **Empirical baseline updated** from M7-Foundation PR1 F4 (21 ms/step at n_dof = 24; projected ~600+ ms/step at n_dof = 72; B6 confirmed required not optional at 12-body scale). |
| **BB-OFFSET-CONNECTOR** | Body-body `LinearConnector` with non-zero attach offset | **New.** Surfaced PR2; disposition pinned at plan Q9; PR4 raises with tracker citation. |

The audit's §7 Foundation block (F1-F4) is fully discharged. All
remaining entries are in §7's "Scenario A enablers" (A1-A3) or
"Scenario B enablers" (B1-B6 + LEVEL2-INTEGRATOR-UNWIRED + BB-
OFFSET-CONNECTOR).

---

## 6. Reassessment — three-tier choice for the next milestone

M7-Foundation is the gate decision point Xabier called for at the
audit close: "At M7-Foundation's close we reassess whether to climb
to Scenario A (L03) — that's a separate decision made with the
empirical data F4 produces."

The data F4 produced (§3) supports three coherent next-step tiers.
This section lays each out neutrally with cost estimates informed
by M7-Foundation's empirical findings; it is **not a
recommendation**.

### Tier 1 — Foundation only (stop here)

**What it gives you.** Single-body trustworthy (M6). Multibody
scaffolding trustworthy in the block-diagonal far-spaced regime
(M7-Foundation). Deck-driven `build_system` as the canonical
entry point.

**What it doesn't give you.** Any of the Scenario A or Scenario B
use cases. No L03 validation. No 12-buoy work. The Phase 2 tracker
sits at 9 entries indefinitely.

**Cost.** Zero new work. Phase 2 tracker accrues entries only as
incidental surfacings arise.

**When this is right.** When the next research question doesn't
need multi-body coupling — e.g., further M6-style cross-checks on
new single-body floaters, deeper M6 OC4 work in regimes M6 itself
didn't reach, M5 reader expansion to additional BEM tools.

### Tier 2 — +Scenario A enablers (4-body structural; L03 path)

**What it gives you.** L03 as a meaningful validation case (subject
to one caveat below). Body-body coupled configurations at N ≥ 3.
Tools for steel-truss structural assemblies.

**Items required.**

- **A1** general 6-DOF rigid-link helper — ~1 week (small surface).
- **A2** connector drift aggregation — ~2 days.
- **A3** N = 4 connector-coupled validation — ~1-2 weeks (requires
  A1).
- **BB-OFFSET-CONNECTOR via the Direct path** — ~1-2 weeks of
  framework surgery, IF the L03 use case requires body-body offset
  connections (likely).

**B4 caveat.** L03's columns are close-packed enough that inter-
column hydrodynamic coupling is non-negligible. The tracker's
B4 (multi-body BEM cross-coupling ingestion) is the gate for L03
being a meaningful validation **at all** — without it, the
comparison would be FloatSim's uncoupled-BEM model against
OrcaFlex's coupled-BEM answer, and any disagreement would be
missing physics rather than a bug
([`phase2-followups.md#B4`](phase2-followups.md)). The
M7-Foundation audit and the M6 closure both flagged this.

So Tier 2 splits:

- **Tier 2a** (Scenario A enablers, no B4): A1 + A2 + A3 +
  BB-OFFSET-CONNECTOR Direct. Total: ~4-6 weeks. Useful for
  far-spaced 4-body cases (e.g. multi-floater wave farms with
  hawser couplings) but **not** for L03's close-packed columns.

- **Tier 2b** (Scenario A + the B4 enabler): Tier 2a +
  **B4** ~3-4 weeks + **B5** ~2 weeks (depends on B4). Total:
  ~9-12 weeks. This unlocks L03 as a meaningful validation case.

**Tier 2a is its own endpoint, not a stepping stone to 2b.** The
two serve different research questions: 2a covers far-spaced
multi-body wave farms with hawser couplings (where inter-body
hydrodynamic coupling is small), while 2b covers L03's close-packed
columns (where it is not). A future milestone can elect 2a → stop,
or 2a → 2b, or 2a → Tier 3, depending on the research priority at
that point.

**When this is right.** Tier 2a: when far-spaced multi-body
configurations are the next research priority and B4-grade
multi-body BEM coupling is not. Tier 2b: when L03 is the next
research priority and the B4 prerequisite is accepted.

### Tier 3 — Full multibody (12-buoy articulated)

**What it gives you.** General articulated-body simulation at the
12-body scale. Survival sea states with moderate-to-large
rotations. The full ARCHITECTURE.md §9.2 Level-2 fidelity.

**Items required.** Everything in Tier 2b plus:

- **B1** selective-DOF joint helpers — ~2-3 weeks.
- **B2** Lagrange-multiplier DAE — ~6-8 weeks (the big one).
  Closes BB-OFFSET-CONNECTOR's free-emergence path automatically.
- **B3** LEVEL2-INTEGRATOR-UNWIRED quaternion integrator wiring —
  ~3-6 weeks. Major rework; ripples through every state-force
  consumer.
- **B6** sparsity-aware linear algebra — ~2-3 weeks. The M7
  PR1 empirical baseline (§3.2) confirms this is required not
  optional at 12-body scale (~600 ms/step → ~100 min for a 10k-step
  run without it).

**Total.** Tier 2b + ~13-19 additional weeks = ~22-31 weeks
end-to-end. Comparable in scope to M1 + M2 + M3 + M4 combined —
i.e., a multi-quarter commitment.

**Estimate-as-planning-aid disclaimer (applies to all three tiers,
called out at the largest because the absolute number is biggest).**
Estimates are order-of-magnitude planning aids, not commitments;
M5 / M6 actuals (including fix branches) ran 3-4× initial per-PR
estimates (see [`m6-closure.md`](m6-closure.md) §6.5). Tier 3
plausibly ranges 30-50+ weeks at the same scope-creep ratio. Tier
2b plausibly ranges 12-18 weeks. Tier 2a plausibly ranges 5-8
weeks. Treat these as "the line that would be written before the
work surfaces what it surfaces," and budget for the same
multiplier when scheduling.

**When this is right.** When 12-body articulated systems are the
research target and the M6 / M7-Foundation single + few-body work
is the stepping stone, not the destination.

---

## 7. Repository state at closure

```
main (post-M6-closure-correction):
  5c4fd62  docs: M6 closure tension-figure correction
  ecebd3d  docs: BB-OFFSET-CONNECTOR tracker entry + Q9 pinned disposition
  eae6ebb  docs: B6 empirical baseline from M7-Foundation PR1 F4 run
  39c79b9  docs: M7-Foundation planning -- audit + plan + Phase 2 tracker
  c71cd9c  docs: M6 closure document
  63b04e1  ← M6 close

milestone-7-foundation (PR1-PR4 commit boundaries annotated):

  --- PR4 (F1 driver, 1 commit) ---
  1a169d1  M7-Foundation PR4 -- F1 build_system driver

  --- PR3 (F3 composer, 1 commit) ---
  2278196  M7-Foundation PR3 -- F3 catenary state-force composer

  --- PR2 (F2 transform, 2 commits) ---
  72bbbed  M7-Foundation PR2 -- diagnostic doc (framework-limit surfacing)
  54703b7  M7-Foundation PR2 -- F2 attachment-offset transform

  --- PR1 (F4 validation, 3 commits) ---
  b34295e  M7-Foundation PR1 -- F4 N=4 block-diagonal validation
  6e6349b  M7-Foundation PR1 -- F4 pre-flight addenda (i)-(v)
  e64c663  M7-Foundation PR1 -- Q8 pre-foundation scaling audit

  --- base ---
  39c79b9  ← shared base with main
```

After this closure doc + the conventions doc land on main, and
`milestone-7-foundation` merges into main, the project state is:

- M6 closed; M7-Foundation closed.
- Single-body trustworthy (M6); multibody scaffolding trustworthy
  in the block-diagonal far-spaced regime (M7-Foundation).
- Phase 2 tracker with 10 distinct entries (LEVEL2-INTEGRATOR-UNWIRED
  + A1, A2, A3 + B1, B2, B4, B5, B6 + BB-OFFSET-CONNECTOR; B3 is a
  cross-reference to LEVEL2-INTEGRATOR-UNWIRED).
- Three-tier choice on the table for Xabier's decision.

---

*Document status: closure draft, awaiting Xabier review (PR5 cadence
Step A). Will be committed to main alongside `multibody-conventions.md`
after review fixes, then milestone-7-foundation merges into main.*
