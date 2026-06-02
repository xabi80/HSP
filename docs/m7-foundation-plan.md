# Milestone 7 Plan — Multi-body Foundation (F1-F4)

Working document. Status: **draft v1, awaiting Xabier review.** Delete or
archive after M7-Foundation merges.

Scope per [`docs/audits/multibody-capability-audit.md`](audits/multibody-capability-audit.md)
§7 Foundation block. M7-Foundation is **strictly bounded** to four items:

- **F1** — `build_system(deck) -> (lhs, kernel, state_force, initial_state)`
  deck-driven composition driver.
- **F2** — `assemble_attachment_transformed_connector` helper for
  anchor-offset transforms (small-angle linear).
- **F3** — `make_catenary_state_force` composer paralleling
  `make_connector_state_force`.
- **F4** — N = 4 block-diagonal validation test (the first N >= 3 test
  the repo has ever had).

Out of scope for M7-Foundation (queued in
[`docs/phase2-followups.md`](phase2-followups.md)): all Scenario A
enablers (A1-A3), all Scenario B enablers (B1-B6), the
LEVEL2-INTEGRATOR-UNWIRED rework. At M7-Foundation's close we
reassess whether to climb to Scenario A (L03) — that's a separate
decision made with the empirical data F4 produces.

Tolerance per CLAUDE.md §5: conservation / hand-derived algebra
identities at `rtol = 1e-12` (driver round-trip, attachment-offset
transform); analytical-physics comparisons at `rtol = 1e-2` (period)
and `rtol = 5e-2` (damping); cross-body coupling silence at
`atol = 1e-10`.

Validation gate: a 4-body block-diagonal deck reproduces the M2
analytical heave-decay period and damping on every body
independently, with no spurious cross-body or cross-DOF coupling,
through the driver. PR1 establishes this via hand-wired assembly
(red gate); PR4 re-confirms it through the deck-driven driver.

Branch: `milestone-7-foundation` (not yet created at the time of
this plan draft). Will be opened off `main` at the post-M6 head
(`63b04e1` — the M6 PR4 xfail-marker flip following the
`fix-make-regular-wave-force-convention` epilogue, which is
already FF-merged into `main`).

---

## M6 discipline carried forward

Every M7 PR follows the four operational disciplines codified in
[`docs/m6-closure.md`](m6-closure.md) §6:

1. **Audit-driven validation (CLAUDE.md §15)** — Q8 below is the
   pinned pre-cross-check audit. No PR2+ assertion fires until Q8 is
   fully green.
2. **Pre-flight diagnostic discipline** — each PR carries a
   first-principles prediction step before its assertion runs; the
   prediction-vs-result agreement is the gate to proceed. Per-PR
   pre-flight contract listed in the PR sequence below.
3. **Decision B xfail markers as evidence-gathering** — any assertion
   that fails for a named-mechanism reason gets an xfail-strict
   marker citing a named follow-up. Generic xfail is forbidden.
4. **Diagnostic-during-implementation** — when implementation
   surfaces an unexpected signal, expand scope to add the diagnostic
   as a runnable artefact rather than note-and-defer.

The pattern lock from CLAUDE.md §13 (six Phase-1 findings) is the
operational backdrop. F4's framing as an Item-19 code-path exercise
is the explicit hypothesis that **the N >= 3 code path will surface
something**.

---

## Decisions to lock with Xabier

### Q1 — Driver API surface and module location

**Proposal: a new `floatsim/driver.py` (single-file module) with one
public entry point.**

**Dimension convention.** Throughout this document, `n_dof = 6N` is
**position-space** dimension for `N` bodies. The full state-space
dimension is `2 * n_dof = 12N`. `CumminsLHS.M_plus_Ainf` and
`CumminsLHS.C` are `n_dof x n_dof`; `RetardationKernel.K` is
`(n_dof, n_dof, N_t)`; `xi`, `xi_dot`, `xi_ddot` are each `(n_dof,)`.
The driver and the integrator both operate in position-space; the
state-space pack `[xi, xi_dot]` is never materialised as a single
`(12N,)` array. The M6-class dimensional ambiguity (`n_dof` vs
`2*n_dof`) is one of the bug-shapes this convention closes.

```python
# floatsim/driver.py
from dataclasses import dataclass

@dataclass(frozen=True)
class SimulationSetup:
    """Output of build_system: everything an integrator needs to run.

    xi0 and xi_dot0 are returned separately to match the
    integrate_cummins contract (xi0, xi_dot0 kwargs are length-n_dof
    each, not a packed length-2*n_dof state vector).
    """
    lhs: CumminsLHS                 # M_plus_Ainf, C: (n_dof, n_dof)
    kernel: RetardationKernel       # K: (n_dof, n_dof, N_t)
    state_force: Callable[[float, ndarray, ndarray], ndarray]
    xi0: ndarray                    # (n_dof,) position IC, post-equilibrium
    xi_dot0: ndarray                # (n_dof,) velocity IC (deck-stated; zeros at rest)
    body_name_to_index: dict[str, int]  # for output-channel resolution

def build_system(
    deck: Deck,
    *,
    bem_databases: dict[str, HydroDatabase],
    dt: float,
    t_max_kernel: float,
    solve_equilibrium: bool = True,
) -> SimulationSetup: ...
```

- `bem_databases` is a dict mapping `Body.hydro_database` reference
  (typically a file path or name from `deck.bodies[k].hydro_database`)
  to a pre-loaded `HydroDatabase`. The driver does NOT itself decide
  which reader to call — that decision is the caller's, and is the
  natural place for OrcaFlex-vs-WAMIT-vs-Capytaine dispatch.
- `dt` and `t_max_kernel` are passed through to
  `compute_retardation_kernel` per body. Per-body kernel `dt` must
  agree (`assemble_global_kernel`'s existing precondition).
- `xi0` carries the **post-equilibrium-solve** position (shape
  `(n_dof,)`) when `solve_equilibrium=True` (default); otherwise the
  deck-stated `InitialConditions` translated into the global
  vector. `xi_dot0` carries the deck-stated velocity IC (typically
  zeros). Both flow directly into `integrate_cummins(xi0=..., xi_dot0=...)`
  with no caller-side unpacking.
- `state_force` is the composed closure summing all connector +
  catenary contributions for this deck.
- Wave-excitation forcing is NOT in `state_force` — wave forcing is a
  `t -> F(t)` callable consumed by `integrate_cummins` via
  `external_force`. The driver may also produce that callable from
  `deck.waves`, but it is returned separately (proposal:
  `SimulationSetup.wave_force` field, optional). This keeps the
  state-force / external-force split clean.

**Open questions for Xabier:**
- `floatsim/driver.py` (single file) or `floatsim/driver/__init__.py`
  (package, anticipating future growth)? Proposal: single file at M7,
  promote to package only if/when the file exceeds ~600 lines.
- Should `bem_databases` be keyed by **body name** (deck field) or
  **hydro_database name** (the BEM file/case name)? Body-name keying
  is simpler at the driver boundary; hydro_database-name keying
  allows two bodies to share a BEM. Proposal: body-name keying for
  M7-Foundation (matches the deck's 1:1 body↔BEM relationship);
  revisit when shared BEM becomes a real case.
- Should `build_system` accept a path string and call `load_deck`
  internally, or accept a pre-loaded `Deck`? Proposal: pre-loaded
  `Deck` only (single responsibility). The caller writes a 2-line
  prologue `deck = load_deck(path); setup = build_system(deck, ...)`.

### Q2 — Body-name → index resolution and the "earth" sentinel

**Proposal: deck-order is index order; "earth" is a magic string.**

- Body `k` in the deck's `bodies:` list occupies global state slots
  `[6k, 6k+6)`. The driver returns a `body_name_to_index` dict for
  the caller's convenience (e.g., extracting body 0's heave channel
  by name rather than index 2).
- Connection endpoints (`LinearSpring.body_a`, `Catenary.body_a`,
  `RigidLink.body_a`) are deck-validated strings. The driver
  resolves them to integer indices for `LinearConnector` /
  `CatenaryAttachment`. The string **"earth"** is reserved and
  resolves to the `-1` sentinel; any other string must match a body
  name (deck validation should catch the mismatch, but the driver
  raises a `ValueError` with a clear message if not).

**Open questions for Xabier:**
- Is "earth" the right magic string? Alternatives: "fixed", "anchor",
  "world". Proposal stays with "earth" (matches the
  `floatsim/io/deck.py:181` `Catenary` docstring's existing usage).
- Should body names be required unique? Pydantic doesn't enforce
  uniqueness on a `list[Body]`. Proposal: enforce at the driver
  boundary with a clear `ValueError` (collect duplicates, raise once).

### Q3 — F2 attachment-offset transform: scope and fidelity

**Proposal: small-angle linear, exact for the linearisation,
hard-error if asked for non-linear range.**

The 6x6 K provided at attachment point `r_attach_body` maps to a
reference-point 6x6 `K_ref` via a small-angle linear pull-back. Let
`r_tilde` be the skew-symmetric cross-product matrix of `r`. By
construction `r_tilde` is skew, so `r_tilde^T = -r_tilde`. We use
`-r_tilde` consistently throughout the derivation.

For a rigid attachment arm `r` (body frame) and small body rotation
`theta` (the Euler 3-vector in `xi[3:6]`), the body-fixed attachment
point translates by::

    delta_attach_trans = delta_ref_trans + theta x r
                      = delta_ref_trans + (-r_tilde) @ theta
                      = delta_ref_trans - r_tilde @ theta

(using the cross-product identity `theta x r = -r x theta = -r_tilde @ theta`).
The attachment-point rotation is the body rotation::

    delta_attach_rot = theta

In 6-DOF block form::

    delta_attach_6 = T @ delta_ref_6
    T = [ I_3   -r_tilde ]   (6x6, the "translation-plus-arm" transform)
        [ 0      I_3      ]

The force at the attachment point is `F_attach = -K_attach @ delta_attach`.
By virtual-work duality, the equivalent generalised force on the
reference point is::

    F_ref = T^T @ F_attach = -T^T @ K_attach @ T @ delta_ref
          = -K_ref @ delta_ref       where K_ref = T^T @ K_attach @ T

Working out `T^T = [I_3, 0; -r_tilde^T, I_3] = [I_3, 0; r_tilde, I_3]`
(applying `r_tilde^T = -r_tilde` once more). The bottom-left
`r_tilde` block is **the moment-arm cross product**: a translational
force at the attachment produces a moment `r x F` about the
reference point. **This block is the whole point of F2.** A
diagonal-only T (dropping the `-r_tilde` and `r_tilde` blocks)
transports force without moment and would silently pass any
force-only test while losing moment fidelity — the PR2 property
test explicitly covers this (see PR2 in the PR sequence).

So `assemble_attachment_transformed_connector` returns a
`LinearConnector` whose `K = T^T @ K_attach @ T` (and analogously for
B). The rest_offset transforms the same way.

**Validity range:** the small-angle linear approximation is exact at
`omega = 0` and accumulates O(theta²) error away. For
`|theta| < 0.1 rad (≈ 5.7°)` the error is < 0.5% (typical Phase 1
regime). Larger angles need the full quaternion-driven transform
(LEVEL2-INTEGRATOR-UNWIRED, Phase 2).

**Open questions for Xabier:**
- Is the small-angle linearisation acceptable for M7-Foundation? It
  matches the integrator's current Level-1 fidelity, so YES is the
  consistent answer. Documenting the bound is mandatory.
- Should F2 also support a non-zero rest position (`r_attach_body`
  at a non-trivial place), or only K transformation? Proposal:
  attach-point-position transformation IS already implicit in the
  `T` matrix derivation; the rest_offset transformation is the
  natural extension. Test it.

### Q4 — F3 catenary state-force composer

**Proposal: a `CatenaryAttachment` dataclass + a state-force builder
that wraps the analytic catenary in the 3D rotation that the M4 PR6 /
M6 PR5 tests currently do by hand.**

```python
# floatsim/mooring/catenary_analytic.py (extends existing module)

@dataclass(frozen=True)
class CatenaryAttachment:
    """One mooring line attached to a body."""
    body_index: int  # -1 for earth (not yet supported; raise)
    fairlead_body: NDArray[np.float64]  # (3,) body-frame attach point
    anchor_global: NDArray[np.float64]  # (3,) inertial-frame anchor
    line: CatenaryLine                  # length, w, EA
    seabed_depth: float                 # for touchdown regime

def make_catenary_state_force(
    attachments: Sequence[CatenaryAttachment],
    n_dof: int,
) -> Callable[[float, ndarray, ndarray], ndarray]: ...
```

The returned closure:
1. For each attachment, extracts body's `xi[6k:6k+6]` and computes
   the inertial-frame fairlead position (small-angle linear, same
   as F2).
2. Forms the 3D vector from fairlead to anchor, projects onto the
   horizontal plane to get the local 2D frame.
3. Calls `solve_catenary(line, anchor_local, fairlead_local,
   seabed_depth)`.
4. Maps the returned `(H, V_fairlead)` back to a 3D force at the
   fairlead in the inertial frame.
5. Translates the 3D force to a 6-DOF generalised force on the body
   reference point (`F_translation, (r_fairlead − r_ref) × F`).
6. Accumulates into the global force vector.

**Modelling assumptions (made explicit, inherited from M6 PR5):**

- **The catenary line lies in the vertical plane containing the
  anchor and fairlead.** Equivalent to "no current, no lateral
  force on the line." Sea current that would deflect the line
  out of the anchor-fairlead plane is out of scope (Phase 2).
- **No line dynamics / dynamic damping.** This is a quasi-static
  analytic catenary; dynamic mooring damping (the MoorDyn-style
  lumped-mass model) is out of scope (Item 26).
- **Explicit-lag treatment.** `make_catenary_state_force` is a
  `state_force` callable, consumed by `integrate_cummins` exactly
  the same way `make_connector_state_force` is: the catenary
  force at step `n` is evaluated at the previous step's
  state `(t_{n-1}, xi_{n-1}, xi_dot_{n-1})`. This O(h) lag is the
  documented integrator convention
  (`docs/openfast-cross-check-conventions.md` Item 11 /
  `floatsim/solver/newmark.py` §"State-dependent force"). The
  PR3 `rtol = 1e-12` identity test against the M6 PR5 hand-wired
  path verifies this invariant: M6 PR5's hand-wired catenary
  force was also lagged one step, so an identity match
  automatically confirms the lag is preserved.

**Open questions for Xabier:**
- Should "earth-attached" catenaries (fairlead on body, anchor on
  earth — the M4 PR6 / M6 PR5 case) be the only supported topology
  in M7-Foundation, with body-to-body catenaries deferred? Proposal:
  YES — no current use case for body-to-body catenaries; the
  geometry / seabed-contact logic gets non-trivial when both
  endpoints move. Raise `NotImplementedError` for the body-to-body
  case until a real fixture demands it.
- What's the failure-mode policy when the catenary has no
  geometrically-valid solution (e.g., fairlead below seabed, zero
  horizontal span)? Proposal: raise `ValueError` from
  `solve_catenary` (existing behaviour), let it propagate. The
  driver wraps with deck-context so the message identifies the
  offending connection.

### Q5 — F4 N = 4 validation test design

**Proposal: four IDENTICAL copies of the M2 heave-only analytical
fixture, block-diagonally stacked, with DISTINCT heave ICs.**

- N = 4 (the smallest "non-trivial" multi-body case beyond what
  M4 PR1 validated at N = 2).
- All four bodies share the same single-body BEM (the M2 narrowband
  `B(omega)` synthetic).
- **Distinct heave ICs per body: 1.0, 0.8, 0.6, 0.4 m.** All other
  DOFs at zero. **The distinct ICs are LOAD-BEARING:** with
  identical bodies AND identical ICs, body k's signal is
  indistinguishable from body j's, so a pack/unpack transposition
  bug (e.g., body 1's state written into body 2's slot in the
  global vector) would PASS every per-body assertion. Distinct ICs
  make the transposition immediately observable as a magnitude
  mismatch on the affected body. This is the pack/unpack
  discriminator and must not be dropped to "simplify" the fixture.
- No connectors (block-diagonal, pure uncoupled).
- Assertions:
  - **(A) Period identity**: each body's heave period matches the M2
    analytical `T_n = 2π√((M+A_inf)/C) = 7.854 s` to `rtol = 1e-2`.
  - **(B) Damping identity**: each body's log-decrement damping
    matches `zeta_n = 0.05` to `rtol = 5e-2`.
  - **(C) Cross-DOF silence (per body)**: each body's surge / sway /
    roll / pitch / yaw history stays at `|xi| < 1e-10 m` (or
    `rad`) **absolute**. Rationale: in a block-diagonal system any
    leakage from the excited heave DOFs into silent DOFs would have
    magnitude `(coupling-fraction) x max_heave_IC = (coupling-fraction)
    x 1.0 m`. The 1e-10 m absolute floor is two decades below
    `1e-8` at the M2 fixture's `M = 1e7 kg` scale (round-off in the
    matrix solve at `np.float64`). Relative-to-silent-DOF-own-IC
    would be meaningless (the silent IC is zero); the relevant
    scale is the **maximum excited-DOF amplitude across all
    bodies**, which is 1.0 m.
  - **(D) IC-scaling identity (pack/unpack discriminator)**: the
    ratio of body k's first heave-peak amplitude to body 0's
    first heave-peak amplitude equals `IC_k / IC_0` (i.e.
    0.8, 0.6, 0.4) within `rtol = 5e-3`. **The specific metric is
    "ratio of first positive peak amplitudes."** Body 0's IC is the
    reference (1.0 m). The tolerance is one decade tighter than
    the period rtol (1e-2) — peak amplitudes are well-defined and
    a transposition would cause a `1.0 m vs 0.8 m` discrepancy,
    far above the floor.
- Static equilibrium pre-step at `n_dof = 24` (with `xi0` = the
  full IC; equilibrium should be trivially zero since `F_state = 0`).
- Sized like `test_m4_two_body_assembly.py` (~150-200 lines).

**Open questions for Xabier:**
- Identical bodies vs heterogeneous? Identical maximises the
  per-body identity assertion; heterogeneous exercises pack/unpack
  more rigorously. Proposal: 4 identical bodies is the right
  starting choice; if F4 surfaces a bug that survives the identical
  case, we'd extend to heterogeneous in a follow-up PR.
- Should F4 also include a heave-coupled connector across two of
  the four bodies (mixed coupled/uncoupled)? Proposal: NO for M7-
  Foundation — F4 is specifically the block-diagonal gate. Connector-
  coupled N >= 3 is a Scenario A enabler (A3 in the audit), out of
  scope.

### Q6 — Per-PR pre-flight prediction contract

**Proposal: every M7 PR carries a Step A / Step B / Step C / Step D
contract per the M6 pre-flight discipline.**

| PR | Step A (predict) | Step B (compare) | Step C (gate) |
|----|------------------|-------------------|---------------|
| PR1 (F4 red) | Hand-derive that 4 uncoupled M2 bodies show 4 independent T_n = 7.854 s decays with first heave peaks scaled by IC = (1.0, 0.8, 0.6, 0.4) | Run the N = 4 integration via hand-wired `assemble_global_lhs([hdb]*4)` + zero connectors; extract all four Q5 assertions A/B/C/D | Q5 (A) `rtol = 1e-2` on period; (B) `rtol = 5e-2` on damping; (C) `atol = 1e-10 m` on per-body silent DOFs (surge/sway/roll/pitch/yaw); (D) `rtol = 5e-3` on first-peak amplitude ratios body_k/body_0 |
| PR2 (F2) | Closed-form derive `K_ref = T^T @ K_attach @ T` for a unit-translation K at a 1.0 m arm; predict the moment block `r x K @ delta` from hand-cross-product | Call `assemble_attachment_transformed_connector` with the same inputs; compare 6x6 matrices AND both translational + moment components of `F_ref` | `rtol = 1e-12` element-wise on K_ref AND on both 3-vector blocks of `F_ref` (force AND moment) |
| PR3 (F3) | Compute the catenary (H, V) at the M6 PR5 OC4 equilibrium offset via direct `solve_catenary` calls (the existing PR5 hand-wired path) | Call `make_catenary_state_force` at the same xi_eq; compare the 6-vector force on body 0 | `rtol = 1e-12` on force components (identity with hand-wired) |
| PR4 (F1) | Predict that `build_system(M4_PR6_deck)` produces an identical `CumminsLHS.M_plus_Ainf, CumminsLHS.C, kernel.K, state_force(0, xi_eq, 0), xi0, xi_dot0` as the hand-wired M4 PR6 setup | Compare driver output to hand-wired output at multiple deck states | `rtol = 1e-12` on matrices and on state-force evaluation; exact equality on `xi0`, `xi_dot0` shapes |

Step D for every PR is the failing assertion that motivates the PR
landing. If Step A and Step B agree at Step C tolerance, the PR is
ready to commit.

### Q7 — Module structure and conventions doc updates

**Proposal:**

- `floatsim/driver.py` — new module (Q1 scope).
- `floatsim/bodies/connector.py` — extended with
  `assemble_attachment_transformed_connector` (Q3 scope). Single
  function; the existing `LinearConnector` dataclass + state-force
  builder are unchanged.
- `floatsim/mooring/catenary_analytic.py` — extended with
  `CatenaryAttachment` dataclass + `make_catenary_state_force`
  (Q4 scope). Existing `solve_catenary` and `CatenaryLine`
  unchanged.
- No changes to `floatsim/solver/state.py`, `floatsim/solver/newmark.py`,
  `floatsim/hydro/radiation.py`, `floatsim/hydro/retardation.py`
  unless F4 surfaces a bug.

**Conventions doc updates** (if F4 surfaces findings):
- New items go into a new `docs/multibody-conventions.md` (rather
  than the OpenFAST cross-check conventions, which is HydroDyn-
  specific). At M7-Foundation close, if no items have accumulated,
  the file is created with a placeholder header anyway, ready for
  Scenario A / B follow-ups.

### Q8 — Pre-foundation audit (CLAUDE.md §15)

**Proposal: before PR1 fires its first assertion, audit the
following code paths.** This is the M7 equivalent of M6 PR1's
HydroDyn pre-flight (Q8 in that plan).

1. **`solve_static_equilibrium` at `n_dof = 24`.** Does
   `scipy.optimize.root(method="hybr")` scale cleanly? The hybr
   default uses a finite-difference Jacobian — fill requires
   `n_dof` residual evaluations (one per perturbed column). If a
   residual eval is `O(n_dof^2)` (matrix-vector products
   dominated by `C @ xi`), then per-Jacobian cost is
   `O(n_dof^3)` flops. At `n_dof = 24` this is ~14k flops per
   Jacobian — negligible. The note here is a **forward-pointing
   warning for the `n_dof = 72` (12-body) reader**: there it
   becomes ~370k flops per Jacobian and is still tractable for
   the residual loop, but `n_dof = 72` with stiffer connectors
   may need many Jacobian recomputes per equilibrium solve. We
   are not solving that here, just establishing the trajectory.
   Pre-flight: run at `n_dof = 6`, 12, 24 with the same physics
   (block-diagonal M2 fixture); record `nfev` and wall-clock so a
   future audit at `n_dof = 72` has a clean scaling baseline.
2. **`np.linalg.solve(A_eff, rhs)` at `n_dof = 24`.** Verify the
   condition number of `A_eff` does not degrade unexpectedly. At
   block-diagonal it must equal the single-body condition number;
   any inflation is a numerical bug.
3. **Explicit-state-force lag at `n_dof = 24`.** With no
   connectors, no state-force is evaluated, so this is trivial. With
   F4's block-diagonal setup the lag does not engage. Re-audit when
   PR4's F1 wires in deck-driven connectors and catenaries.
4. **`assemble_global_lhs` docstring contract.** It says "block-
   diagonal stacking assumes no hydrodynamic coupling between
   bodies." F4 honors this. F1 must propagate the same assumption
   via the driver's BEM-database-per-body input contract.
5. **`assemble_global_kernel` docstring contract.** "Every kernel
   must share the same `dt` and `n_lags`." F4 trivially satisfies
   (identical bodies). F1 must enforce in the driver.
6. **`make_connector_state_force` body-index validation.** "Body
   indices outside `[-1, n_dof // 6)` raise at construction." F1's
   deck-driven path must produce indices that satisfy this; the
   deck has named body references and a connection's
   `body_a / body_b` strings must resolve to in-range integers.
7. **`make_catenary_state_force` (new in F3)** body-index range:
   same precondition. Q4 proposes "earth attachment not yet
   supported"; PR3's audit must confirm the M6 PR5 fixture exercises
   only `body_index = 0` (body-to-earth via anchor_global).
8. **Pack/unpack indexing.** F4's assertion (C) (cross-body silence)
   tests this. Verified by F4's pass.

Items 1, 2, and 6 are the highest-risk. The audit checklist lives in
this Q8 section; each item is marked verified / pending / N/A by
PR1's close.

### Q9 — Scope discipline: what is OUT of M7-Foundation

**Proposal: explicit exclusions, all queued in
[`docs/phase2-followups.md`](phase2-followups.md):**

- **A1** — General 6-DOF rigid-link helper. The `heave_rigid_link`
  helper remains; a general N-DOF rigid link is queued.
- **A2** — Connector drift diagnostic aggregation. Per-DOF drift
  exists; no aggregator change in M7-Foundation.
- **A3** — N = 4 structural-member (connector-coupled) validation.
  F4 is uncoupled-only.
- **B1** — Selective-DOF joint helpers (hinge, ball, prismatic).
- **B2** — Lagrange-multiplier DAE constraint formulation.
- **B3** — LEVEL2-INTEGRATOR-UNWIRED: wiring the quaternion
  integrator into the time-domain solver. The audit's named
  follow-up is logged in `phase2-followups.md`.
- **B4** — Multi-body BEM cross-coupling ingestion.
- **B5** — Coupled retardation kernel transform on multi-body BEM.
- **B6** — Sparsity-aware linear algebra.
- **BB-OFFSET-CONNECTOR** — Body-body `LinearConnector` with
  non-zero attachment offset. Surfaced at PR2 (commit `54703b7`)
  during F2's derivation. See
  [`phase2-followups.md`](phase2-followups.md) entry.

Each line above gets a tracker entry. Reassessment at M7-Foundation
close decides which (if any) of these to promote to a successor
milestone.

**Pinned PR4 (F1) disposition for BB-OFFSET-CONNECTOR
(Xabier, 2026-06-01).** `build_system` raises
`NotImplementedError` on a deck `LinearSpring` with body-body
(neither endpoint earth) AND any non-zero `attach_a_body` or
`attach_b_body`, with a message citing
`docs/phase2-followups.md#bb-offset-connector`. Decks where both
offsets are zero, OR where one endpoint is earth and the other
has a single offset, are supported (F2's locked scope at PR2).

Rationale recorded for the audit trail:

1. **Schema validation should validate schema.** Framework
   limits live where the framework lives, which is `build_system`
   (the consumer of the schema). Pushing them into the deck
   validator buries the constraint in a "your YAML is wrong"
   error rather than "this framework path is open work, here's
   the tracker entry."
2. **Tracker entries are the institutional response to Item 23.**
   Option (b) produces one; option (a) (schema-level rejection)
   does not.
3. **Reversibility.** When the framework extends (Direct or
   B2-derived per the tracker entry), option (b) is a one-line
   removal in `build_system`. Option (a) would require deck
   schema migration.
4. **UX parity.** A clear `NotImplementedError` from
   `build_system` with deck-context in the message gives the
   same operational signal to the user as a schema rejection;
   the "catch at earliest point" argument for (a) is mostly
   cosmetic.

---

## PR sequence

### PR1 — F4 N = 4 block-diagonal validation test (red gate)

Test-first per CLAUDE.md §5. Uses hand-wired `assemble_global_lhs`,
no F1 / F2 / F3 helpers — this PR is intentionally the
**Item 19 code-path exerciser**.

- `tests/validation/test_m7_n4_block_diagonal.py` — Q5 scope, ~180
  lines.
- Per Q8 pre-foundation audit: PR1 also runs the
  `solve_static_equilibrium` scaling check (n_dof = 6, 12, 24) and
  the `np.linalg.solve` condition-number sanity check as part of the
  pre-flight diagnostics. Diagnostic findings (if any) committed at
  `docs/diagnostics/m7-pr1-multibody-scaling.md`.
- ~250 lines including diagnostic doc.

### PR2 — F2 attachment-offset transform

- `floatsim/bodies/connector.py` extended with
  `assemble_attachment_transformed_connector`.
- `tests/unit/test_connector_attachment_transform.py` — Q6 PR2
  pre-flight pinning (closed-form `T^T @ K @ T` identity at
  rtol = 1e-12 for unit-translation K, rotation-translation
  coupling, and rest_offset transformation).
- **Moment-transfer property test (`hypothesis`):** for random
  body-frame K, random attachment offset `r != 0`, random small-angle
  `xi`, the transformed connector's full 6-DOF generalised force
  (translational components `F_ref[0:3]` AND moment components
  `F_ref[3:6]`) at the reference point equals the un-transformed K's
  force at the attachment, pulled back via `T^T`. Both blocks of
  `F_ref` are asserted at rtol = 1e-12. **A translational-only
  assertion would pass against a T that drops the `-r_tilde` /
  `r_tilde` blocks; the moment block is the whole point of F2 (Q3)
  and gets its own explicit assertion.**
- Discriminator case: a translational K at a 1.0 m arm under a unit
  attachment-point translation must produce a moment of exactly
  `r x K @ delta` at the reference. Hand-derived, rtol = 1e-12.
- ~250 lines.

### PR3 — F3 catenary state-force composer

- `floatsim/mooring/catenary_analytic.py` extended with
  `CatenaryAttachment` + `make_catenary_state_force`.
- `tests/unit/test_catenary_state_force.py` — Q6 PR3 pre-flight
  pinning (identity with hand-wired `solve_catenary` calls at the
  M6 PR5 OC4 fixture geometry, rtol = 1e-12).
- Refactor `tests/validation/test_m6_openfast_moored_eq.py` to use
  `make_catenary_state_force` instead of the hand-wired per-line
  closure. The cross-check assertions are unchanged; this is a
  composition refactor only. Re-running the M6 PR5 suite must
  reproduce the existing sub-0.15% tension agreement to the same
  precision.
- ~300 lines.

### PR4 — F1 driver `build_system`

- `floatsim/driver.py` — Q1 scope, ~400 lines.
- `tests/unit/test_driver_build_system.py` — Q6 PR4 pre-flight
  pinning. Tests:
  - Round-trip identity: `build_system(M4_PR6_deck, ...)` →
    `(lhs, kernel, state_force, initial_state)` identical to the
    hand-wired M4 PR6 setup at rtol = 1e-12 on matrices and
    rtol = 1e-12 on `state_force(0, xi_eq, 0)`.
  - Body-name → index resolution honors deck order.
  - "earth" sentinel resolves to -1.
  - Duplicate body names raise with a clear message.
  - Unknown connection endpoint raises.
- Refactor `tests/validation/test_m4_two_body_moored.py` and
  `tests/validation/test_m7_n4_block_diagonal.py` (from PR1) to
  use `build_system` (the deck-driven path). Existing assertions
  unchanged. Existing pass criteria preserved (no tolerance
  loosening).
- ~600 lines.

### PR5 — M7-Foundation closure

- `docs/m7-foundation-closure.md` — audit-trail record paralleling
  [`docs/m6-closure.md`](m6-closure.md). Cross-references PR1-PR4
  retrospectives + any post-mortems.
- `docs/multibody-conventions.md` — placeholder file if no
  conventions items accumulated; populated entries if F4 / PR2 /
  PR3 / PR4 surfaced any.
- Post-mortems in `docs/post-mortems/` for any latent bugs
  surfaced by the Item 19 PR1 framing.
- Plot regeneration (if applicable) into
  `docs/figures/m7-foundation/`.
- Decision-point reassessment: should the project promote any of
  A1-A3 / B1-B6 to a successor milestone, or pause for
  user-driven study work? This goes to Xabier as a separate review.
- ~200 lines.

---

## Ordering rationale

PR1 first by Item-19 design — the user's explicit framing is that F4
should "surface whatever breaks in the size-agnostic-but-untested
solver/integrator at N >= 3." Putting F4 LAST would conflate
"did F1/F2/F3 introduce a bug" with "did N >= 3 surface a latent
bug." PR1 lands the red gate with the existing code paths
unchanged.

PR2 (F2) and PR3 (F3) are independent helpers. Either order works.
PR2 before PR3 is proposed because F2 is the smaller surface area
(pure 6x6 linear algebra) and its pre-flight has zero geometric
edge cases — useful as a warm-up for the conventions-discipline
muscle in the milestone.

PR4 (F1) depends on PR2 (uses the F2 helper for `LinearSpring`
entries) and PR3 (uses the F3 helper for `Catenary` entries). PR4
is the deck-driven composition driver, the highest-leverage piece
per the audit's framing.

PR5 closes the milestone with the audit-trail doc + any
post-mortems + reassessment.

## Risks

- **F4 latent-bug surfacing.** This is the *expected* outcome of the
  Item-19 framing, not a risk per se. The risk is that whatever
  surfaces requires a Scenario-A or Scenario-B enabler to fix; if
  so, M7-Foundation pauses and we open a tracker entry rather than
  expanding scope.
- **F2 small-angle linearisation bound.** Documented as
  `|theta| < 0.1 rad` for <0.5% error. If a future fixture demands
  10°+ rotations under F2 transforms, the linearisation is
  inadequate — that's LEVEL2-INTEGRATOR-UNWIRED territory.
- **F3 catenary edge cases.** The existing `solve_catenary` handles
  suspended + touchdown regimes. Edge cases (fairlead at seabed,
  zero horizontal span, line too short for static reach) raise
  `ValueError` from the existing solver; F3 propagates with deck
  context. The PR3 test set must exercise at least one of each.
- **F1 API lock-in.** `build_system` becomes the public composition
  entry point. Getting the signature wrong (positional vs keyword,
  the `bem_databases` keying choice in Q1) makes the API painful to
  change later. Q1 proposes the contract; review locks it.
- **Deck schema → runtime semantics coverage.** The deck has
  `LinearSpring`, `Catenary`, `RigidLink` connection types. F1 must
  cover all three:
  - `LinearSpring` → `LinearConnector` via F2 (attachment
    transform).
  - `Catenary` → `CatenaryAttachment` via F3.
  - `RigidLink` → `heave_rigid_link` (the existing helper). The
    deck schema is heave-only at M4 PR3 (Q9 A1 deferred); F1 does
    not promote.
  PR4's tests must exercise all three connection types.
- **Single-body decks still work.** The driver must produce sane
  output at `N = 1` (single-body, no connections). M2-M6 cross-
  check decks fall into this category. PR4's tests must include a
  single-body case (OC4 marin_semi unmoored) to confirm.

## Session-continuity notes

If a fresh session picks this up: M6 closed at commit `63b04e1` on
`main` (the "M6 PR4 -- flip xfail markers post-F-WAVE-FORCE-CONV
closure" tip). The M6 epilogue branch
`fix-make-regular-wave-force-convention` is **FF-merged into main**
at commit `3f84d7b` (an ancestor of `63b04e1`); both M6 PR4/PR5/PR6
and the convention fix are in main's linear history. `main` and
`origin/main` agree at `63b04e1`. The M6 closure document is at
`docs/m6-closure.md`. M7-Foundation has no code yet, only this
plan + the Phase 2 tracker (`docs/phase2-followups.md`). Both
planning docs are committed directly to `main` (project-level
artifacts, paralleling the M6 closure doc's landing). The branch
`milestone-7-foundation` is created off `main` at the head once
review locks Q1-Q9.

The audit motivating M7-Foundation is
[`docs/audits/multibody-capability-audit.md`](audits/multibody-capability-audit.md);
the Phase-2 tracker that holds the deferred items is
[`docs/phase2-followups.md`](phase2-followups.md). M6 plan
[`docs/milestone-6-plan.md`](milestone-6-plan.md) is the style
reference for this document.

Q1-Q9 above are **proposals** — do not implement until Xabier has
reviewed and locked them. The first implementation action is PR1's
red test (F4 N = 4 block-diagonal), preceded by the Q8 pre-foundation
audit committed as a diagnostic doc.
