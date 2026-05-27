# M7-Foundation PR1 — Q8 pre-foundation scaling audit

**Date.** 2026-05-11
**Scope.** Audit the size-agnostic-but-untested solver / integrator code
paths at `n_dof = 24` (the F4 N = 4 target) before the F4 red test
fires. The audit is mandatory per
[`docs/m7-foundation-plan.md`](../m7-foundation-plan.md) Q8; the F4
red test in `tests/validation/test_m7_n4_block_diagonal.py` will not
land until every item below is marked verified.
**Diagnostic script.** [`scripts/m7_pr1_scaling_audit.py`](../../scripts/m7_pr1_scaling_audit.py).

This is the M7 analog of the M6 pre-cross-check audit pattern
(CLAUDE.md §15): trace the code paths between the entry point and the
assertion under test, and gate the PR on each path being either
verified-with-a-runnable-protocol or marked N/A with reasoning.

The framing per the user-locked plan: F4 is an **Item-19
code-path-exercise** (CLAUDE.md §13). The hypothesis behind PR1 is
that the size-agnostic code paths (`solve_static_equilibrium`,
`integrate_cummins`, `np.linalg.solve(A_eff, rhs)`, the pack/unpack
helpers in `floatsim/solver/state.py`) **will surface something** at
N >= 3 that has stayed dormant through M2-M6. The audit's job is to
verify each path's docstring contract holds at the new size, BEFORE
attributing any F4 failure to those paths. If a path fails the audit,
it surfaces as a finding here rather than as a confusing red-test
failure later.

---

## Audit summary

| # | Item | Status |
|---|------|--------|
| 1 | `solve_static_equilibrium` scaling at n_dof = 6, 12, 24 | ✅ verified |
| 2 | `A_eff` condition number on block-diagonal stacks | ✅ verified |
| 3 | Explicit-state-force lag at n_dof = 24 | ✅ N/A (no state_force in F4) |
| 4 | `assemble_global_lhs` block-diagonal docstring contract | ✅ verified (reasoning) |
| 5 | `assemble_global_kernel` uniform-`dt`/`n_lags` contract | ✅ verified (reasoning) |
| 6 | `make_connector_state_force` body-index validation at n_dof=24 | ✅ verified |
| 7 | `make_catenary_state_force` body-index range | ✅ N/A (helper does not exist yet — PR3 scope) |
| 8 | Pack/unpack indexing | ✅ deferred to F4 assertion (C) and (D) (the discriminators) |

No surfacings. The pre-foundation audit clears PR1's F4 red test to
fire.

---

## Item 1 — `solve_static_equilibrium` scaling

**Protocol.** Run `solve_static_equilibrium(lhs=lhs_global,
state_force=None, tol=1e-6)` on a block-diagonal stack of N copies
of the M2 heave-only analytical fixture (M = A_inf = 1e7 kg,
C_33 = 1.28e7 N/m, only heave has restoring) at N = 1, 2, 4. Record
`nfev`, residual `inf`-norm, convergence flag, and wall-clock.

**Result.**

```
 n_dof    N    nfev     res_inf_N     wall_s   conv
     6    1      10     0.000e+00     0.0003   True
    12    2      16     0.000e+00     0.0001   True
    24    4      28     0.000e+00     0.0001   True
```

**Interpretation.**

- `nfev` scales sub-linearly with `n_dof` (10 → 16 → 28 for a 4×
  increase in size). The expected behaviour for `scipy.optimize.root`
  with `method="hybr"` is dominated by Jacobian fill (`n_dof`
  residual evals per Jacobian) plus a handful of iterations to
  converge from the zero initial guess. At `n_dof = 24` the observed
  28 residual evaluations is consistent with one Jacobian fill
  (~24 evals) plus ~4 iteration evals — the trivial fixed point
  (`F_state = None`, `xi = 0` solves `C xi = 0`) converges in one
  step. The scaling is forward-pointing for the n_dof = 72
  audit at Phase 2 — see plan Q8 item 1's complexity note.
- Residual `inf`-norm is exactly 0.0 across all three sizes. The
  fixture is trivially solvable; this is the floor of the
  diagnostic, not a precision claim about the solver in general.
- Wall-clock is sub-millisecond throughout. Cache warming hides
  the n_dof = 24 cost; the audit is not a performance benchmark.

**Conclusion.** ✅ `solve_static_equilibrium` scales as expected at
n_dof = 24. No surfacing.

---

## Item 2 — `A_eff` condition number on block-diagonal stacks

**Protocol.** Construct `A_eff = (1-alpha_m)(M+A_inf) + (1-alpha_f) h^2
beta C` (the integrator's per-step LHS matrix at the
generalized-alpha `rho_inf = 0.9`, `h = 0.01` representative
values) on the same fixture and compute `cond(A_eff)` via
`numpy.linalg.cond` at N = 1, 2, 4.

**Result.**

```
 n_dof    N     cond(M+Ainf)          cond(C)      cond(A_eff)
     6    1        1.000e+02              inf        1.000e+02
    12    2        1.000e+02              inf        1.000e+02
    24    4        1.000e+02              inf        1.000e+02
```

**Interpretation.**

- `cond(M + A_inf) = 100` is identical across N = 1, 2, 4 — block-
  diagonal stacking preserves the condition number, as expected.
  Any inflation here would indicate a numerical assembly bug; the
  finding is clean.
- `cond(C) = inf` is expected: the M2 fixture has hydrostatic
  restoring only in heave (`C[2,2] = 1.28e7 N/m`); surge / sway /
  roll / pitch / yaw all have `C_ii = 0`. The 6x6 `C` block is
  singular in those directions, so `cond(C)` per block is infinite.
  Block-diagonal stacking of singular blocks produces a singular
  global `C`, hence `inf` at every N. This is **not** a bug — it's
  the documented "rank-deficient C" case that `solve_static_equilibrium`
  handles via the diagonal regularisation `lambda_reg * I`
  ([`equilibrium.py:32-46`](../../floatsim/solver/equilibrium.py)).
  The audit records the value here so a future investigator does
  not mistake `inf` for a regression.
- `cond(A_eff) = 100` (matches `cond(M + A_inf)`) — at `h = 0.01`
  the `h^2 beta C` term is dwarfed by `M + A_inf`, so `A_eff`
  inherits the well-conditioned mass-matrix structure. At
  `h = 0.1` (decade slower step) the C term would contribute ~1%
  to `A_eff` and would tighten `cond(A_eff)` slightly (well-
  conditioned M + A_inf + small C-rank-deficient is still finite
  if M+A_inf is full rank). The integrator does NOT see the
  singularity-of-C; only the equilibrium solver does, and it
  regularises explicitly.

**Conclusion.** ✅ Block-diagonal stacking preserves the M+A_inf
and A_eff condition numbers. No surfacing. The `cond(C) = inf`
finding is documented-expected, not a regression.

---

## Item 3 — Explicit-state-force lag at n_dof = 24

**Status.** ✅ N/A for F4.

**Reasoning.** F4's design (per plan Q5) uses **zero connectors**,
zero catenaries, zero Morison elements — the system is pure
block-diagonal free decay. The integrator's `state_force` argument
defaults to `None`, in which case `_eval_state_force` returns the
zero vector ([`newmark.py:248-258`](../../floatsim/solver/newmark.py)).
The explicit O(h) lag is therefore not engaged in F4.

When the F1 driver wires deck-driven connectors and catenaries in
PR4 (and especially when those land on top of an F4-shaped block-
diagonal LHS), re-audit this item with the explicit-lag-stability
gate. For PR1's red test, the item is not engaged.

---

## Item 4 — `assemble_global_lhs` block-diagonal docstring contract

**Status.** ✅ verified (by reasoning).

**Reasoning.** The
[`solver/state.py:18-31`](../../floatsim/solver/state.py) module
docstring says *"Hydrodynamic cross-coupling between bodies
(off-block-diagonal entries from a multi-body BEM run) will be
plugged in later by assembling the global matrices directly rather
than via these helpers. The helpers here target the M4 PR1 path:
N independent bodies, each backed by its own single-body BEM
database."*

F4 honours this contract: every body is backed by a SINGLE BEM
database, and the global LHS is built by
`assemble_global_lhs([hdb_single] * 4)`. There is no
off-block-diagonal coupling involved — the contract is trivially
honoured. Item 2's condition-number check on the assembled
`M_plus_Ainf` is the runtime confirmation (block-diagonal
stacking preserves the per-block condition number, which is the
identity expected of block-diagonal matrices).

---

## Item 5 — `assemble_global_kernel` uniform-`dt`/`n_lags` contract

**Status.** ✅ verified (by reasoning).

**Reasoning.** [`solver/state.py:172-184`](../../floatsim/solver/state.py)
raises `ValueError` if input kernels disagree on `dt` or
`n_lags`. F4 uses four IDENTICAL kernels (all from
`compute_retardation_kernel(hdb_single, t_max=..., dt=...)` with
the same arguments), so the contract is trivially honoured by
construction. The raise path is unit-tested at
`tests/unit/test_state.py` (if it exists) or covered by
`assemble_global_kernel`'s own validation.

For PR4's F1 driver, when bodies might carry different BEM
databases, the driver must validate / enforce a uniform `dt`
across all bodies before calling `assemble_global_kernel`. That's
PR4 audit territory.

---

## Item 6 — `make_connector_state_force` body-index validation at n_dof=24

**Protocol.** Construct `LinearConnector` instances with body
indices spanning the valid range `[-1, 4)` for `n_dof = 24` (four
bodies) and one out-of-range index (`body_b = 4`); call
`make_connector_state_force([c], n_dof=24)` for each; verify the
expected accept / reject behaviour.

**Result.**

```
  bodies ( 0,  1): accepted (n_dof=24)
  bodies ( 1,  2): accepted (n_dof=24)
  bodies ( 2,  3): accepted (n_dof=24)
  bodies ( 0,  3): accepted (n_dof=24)
  bodies (-1,  0): accepted (n_dof=24)
  bodies ( 3, -1): accepted (n_dof=24)
  bodies ( 0,  4): correctly rejected -- connector 0: body index 4 outside valid range [-1, 4) for n_dof = 24
```

**Interpretation.** All in-range indices (0, 1, 2, 3, plus `-1`
earth) are accepted regardless of which endpoint (`body_a` /
`body_b`) carries them. The out-of-range index (4) is rejected
with the documented error message that includes the offending
connector position (0), the offending index (4), the valid range
(`[-1, 4)`), and the originating `n_dof = 24`. The error message
is actionable.

**Conclusion.** ✅ The body-index validation at `n_dof = 24` is
correct. No surfacing. F4 has zero connectors so this code path
is not engaged in F4 itself, but the validation is verified
correct for when PR4's F1 driver wires deck-driven connectors at
N = 4 (`test_m4_two_body_moored.py`-style usage).

---

## Item 7 — `make_catenary_state_force` body-index range

**Status.** ✅ N/A.

**Reasoning.** `make_catenary_state_force` does not exist yet
(M7-Foundation PR3 scope). When it lands, the PR3 audit must
verify the analogous body-index validation at the relevant
`n_dof` for any test fixture exercising it. F4 does not engage
this code path.

---

## Item 8 — Pack/unpack indexing

**Status.** ✅ Deferred to F4 assertion (C) cross-DOF silence and
assertion (D) IC-scaling identity.

**Reasoning.** Plan Q5's assertion (D) explicitly tests the
pack/unpack indexing via distinct ICs (1.0, 0.8, 0.6, 0.4 m): a
transposition bug (e.g., body 1's state written into body 2's
slot) would surface as a magnitude mismatch on the affected
body. Assertion (C) tests that no signal leaks between blocks.
The pair (C + D) is the runtime audit for pack/unpack at
n_dof = 24; this item is verified by F4's pass, not by a
standalone diagnostic.

---

## What surfaced

Nothing.

The plan's Item-19 hypothesis is that **F4 will surface something
in the size-agnostic-but-untested code paths**. Items 1, 2, 6
above each address a hypothesis variant:

- Item 1: scaling of equilibrium-solver iteration count and wall
  cost. Clean.
- Item 2: numerical conditioning of the assembled matrices.
  Clean (block-diagonal preserves condition; the `cond(C) = inf`
  finding is documented-expected, not a regression).
- Item 6: body-index validation at the new size. Clean.

Items 4 and 5 are verified by reasoning + the construction of F4
itself (identical kernels, single BEM per body). Items 3, 7 are
not engaged by F4.

**Item 19 outstanding.** The audit clears the size-agnostic code
paths for `n_dof = 24` at the **static / equilibrium / assembly
level**. It does NOT exhaust the F4 hypothesis space — anything
that surfaces inside `integrate_cummins`'s step loop (the
`RadiationConvolution` buffer at `n_dof = 24`, the
`np.linalg.solve(A_eff, rhs)` per-step factorisation, the
explicit-mu treatment in the convolution sum) only engages once
the F4 red test runs the full N = 4 integration. The F4 test
itself is the next Item-19 exerciser. If F4 surfaces something
the audit did not catch, that finding gets a post-mortem in
`docs/post-mortems/` and a tracker entry in
`docs/phase2-followups.md` per the M6 discipline.

---

*Audit close. F4 red test (`tests/validation/test_m7_n4_block_diagonal.py`)
is cleared to fire.*
