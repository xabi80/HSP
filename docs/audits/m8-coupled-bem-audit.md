# M8 pre-milestone audit — coupled multi-body BEM

**Per plan Q7 lock** ([`m8-coupled-bem-plan.md`](../m8-coupled-bem-plan.md)),
formalised before PR1, following the M7.5 precedent (audit before the
first PR). Content is the Phase-1 Measurement C reads, re-derived from
the repo at drafting time (2026-07-17), plus the PR1 Step-A
construction-site classification.

**Purpose.** Establish exactly which code assumes a single body, so the
N-body extension (Q1) can be scoped as a data-model change rather than
a solver change — and so the legacy path can be shown untouched.

---

## 1. What is already N-body generic

The M7-Foundation F4 work (N = 4 block-diagonal) left the solver stack
size-agnostic. These need **no M8 change**:

| component | evidence |
|---|---|
| `CumminsLHS` | `n_dof` / `n_bodies` properties (radiation.py:77,83); validates `n_dof = 6N` |
| `RetardationKernel` | validates `n_dof = 6N` for `N >= 1` (retardation.py:138) |
| `integrate_cummins` | "agnostic to the number of bodies" (newmark.py:15); `n_dof` read from `lhs` |
| `assemble_global_lhs` | block-diagonal stacking, body `k` at `[6k:6k+6]` (state.py:142-144) |
| `solve_static_equilibrium` | operates on `lhs.C` at whatever size (equilibrium.py:89) |

**Consequence:** M8 is concentrated in the ingestion layer. The
integrator does not change.

## 2. What assumes a single body (the M8 work)

| component | assumption | file:line |
|---|---|---|
| `HydroDatabase.__post_init__` | hardcoded `(6,6,n_w)` / `(6,6)` / `(6,n_w,n_h)` shape validators | database.py:187-196 (pre-PR1) |
| `compute_retardation_kernel` | consumes a 6×6 database; returns `(6,6,N_t)` | retardation.py:210 |
| `build_system` | `bem_databases: dict[str, HydroDatabase]` — one single-body db **per body**, composed **block-diagonally only**; no coupled path | driver.py:398,446,454-463 |

## 3. Construction-site census (PR1 Step A)

`git grep "HydroDatabase("` returns **31 hits**, of which **29 are
actual constructions**; the other 2 are *test function names*
(`test_read_wamit_returns_valid_HydroDatabase`,
`test_marin_semi_trimmed_loads_into_HydroDatabase`) in
`tests/unit/test_wamit_reader.py:344,561`.

> **Correction to the plan.** `m8-coupled-bem-plan.md` cites "31
> construction sites"; that is the raw grep-hit count. The re-derived
> construction count is **29**. The classification conclusion is
> unchanged (all 29 are single-body, all survive the extend). Recorded
> here per the program plan's re-derivation rule.

| area | file | constructions |
|---|---|---:|
| production reader | `floatsim/hydro/readers/capytaine.py:202` | 1 |
| production reader | `floatsim/hydro/readers/orcaflex_vessel_yaml.py:150` | 1 |
| production reader | `floatsim/hydro/readers/wamit.py:243` | 1 |
| test support | `tests/support/synthetic_bem.py:116` | 1 |
| unit | `tests/unit/test_excitation.py:30` | 1 |
| unit | `tests/unit/test_hydro_database.py` (14 sites) | 14 |
| unit | `tests/unit/test_retardation_kernel.py:109,301` | 2 |
| unit | `tests/unit/test_retardation_kernel_extension.py:79,104` | 2 |
| validation | `tests/validation/test_m3_regular_wave_steady_state.py:114,337` | 2 |
| validation | `tests/validation/test_m6_openfast_drag_decay.py:192` | 1 |
| validation | `tests/validation/test_m6_openfast_free_decay.py:300` | 1 |
| validation | `tests/validation/test_m6_openfast_regular_wave.py:253` | 1 |
| validation | `tests/validation/test_oc4_heave_free_decay.py:91` | 1 |
| **total** | | **29** |

### Breakage classification: extend vs new class

**All 29 sites construct single-body databases using keyword arguments
only** (no positional construction anywhere).

- **Under the LOCKED extend (Q1)** — `body_labels` is a new field with
  default `None`, declared **last** (after `metadata`, which already
  has a default). All 29 sites omit it, take the default, and hit the
  legacy branch, whose shape checks are the pre-M8 code verbatim.
  **Breakage: zero, structurally** — not merely "tests pass".
- **Under the rejected separate-class alternative** — the 29 sites are
  likewise untouched, but every *consumer*
  (`compute_retardation_kernel`, `build_system`,
  `assemble_cummins_lhs`) would need a type branch. The extend moves
  the branch to one place (the data model) instead of N places.

This census is the evidence for the Q1 lock: the decision is driven by
**consumer count**, not aesthetics.

## 4. Risk carried into PR1

The legacy path must be bit-identical. The plan's red gate
(`marin_semi` byte-diff + full suite unchanged against the 662 pre-M8
baseline) is **confirmation**, not the guarantee — the guarantee is
that the `body_labels is None` branch contains the original code
unmodified. Both are recorded in the PR1 commit.

## 5. Baseline validity note

The 662 full-suite baseline was measured at M7.5 PR4-A. Between that
measurement (`3a6d00f`) and the PR1 branch point (`11b2d8a`) the only
change under `floatsim/` or `tests/` is a **docstring-only** edit to
`floatsim/hydro/mesh_hygiene.py` (`907a2b2`, `VolumeReport.signed_volume`
clarification; verified: every changed line is inside the docstring).
The baseline therefore still applies without a re-run.
