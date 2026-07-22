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

---

# PR2 Step A — excitation phase-convention audit

**Per plan PR2 Step A** (the M6 Item-16 JAxCd-class lesson: a hidden
convention factor is invisible until real data exercises it). Audited
before the multi-body reader path was written, against the real
spar-fin Capytaine output (`studies/spar-fin-decay/capytaine_bem.nc`).

## Convention under test

| | convention |
|---|---|
| Capytaine (native) | `x(t) = Re[X e^{-i omega t}]` — **lags** |
| FloatSim `HydroDatabase.RAO` | `F(t) = Re[X A_wave e^{+i omega t}]` — **leads** |
| translation | `RAO_floatsim = conj(F_capytaine)` — `capytaine.py`, `_extract_excitation` |

## Measured results

| check | result | reads |
|---|---|---|
| conjugation applied exactly | `max\|RAO − conj(raw)\| = 0.0` | translation is exact, not approximate |
| conjugation is not a no-op | `max\|RAO − raw\| = 1.2414` | the conj genuinely fires (a symmetric-phase fixture could hide this) |
| **no hidden magnitude factor** | `\|RAO\|/\|raw\|` = **1.0** (min = max = 1.0) | pure phase operation; no rho/g/area scaling sneaks in |
| **physical magnitude guard** | long-wave heave `\|F\|/C33` = **0.9977** at omega = 0.1, rising toward 1 as omega -> 0 | independent physics check |

The last row is the Item-16-class guard. In the long-wave limit the
heave excitation per unit wave amplitude tends to `rho g A_wp` (the
wave lifts the body hydrostatically), which is exactly `C33`. Measuring
`0.9977` confirms **no missing rho, g, or waterplane-area factor**
anywhere in the chain — the failure mode that Item 16 recorded.

## Consequence for the multi-body path (PR2 Step B)

`_extract_radiation` and `_extract_excitation` are **permutation-length
generic** — they index with `dof_perm` and never hardcode 6. The
multi-body path therefore reuses them unchanged, with a `6N`
permutation. This is deliberate: **the conjugation cannot diverge
between the single-body and multi-body paths, because there is only one
copy of it.** Only three functions needed generalizing
(`_resolve_dof_permutation`, `_extract_hydrostatic`, `_resolve_a_inf`),
none of which touch phase.

## Standing finding — pre-existing `mypy --strict` errors

`mypy --strict floatsim/hydro/readers/capytaine.py` reports **4 errors,
all pre-existing and unchanged by M8** (verified by stashing the PR2
diff and re-running: identical set, the capytaine one merely
line-shifted 446 -> 516 by added code):

- `readers/wamit.py:228,229,230` — `**dict[str, float]` passed where
  `bool` expected (3 errors).
- `readers/capytaine.py` `_resolve_a_inf` — `Returning Any from
  function declared to return ndarray[float64]` (1 error).

M8 introduces **zero** new type errors. These are **not** fixed here:
CLAUDE.md §9 forbids mixing unrelated refactors into a feature PR, and
CLAUDE.md §3 nominally requires `mypy --strict` to pass on `floatsim/`,
so this is a genuine pre-existing debt. Recorded here rather than
silently absorbed; it wants its own `fix-` branch.

---

# PR3 Step A — coupled-kernel gate audit

**Per plan PR3 Step A.** Confirms the Item-25 override semantics and
enumerates the Q3 gate changes, with the PSD tolerance re-derived from
the committed fixtures (measure-before-threshold).

## Override semantics (Q3 i)

`compute_retardation_kernel`'s `asymptote_check_override` is **per-call
and global** to the computation (retardation.py:346/370): a non-None
rationale skips `_validate_input_gates` entirely and zero-fills the
tail; Check 3 (`_validate_kernel_decay`) still runs. There is no
per-entry override API and the lock says not to build one. **Confirmed
already correct — no change.**

## 6-hardcodes to generalize (all -> n_dof = B.shape[0])

- prepend `np.zeros((6, 6, 1))` (retardation.py:344)
- `_validate_input_gates`: `diag_max` loop and both `range(6)` loops
  (retardation.py:410,419,454-457)
- `_validate_kernel_decay`: `range(6)` diagonal loop (retardation.py:517)

`filon_trap_cosine` already broadcasts over leading axes, so the kernel
integral itself is size-generic.

## Sign handling (Q3 ii) — there is no existing B>=0 check

Neither `_validate_input_gates` (std/mean asymptote) nor
`_validate_kernel_decay` (|K| decay) assumes non-negativity today. So
Q3(ii) is "add the sign invariant in its correct form", not "un-break"
an over-strict one. The **PSD check is that correct form**: a PSD
matrix has non-negative diagonal (subsuming `B_ii >= 0`) while placing
**no constraint on off-diagonal signs** -- exactly what the measured
`B_cross` range (down to -3.99e-04) requires. A separate per-entry
diagonal check would be redundant with PSD.

## PSD check (Q3 iii) — measured tolerance + scope

Eigenvalues of the symmetrized `B(omega)` must be `>= -tol`, but only
where `B(omega)` is **physically significant** -- the same noise/noise
lesson as the excitation gate. Measured min-eigenvalue / global-max|B|:

| fixture | 1% floor | 5% floor |
|---|---|---|
| 18-DOF cluster | -6.12e-6 | -6.12e-6 |
| spar-fin single | **-7.15e-2** | -6.26e-6 |
| marin_semi | +2.4e-7 | +6.3e-5 |

The spar-fin `-7%` at the 1% floor is entirely the omega=20.9 tail
(B there is ~4% of peak -- the known high-omega mesh-resolution noise);
it vanishes at a 5% significance floor. **Locked values:**
`_PSD_SIGNIFICANCE_FLOOR = 0.05` (only enforce where
`max|B(omega)| >= 5% * global max|B|`); `_PSD_REL_TOL = 1e-3` (~150x
margin over the measured ~6e-6 residual, still catching gross
non-PSD).

**Scope: multi-body only** (`hdb.body_labels is not None`). The
existing single-body kernel tests build *random symmetric* B (e.g.
`test_retardation_kernel.py:96`), which is non-PSD by construction; an
unconditional PSD gate would break them. This matches the plan's
framing ("the multi-body generalization of B >= 0") and puts the check
exactly where the block-misassembly risk it guards against lives. The
single-body path retains no PSD check, as before.

## PR3 Step C — the PSD gate caught a real BEM defect (finding)

When the gate first ran on the **production-grid** 18-DOF fixture it
fired. Diagnosis (Steps 1-2 of the Step-C redirect,
`defect_diagnostic.py`): a **whole-matrix contaminated frequency slice
at ω≈4.934** on the cluster-draft spar hull — a near-singular
boundary-integral solve, **not** a lid-removable irregular frequency.
Full characterization and the M11 detection-gap note live in tracker
`BEM-CONTAMINATED-FREQUENCY-SLICE-CLUSTER-DRAFT`
(`docs/phase2-followups.md`, committed to `main`). Step C was
re-specified (positive gate on the contaminated-ω-excluded grid;
permanent negative gate on the unmodified fixture); see the plan.

**Falsified hypothesis (recorded, not struck — it never reached this
doc; it lived only in a working note).** An early guess framed ω≈4.934
as the spar's internal free-surface irregular frequency via the
vertical-cylinder condition `ka ≈ j₀₁ = 2.405`. **Falsified by
measurement:** the lid waterplane centroid radius is **0.063 m**
(`defect_diagnostic.py`, `generate_lid` faces_centers), so at ω=4.934
(`k = ω²/g = 2.48 m⁻¹`) **ka ≈ 0.16** — two orders below 2.405. The
lid workflow is separately confirmed sound: it removes the *genuine*
Capytaine-flagged irregular frequency at ω≈20.909 (surge −1.294 →
+6.144) but leaves ω=4.934 **unchanged** (−0.0856 → −0.0856). Mechanism
**unknown**; see the tracker.
