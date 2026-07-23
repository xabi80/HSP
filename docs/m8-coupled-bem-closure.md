# M8 — Coupled multi-body BEM: closure

**Milestone:** M8 (first of the Tier 3 program,
[`tier3-program-plan.md`](tier3-program-plan.md)); plan
[`m8-coupled-bem-plan.md`](m8-coupled-bem-plan.md) (Q1–Q8 locked
2026-07-17). Branch `milestone-8-coupled-bem`; closure drafted
2026-07-21 (PR4 Phase A), committed at PR4 Phase B.

---

## S1 — Scope and deliverables

**Goal (plan):** ingest coupled multi-body BEM — the N-body
`HydroDatabase` (B4), coupled retardation kernels (B5), and per-body
excitation — with the consuming machinery exercised by condensation
tests, not new solver features.

| PR | commit | size | closed |
|---|---|---|---|
| PR1 — N-body data model | `a1399b2` | 3 files, +291/−12 | `body_labels: tuple[str, ...] \| None` on `HydroDatabase`; `None` = legacy branch (pre-M8 code verbatim); labels = 6N shapes; `n_bodies` from labels, never shape arithmetic. Tracker **B4** (with PR2). |
| PR2 — reader multi-body path | `14a447e` | 5 files, +277/−24 | `read_capytaine` multi-body detection by **distinct body-prefix count**; 6N permutation in first-appearance order; excitation ingested through the same (permutation-generic) extraction as single-body. Tracker **B4** (with PR1). |
| PR3 — coupled kernels + PSD gate | `2d59907` | 7 files, +752/−54 | `compute_retardation_kernel` generalized to 6N×6N (four 6-hardcodes → `n_dof`); **new multi-body PSD gate** on `B(ω)` (Q3 iii). Step C re-specified mid-PR (S4). Tracker **B5**. |
| PR4 — condensation gates + closure | *(this commit)* | — | `T`-map with the **Q5 label contract** (`tests/support/condensation.py`); both terminal gates as permanent tests (`tests/validation/test_m8_condensation_gates.py`); `composite_bem.nc` / `reference_single_bem.nc` regenerated at the 80-point production grid (identical-grids-by-construction for the excitation gate); this closure. |
| tracker (on `main`) | `bce7aea` | +83 | `BEM-CONTAMINATED-FREQUENCY-SLICE-CLUSTER-DRAFT` opened (S5). |

**PR4 Step-0 determinations** (recorded, they shaped the gates):
(a) the 18-DOF fixture carries full excitation but **no hydrostatic
block** — the decay gate assembles `C₁₈` block-diagonally from the
committed single-hull hydrostatic (`reference_single_bem.nc`,
C33 = 221.0807 N/m; `TᵀC₁₈T` → 663.2420 N/m, asserted);
(b) the committed composite fixture was on a 40-point grid — the
excitation gate's identical-grids-by-construction lock required
regenerating it at the 18-DOF fixture's 80-point grid
(`cluster_bem.py` `_N_OMEGA` 40 → 80; solve 566 problems / 511 s;
A_inf(heave) = 64.0738, C33 = 663.2420 — both bit-consistent with the
study record);
(c) **Q5's `build_system` coupled path is deferred to M9** (S7).

## S2 — Terminal gates (both PASSED, measured)

> **Honesty framing (plan Q4, carried verbatim in both gate
> docstrings):** these gates validate the ingestion and assembly path,
> not the underlying BEM physics — the two models share an influence
> matrix, so agreement is a linear-algebra identity. Independent
> validation of coupled hydrodynamics does not exist in this program
> before M10.

**DECAY GATE** (`test_decay_gate_condensed_18dof_reproduces_cluster_period`).
The coupled 18-DOF database (contaminated-ω-excluded grid), condensed
through `T` (`TᵀMT`, `TᵀA(ω)T`, `TᵀB(ω)T`, `TᵀCT`), run as a 6×6
heave free-decay exactly mirroring the composite study
(dt 0.01 / 50 s / kernel t_max 30 s / IC 0.10 m):

- condensation pins: `M₃₃ = 98.0100` kg (exact), `C₃₃ = 663.2420` N/m
  (exact identity vs 3 × single-hull), `A∞₃₃ = 64.0738` kg
  (reproduces `interaction.json` A33_composite_inf);
- **measured condensed T_n = 3.10533 s** vs reference **3.106 s** —
  rel diff **0.0215 %**, well inside the rtol 1e-2 gate. (The
  composite study's own FloatSim run measured 3.10533 s by
  zero-crossings / 3.1067 s by peaks; the condensed 18-DOF path lands
  on the same period to five significant figures.)

**EXCITATION GATE**
(`test_excitation_gate_condensation_identity_matched_grids`).
`TᵀF_exc,18(ω)` vs composite `F_exc(ω)`, identical 80-point grids
**asserted** (not assumed), physically-excited DOF only
(surge/heave/pitch at β=0), floor `1e-6 × max|F_comp|` per ω.
Measured worst residuals, two-tier by operator conditioning:

| band | worst magnitude rel | worst phase |
|---|---|---|
| resolved (ω ≤ 15.665) | **1.46e-5** | 0.0005° |
| Capytaine-flagged mesh-resolution band (ω > 15.665) | 8.7e-4 | 0.015° |

**At the contaminated frequency ω = 4.934 (deliberately included):
surge 1.8e-8, heave 2.4e-8, pitch 4.2e-6.** See S3.4.

## S3 — Empirical findings

### 3.1 PR1 — byte-identity on the legacy path

All **29** `HydroDatabase(` construction sites (the plan's "31" was
the raw grep count; two hits are test *names* — corrected in the
audit) construct single-body by keyword and take the `body_labels=None`
default, whose branch is the pre-M8 validator code verbatim.
Confirmation: `marin_semi` sha256 of A/B/A_inf/C/RAO bit-identical
pre/post-change; full suite unchanged.

### 3.2 PR2 — phase-convention audit + a design finding

Measured on the real spar-fin NetCDF: conjugation exact
(`max|RAO − conj(raw)| = 0.0`), genuinely firing
(`max|RAO − raw| = 1.2414`), **no hidden magnitude factor**
(`|RAO|/|raw| = 1.0`, min = max), and the Item-16-class physical
guard: long-wave heave `|F|/C33 = 0.9977 → 1` as ω → 0 (no missing
ρ/g/A_wp). **Design finding:** `_extract_radiation` /
`_extract_excitation` were already permutation-length generic, so the
multi-body path reuses them unchanged — the lags→leads conjugation
*cannot* diverge between the single- and multi-body paths because
there is exactly one copy of it.

### 3.3 PR3 — the PSD gate caught a real defect on first contact

The PSD check (the one M8 gate that is NOT a construction identity —
it constrains the shared BEM solve) fired the first time it met a
production-grid fixture, and it was right: a whole-matrix contaminated
solve at ω ≈ 4.934 (S3.4, S4, tracker). The MB kernel measurement was
re-locked through the full gated kernel path on the excluded-grid
input: decay-to-10 % **0.56 s (diag) / 0.76 s (cross)**; cross
K(0)/diag K(0) = **0.74**; cross/diag B-peak ratio **0.986** — single
`t_max` confirmed adequate.

### 3.4 The worked example: identities pass perfectly on wrong data

At ω = 4.934 the **entire 18×18 solve is contaminated** — measured
against neighbour trends: surge **−4.18 %**, roll **−5.26 %**, pitch
**−6.76 %**, heave **sign-flipped** (−0.0856 vs physical ≈ +0.008).
On that same slice, the condensation identity holds at
**1.8e-8 / 2.4e-8 / 4.2e-6** (surge/heave/pitch) — machine-grade
agreement on data that is wrong by ~5 % across the whole matrix,
because both models share the contaminated influence matrix and the
identity is linear algebra on top of it.

**Stated plainly: every identity gate in this milestone would pass on
a wholly wrong BEM solve.** This is what the plan's
no-independent-reference gap means concretely — 0.000 % agreement is
a property of the construction, not evidence about the physics. The
first genuine tests of the coupling are M10 (articulated regime, where
the rigid identity no longer holds) and M11 Stage 2 (tank data).

### 3.5 Full-suite counts

`688 passed / 50 skipped / 20 xfailed / 0 failed` including slow
(PR4 Phase A run). Delta vs PR3's 683/50/20: **+5 = the PR4 gate
tests** (3 label-contract + decay gate + excitation gate); skipped and
xfailed unchanged. Lineage: 662 (pre-M8) → 671 (PR1, +9) → 676
(PR2, +5) → 683 (PR3, +7) → 688 (PR4, +5).

## S4 — The PR3 mid-PR re-specification (on the record)

The locked PR3 Step C assumed the full kernel runs clean end-to-end on
the 18-DOF fixture. Two realities intervened. (1) The Phase-1 fixture
was a **reduced-grid diagnostic** (geomspace 0.5–8, 12 pts): heave was
resolved, but surge/sway/roll/pitch sat at **100 % of B-peak at
ω_max** (their peaks are at ~9.5–10 rad/s) and their truncation-ringing
kernels could not pass Check 3 at any physical `t_max`. The fixture was
regenerated on the production grid (geomspace 0.1–30, 80 pts). (2) The
production-grid solve **exposed a contaminated frequency slice at
ω ≈ 4.934** (plus a genuine, lid-removable irregular frequency at
ω ≈ 20.909), and the new PSD gate refused the fixture — correctly.

Step C was re-specified rather than forced: the fixture is retained
**unmodified**; the positive gate runs on the fixture with the two
contaminated ω **excluded from the grid at read/test level**, and a
new permanent negative gate asserts PSD fires on the unmodified
fixture at 4.934 and that the contamination is whole-matrix.

**Why grid exclusion is not interpolation:** exclusion uses only
values the solver actually produced — a contaminated solve is simply
*not used*. Interpolation would *invent* replacement values to make a
gate pass, which is the tuning pattern this project refuses; it would
also have shipped the defect silently into M11. Exclusion is
additionally the detect-and-exclude half of the M11 mitigation
(tracker: detect-and-re-solve at a perturbed ω).

## S5 — Tracker dispositions (`docs/phase2-followups.md`)

- **B4 — Multi-body BEM cross-coupling ingestion: CLOSED** by PR1
  (`a1399b2`, data model) + PR2 (`14a447e`, reader). Struck at PR4.
- **B5 — Coupled retardation kernel transform: CLOSED** by PR3
  (`2d59907`). Struck at PR4.
- **BEM-CONTAMINATED-FREQUENCY-SLICE-CLUSTER-DRAFT: OPENED**
  (`bce7aea`, on `main`) and remains **OPEN — an M11 blocker** with
  the detection-gap note (S7).
- **ITEM25-SMALL-BODY-APPLICABILITY:** referenced (kernel override
  rationale in PR3/PR4 tests and the study scripts); no disposition
  change.

## S6 — Deviations from plan / process

1. **Branch topology.** The contaminated-slice tracker was committed
   to `main` (`bce7aea`) mid-milestone (its scope is cross-milestone —
   an M11 blocker — so it did not belong to the branch). Consequence:
   `main` and `milestone-8-coupled-bem` diverged; the branch was
   **rebased onto `main` before the FF-merge** at close (Phase B),
   which rewrote the PR1–PR3 hashes — the citations in this document
   are the post-rebase (final, on-`main`) hashes.
2. **Mid-PR Step C re-specification** (S4) — recorded in the PR3
   commit message at the time, not discovered at closure.
3. **Inherited lint debt — one tracked decision, not silent carry.**
   Measured during M8, deferred TOGETHER to a dedicated `fix-` branch
   after M8 closes (CLAUDE.md §9 forbids folding unrelated cleanup
   into feature PRs):
   - `mypy --strict`: **4 pre-existing errors** on reader modules
     (`readers/wamit.py:228-230` ×3; `readers/capytaine.py`
     `_resolve_a_inf` return-Any ×1). Verified pre-existing by
     stashing the PR2 diff; M8 added zero.
   - `ruff` on `studies/`: **de-facto ungated** — the committed
     `crosskernel_diagnostic.py` alone carries 21 violations (N806
     scientific-naming class). Either `studies/**` joins the
     per-file-ignores like `scripts/**` (same "mirrors scientific
     naming" rationale already in `pyproject.toml`), or the study
     scripts get cleaned; decide on the fix- branch.

## S7 — What M8 hands forward

- **To M9 — the unimplemented half of Q5, stated:** the Q5 lock
  specifies a `build_system` coupled path (deck declares one shared
  N-body database; assembly builds the coupled 6N×6N LHS + kernel
  directly). The Q6 PR sequence allocated no PR to it and M8's scope
  exclusion is explicit ("the consuming machinery is exercised by the
  condensation scripts, not by new solver features"), so **M9 inherits
  an unimplemented lock** — this sentence is the required statement.
  The ingestion side (data model, reader, kernels, PSD gate) is done;
  M9 wires consumption when joints first need it.
- **The label contract as the defensive pattern.**
  `tests/support/condensation.py` is the reference implementation:
  label→block maps built from `body_labels`, hard raises on mismatch /
  missing / duplicate, permanent raise-path tests. M9's `build_system`
  coupled path must adopt it verbatim — block misalignment at 6N×6N
  is silent and plausible-looking, which is why mapping is by label,
  never positional.
- **To M11 — the contaminated-slice class, with a detection gap.**
  Tracker `BEM-CONTAMINATED-FREQUENCY-SLICE-CLUSTER-DRAFT`: a 72-DOF
  production BEM will meet this class at some frequency. The PSD gate
  caught *this* instance only because heave's near-zero magnitude
  turned a ~5 % shared perturbation into a sign flip; a milder
  pole-straddle would pass PSD undetected. **M11 needs
  frequency-slice smoothness screening (neighbour-trend deviation
  across the grid), not PSD alone**; the mitigation candidate is
  detect-and-re-solve at a perturbed ω (feature width ~0.02 rad/s ≪
  grid spacing), since the mesh/lid path measurably does not touch it.
- **An empirical note on identity gates** (excitation gate, S2): even
  a construction identity has a noise floor set by operator
  conditioning — machine-grade below the mesh-resolution limit,
  ~50× looser above it. Future identity gates on BEM outputs should
  expect (and tier) this rather than assuming uniform round-off.
