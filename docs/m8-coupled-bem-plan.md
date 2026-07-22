# M8 — Coupled multi-body BEM (B4/B5 + excitation) — PLAN

**Status: LOCKED (Q1-Q8), 2026-07-17.** First milestone of the Tier 3
program ([`tier3-program-plan.md`](tier3-program-plan.md), locked
2026-07-05, `a623bda`). Inherits that plan's locks; this plan adds the
M8-specific ones and the per-PR contracts.

**Milestone goal.** Ingest **coupled multi-body BEM** into FloatSim:
- **B4** — reader + N-body `HydroDatabase` data model carrying the full
  `6N×6N` added-mass / radiation-damping matrices with body labels;
- **B5** — coupled retardation kernels (full-matrix Filon);
- **excitation** — per-body Froude-Krylov + diffraction (with
  inter-body scattering) ingestion.

**Scope exclusions.** NO joints (M9); NO articulated runs (M10); NO
time-domain wave-response *simulation* — kernel + excitation
**INGESTION** only. The consuming machinery is exercised by the
condensation scripts (linear algebra on M8 outputs), not by new solver
features. The solver stack is already 6N-generic (MC).

---

## ⚠ What the M8 gates do and do not prove

**Both terminal gates are construction-consistency identities, not
independent physics validation.** The composite single-body run and the
18-DOF multi-body run **solve the same influence matrix over the same
panels** — `cluster3_fullfix.gdf` is literally the concatenation of the
three translated hull meshes — differing only in the prescribed DOF
velocity distributions. The rigid mode is exactly a linear combination
of the 18 body DOFs, so by linearity of the radiation problem and of
force integration:

    A_composite  ≡  Tᵀ A₁₈ T            (identically, to solver round-off)

That is why the measurements came out at **0.000 % / 0.0000 %** — an
identity, not an agreement. The gates therefore **do** catch:
mis-indexed blocks, label-parsing errors, a wrong `T`, phase-assembly
mistakes, block misalignment — precisely the failure modes an
*ingestion* milestone must catch, which is why they are the right gates
for M8.

They **cannot** catch anything wrong inside the shared BEM solve
itself. **M8 has no independent reference for coupled multi-body
hydrodynamics.** That gap is real and is mitigated only downstream —
see the risk register. A future reader must not mistake 0.000 % for
validated coupled hydrodynamics.

---

## Phase-1 measurements (grounding)

| id | measurement | result | script |
|---|---|---|---|
| **MA** | 18-DOF excitation dataset structure | `excitation_force(omega, wave_direction, influenced_dof)`; 18 `buoyK__DOF` labels; data_vars {`Froude_Krylov_force`, `diffraction_force`, `excitation_force`} | `capytaine_excitation_diagnostic.py` |
| **MA-gate** | excitation condensation `Tᵀ F₁₈` vs composite, all 6 DOF | excited DOF (β=0): **surge / heave / pitch = 0.0000 %** in magnitude **and phase** (0.0000°) at all 6 ω. Forbidden DOF (sway/roll/yaw) at the numerical floor in **both** models — not compared (magnitude-floor risk row). Heave gate pin 2 %, HARD STOP 5 % — **PASSED**. *Identity, not validation (see box above).* | same |
| **MB** | cross-term damping-kernel decay | **RE-LOCKED at PR3 Step C** (see note): cross `B_ij` peak = **0.986×** diagonal; cross `K(0)` = **74 %** of diagonal `K(0)`; decay-to-10 % **0.76 s (cross) vs 0.56 s (diag)** → **single `t_max` adequate**. `B_cross` range includes negative off-diagonals (physical). | `test_retardation_kernel.py` (positive gate) |
| **MC** | single-body-assumption audit | solver stack already 6N-generic; bottleneck = HydroDatabase + kernel-input + build_system | Step-3 reads |
| **MD** | 18×18 reciprocity (from program G5) | `max\|A_ij − A_ji\| / max\|A\|` = **1.08e-4** — a genuine physical invariant, *not* a construction identity | 18-DOF diagnostic |

**MB re-lock note (PR3 Step C, 2026-07-20).** The original MB figures
(cross `B_ij` peak 0.981×, cross `K(0)` 74 %, decay-to-10 % 0.77 s /
0.56 s) were measured **heave-only, via raw `filon_trap_cosine` with no
gate, on the coarse `geomspace(0.5, 8, 12)` grid**. They are superseded
by a fresh measurement through the **full gated
`compute_retardation_kernel`** path on the PR3 positive-gate input (the
production-grid 18-DOF fixture with the two contaminated frequencies
excluded — see PR3 below). The numbers barely moved (heave was
well-resolved on both grids), but the measurement basis changed: grid
widened `0.5–8 → 0.1–30`, and raw filon → full kernel path (Filon +
zero-fill tail + Check 3). The re-locked values are in the MB row above;
`MD` (reciprocity) is an `A_inf` quantity and is **unchanged** by the
regrid (bit-identical `1.080e-4`).

**MC detail (file:line).**
- **Already 6N-generic (F4 legacy):** `CumminsLHS.n_dof` / `n_bodies`
  (radiation.py:77,83); `RetardationKernel` validates n_dof=6N
  (retardation.py:138); `integrate_cummins` body-agnostic
  (newmark.py:15); `assemble_global_lhs` block-diagonal stack
  (state.py:142-144).
- **Single-body-bound (the M8 work):** `HydroDatabase.__post_init__`
  hardcodes `(6,6,n_w)` / `(6,6)` validators (database.py:187-196);
  `compute_retardation_kernel` consumes a 6×6 database
  (retardation.py:210); `build_system` takes
  `bem_databases: dict[str, HydroDatabase]`, composed
  **block-diagonally only** (driver.py:398,446,454-463) — **no coupled
  path exists**.
- **31 `HydroDatabase(` construction sites** (3 production readers +
  ~11 test files + synthetic support).

---

## Locks

### Q1 — Data model: **extend `HydroDatabase`** (LOCKED)

Add `body_labels: tuple[str, ...] | None = None`.

- **`None` → legacy path.** Shape must be `(6,6,n_w)` / `(6,6)`. The
  code path is **bit-identical to today by construction, not by test** —
  the legacy branch is the existing code, untouched, not a special case
  of a generalized branch.
- **Provided → N-body path.** `N = len(body_labels)`; shapes
  `(6N,6N,n_w)` / `(6N,6N)`.
- **`n_bodies` derives from `body_labels`, never from shape
  arithmetic.** Shape alone cannot be trusted to split DOF into bodies
  (a `12×12` array admits more than one reading, and any future
  generalized/flexible modes would silently mis-split). Labels are
  authoritative.

**Red gate (PR1):** `marin_semi` byte-diff invariants + the full test
suite unchanged — now as **confirmation** of the by-construction legacy
path, not as its sole guarantee.

**GROUNDING:** MC — 31 construction sites and 3 readers all build
single-body; the solver stack already accepts 6N, so the extend meets
it. Consumer-count argument, not aesthetic.

### Q2 — Reader: detection by **distinct body-prefix count** (LOCKED)

Multi-body detection keys on the **number of distinct body prefixes**,
**not** on prefix presence. A single-body Capytaine run with a *named*
body emits `spar_fin__Heave`-style labels — that is still **N = 1** and
**must** produce a database byte-identical to the unlabeled path.

Algorithm: parse DOF-label prefixes → count distinct → **N = 1 ⇒ legacy
construction** (no `body_labels`, 6×6 shapes); **N > 1 ⇒ N-body
construction**. Single-body path otherwise untouched.

**Symmetrization:** full-matrix at `__post_init__` (program Q2;
grounded on MD/G5 noise scale 1.08e-4). **Metadata residuals: keep the
4 full-matrix scalar keys** as today (approved) — they already
summarize the worst off-diagonal.

### Q3 — Kernel: full-matrix Filon, single `t_max`, per-entry gate (LOCKED)

Full-matrix **Filon** over the `6N×6N` `B(ω)`; **single `t_max`**
(MB: cross and diagonal decay on the same sub-second timescale);
Item-25 gate evaluated **per entry**. Three specifications:

1. **The Item-25 override is per-call, global to the kernel
   computation.** Do **not** build a per-entry override API — one
   rationale string covers the whole `compute_retardation_kernel` call.
2. **Sign handling differs by position.** Diagonal entries require
   `B_ii(ω) ≥ 0`. **Off-diagonals carry no sign requirement** —
   measured `B_cross` range includes **−3.99e-04**, which is physically
   fine. The current gate assumes non-negativity; applying it per-entry
   *unmodified* would spuriously fire on every negative cross-term.
   The per-entry gate must therefore branch on `i == j`.
3. **NEW — PSD check on `B(ω)`.** The multi-body generalization of
   `B ≥ 0`: eigenvalues of the full `6N×6N` `B(ω)` at each ω must be
   `≥ −tol`. Cheap, and it is the correct physical invariant for a
   damping matrix. **This is one of the few M8 checks that is NOT a
   construction identity** — it constrains the shared BEM solve itself.

### Q4 — Excitation (LOCKED)

Per-body `F_exc` blocks per MA's structure (`6N` complex vector per
(ω, heading)); ingest FK + diffraction. Gate = the MA condensation
cross-check as a permanent test, on the physically-excited DOF, with
the magnitude floor.

**GATE SPECIFICATION (structural, not an implementation detail).**
The gate runs both models on **IDENTICAL frequency grids BY
CONSTRUCTION**: one `omega` array defined **once** and passed to *both*
solves, exact `.sel(omega=w)` extraction, **no interpolation path**.
This is already the structure of `capytaine_excitation_diagnostic.py` —
`_OMEGAS = np.geomspace(0.5, 8.0, 6)` defined once (line 24), consumed
by both the 18-DOF problem list (`for w in _OMEGAS`, line 64) and the
composite problem list (same expression, line 83); extraction is
`E.sel(omega=w, influenced_dof=d)` over `for w in _OMEGAS`
(lines 40-41). Explicit here to **prevent silent regression** when the
diagnostic is promoted to a permanent test (e.g. someone later
resamples one side onto a production grid). Grid matching is part of
the gate's *definition*.

**HONESTY CLAUSE (must survive into the test docstring).**
> This gate validates the ingestion and assembly path, not the
> underlying BEM physics — the two models share an influence matrix, so
> agreement is a linear-algebra identity. Independent validation of
> coupled hydrodynamics does not exist in this program before M10.

### Q5 — Deck / assembly: shared database + **label-mapping contract** (LOCKED)

A deck declares **one shared N-body database** across its bodies.
`build_system` gains a coupled path: when bodies share an N-body
database, assemble the coupled `6N×6N` LHS + kernel directly (bypassing
the block-diagonal `assemble_global_lhs` / `assemble_global_kernel`);
the block-diagonal path stays for independent single-body databases.

**LABEL-MAPPING CONTRACT (the highest-value defensive lock in M8).**
Deck-body → database-block mapping is **by label, never positional**:
- each deck body declares `hydro_body_label`;
- `build_system` builds the index map **from labels**;
- it **raises** on any label mismatch, missing label, or duplicate.

**Rationale:** block misalignment at `6N×6N` produces *plausible-looking
wrong answers that pass smoke tests* — the failure is silent and the
result is dimensionally valid. Positional mapping cannot detect it;
label mapping makes it impossible.

### Q6 — PR sequence (LOCKED)

`PR1` data model → `PR2` reader → `PR3` kernels → `PR4` condensation
gates + closure. Per-PR contracts below.

**Strengthened PR2 gate.** Read the **composite single-body NetCDF
through the multi-body path** (N = 1 *with* labels) and require
**byte-identity** against the single-body path. This catches
phase/sign convention errors **more sharply than the condensation
check**, because it isolates the *reader* from the *assembly* — a
convention error that the condensation identity would absorb on both
sides shows up here as a byte difference.

### Q7 — Pre-M8 audit doc (LOCKED)

Formalize `docs/audits/m8-coupled-bem-audit.md` before PR1; the MC
content above is its draft. Per the M7.5 precedent (audit before PR1).

### Q8 — Estimate (LOCKED format)

**Planning unit = AI-assisted calendar days**; the human-effort column
is retained for context only.

| | AI-assisted calendar (planning unit) | human-effort (context) | ×3-4 (context) |
|---|---|---|---|
| M8 total | **~4-6 days** | 3-5 wk | 9-20 wk |

**Variance driver — amendment cycles, not PR count.** M7.5 closed in
~2 calendar days (`a14a1ef` 2026-07-02 → `3a6d00f` 2026-07-04) while
absorbing **six Q2 algorithm amendments**. PR count was not the
variance; the amendment cycles were. **M8's equivalent risk is the
coupled-data-model design** (Q1/Q2) — if `body_labels` semantics or the
detection rule need re-cutting mid-milestone, that is the schedule.

---

## Per-PR contracts

### PR1 — Data model (`body_labels`, N-body shapes)

- **Step A (pre-flight).** Enumerate and classify all 31
  `HydroDatabase(` construction sites; confirm each lands on the legacy
  branch untouched. Formalize the Q7 audit doc.
- **Step B (implement).** Add `body_labels`; generalize the validators
  to `(6N,6N,n_w)` when labels are present; `n_bodies` from labels;
  legacy branch left as existing code.
- **Step C (gate).** RED GATE: `marin_semi` byte-diff invariants
  **bit-identical** + full test suite unchanged (662 pre-M8 baseline).
  Any diff on the legacy path is a stop.

### PR2 — Reader (radiation + excitation ingestion)

- **Step A (pre-flight).** **Excitation phase-convention audit**:
  Capytaine excitation phase/sign vs FloatSim's `+i` convention,
  documented with a worked single-body comparison (the M6 Item-16
  JAxCd-class factor lesson).
- **Step B (implement).** Distinct-prefix-count detection; parse
  `6N×6N` radiation blocks + labels; parse excitation blocks per MA
  structure; full-matrix symmetrization.
- **Step C (gate).** **Composite single-body NetCDF read through the
  multi-body path (N=1 with labels) must be byte-identical to the
  single-body path.** Plus: the 18-DOF fixture ingests and reproduces
  the measured reciprocity residual (1.08e-4, MD).

### PR3 — Coupled kernels

- **Step A (pre-flight).** Confirm per-call Item-25 override semantics;
  enumerate the gate changes required by Q3 (i)-(iii); define `tol` for
  the PSD check.
- **Step B (implement).** Full-matrix Filon over `6N×6N`; single
  `t_max`; per-entry gate with `i == j` branch (diagonal non-negativity
  only); **PSD check on `B(ω)`**.
- **Step C (gate). RE-SPECIFIED mid-PR (2026-07-20).** The locked
  contract assumed the full kernel runs clean end-to-end on the 18-DOF
  fixture. It does not: the production-grid regeneration (needed so the
  non-heave DOFs reach their `B`-peaks and pass Check 3 — the coarse
  `0.5–8` grid left surge/roll/pitch at 100 % of peak at `ω_max`)
  exposed a **contaminated frequency slice at ω≈4.934** (and a genuine
  Capytaine-flagged irregular frequency at ω≈20.909). The PSD check
  (Q3 iii) **correctly refuses** the full fixture — the one M8 gate that
  constrains the shared BEM solve caught a real defect in real data on
  first contact with a production-grid fixture. The fixture is retained
  **unmodified**; the gate is split:

  - **POSITIVE gate** — on the 18-DOF fixture with the two contaminated
    ω **excluded from the grid at read/test level** (a grid *selection*,
    not a value modification: a contaminated solve is simply not used;
    this is also the detect-and-exclude half of the M11 mitigation).
    Exercises full-matrix Filon at `6N=18`, single `t_max`, the
    per-entry gate with the `i==j` branch, and **PSD passing**.
    *Verified:* excluded-grid Check 3 worst-DOF decay = **0.0249 %** at
    `t_max = 60 s` (< 0.1 % threshold). Reproduces the re-locked MB
    decay (0.76 s cross / 0.56 s diag to the 10 % floor).
  - **NEGATIVE gate** — new and permanent, on the **unmodified** 18-DOF
    fixture: (i) PSD **fires** at ω=4.934 (min eig **−0.1201** vs max eig
    **+20.40**; min-eig/max\|B\| = **−0.250 %**); (ii) the contamination
    is **whole-matrix, not heave-only** — surge / roll / pitch each
    deviate **> 3 %** from their neighbour midpoints at ω=4.934 (measured
    **−4.18 % / −5.26 % / −6.76 %**). A test that proves the gate catches
    a real defect in real data is worth more than a passing fixture.

  See tracker `BEM-CONTAMINATED-FREQUENCY-SLICE-CLUSTER-DRAFT`
  (`docs/phase2-followups.md`) for the full characterization and the
  M11 detection-gap note.

### PR4 — Condensation gates + closure

- **Step A (pre-flight).** Build `T` (18×6) via the **Q5 label
  contract**; verify label-mismatch and duplicate-label both raise.
- **Step B (implement).** Both condensation gates as permanent tests,
  each carrying the Q4 honesty clause in its docstring.
- **Step C (gate).** Decay: condensed 6×6 reproduces `T_n = 3.106 s` at
  **rtol 1e-2**. Excitation: `Tᵀ F_exc` vs composite at **0.0000 %** on
  the excited DOF **with the magnitude floor applied**. Closure doc.

---

## Risks

| risk | evidence | mitigation |
|---|---|---|
| **NO independent reference for coupled multi-body BEM** | Both M8 terminal gates are construction-consistency identities (`A_comp ≡ Tᵀ A₁₈ T`, same influence matrix, same panels) — they cannot detect an error inside the shared BEM solve. M8 ships coupled hydrodynamics with **zero independent validation**. | **Mitigation ladder:** (1) reciprocity, already measured at **1.08e-4** (MD) — a genuine physical invariant; (2) the **new PSD check on `B(ω)`** (Q3 iii) — also not a construction identity; (3) **M10's articulated regime**, where the joints free relative motion and the rigid identity **no longer holds** — the first genuine test of the coupling; (4) **tank data at M11 Stage 2**. State the gap plainly in the M8 closure doc. |
| **block misalignment (silent, plausible-looking)** | A mis-mapped body→block index at `6N×6N` yields dimensionally valid, smoke-test-passing, wrong answers | **Q5 label-mapping contract**: mapping by `hydro_body_label` only, never positional; `build_system` raises on mismatch / missing / duplicate. PR4 Step A tests both raise paths. |
| **back-compat surface** | 31 construction sites + full suite (662 pre-M8) all single-body | Q1 legacy path bit-identical **by construction**; PR1 Step C byte-diff gate as confirmation |
| **excitation phase conventions** | Capytaine vs FloatSim excitation phase/sign (M6 Item-16 factor lesson) | PR2 Step A dedicated audit; PR2 Step C byte-identity gate isolates reader from assembly — sharper than the condensation identity |
| **cross-model gates: relative error on unexcited DOF** | Relative error on symmetry-forbidden channels is **noise / noise** — measured at M8 Phase 1 as **198 % / 247 % / 71 %** apparent disagreement on sway / roll / yaw at β=0, where **both** models sit at the numerical floor (\|F\| ≈ 0). Same artifact class as `cbc0dc1`. **This was a DIAGNOSTIC REPORTING bug — denominator guard too loose at `1e-12` — not a model disagreement:** corrected per-DOF result is surge / heave / pitch at **0.0000 %** magnitude *and* phase. | floor: **`1e-6 × max\|F_comp\|` per omega**; sub-floor DOF reported "below floor / not compared". Applies to **every cross-model gate in this program**, including **M11 Stage 1** (digitized OrcaFlex plots carry near-zero lateral channels). |
| **12-buoy-scale array memory** | N=12 (n_dof=72): A/B each `72² × 16 B × N_ω` = 5184 × 16 × 40 ≈ **3.3 MB**; **kernel** `72² × 8 B × N_t` = 5184 × 8 × 3001 ≈ **124 MB**. The BEM *solve* influence matrix (17,856², ~5 GB/ω) is the solver's concern — program risk register. | stored DB is small; 124 MB kernel manageable; monitor at PR3 |
| **contaminated BEM frequency slice (M11 detection gap)** | PR3 surfaced a whole-matrix contaminated solve at ω≈4.934 on the cluster-draft spar hull (a near-singular boundary-integral operator; **not** a lid-removable irregular frequency — Capytaine does not flag it). PSD caught **this** instance only because heave's near-zero magnitude turned a coherent ~5 % perturbation into a **sign flip**; a less severe pole-straddle would produce ~5 % undershoot with no sign change and **pass PSD undetected**. | Tracker **`BEM-CONTAMINATED-FREQUENCY-SLICE-CLUSTER-DRAFT`** (`docs/phase2-followups.md`). PR3 excludes the contaminated ω from the positive gate and asserts PSD fires on the unmodified fixture (negative gate). **M11 needs frequency-slice smoothness screening (neighbour-trend deviation), not PSD alone**; mitigation candidate is detect-and-re-solve at a perturbed ω (feature width ≪ grid spacing → ~0.1 rad/s shift suffices), since the mesh/lid path demonstrably does not touch it. |

---

## Session-continuity notes

- Program plan reference: `docs/tier3-program-plan.md` @ **`a623bda`**.
  (Instructions have three times cited artifacts that do not exist in
  the repo — `48b2b25`, plus two number sets. **Standing rule: no
  citation is authoritative without a repo check**; discrepancy is a
  stop condition — program plan, Milestone-plan protocol.)
- Phase-1 diagnostics live at
  `studies/cluster-3buoy-rigid/{capytaine_excitation_diagnostic,crosskernel_diagnostic}.py`;
  the 18-DOF radiation fixture `capytaine_multibody_18dof.nc` is on
  `main` at `a623bda`.
- **Next: PR1** (data model). Open the milestone branch and the Q7
  audit doc first, per the M7.5 precedent.
