# Milestone 6 Plan — OpenFAST / HydroDyn Cross-Check

Working document. Status: **draft v2 (post-OrcaFlex-license-loss pivot), awaiting Xabier review.** Delete or archive after M6 merges.

Scope per ARCHITECTURE.md §8: take a representative deck, run it in
both FloatSim and a reference time-domain tool, document discrepancies.

**Reference tool: OpenFAST v3.x** (latest stable, open source, free).
Specifically the **HydroDyn** submodule plus **MAP++** (or MoorDyn)
for mooring. OrcaFlex/OrcaWave are no longer available; OpenFAST is
the open-source industry baseline used by NREL, DOE, and the
research community for floating-platform cross-checks.

**Reference deck: OC4 DeepCwind** semi-submersible from the OpenFAST
r-test repo. The same `marin_semi.{1,3,hst,4}` WAMIT files we
already trimmed into `tests/fixtures/bem/wamit/` for M5 PR1. The
BEM hydrodynamics are therefore byte-identical between FloatSim and
OpenFAST — any discrepancy lives downstream of the BEM input, in
the time-domain assembly or numerical integration.

Tolerance per CLAUDE.md §5: `rtol=5e-2` is acceptable for tool
cross-checks; investigate anything worse. Tighter where the physics
allows (statics, free decay) — see Q4.

Validation gate per ARCHITECTURE.md §7: full-system OpenFAST
agreement on the OC4 DeepCwind semi-submersible across statics,
free decay, regular-wave RAO, moored statics, and drag-on decay.

Branch: `milestone-6-openfast-cross-check` (renamed from the v1
draft `milestone-6-orcaflex-cross-check` after the OrcaFlex license
loss). Created off `main` at the M5 partial-close merge `8f5225d`.

---

## Decisions to lock with Xabier

### Q1 — Reference deck

**Proposal: lock OC4 DeepCwind from OpenFAST r-test as the
canonical M6 case.**

- Path in upstream:
  `5MW_OC4Semi_Linear/` in
  [OpenFAST/r-test](https://github.com/OpenFAST/r-test/tree/main/glue-codes/openfast/5MW_OC4Semi_Linear).
- Hydrodynamics: OpenFAST's `5MW_Baseline/HydroData/marin_semi.{1,3,hst,4}`
  — already in our repo at
  `tests/fixtures/bem/wamit/marin_semi_trimmed.{1,3,hst,4}` (the
  trimmed subset). For M6 we commit the **full** `marin_semi.*`
  alongside (under `tests/fixtures/openfast/oc4_deepcwind/`) so
  OpenFAST sees a complete BEM grid; the trimmed copy stays in
  `tests/fixtures/bem/wamit/` for M5's parser unit tests.
- Geometry reference: Robertson et al., *Definition of the
  Semisubmersible Floating System for Phase II of OC4*, NREL
  TP-5000-60601 (2014). Already cited in
  `docs/wamit-fixture-attribution.md`.
- Why OC4 DeepCwind?
  - Multi-tool validated (HydroDyn, AQWA, ProteusDS, WEC-Sim) — a
    discrepancy points at the parser, not the reference.
  - Already half-committed (BEM data in M5).
  - OpenFAST r-test ships scenario decks for free decay, regular
    waves, irregular waves — most of our scenarios drop in.
  - Apache 2.0 — safe to commit alongside attribution.
- Why not the "linear" variant specifically? `5MW_OC4Semi_Linear`
  uses **linear hydrostatic restoring** — exactly FloatSim's Phase 1
  fidelity level. Other variants (`5MW_OC4Semi_WSt_WavesWN` etc.)
  add nonlinear restoring or wind-turbine coupling — out of scope.

**Open questions for Xabier:**
- Is `5MW_OC4Semi_Linear` indeed the right variant, or is there an
  even-more-stripped-down hydrodynamics-only variant in the OpenFAST
  test suite worth considering?
- Are we OK keeping the wind turbine **disabled** (CompElast,
  CompAero, CompInflow, CompServo all = 0) for the cross-check?
  This isolates HydroDyn but means the comparison is platform-only,
  not coupled FOWT.

### Q2 — Cross-check scenario menu

**Proposal: five scenarios in increasing-complexity order
(reordered per Xabier's guidance):**

| # | Scenario | Physics validated | Variant |
| - | -------- | ----------------- | ------- |
| **S1** | Static equilibrium | Restoring + gravity/buoyancy balance | OC4Semi, no waves, no wind, no mooring |
| **S2** | Free decay (heave + pitch) | Cummins free response, period and damping | Same as S1, with small IC |
| **S3** | Regular-wave RAO sweep | Excitation + radiation across freq band | OC4Semi with regular wave, no mooring, no wind |
| **S4** | Moored static equilibrium | Catenary mooring + 6-DOF assembly | OC4Semi with 3 catenary lines (DeepCwind config), no waves, no wind |
| **S5** | Drag-on free decay | Morison drag + integrator | Same as S2 with HydroDyn's `Morison Members` block populated |

S1–S3 are mandatory (validate the M2/M3 hydrodynamic core).
S4 cross-checks M4 PR4 catenary against MAP++/MoorDyn statics.
S5 cross-checks M5 PR4 Morison drag.

**Out of scope (Phase 2):**
- Irregular seas (JONSWAP, PM): Phase 2.
- Coupled FOWT (turbine + platform): Phase 3.
- Second-order drift: Phase 2.
- Dynamic mooring: Phase 2 — Q7 below.

### Q3 — Fixture generation

**Proposal: commit OpenFAST inputs + extracted CSV time histories +
companion JSON metadata, with a regeneration script.**

```
tests/fixtures/openfast/oc4_deepcwind/
├── inputs/
│   ├── OC4Semi.fst              # driver
│   ├── OC4Semi_HydroDyn.dat
│   ├── OC4Semi_ElastoDyn.dat
│   ├── OC4Semi_MAP.dat          # mooring (S4)
│   └── HydroData/
│       └── marin_semi.{1,3,hst,4}
├── outputs/
│   ├── s1_static_eq.csv
│   ├── s1_static_eq.json        # metadata: version, dt, duration, ...
│   ├── s2_free_decay.csv
│   ├── s2_free_decay.json
│   ├── s3_rao_T08.csv           # regular-wave runs, one per period
│   ├── s3_rao_T08.json
│   ├── s3_rao_T10.csv
│   ├── ...
│   ├── s4_moored_eq.csv
│   ├── s4_moored_eq.json
│   ├── s5_drag_decay.csv
│   └── s5_drag_decay.json
└── README.md
```

- **Inputs are committed verbatim** from OpenFAST r-test (Apache 2.0,
  attribution in README).
- **Outputs are pre-extracted and committed** so CI does not need
  OpenFAST installed. CSV format: `time, surge, sway, heave, roll,
  pitch, yaw, ...` per scenario, units SI throughout.
- **JSON metadata** per CSV records: `openfast_version`, `dt`,
  `duration`, `seed_if_random`, `compile_flags`, regeneration
  command. Mirror M5's "JSON sidecar" pattern used for the WAMIT
  fixtures.
- **Regeneration**: `scripts/extract_openfast_fixtures.py` runs
  OpenFAST locally for each scenario and writes the CSV + JSON
  pair. Documented requirement: OpenFAST executable on the user's
  PATH (the script does not vendor or compile OpenFAST).
- **Total fixture footprint**: estimated ~500 KB committed —
  inputs + 5 CSVs (~50 KB each at 0.1 s × 600 s × 8 cols) + tiny
  JSON files. Comfortable in repo.

**Why pre-extract rather than run OpenFAST in CI?**
- Reproducibility: pin the OpenFAST version that produced the
  reference, freeze the comparison.
- CI speed: a single OC4 free decay run is ~30 s of wall time;
  five scenarios × CI matrix would dominate runtime.
- Repo portability: contributors don't need OpenFAST to run the
  fast-gate tests.
- Anyone with OpenFAST installed can regenerate via the script —
  no license, no proprietary dependency, no GUI clicks.

### Q4 — Comparison metrics and tolerances (revised at PR1.1)

**Locked per the M6 baseline sanity-check (Xabier, 2026-05-01).**
Tolerances on the static-equilibrium scenarios (S1, S4) were
**loosened** from the v1 draft to accommodate the residual
oscillation present in the OpenFAST reference at TMax=200s in
still water (heave natural period ~17 s, light radiation damping
in still water). Reference values are computed as the
**time-average over the last 30 s** of each channel, NOT the
instantaneous final-sample value (per
`docs/openfast-cross-check-conventions.md` Item 12).

| Scenario | Primary metric | Tolerance |
| -------- | -------------- | --------- |
| **S1** static eq. | 6-DOF displaced position, time-averaged over the last 30 s of the OpenFAST reference | `atol = 0.15 m` (heave), `atol = 0.05 m` (surge/sway), `atol = 0.5°` (rotations) |
| S2 free decay | (a) heave/pitch period from upward zero crossings; (b) log-decrement damping ratio over first 5 cycles | period: `rtol = 2e-2`; damping: `rtol = 5e-2` |
| S3 RAO sweep | per-frequency steady-state amplitude (mean of last 3 cycles); phase against wave elevation | amplitude: `rtol = 5e-2`; phase: `atol = 5°` |
| **S4** moored eq. | platform horizontal offset (last-30-s mean); per-line top tension (last-30-s mean) | offset: `atol = 0.7 m` (surge), `atol = 0.3 m` (sway/heave); tension: `rtol = 5e-2` |
| S5 drag decay | first 5 peak amplitudes against OpenFAST peaks | `rtol = 5e-2` per peak |

The S1 heave tolerance accommodates the ~0.13 m last-10%-std
observed in the reference (reference value ~0.65 m, so tolerance
is ~23% of signal). The S4 surge tolerance accommodates the
~0.71 m last-10%-std observed in the reference (MoorDyn took
~48 s of 200 s init time, leaving only ~152 s of usable sim).
Tensions converged faster (line stiffness >> hydrostatic) but
inherit the surge oscillation envelope.

Tighter on S2 / S3 / S5 — these are *dynamic* scenarios where the
metric is per-cycle peak amplitude or steady-state amplitude over
multiple cycles, not a quasi-static mean; the integrator alone
owns the residual.

### Q5 — Discrepancy report

**Proposal (unchanged from v1): `docs/openfast-cross-check-report.md`
+ figure regeneration script.**

- Markdown report with one section per scenario:
  - Setup (deck variant, IC, duration, dt, OpenFAST version)
  - Comparison table (FloatSim vs OpenFAST, abs err, rel err,
    pass/fail)
  - Time-history overlay plot (PNG)
  - Resolved / open discrepancies subsections
- Plots regenerated by `scripts/plot_openfast_crosscheck.py` into
  `docs/figures/openfast_crosscheck/`. Plots are committed
  (small PNGs, ~50 KB each); script is the single regeneration source.
- Report is one-time per milestone, not continuously updated.

### Q6 — Workflow integration / CI

**Proposal (unchanged from v1): all M6 tests marked
`@pytest.mark.slow`.**

- Per CLAUDE.md §5, `slow` runs nightly only.
- M6 tests are CSV-loader-cheap but FloatSim runs 600 s simulated
  time × Cummins convolution — ~30 s wall-clock per scenario,
  ~3 min for the full M6 set. Too slow for the per-PR fast gate
  (`pytest -q` should stay under 5 min total; currently ~2 min).

### Q7 — Drag and mooring physics differences

**Proposal: cross-check at the configurations where modeling
choices align, document divergences:**

- **Drag (S5):** OpenFAST's HydroDyn implements Morison elements
  via a `Members` table with `MemberID, MJointID1, MJointID2, MDiv,
  MPropSetID1, MPropSetID2, ...`. Each member has axial and
  transverse `Cd` and `Ca` coefficients. FloatSim's
  `floatsim.hydro.morison.MorisonElement` carries `diameter, Cd,
  Ca, include_inertia` and applies member-normal projection.
  - **Same physics: drag coefficient applied to relative-velocity
    member-normal projection.** Cross-check is fair if the deck
    matches `Cd, D, L` exactly and `include_inertia` is set
    consistently with HydroDyn's `MCa` value.
  - **Caveat**: HydroDyn supports per-member axial drag (`MCdAx`)
    in addition to transverse. FloatSim does not (member-normal
    only by design). Decks that exercise axial drag must be
    flagged or the axial term zeroed in the OpenFAST input.
- **Mooring (S4):**
  - OpenFAST options: **MAP++** (quasi-static analytical catenary,
    similar to FloatSim's M4 PR4) or **MoorDyn** (lumped-mass
    dynamic line).
  - **For S4 (statics) we use MAP++** — direct analytical-catenary
    cross-check against FloatSim's Irvine implementation. Should
    agree to numerical noise; `rtol = 1e-2` may be achievable, but
    we hold the budget at `5e-2` per CLAUDE.md.
  - Dynamic mooring (MoorDyn vs FloatSim's analytical) is **out of
    scope** — different physics. Phase 2.

### Q8 — Reference-point and sign-convention sanity check

**Proposal: a one-time pre-flight check committed as
`docs/openfast-cross-check-conventions.md` before any scenario PR
lands.** Now HydroDyn-specific (was OrcaFlex-specific in v1).

References:
- *OpenFAST documentation* —
  https://openfast.readthedocs.io/en/main/source/user/hydrodyn/index.html
- *HydroDyn Theory Document* — Jonkman, Robertson, Hayman, NREL
  technical report TP-5000-XXXX (referenced in the OpenFAST docs).

Pre-flight check items (verify against the OpenFAST manual, document
findings before PR2 lands):

1. **Reference point**: OpenFAST's `PtfmRefzt` is the platform
   reference height (z-coordinate of the reference for inertia
   and motion outputs). For OC4Semi this is typically
   `PtfmRefzt = 0.0 m` (at SWL) — must match FloatSim's body
   `reference_point[2] = 0.0`. Confirm in the `.fst` driver file.
2. **Wave heading**: HydroDyn `WaveDir` is degrees, with `0°` =
   +X propagation. Same as FloatSim. Verify in `HydroDyn.dat`.
3. **Euler order for output**: HydroDyn's platform rotations are
   reported as `PtfmRoll, PtfmPitch, PtfmYaw` in degrees. The
   underlying convention is **ZYX intrinsic** (yaw-pitch-roll) —
   matches FloatSim's deck I/O. Must verify in HydroDyn theory
   doc; OpenFAST's ServoDyn module historically uses a different
   convention for blade pitches but HydroDyn platform output is
   ZYX intrinsic per the FAST modularization document.
4. **Time origin**: OpenFAST starts at `t = 0` (no implicit ramp);
   if HydroDyn's `WaveModH = 5` (irregular ramp-up) is set, the
   ramp is in negative time. For M6 we use regular waves only
   (S3) which start cleanly at `t = 0`; align FloatSim's
   `ramp_duration = 0` for cross-check runs only (Phase 1 default
   is 20 s; per-scenario override is the existing path).
5. **Hydrostatic stiffness**: HydroDyn's `PtfmCMatrix` carries the
   **buoyancy/waterplane** contribution only. Same convention as
   the M5 WAMIT and Capytaine readers. Gravity (`m·g·z_G` on
   roll/pitch) is added by OpenFAST's ElastoDyn from the platform
   mass and CoG. FloatSim's `Body` assembly adds gravity from the
   deck's `mass` and `centre_of_gravity`. Must verify the two
   gravity contributions match per scenario.
6. **Wave elevation reference**: HydroDyn `WaveOriginZ = 0.0` is the
   default — wave elevation referenced to SWL. Matches FloatSim.
7. **Output sample rate**: HydroDyn `OutFileFmt = 1` (text output)
   at `DT_Out = 0.05 s` (matches our typical FloatSim dt). If
   OpenFAST runs at finer dt internally, the output is decimated
   — fine, we just align.
8. **Coordinate sign**: OpenFAST's surge/sway/heave are the
   inertial-frame translations of the platform reference point.
   Same as FloatSim's `xi[0:3]`. No sign flip.

Items 3 and 5 are the highest-risk — confirm against the manual
before any PR2 assertion fires. Document evidence (page numbers,
URLs) in `docs/openfast-cross-check-conventions.md`.

### Q9 — OrcaWave reader status

**Per Xabier's directive: keep `floatsim/hydro/readers/orcawave.py`
but mark it unvalidated.**

- The reader was a stub from M5 PR1 / earlier; without OrcaWave we
  cannot generate end-to-end test fixtures for it.
- Action items for PR1:
  - Add a module-level docstring to `floatsim/hydro/readers/orcawave.py`
    stating: "(1) status: unvalidated end-to-end as of M6; (2) no
    working test fixture available; (3) M5's OrcaFlex VesselType
    YAML reader (`orcaflex_vessel_yaml.py`) is the verified path
    for OrcaWave-derived BEM data; (4) this module remains for
    future re-introduction if Xabier reacquires an OrcaWave
    license, or if a community OrcaWave fixture lands."
  - Add a matching note to CLAUDE.md §7 (BEM Database Readers).
  - Drop `model3small_nomor` references from CLAUDE.md §12; replace
    with the OC4 DeepCwind reference.
- The reader is not deleted — that decision is reversible if
  OrcaWave returns to the project.

---

## PR sequence

Reordered per Xabier's directive (statics → decay → RAO → moored
statics → drag decay). Numbered by deliverable order, not test
complexity.

### PR1 — Fixture import + CSV reader + conventions doc

- Commit OpenFAST input files at `tests/fixtures/openfast/oc4_deepcwind/inputs/`
  (full `marin_semi.*` BEM, `OC4Semi.fst`, submodule `.dat` files).
- Commit `tests/fixtures/openfast/oc4_deepcwind/README.md` with
  upstream attribution and reproduction recipe.
- `scripts/extract_openfast_fixtures.py` — runs OpenFAST locally,
  writes per-scenario CSV + JSON pairs to `outputs/`.
- `tests/support/openfast_csv.py` — small loader: `(t, xi, ...) =
  load_openfast_history(path)`. Mirrors any future `orcaflex_csv.py`.
- `docs/openfast-cross-check-conventions.md` — Q8 pre-flight log
  with manual-page citations.
- **CLAUDE.md updates** (per Q9): drop `model3small_nomor` from §12;
  replace with OC4 DeepCwind. Add §7 note on the unvalidated
  `orcawave.py` reader.
- **`floatsim/hydro/readers/orcawave.py`**: add module docstring
  per Q9.
- ~300 lines (mostly fixture-import + loader scaffolding; the
  conventions doc is the bulk of the prose).

### PR2 — S1 static equilibrium

- `tests/validation/test_m6_openfast_static_eq.py`
- Run FloatSim's `static_equilibrium_solver` on the OC4 deck;
  compare displaced position to OpenFAST's reported equilibrium
  (extracted as the steady-state of a no-wave run).
- Tolerance: `atol = 1e-3 m` translations, `atol = 1e-2°` rotations.
- ~150 lines.

### PR3 — S2 free decay (heave + pitch)

- `tests/validation/test_m6_openfast_free_decay.py`
- Run FloatSim free decay from the same IC OpenFAST used; extract
  peaks and period; compare to the CSV reference.
- Heave decay first (simpler), then pitch decay (cross-coupling with
  surge expected — expand assertion budget if disagreement
  is in the cross-coupling channel, not heave/pitch directly).
- ~250 lines.

### PR4 — S3 regular-wave RAO sweep

- `tests/validation/test_m6_openfast_regular_wave.py`
- Run FloatSim at each of OpenFAST's swept periods; compare
  steady-state amplitudes and phases per DOF.
- Period grid: 8, 10, 12, 14, 16, 18 s (covers OC4 heave natural
  ~17 s, pitch natural ~26 s in roll/pitch coupling). Single
  heading 0°.
- One CSV per period in the fixture set.
- ~300 lines.

### PR5 — S4 moored static equilibrium

- `tests/validation/test_m6_openfast_moored_eq.py`
- 3 catenary lines per the DeepCwind mooring spec. FloatSim's
  Irvine analytical catenary (M4 PR4) vs OpenFAST's MAP++ (also
  analytical). Should agree to better than `5e-2`; report what we
  actually achieve.
- ~250 lines.

### PR6 — S5 drag-on free decay

- `tests/validation/test_m6_openfast_drag_decay.py`
- HydroDyn's `Members` block populated; same Morison elements in
  FloatSim deck. Cross-checks the M5 PR4 wiring.
- ~250 lines.

### PR7 — Cross-check report + plots

- `docs/openfast-cross-check-report.md` — full results table, one
  section per scenario.
- `scripts/plot_openfast_crosscheck.py` regenerates
  `docs/figures/openfast_crosscheck/*.png`.
- Closes M6.
- ~300 lines.

---

## Ordering rationale

PR1 lands the scaffolding + the locked Q9 housekeeping (CLAUDE.md
edits, orcawave.py docstring) — independent of any scenario.

PR2 (statics) is the simplest physics check — no integrator
dynamics, just the equilibrium solver. If statics disagree, every
later PR is suspect, so this is the gate.

PR3 (free decay) before PR4 (RAOs) — free response isolates
restoring + radiation; RAO comparison adds excitation on top.
Without correct free decay, RAO disagreements are confusing.

PR5 (moored) and PR6 (drag) are physics-extension checks; either
order is fine. The plan lists them in scenario-order for
report-readability.

PR7 closes the milestone with the report.

## Risks

- **Deck identity drift across tools.** Any difference between the
  FloatSim deck and the OpenFAST deck — mass, BEM data, drag
  coefficients, mooring pretension — produces a discrepancy that
  looks like a physics bug. Mitigation: PR1's conventions doc is
  exhaustive; every scenario PR starts with a deck-identity check
  that asserts FloatSim's `Body.mass`, `Body.inertia`, `(M+A_inf)`,
  `C` match OpenFAST's `Platform*` inputs to numerical noise
  before any time-domain assertion fires.
- **OpenFAST version drift.** The committed CSVs are
  version-locked. Each `outputs/*.json` records the OpenFAST
  version that produced it; the regeneration script logs the
  detected version on launch. Re-running on a different OpenFAST
  version is a documented full-cycle operation.
- **Phase convention slip.** HydroDyn's RAO phase convention is
  documented in the manual; we assume "leads" (matches our
  `HydroDatabase.RAO`) but **must verify** in PR1 — if HydroDyn
  uses lags internally, the comparison conjugates one side. Q8
  item 3 covers this.
- **OpenFAST install.** Anyone regenerating fixtures needs
  OpenFAST. Binaries are available at
  https://github.com/OpenFAST/openfast/releases. Windows users:
  `openfast_win64.exe`. Linux users: build from source or use
  conda-forge. The script's installation requirements are
  documented but not enforced (no auto-install).
- **HydroDyn nonlinearity assumptions.** The `_Linear` variant of
  the OC4Semi case uses linear hydrostatic restoring — matches
  FloatSim's Phase 1 fidelity. If a future scenario needs
  nonlinear restoring (large-angle pitch, wave-modulated
  waterplane), it's out of scope; flag as Phase 2.
- **MoorDyn vs MAP++ choice.** S4 cross-check uses MAP++ (analytical
  catenary). If the OC4 reference deck ships with MoorDyn instead,
  PR5 must replace MoorDyn with MAP++ in a per-scenario deck
  override (small input file edit, documented in PR5).

## Session-continuity notes

If a fresh session picks this up: M5 closed partially at merge
commit `8f5225d` on main (PR3 sphere fixtures deferred on branch
`milestone-5-pr3-sphere-fixtures`). M6 has no code yet, only this
plan (v2, post-OrcaFlex-license-loss). Branch
`milestone-6-openfast-cross-check` is created off main and tracks
main; the v1 draft branch `milestone-6-orcaflex-cross-check` was
deleted on the remote when v2 landed.

Q1–Q9 above are **proposals** — do not implement until Xabier has
reviewed and locked them.

The reference OpenFAST case lives in the OpenFAST/r-test repo.
Inputs will be vendored at
`tests/fixtures/openfast/oc4_deepcwind/inputs/`; reference time
histories must be extracted by `scripts/extract_openfast_fixtures.py`
(Xabier's side, on a machine with OpenFAST installed) before PR2
can write its first failing test.
