# Milestone 6 Plan — OrcaFlex Cross-Check

Working document. Status: **draft, awaiting Xabier review.** Delete or archive after M6 merges.

Scope per ARCHITECTURE.md §8: take a representative deck, run it in both
FloatSim and OrcaFlex, document discrepancies. This is the closing
validation of Phase 1 — every other milestone validated against
analytical references; M6 validates against the industry-standard
licensed tool. Tolerance per CLAUDE.md §5: `rtol=5e-2` is acceptable;
investigate anything worse.

Validation gates per §7: full-system OrcaFlex agreement on the
canonical HSFP (Hi-Stability Floating Platform) deck — `model3small_nomor.*`
per CLAUDE.md §12 — across free decay, regular-wave response, static
equilibrium, drag, and mooring scenarios.

Branch: `milestone-6-orcaflex-cross-check` (created off `main` at the
M5 partial-close merge `8f5225d`).

---

## Decisions to lock with Xabier

### Q1 — Reference deck

**Proposal: lock `model3small_nomor` as the canonical M6 case.**

- Path: `C:\Users\xlama\OneDrive\Documents\buoy\Orca\orcawave\model3small_nomor.{owr,sim,dat}`
  (and `_mooring`, `_nofin` siblings). Single body, no-mooring variant
  is the primary (clean free-floating dynamics); the `_mooring`
  variant exercises catenary cross-check.
- Geometry: HSFP — Spar-Fin Platform per `HSFP_OrcaWave_Setup_Guide_RevB.docx`.
  Single body, axisymmetric-ish, fits FloatSim's Phase 1 single-body scope.
- Why not OC4 DeepCwind? We already have the OC4 `platform_small.yml`
  fixture from M2, but it carries the OrcaFlex VesselType already
  exported — using it for M6 would partially circle back on its own
  reference. `model3small_nomor` is independent BEM data and is the
  case CLAUDE.md §12 explicitly nominates for M6.
- Why not `model3small.owr` (full case)? The full case includes
  whatever OrcaFlex extras Xabier had on top — mooring tweaks,
  damping coefficients, etc. The `_nomor` variant is the cleanest
  starting point; siblings get added per scenario.

**Open questions for Xabier:**
- Is `model3small_nomor` indeed the right primary case, or has it
  been superseded by another variant in your workflow?
- Do the `_nomor`, `_nomor_mooring`, `_nomor_nofin` siblings share
  identical hydrodynamics (only mooring/fin geometry changes between
  them)? If yes, one BEM database supports all scenarios.

### Q2 — Cross-check scenario menu

**Proposal: five canonical scenarios.** Each scenario is one
PR/test file:

| Scenario | Physics validated | Variant |
| -------- | ----------------- | ------- |
| **S1 — Free decay (heave + pitch)** | Cummins free response, period and damping | `_nomor` |
| **S2 — Regular-wave RAO sweep** | Excitation, radiation across frequency band | `_nomor` |
| **S3 — Static equilibrium** | Restoring + gravity/buoyancy balance | `_nomor` |
| **S4 — Drag-on free decay** | Morison drag + integrator | `_nomor_nofin` (drag-dominated heave plate removed) vs `_nomor` (with plate) |
| **S5 — Moored static equilibrium** | Catenary + 6-DOF spring connectors | `_nomor_mooring` |

S1–S3 are mandatory (validate the M2/M3 hydrodynamic core).
S4 is contingent on a clean drag-isolation comparison existing
between two HSFP variants. S5 is contingent on the catenary
geometries in `_nomor_mooring` matching the Irvine analytical form
FloatSim implements (M4 PR4).

**Out of scope (would need new physics):**
- Multi-body cross-check — HSFP is single-body; covered by M4 unit tests.
- Irregular seas — Phase 2 per ARCHITECTURE.md §1.2.
- Second-order drift forces — Phase 2.

### Q3 — Producing OrcaFlex time-history fixtures

**Proposal: pre-extracted CSVs committed to the repo.**

- Xabier runs each scenario in OrcaFlex once.
- Time histories exported to CSV via OrcaFlex GUI's
  *Time History → Export* (or via OrcFxAPI script — Xabier's full
  license accepts this; CLAUDE.md §10's "demo refuses API access"
  applies only to the BEM `.owr` reading path, not to running
  simulations).
- One CSV per scenario, committed at
  `tests/fixtures/orcaflex/{scenario_name}.csv`.
- Header row carries column names; first column is time in seconds;
  remaining columns are 6-DOF body state (per OrcaFlex naming) plus
  any per-line tension columns for S5.
- File sizes: ~50 KB per scenario at 0.1 s sampling × 600 s × 7 cols.
  Five scenarios → ~250 KB total. Comfortable in-repo.

**Why not run OrcaFlex live in CI?**
- License requirement makes CI runners impossible.
- Test reproducibility: pinning a fixture removes any chance of
  OrcaFlex version drift silently changing the reference.
- Speed: live OrcaFlex would dominate test runtime.

**OrcFxAPI dependency:** none in FloatSim. The export step is on
Xabier's side only. The CSV format is plain ASCII and consumed via
`numpy.loadtxt` plus a thin column-mapping helper.

### Q4 — Comparison metrics

**Proposal: per-scenario closed-form metrics, with phase-aligned
time-history overlay plots as supplementary diagnostics.**

| Scenario | Primary metric | Tolerance |
| -------- | -------------- | --------- |
| S1 free decay | (a) heave/pitch period from upward zero crossings; (b) log-decrement damping ratio over first 5 cycles | period: `rtol=2e-2`; damping: `rtol=5e-2` |
| S2 RAO sweep | per-frequency steady-state amplitude (mean of last 3 cycles); phase against wave elevation | amplitude: `rtol=5e-2`; phase: `atol=5°` |
| S3 static eq. | 6-DOF displaced position from origin | `atol = 1e-3 m` (translations), `atol = 1e-2°` (rotations) |
| S4 drag decay | first 5 peak amplitudes against OrcaFlex peaks | `rtol=5e-2` per peak |
| S5 moored eq. | platform horizontal offset; per-line top tension | offset: `rtol=5e-2`; tension: `rtol=5e-2` |

Tighter tolerances on S1/S3 (`rtol=2e-2`, `atol=1e-3 m`) reflect
that no drag, no mooring, no waves means the integrator alone owns
the residual — and our M2 free-decay validates that integrator at
`rtol=3e-2` already, so this should be tighter, not looser, than
the analytical-reference gates.

### Q5 — Discrepancy report

**Proposal: `docs/orcaflex-cross-check-report.md` + figure regeneration script.**

- Markdown report with one section per scenario:
  - Setup (deck variant, IC, duration, dt)
  - Comparison table (FloatSim vs OrcaFlex, abs err, rel err, pass/fail)
  - Time-history overlay plot (PNG, regenerated)
  - "Resolved discrepancies" subsection if a mismatch was traced and
    explained (e.g., different drag-coefficient defaults, different
    mooring-pretension reference)
  - "Open discrepancies" subsection if any remain unresolved at PR
    time (must be `rtol < 5e-2` to be acceptable per CLAUDE.md §5)
- Figures regenerated by `scripts/plot_orcaflex_crosscheck.py`,
  output under `docs/figures/orcaflex_crosscheck/`. Plots are
  committed (small PNGs, ~50 KB each); the script is the
  single-source regeneration.
- Report is a one-time deliverable per milestone, not a continuously
  updated artifact. Re-running it requires re-running OrcaFlex (Xabier's
  side) and is not in CI.

### Q6 — Workflow integration / CI

**Proposal: all M6 tests marked `@pytest.mark.slow`.**

- Per CLAUDE.md §5, `slow` tests run nightly in CI, not on every PR.
- M6 tests load CSV fixtures (cheap) but FloatSim runs full
  time-domain simulations — typically 600 s simulated time at
  `dt = 0.05 s` = 12000 steps per scenario. With Cummins convolution
  this is ~30 s wall-clock per scenario locally; over five scenarios,
  ~3 min. Too slow for the fast PR gate (`pytest -q` should stay
  under 5 min total, currently ~2 min).
- The `slow` marker is the standard escape hatch — same one used by
  M2's OC4 free-decay test.

### Q7 — Drag and mooring physics differences

**Proposal: cross-check on configurations where the modeling
choices align, document where they diverge.**

- **Drag (S4):** OrcaFlex uses Morison-like elements with the same
  `Cd`, `Ca`, projected-area formulas. If Xabier's HSFP deck
  applies drag only via Morison elements (not a body-level damping
  matrix), the cross-check is fair. If OrcaFlex adds linear viscous
  damping to the body matrix in addition, FloatSim must mirror that
  (or the deck must be edited for the cross-check).
- **Mooring (S5):** OrcaFlex uses lumped-mass dynamic lines by
  default; FloatSim uses analytical Irvine elastic catenary (M4 PR4).
  These agree at static equilibrium (closed-form catenary is the
  static limit of a dynamic line) but diverge under transients.
  Cross-check is therefore static-only for S5; transient mooring
  agreement is a Phase 2 deliverable.
- Document divergences in `docs/orcaflex-cross-check-report.md`
  per scenario.

### Q8 — Reference-point and sign convention sanity check

**Proposal: a one-time pre-flight check committed as Q8 documentation,
not a test.** Before running any scenario:

1. Compute the OrcaFlex VesselType reference point from the `.owr`
   header / VesselType YAML export. Compare to FloatSim's body
   `reference_point` after deck assembly.
2. Verify OrcaFlex wave heading 0° = +X (matches ARCHITECTURE.md §3).
3. Verify OrcaFlex Euler order ZYX intrinsic (yaw-pitch-roll) matches
   FloatSim's deck I/O convention.
4. Verify time origin: OrcaFlex `Stage 0` = `t = 0`, ramp-up runs
   into negative time domain by convention; FloatSim ramps from
   `t = 0` forward. Either align ramp endpoints or skip the ramp
   region in comparison.

Document each check in `docs/orcaflex-cross-check-conventions.md`
before scenario PRs land. Three pages, no code.

---

## PR sequence

### PR1 — Fixture import + CSV reader

- Commit `model3small_nomor.{owr→.yml VesselType export}` as
  `tests/fixtures/bem/orcaflex/hsfp_nomor.yml` (the BEM hydrodynamics
  shared by S1–S4).
- Commit one CSV time-history per scenario at
  `tests/fixtures/orcaflex/{s1_free_decay, s2_rao_sweep, s3_static_eq,
  s4_drag_decay, s5_moored}.csv`. Empty placeholders if Xabier's
  exports are pending; the per-fixture skip pattern (mirroring the
  M5 PR3 skeleton) lets each scenario activate as its CSV lands.
- `tests/support/orcaflex_csv.py` — small loader returning a
  named-tuple of `(t, surge, sway, heave, roll, pitch, yaw, ...)`
  arrays. ~100 lines.
- `docs/orcaflex-cross-check-conventions.md` — Q8 sanity-check log.
- ~250 lines.

### PR2 — S1 free decay (heave + pitch)

- `tests/validation/test_m6_orcaflex_free_decay.py`
- Run FloatSim free decay from the same IC OrcaFlex used; extract
  peaks and period; compare to the CSV reference.
- Pre-flight: confirm both share the same `(M + A∞)`, `C` matrices.
- ~250 lines.

### PR3 — S2 regular-wave RAO sweep

- `tests/validation/test_m6_orcaflex_regular_wave.py`
- Run FloatSim at each of OrcaFlex's swept periods; compare
  steady-state amplitudes and phases per DOF.
- Period grid: 5–20 s in 5 steps (matches OrcaFlex's typical RAO
  sweep). Single heading 0°.
- ~250 lines.

### PR4 — S3 static equilibrium

- `tests/validation/test_m6_orcaflex_static_eq.py`
- Run FloatSim's `static_equilibrium_solver` on the HSFP deck;
  compare displaced position to OrcaFlex's reported equilibrium.
- ~150 lines.

### PR5 — S4 drag-on free decay

- `tests/validation/test_m6_orcaflex_drag_decay.py`
- Same IC as S1, but with Morison elements active. Compare peak
  amplitudes (drag-dominated decay) to OrcaFlex's response.
- Cross-checks the M5 PR4 Morison wiring against an industry tool.
- ~200 lines.

### PR6 — S5 moored static equilibrium

- `tests/validation/test_m6_orcaflex_moored_eq.py`
- Catenary mooring lines from `_nomor_mooring` deck, FloatSim's
  Irvine analytical catenary; compare offset and per-line tensions.
- Static-only per Q7 — dynamic mooring response is Phase 2.
- ~200 lines.

### PR7 — Cross-check report + plots

- `docs/orcaflex-cross-check-report.md` — full results table, one
  section per scenario, with overlay plot links.
- `scripts/plot_orcaflex_crosscheck.py` — regenerates PNGs into
  `docs/figures/orcaflex_crosscheck/`.
- Closes M6.
- ~300 lines.

---

## Ordering rationale

PR1 first — readers and CSV loader scaffolding are independent of
scenarios, and PRs 2–6 all need the loader. PR1 also commits the
conventions doc (Q8), which is a hard prerequisite — running a
scenario without verified frames is wasted work.

PR2 (free decay) before PR3 (RAOs) — if the free response is wrong,
the RAO comparison will be confusing. Free decay isolates the
restoring + radiation; RAOs add excitation on top.

PR4 (statics) is independent of PR2/PR3 dynamics and could land in
parallel; sequencing here is for review-bandwidth reasons (one PR
in flight at a time).

PR5 (drag) needs the M5 Morison wiring, already on main.

PR6 (mooring) needs the M4 catenary, already on main.

PR7 closes the milestone.

## Risks

- **Deck identity drift.** Any difference between FloatSim's deck
  and OrcaFlex's deck — mass, BEM data, drag coefficients, mooring
  pretension — produces a discrepancy that looks like a physics
  bug. Mitigation: PR1's conventions doc is exhaustive; every PR
  starts with a deck-identity sanity check.
- **OrcaFlex version drift.** Xabier's OrcaFlex install pins a
  specific version. If the CSV fixtures are regenerated against a
  newer OrcaFlex, results may shift slightly. Mitigation: Capture
  OrcaFlex version in `docs/orcaflex-cross-check-conventions.md`;
  re-run is a documented full-cycle operation, not a silent bump.
- **Phase convention slip.** OrcaFlex stores RAO phase in degrees
  with the lags-by-default convention; FloatSim stores radians with
  leads. The OrcaFlex VesselType reader handles the BEM database
  side, but S2's per-frequency phase comparison is a fresh path —
  the comparison must explicitly conjugate one side. Bug-catcher
  test: a regular wave at heading 0° should produce surge with the
  same phase in both tools.
- **OrcFxAPI not strictly required, but helpful.** If Xabier has
  the full license, an OrcFxAPI-driven CSV export script
  (`scripts/extract_orcaflex_timeseries.py`) avoids GUI clicks for
  every fixture regeneration. Optional.
- **Moored cross-check sensitivity.** Catenary pretension depends
  exquisitely on line length, weight per unit length, and EA. Tiny
  deck-input differences propagate to ~20% tension differences.
  Mitigation: S5's tolerance budget is `rtol=5e-2` on tension,
  acknowledging that anything tighter would over-fit the deck
  match. If we see >20% disagreement, the deck is wrong, not the
  physics.

## Session-continuity notes

If a fresh session picks this up: M5 closed partially at merge
commit `8f5225d` on main (PR3 sphere fixtures deferred on branch
`milestone-5-pr3-sphere-fixtures`). M6 has no code yet, only this
plan. Branch `milestone-6-orcaflex-cross-check` is created off main
and tracks main. Q1–Q8 above are **proposals** — do not implement
until Xabier has reviewed and locked them.

The reference OrcaFlex case lives outside the repo at
`C:\Users\xlama\OneDrive\Documents\buoy\Orca\orcawave\model3small_nomor.*`.
The `.sim` time histories must be exported to CSV (Xabier's side,
either GUI or OrcFxAPI) before PR2 can write its first failing test.
