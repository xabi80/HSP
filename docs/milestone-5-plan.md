# Milestone 5 Plan — WAMIT + Capytaine Readers, Morison Drag

Working document. Status: **locked, not yet started.** Delete or archive after M5 merges.

Scope per ARCHITECTURE.md §8: pluggable BEM readers (WAMIT, Capytaine), Morison drag elements, full Phase 1 validation suite.
Validation gates per §7:

1. Same physical case (fully submerged sphere) loaded via all three readers (OrcaFlex YAML, WAMIT, Capytaine) → identical `HydroDatabase` to a tight tolerance, with analytical added mass `A_ii = (2/3)·π·ρ·R³` as the absolute reference.
2. Heave free decay with a Morison drag disk → analytical hyperbolic-envelope decay (closed-form quadratic damping).
3. M2/M3/M4 gates re-run with no regression after the reader dispatch + drag wiring lands.

Branch: `milestone-5-readers-drag` (created off `main` at `6b2ea41`).

---

## Decisions (from Xabier, locked)

### Q1 — Morison kinematics

- **Relative velocity** `(u_fluid − u_body)`. Standard Morison; non-negotiable for floating bodies.
- **Wave kinematics: linear Airy clipped at MWL.** Wheeler stretching deferred to Phase 2 — emit a one-line `# TODO(phase-2): Wheeler stretching` comment in the kinematics helper.
- **Midpoint quadrature** per `morison_member`. Slender-body assumption — members must be short relative to the wavelength.
- **Member-normal projection.** The drag formula is

  ```
  dF_drag/dl = 0.5 · ρ · D · Cd · |u_n| · u_n
  ```

  where `u_n` is the component of `u_rel` projected onto the plane normal to the member axis (`u_n = u_rel - (u_rel · ê_axis) · ê_axis`). Members aligned with flow contribute negligible drag — the full-vector formula is wrong for arbitrary orientations.
- **Drag-only by default.** The Morison force has both drag and inertia (`ρ·V·(1+Ca)·a_fluid − ρ·V·Ca·a_body`). For BEM-coupled bodies, the inertia is already in `A(ω)` and `F_exc` — applying both double-counts. Add deck flag `include_inertia: bool = False` to `MorisonMember`. When the user sets `true` AND the body has a non-empty BEM database, emit a startup-time warning naming the affected member.

### Q2 — WAMIT scope and fixture

- **Files read:**
  - `.1` — `A(ω)`, `B(ω)`, plus the `Infinity` and `Zero` rows for `A∞`. Primary input.
  - `.3` — `F_exc(ω, β)` complex. Primary input.
  - `.hst` — hydrostatic stiffness `C` (6×6). Primary input.
  - `.4` — motion RAOs. Read as a cross-check vs RAO assembled from `(M, A∞, B, C, F_exc)`. Optional but valuable.
- **Fixture:** OC4 DeepCwind `marin_semi.{1,3,hst,4}` from the OpenFAST r-test repo:
  - Source: <https://github.com/OpenFAST/openfast/tree/main/reg_tests/r-test/glue-codes/openfast/5MW_Baseline/HydroData>
  - Geometry: Robertson et al., *Definition of the Semisubmersible Floating System for Phase II of OC4*, NREL/TP-5000-60601, 2014.
  - License: NREL output, Apache 2.0 — safe to commit with attribution.
  - Multi-tool-validated (HydroDyn, ProteusDS, WEC-Sim, AQWA): parser disagreements point at parser bugs, not reference uncertainty.
- **Trim policy.** Commit a frequency-trimmed, single-heading subset as `tests/fixtures/bem/wamit/marin_semi_trimmed.{1,3,hst,4}` — keeps repo size sane while preserving the file-format edge cases (Inf/Zero rows, header layout). Full files (if needed for manual sanity checks) live outside the repo.
- **Attribution:** ship `docs/wamit-fixture-attribution.md` alongside the fixture noting source URL, license, geometry citation, and the trim recipe.

### Q3 — Capytaine output

- **Add `netCDF4` and `xarray` to the baseline dep list.** Capytaine's documented output API is xarray-backed; raw NetCDF parsing is needlessly verbose.
- `floatsim/hydro/readers/capytaine.py` opens the file via `xarray.open_dataset(...)` and indexes by labeled dims (`omega`, `radiating_dof`, `influenced_dof`, `wave_direction`).
- CLAUDE.md §9 baseline list updated as part of PR1.
- Capytaine itself is **not** a runtime dep — we only consume its NetCDF output. Any sphere fixture is generated offline by a `scripts/` helper.

### Q4 — Cross-check fixture

- **Fully submerged sphere, depth ≥ 5R** (no free-surface effects). Analytical:
  - `A_ii = (2/3)·π·ρ·R³` for all three translational DOFs (Lamb 1932, §92).
  - `B_ii = 0` (deep submergence, no radiation damping).
  - `C = 0` (neutrally buoyant, fully submerged).
- All three readers must agree with each other AND with the analytical reference — that's what makes this a real test of the readers rather than a tool-vs-tool consistency check.
- **Synthetic per-reader unit fixtures** sit alongside as scaffolding: hand-authored small files exercising header variations, frequency counts, complex-number formatting. These catch parser bugs the sphere case does not (header edge cases, missing rows, etc.).

### Q5 — Drag validation gate

- **Heave free decay with a horizontal drag disk.** Closed-form quadratic-damping envelope:

  ```
  ξ_n = ξ_0 / (1 + n · ξ_0 · δ)
  ```

  where `δ = (4/3) · Cd · ρ · A_disk / (m + A∞_zz)` (Faltinsen Ch. 4) and `n` is the cycle index. **Hyperbolic, not exponential.** Linear damping gives `ξ_n = ξ_0 · e^{−nζ}` — superficially similar over 1–2 cycles, sharply different over 5.
- Tolerance: `rtol=1e-3` on the first 5 peak amplitudes against the analytical envelope. Tighter and you fight integrator error; looser and the test stops detecting drag bugs.
- Bonus check: at small amplitude the radiation-damping (linear) regime dominates; confirm the small-amplitude decay rate matches `B(ω_n)` to M2 tolerance. One test, two physical mechanisms, both with closed-form references.

### Q6 — Reader factory dispatch

- **Plain `if/elif`** in `floatsim/hydro/readers/__init__.py` exposing `load_hydro_database(path: Path, format: Literal["orcaflex", "wamit", "capytaine"]) -> HydroDatabase`. Twenty lines, zero magic. Revisit only if external users need plugin support (Phase 4+).

### Process scope

- **No deck-schema changes outside the `MorisonMember.include_inertia` flag.** This milestone is reader + drag, not a YAML overhaul. `MorisonMember`, `Body.drag_elements`, `connection` types — all already exist from earlier milestones. M5 wires drag into the integrator and adds the one boolean.

---

## PR sequence

### PR1 — Reader dispatch + WAMIT reader + dep update

- `floatsim/hydro/readers/__init__.py` — `load_hydro_database(path, format)` if/elif dispatch (Q6).
- `floatsim/hydro/readers/wamit.py` — parses `.1`, `.3`, `.hst`, `.4` (Q2).
- `tests/fixtures/bem/wamit/marin_semi_trimmed.{1,3,hst,4}` — trimmed OC4 reference.
- `tests/fixtures/bem/wamit/synthetic_*.{1,3,hst,4}` — hand-authored unit fixtures.
- `docs/wamit-fixture-attribution.md` — source URL, license, geometry citation, trim recipe.
- `CLAUDE.md` §9 baseline dep update: add `xarray` + `netCDF4`.
- `pyproject.toml` — add the two deps.
- **Red tests:**
  - `.1` parser: `A∞` (Inf row), zero-frequency row, `A(ω)`, `B(ω)` match expected to `rtol=1e-12`.
  - `.hst` parser: 6×6 `C` matches expected.
  - `.3` parser: complex `F_exc(ω, β)` magnitude and phase match.
  - `.4` parser: motion RAO at one frequency matches assembled prediction from `(M, A∞, B, C, F_exc)` to `rtol=1e-3`.
  - `load_hydro_database(..., format="wamit")` returns a `HydroDatabase` with the right shape; `A`, `B`, `C` symmetry validated.
- ~600 lines.

### PR2 — Capytaine reader

- `floatsim/hydro/readers/capytaine.py` — `xarray.open_dataset` driven, indexed by labeled dims (Q3).
- `tests/fixtures/bem/capytaine/sphere_submerged.nc` — fully submerged sphere case (Q4), generated offline by `scripts/build_sphere_capytaine_fixture.py` and committed.
- `tests/fixtures/bem/capytaine/synthetic_*.nc` — hand-authored unit fixtures.
- **Red tests:** mirror PR1's parser pattern; one synthetic NetCDF authored by the test, one real Capytaine output. Sphere case must reproduce the analytical `(2/3)·π·ρ·R³` to BEM panel-density tolerance (`rtol≈1e-2` for a coarse mesh).
- ~450 lines.

### PR3 — Three-reader cross-check (validation gate, Q4)

- `tests/validation/test_bem_reader_cross_check.py` — load fully-submerged sphere through all three readers; assert `HydroDatabase` agreement and analytical agreement.
- OrcaWave sphere fixture: run a sphere case in OrcaWave (Xabier), commit the YAML.
- WAMIT sphere fixture: run the matching sphere case in WAMIT (Xabier), commit trimmed `.1/.3/.hst`.
- Capytaine sphere fixture: built by the script in PR2.
- **Red tests:**
  - All three readers report `A_ii` agreeing with analytical `(2/3)·π·ρ·R³` to `rtol=5e-3` (numerical-method scatter expected).
  - `B_ii(ω) → 0` to `atol=1e-2 · ρ·R³·ω` for all three readers (deep submergence).
  - `C ≈ 0` to `atol=1e-3 · ρ·g·R²` (neutral buoyancy).
- ~300 lines.

### PR4 — Morison drag element

- `floatsim/hydro/morison.py` — element-wise Morison force per Q1.
  - Member-normal projection of `u_rel` before applying drag.
  - `include_inertia=False` default; `True` adds `ρ·V·(1+Ca)·a_fluid_normal − ρ·V·Ca·a_body_normal`.
  - Linear Airy clipped at MWL for fluid kinematics; `# TODO(phase-2): Wheeler stretching` comment in the kinematics helper.
- `floatsim/io/deck.py` — extend `MorisonMember` with `include_inertia: bool = False` (only schema change in M5).
- Integrator wiring: `Body.drag_elements` consumed in the per-step state-force assembly (currently a no-op).
- Body-frame transforms: each member's `node_a`, `node_b` rotate with the body's quaternion (`R(q)` from M4 PR2).
- **Red tests:**
  - Single cylinder, fixed body, uniform flow normal to member → analytical `F = 0.5·ρ·D·L·Cd·u²`.
  - Single cylinder, fixed body, uniform flow parallel to member → `F ≈ 0` (member-normal projection check; the bug-catcher test).
  - Single cylinder, oscillating fluid, fixed body → KC-number behaviour (drag + optional added-mass term agree with analytical).
  - Startup warning emitted when `include_inertia=True` AND the body has a non-empty BEM database.
- ~700 lines.

### PR5 — Drag validation gate (Q5)

- `tests/validation/test_m5_drag_free_decay.py`.
- Compare integrator output peak amplitudes to analytical hyperbolic envelope `ξ_n = ξ_0 / (1 + n·ξ_0·δ)`.
- Bonus: low-amplitude regime matches B(ω_n) prediction.
- **Red test:** first 5 peaks match analytical envelope to `rtol=1e-3`. **The envelope shape is the test** — do not fit an exponential.
- ~300 lines.

### PR6 — Phase 1 validation re-run + docs

- Run all M2/M3/M4 validation gates; confirm no regression from PR1–PR5.
- `docs/wamit-format-notes.md`, `docs/capytaine-format-notes.md`, `docs/morison-conventions.md` — short pages on units, sign conventions, fields read/ignored, the `include_inertia` rule.
- No new tests; pure regression + docs.
- ~150 lines.

---

## Ordering rationale

PR1 first — readers are independent of drag, and PR3 needs both new readers. PR2 follows PR1 to share dispatch + dep scaffolding. PR3 caps the reader work with a validation gate. PR4 (drag) is independent of readers but depends on M4 PR2's body-frame transforms. PR5 caps drag with a validation gate. PR6 closes the milestone.

## Risks

- **PR1 fixture license/trim.** OpenFAST is Apache 2.0 (compatible with our project), but every committed fixture needs an attribution entry. `docs/wamit-fixture-attribution.md` is the single source.
- **PR2/PR3 sphere fixture lift.** Three sphere computations needed: OrcaWave (Xabier), WAMIT (Xabier or NREL example), Capytaine (script). Whichever lags becomes PR3's gating item.
- **PR3 numerical scatter.** Three independent BEM solvers won't agree to machine precision. `rtol=5e-3` for `A∞` is realistic; if scatter is wider in practice, investigate panel density before widening tolerance.
- **PR4 implicit-stability of `−|u_n|·u_n`.** Quadratic drag is non-Lipschitz at `u_n=0`. Explicit treatment is fine for typical offshore Cd, but emit a startup-time damping diagnostic mirroring M4 PR3's connector-stability check.
- **PR5 envelope-vs-exponential trap.** Hyperbolic and exponential decay look similar over 1–2 cycles; over 5 they diverge sharply. Test must compare to the hyperbolic form, not an exponential fit.

## Session-continuity notes

If a fresh session picks this up: M4 is merged at commit `6b2ea41` on `main`. No M5 code exists yet. Q1–Q6 are **locked** above. Branch `milestone-5-readers-drag` is created and tracks `main`. Start PR1 by writing failing `.1` parser tests against a synthetic mini-fixture, then add the trimmed `marin_semi.*` fixtures and the production reader.
