# OpenFAST Cross-Check Report (M6)

Living document, updated per scenario PR. Captures the design
decisions, quantitative results, and known discrepancies for the
M6 OpenFAST/HydroDyn cross-check. The companion conventions doc
(`docs/openfast-cross-check-conventions.md`) is the rules of the
game; this report is the play-by-play.

| Scenario | PR | Status | Date |
| -------- | -- | ------ | ---- |
| S1 -- unmoored static equilibrium | PR2 | ✅ landed (post-fix-pr2-cmzt audit) | 2026-05-05 |
| S2 -- pitch free decay (radiation-only, post-Option-A) | PR3 | ✅ landed (period xfail-strict under F1-revised post-fix-wamit-dim) | 2026-05-05 |
| S3 -- regular-wave RAO sweep | PR4 | ✅ landed (impedance-only path; time-domain xfail-strict pending F-WAVE-FORCE-CONV + F-DAMP-MATCH) | 2026-05-09 |
| S4 -- moored static equilibrium | PR5 | ⏭ pending | -- |
| S5 -- drag-on heave free decay | PR6 | ✅ landed (P2 narrowing: equivalent Morison aggregate; hyperbolic-envelope regime classification + δ rel-err 3.88 %) | 2026-05-10 |

---

## Named follow-ups (cross-cutting)

### F1 -- Deck-identity refinement: rebuild C_55 for OpenFAST's effective mass distribution

**Surfaced by**: PR2 pitch borderline.

**Symptom**: FloatSim's `platform_small.yml` C_55 was authored
against Robertson 2014's "platform-only" mass (m=1.347e7 kg,
z_G=−13.46 m). OpenFAST's S1 deck has the platform + tower + RNA
combined CoG at z_G ≈ −5.7 m. The pitch restoring stiffness is
dominated by ``-m·g·z_G``, which differs between the two
conventions by ≈ 1e9 N·m/rad.

**Phase 1 disposition**: ship with documented gap. The PR2 cross-
check passes by relying on small-magnitude pitch (both predicted
and observed are < 0.5°), absorbed by Item 13's tolerance.

**Phase 2 fix**: rebuild the FloatSim deck's C with OpenFAST's
*combined* (platform + locked tower + locked RNA) mass and
combined CoG when running cross-check scenarios. Either:
1. Author a new fixture `oc4_combined_with_tower.yml` that pre-
   computes the combined-CoG C, OR
2. Extend `assemble_cummins_lhs` to accept an additional rigid
   mass term that propagates into the gravity-decomposition step.

**Tracking**: this follow-up may also surface dynamically in PR3
(free-decay period scales as `sqrt(M/C)`; a 5% C_55 mismatch
gives a 2.5% period mismatch, right at PR3's `rtol=2e-2`
tolerance edge). If PR3's period assertion fails cleanly to this
mechanism, the failure is classified F1 (not a new entry) and
documented in PR3's Known Discrepancies, not papered over by
widening tolerance.

---

## PR2 -- S1 unmoored static equilibrium (closed 2026-05-04)

### Path-selection retrospective

The static-equilibrium cross-check needed one of four framings:

- **Path I -- Apply OpenFAST's net static residual as external
  force**. Compute `F_static = m_total·g − ρgV` from OpenFAST input
  files; pass through FloatSim's `state_force`; assert `xi_eq`
  matches OpenFAST's last-30-s mean. Pro: directly validates the
  linearised assembly + gravity decomposition. Con: needs an
  ElastoDyn / HydroDyn input parser.

- **Path II -- Use the same masses/inertias OpenFAST uses, with
  WAMIT marin_semi.{1,3,hst} via M5's reader**. Hand-author a
  FloatSim deck from the OpenFAST inputs. Pro: maximum deck
  identity. Con: same parsing scope; needs combined-system rigid
  mass matrix.

- **Path III -- Defer to follow-up: PR2 just notes the
  equilibrium-offset gap as a Known Discrepancy**. Pro: scoped
  to one PR. Con: not a real cross-check.

- **Path IV.b -- Reframe the test from "validate equilibrium
  agreement" to "validate linear response to deck residual"**.
  FloatSim's solver returns `xi=0` because equilibrium IS the
  linearisation point of Cummins (now codified as conventions
  doc Item 15). The 0.488 m OpenFAST heave offset is a deck-
  bookkeeping artifact (mass / displacement imbalance at the WAMIT
  reference), not physics disagreement. Compute the residual,
  apply it, assert displacement -- this is what tests the
  *linearised assembly + gravity decomposition* that the M5 PR1
  audit fixed.

**Selected: Path IV.b**. Locked by Xabier 2026-05-04.

Rationale: Path IV.b is the tightest test of what M6 is supposed
to validate. Paths I and II are equivalent in physics but I has
smaller scope (Path II requires reconstructing the rigid mass
matrix, Path I just adds a force). Path III isn't a cross-check.
Path IV.b reframes I to make the conceptual structure explicit:
the cross-check is *not* "do both tools settle to the same
equilibrium" -- it's "given the deck's residual force, do they
agree on the linearised displacement response."

The reframing also surfaced Item 15 ("Static equilibrium under
Cummins linearisation"), which is an institutional-memory
finding worth more than the test itself: future cross-check
scenarios on other tools will hit the same conceptual gap, and
the conventions doc now captures it.

### Implementation

- `tests/support/openfast_deck.py` -- 360-line OpenFAST input
  parser scoped to the residual computation. Reads 13 named
  scalars from `.fst` / HydroDyn / ElastoDyn, plus integrates
  tower and blade `TMassDen` station tables trapezoidally.
- `tests/unit/test_openfast_deck.py` -- 12 unit tests pinning
  the parser against the committed S1 fixture and synthetic
  closed-form references for the integration helper.
- `tests/validation/test_m6_openfast_static_eq.py` -- 6 cross-
  check assertions on heave / roll / pitch only (Item 14).

The one literature constant the parser depends on:
`OC4_PLATFORM_TOTAL_MASS_KG = 1.3473e7` (Robertson 2014 Table
3-1; total platform mass *including* fixed water ballast). The
alternative -- parsing HydroDyn's `FillGroups` plus member
geometry -- adds ~300 lines of OC4-specific column-geometry
handling for the same number, and was rejected as scope creep.

### Quantitative results

S1 reference (OpenFAST `s1_static_eq.csv`, TMax=600 s, last-30-s
mean):

| DOF | OpenFAST mean | last-30-s std |
|-----|---------------|----------------|
| heave | +0.4882 m  | 0.0552 m |
| roll  | −0.0000°   | 0.0005°  |
| pitch | −0.0814°   | 0.0094°  |

FloatSim with deck-residual external force (Path IV.b):

| DOF | FloatSim | Δ vs OpenFAST | Tolerance | Margin |
|-----|----------|---------------|-----------|--------|
| heave | +0.514 m | 0.026 m | ±0.15 m | **6× inside** |
| roll  | 0° (by symmetry) | <0.001° | ±0.5° | comfortable |
| pitch | +0.371° | 0.452° | ±0.5° | **0.05° margin** |

The deck residual itself, parsed from S1 inputs:
- Total mass = 1.407e7 kg (platform-with-ballast 13.473e6 +
  tower 0.250e6 + hub 56.78k + nacelle 240k + 3 blades 54.1k)
- Buoyancy at BEM reference = ρ·V·g = 1.399e8 N
- Weight = m·g = 1.380e8 N
- Net F[2] = +1.876e6 N upward
- F[4] (pitch moment from off-axis NacCMxn=+1.9 m) = +6.54e6 N·m

### Known discrepancies

**KD-1: pitch agreement is borderline (0.05° margin).**
Classified under follow-up F1. FloatSim's predicted pitch is
+0.371° (driven by NacCMxn); OpenFAST observes −0.0814°. The
difference (0.452°) sits inside Item 13's ±0.5° tolerance but
suggests OpenFAST sees a much larger effective pitch stiffness
than `platform_small.yml` carries -- consistent with the
combined-CoG hypothesis described in F1.

The test passes on agreement (both small in magnitude), not on
precise value. A future deck-identity refinement (F1) would
tighten this and likely flip pitch into the same comfortable-
margin status as heave.

### Conventions activated / closed at PR2

| Item | Description | PR2 status |
|------|-------------|------------|
| 12 | Last-30-s averaging for S1 / S4 | ✅ verified (S1 implementation in `_last_30s_mean`) |
| 13 | Tolerances accommodate residual oscillation | ✅ verified (heave 6× inside, pitch borderline) |
| 14 | Static eq tests assert only on restored DOFs | ✅ verified (skipped surge/sway/yaw per the OC4 unmoored topology) |
| 15 | Static equilibrium under Cummins linearisation | ✅ verified (zero-F_external returns xi=0; residual-driven assertions) |

### Post-landing audit and fix (`fix-pr2-cmzt`, 2026-05-05)

The PR3 diagnostic (Pre-1) flagged a convention inconsistency in
`compute_openfast_deck_residual`: the parser was reading OpenFAST's
`PtfmCMzt = -8.66 m` (steel-only platform CoG) for the
`platform_with_ballast` component while pairing it with Robertson
2014's `1.347 × 10⁷ kg` (with-ballast mass). The mismatched
`(M, z_G)` pair entered only the diagnostic `cog_total_z_m` field;
F-vector elements were unchanged because the moment formulas at
`xi=0` reference only horizontal CoG offsets (verified
numerically: F[2:5] deltas under -8.66 m → -13.46 m substitution
were ≤ 10⁻⁹ N).

**Disposition**: small fix on `fix-pr2-cmzt` branch off main,
merged before M6 PR4 starts. Same precedent as the M5 hydrostatic-
gravity fix (caveats live in code gates, not in prose).

**Changes**:

- `tests/support/openfast_deck.py`: removed the `PtfmCMzt` scan
  for the platform-with-ballast component; replaced with the
  Robertson constant `OC4_PLATFORM_COG_Z_M = -13.46 m`. Added
  `platform_cog_z_m` kwarg paired with `platform_total_mass_kg`
  for non-OC4 deck overrides.
- `tests/unit/test_openfast_deck.py`: two new unit tests pinning
  the algebraic invariant (`F_residual[2:5]` independent of
  `platform_cog_z_m` for axisymmetric on-axis-CoB decks) and the
  default consistency (Robertson mass pairs with Robertson CoG;
  guards against a future swap to OpenFAST's steel-only `PtfmCMzt`).
- Conventions doc Item 17 ("z_G must be consistent with mass M
  and stiffness C across all uses").

**Numerical impact on PR2 assertions**: zero (within 10⁻⁹ N float
noise). All 6 PR2 assertions still pass; `cog_total_z_m`
diagnostic now reports -10.701 m (all-Robertson convention)
instead of -5.308 m (mixed convention).

The fix is forward-looking: PR4 will inherit a consistent
convention, and any future scenario PR that introduces a
vertical-lever-arm moment formula (e.g., S4's mooring lines
attaching at depth, or a non-axisymmetric body) is protected by
the new invariant test.

See `docs/diagnostics/m6-pr4-pre1-cmzt-audit.md` for the full
audit.

---

## PR3 -- S2 pitch free decay (closed 2026-05-05)

### Re-scope retrospective (Option A)

PR3's original plan (period + tight `ζ` cross-check, `rtol = 5e-2`
on damping per Q4) was paused after pre-flight diagnostics
surfaced two findings.

**Mod 1** added a missing `PtfmSurge = 0.0` IC override to the S2
scenario (the baseline ElastoDyn carried a non-zero default
inherited from the OpenFAST baseline deck). Surge has zero
hydrostatic stiffness on unmoored OC4, so a 5 m IC drifts
indefinitely and contaminates the pitch response through cross-
coupling. This is the dynamic-test analogue of conventions doc
Item 14 (which forbids static-equilibrium cross-checks on
unrestored DOFs); now codified by `morison_drag_disabled` /
`PtfmSurge=0.0` in `scenario_config.py`.

**Mod 2** measured the OpenFAST pitch ζ over peaks 1-5 / 5-10 /
10-20 of the regenerated S2 reference. Result: 0.02968 / 0.01494
/ 0.00880, relative spread 117 % — far outside the 5 % stability
gate. Successive peaks decayed much faster early than late: the
**hyperbolic envelope** signature of quadratic Morison drag
(Faltinsen 1990 §4). Confirmed by inspection of the S2
`_HydroDyn.dat` (`NMembers = 25`, `MCoefMod = 3`, `PropPot = True`
on every member; OC4 published `C_D` values active on all 25
members). OpenFAST S2 dissipation was **radiation + quadratic
Morison drag**, not the radiation-only physics PR3 was scoped
against.

**Mod 4** ran a kernel-`t_max` convergence diagnostic on the
post-fix kernel (FloatSim PR3 setup with marin_semi.1):
``t_max = 100 s`` already converged to ``rtol < 1e-3`` on both
period and ζ. Locked ``t_max = 200 s`` (with headroom) for PR3.
The values themselves were striking: FloatSim period = 18.43 s
vs. OpenFAST 26.81 s (-31 %), FloatSim ζ ~ 1 × 10⁻⁹ vs.
OpenFAST 0.0297-0.0088 (3-4 orders of magnitude apart) — the
combined-deck mismatch (F1) plus the radiation-only-vs-drag
mismatch.

Three options were on the table for re-scope. **Option A** was
selected: disable Morison drag in the S2 reference so both tools
run radiation-only physics, accept that radiation damping at OC4
pitch resonance is too small to assert tightly, replace the ζ
match with non-negativity. See
`docs/diagnostics/m6-pr3-damping-stability.md` for the full
analysis and the option matrix.

### Pre-step period gap diagnostic

Before any test code landed, the period gap between FloatSim and
OpenFAST (drag-off) was classified per the locked decision tree.
Two FloatSim setups against OpenFAST's drag-off reference:

- **Setup A** (Robertson platform-only mass + Robertson
  ``C_55 = 1.078e9``): period 18.43 s, rel-err **-31.3 %**.
- **Setup B** (combined deck — platform + tower + RNA from
  OpenFAST ElastoDyn parsing + ``C_55`` recomputed with combined
  CoG): period 25.67 s, rel-err **-4.29 %**.

Setup B is **within ``rtol = 5e-2`` but beyond ``rtol = 2e-2``**
→ F1-mostly-explains; period assertion fires xfail-strict under
"F1-residual". Full numerics, residual hypotheses, and the
`PtfmCMzt` vs. Robertson convention note in
`docs/diagnostics/m6-pr3-period-gap-diagnostic.md`.

### Implementation

- `scenario_config.py` — added `morison_drag_disabled: bool` to
  `Scenario`; locked S2 with `morison_drag_disabled=True` plus
  the explicit `PtfmSurge=0.0` IC override. Mirror update in
  `openfast_setup/scenario_config.py`.
- `generate_scenario_decks.py` — added `_zero_morison_drag` hook
  invoked when `scenario.morison_drag_disabled` is set; zeroes
  all `CylMember*Cd*`, `AxCd`, `CylSimpl*Cd*` entries on the
  HydroDyn `fst_vt`. Members and joints stay in place
  (kinematics + PropPot member buoyancy via the BEM still run);
  only the quadratic Cd-based viscous drag is removed. Mirror
  update in `openfast_setup/`.
- `tests/validation/test_m6_openfast_free_decay.py` — 7
  assertions on the regenerated drag-off S2 reference:
  finiteness, IC-application (Mod 5: ``pitch[0] = 5°`` exact,
  ``pitch[1]`` essentially unchanged), pitch period
  (xfail-strict under F1-residual), pitch ζ ≥ -1e-6
  (non-negativity, kernel-fix validation), envelope-trend
  (Mod 3: geometric-mean comparison across all adjacent triples,
  no growth > 1 %), and a diagnostic-log test that emits
  per-window ζ and the first-N peaks for the report.

### Quantitative results

OpenFAST S2 reference (drag-off, regenerated 2026-05-05):

| Quantity | Value |
|----------|------:|
| Pitch period (10 inter-zero-crossing intervals) | **26.83 s** |
| Pitch peak[0] | 5.00° (essentially undecayed at first peak) |
| ζ over peaks 1-5 | 1.5 × 10⁻⁴ (numerical noise floor) |

FloatSim Setup B (combined deck on marin_semi.1, post-fix kernel):

| Quantity | FloatSim | Δ vs OpenFAST | Tolerance | Result |
|----------|---------:|--------------:|----------:|--------|
| Pitch period (pre-fix-wamit-dim) | 25.67 s | -4.29 % | rtol = 2e-2 | **xfail-strict** (F1-residual, FALSIFIED — see KD-2-revised) |
| Pitch period (post-fix-wamit-dim, locked) | 32.34 s | +20.54 % | rtol = 2e-2 | **xfail-strict** (F1-revised / KD-2-revised) |
| Pitch ζ (first 5 cycles) | ~1 × 10⁻⁹ | n/a | ζ ≥ -1e-6 | **PASS** (kernel-fix validation) |
| Envelope trend (any growth) | none | n/a | ≤ 1 % per triple | **PASS** |
| IC application | exact | n/a | abs = 1e-6° | **PASS** |

### Known discrepancies

**KD-2: F1-residual (FALSIFIED at fix-wamit-dimensionalisation).**
At PR3 land time the combined-deck FloatSim period was 4.29 %
short of OpenFAST. The dominant hypothesis was the platform-
with-ballast distributed-inertia treatment: Setup B uses a single
point-mass at Robertson's CoG of -13.46 m, but the actual ballast
water is distributed in column-fill members with their own volume
distribution. A proper F1-residual fix would parse HydroDyn's
`FillGroups` and member geometry to compute the actual ballast
inertia.

**This classification was falsified by the
fix-wamit-dimensionalisation branch (2026-05-07).** The pre-fix
WAMIT reader returned non-dimensional ``A(omega)`` values
verbatim, so ``A_55(omega_n)`` was ~1000x smaller than the
physical value. With ``M >> A`` for the rigid platform mass,
the period assertion was insensitive to the missing factor and
the small residual coincidentally landed near rtol=5e-2. After
the dim-fix, ``A_55(omega_n)`` is correctly ~7.7e9 kg·m² and
the FloatSim period jumps to 32.34 s — a +20.54 % residual in
the OPPOSITE direction. **The original distributed-inertia
hypothesis cannot be the dominant explanation** because
distributed inertia would shorten the period by adding
rotational mass, not lengthen it. The post-fix gap is genuinely
unexplained. Tracked as **KD-2-revised** below; xfail-strict
under "F1-revised" in the test reason string.

**KD-2-revised: F1-revised — post-fix-wamit-dim period gap.**
Post-WAMIT-dim-fix combined-deck FloatSim pitch period is
32.34 s vs OpenFAST 26.83 s = **+20.54 % rel-err**. Possible
causes (none yet confirmed):

1. **Mass bookkeeping**: Setup B's combined-CoG aggregation may
   double-count or under-count the platform-with-ballast moment
   of inertia. The Robertson values for ``I_55_CoG`` are at the
   platform-only CoG; combined-deck moves the reference to the
   system CoG via parallel-axis. A sign or factor error here
   would leave M+A oversized by ~ 50 % at omega_n, producing a
   ~ 25 % period overshoot — close to the observed +20 %.
2. **BEM frequency interpolation**: ``A_55(omega_n=0.194 rad/s)``
   is interpolated linearly in the marin_semi grid. At low
   omega the grid is dense (Δω = 0.01) and interpolation is
   not the dominant error. Likely small (< 1 % period effect).
3. **Cummins reference-point**: ``M+A(omega_n)`` and ``C`` are
   computed at different reference points; if the parallel-axis
   transform of A is incorrect, the period shifts. Worth
   investigating but less likely the dominant term.
4. **Hydrostatic stiffness sign / factor on the asymmetric
   off-diagonals** (e.g., ``C_15``, ``C_24``, ``C_46``).
   Robertson 2014 publishes only the diagonals; FloatSim's
   ``platform_small.yml`` zeroes the off-diagonals. If
   OpenFAST's ``HstFile`` has non-zero off-diagonals that
   contribute at omega_n, the period shifts. Check the marin_semi
   .hst file.

The investigation needs its own audit (per the M6 PR2-style
Pre-step pattern) before it can be scoped as a PR. Tracked for
M6 epilogue or post-M6.

**This is the dominant Phase 2 follow-up out of M6.** Once the
+20.54 % gap is closed, the radiation-only pitch free-decay
test will start passing rtol=2e-2; the xfail-strict marker
must come off when that happens.

**KD-3: Radiation-only OC4 pitch ζ is too small to test
tightly.** Both tools report ζ at the numerical noise floor
(~10⁻⁹ in FloatSim, ~10⁻⁴ in OpenFAST). Quantitative damping
cross-checks belong to scenarios where the dominant dissipation
is matched in both tools (S5 drag-on heave decay, M6 PR6).
Codified as conventions doc Item 16.

### Convention adds

**Item 16** (`docs/openfast-cross-check-conventions.md`): Damping
cross-check tolerance depends on dissipation regime. Three
regimes (quadratic-drag-hyperbolic, linear-radiation-exponential,
radiation-only-noise-floor); decision rule for scoping new
damping cross-checks; protocol for switching regimes when the
dominant mechanism doesn't match.

### What this PR did NOT do

- No F1 / F1-residual fix (Phase 2 follow-up). Note: the
  F1-residual classification was falsified at
  fix-wamit-dimensionalisation; see KD-2-revised above for the
  post-fix-wamit-dim story.
- No tolerance widening to mask the period mismatch (xfail-strict
  is the disposition; ``rtol = 2e-2`` stays as the M6 plan
  target).
- No heave-decay variant (separate fixture; future PR if S5 needs
  per-DOF cross-checks).
- No drag-on damping cross-check (deferred to S5 / PR6 by
  design).

---

## PR4 -- S3 RAO sweep (in flight, post-fix-s3-wavemod)

### Pre-2 WaveMod misconfiguration (closed 2026-05-06 on `fix-s3-wavemod`)

Before any PR4 RAO test code lands, the Pre-2 audit caught a
misconfiguration that had been latent across S1-S3 commits: the
S3 scenario carried `seastate_edits = {"WaveMod": 2, ...}` with a
comment `# regular Airy`. `WaveMod = 2` is JONSWAP irregular
spectrum; `WaveMod = 1` is regular Airy. All 14 S3 scenarios had
been generating irregular waves; the long-period scenarios
(WaveTp ≥ 20 s) had their JONSWAP peak entirely below
`WvLowCOff = 0.314 rad/s` and produced wave trains with **no
spectral content at the labeled frequency at all**.

Disposition (`fix-s3-wavemod` branch off main, merged before
PR4 starts):

1. **Vendored generator removal**: the duplicate generator at
   `tests/fixtures/openfast/oc4_deepcwind/baseline/case/`
   (`scenario_config.py`, `generate_scenario_decks.py`,
   `run_scenarios.py`) had drifted to a non-functional state
   (still on the OpenFAST v3 schema with `WaveMod` in
   `hydrodyn_edits` while the authoritative
   `openfast_setup/scenario_config.py` migrated to v4+ with
   `seastate_edits`). The vendored copy would have raised
   `KeyError` if anyone had run it. Removed in this branch
   rather than re-synced — maintaining two copies is a
   perpetual drift risk for no benefit since the
   contributor-machine generator is the only one that runs.
   `CLAUDE.md` §14 updated with the "deck regeneration is a
   contributor-machine task; no in-tree generator" note;
   `tests/fixtures/openfast/oc4_deepcwind/baseline/case/README.md`
   rewritten to say so.
2. **WaveMod fix**: one-line change in
   `openfast_setup/scenario_config.py`:
   `WaveMod: 2 → WaveMod: 1`. Comment corrected.
3. **Smoke test on WaveTp = 10 s** (regenerate one, run, FFT):
   wave-elevation FFT showed clean monochromatic peak at exactly
   10 s (secondaries < 0.03 % of main); the pitch response showed
   the main peak at 10 s plus an 8.2 % secondary at T ≈ 28.6 s
   (the OC4 pitch natural-period transient — regime 3 per
   Item 16). The pitch contamination is a known consequence of
   the OC4 pitch's low effective damping at small amplitude; it
   is handled in PR4's RAO extractor by sinusoidal lstsq fit
   at the wave frequency, not by extending TMax.
4. **Full S3 regeneration**: all 14 sweep variants regenerated
   with `WaveMod = 1`. CSVs re-extracted.
5. **Deck-generation regression test** in
   `openfast_setup/tests/test_scenario_decks.py` asserts that
   the generated `*_SeaState.dat` `WaveMod` value matches what
   `seastate_edits` declared for every scenario. This is the
   regression gate that catches the bug at deck-generation time
   rather than at PR4 RAO-extraction time. Same precedent as
   PR4 Pre-1's `test_F_residual_invariant_to_platform_cog_z`.
6. **Conventions doc Items 18, 19, 20, 21**:
   - Item 18: wave-mode value must match intent.
   - Item 19: code-path exercise principle (generalised across
     four findings).
   - Item 20: RAO extraction requires frequency-selective
     filtering (sinusoidal lstsq over band-pass; phase-shift-
     free, residual-diagnostic-friendly).
   - Item 21: OpenFAST quantises ``WaveTp`` to the nearest IFFT
     bin (``WaveDOmega = 2π / WaveTMax``); RAO fits must use
     the quantised period, not the labelled one. Surfaced when
     the post-fix Pre-2 gate flagged elevated residuals on
     WaveTp = 16/18/22 (non-divisors of WaveTMax = 600 s) and
     was traced via inter-zero-crossing measurements showing
     the actual wave was at the IFFT-snapped bin.
7. **CLAUDE.md §13 updated** with the four-bug pattern lock.

### Pre-2 closure quantitative summary

After fix + regen + IFFT-bin-quantisation fit correction, all 14
wave-elevation channels produce clean monochromatic content at
the configured frequency. Worst case: WaveTp = 4 s with
amp rel-err = -0.31 % and fit-residual / signal = 0.0017. Both
gates (rel-err < 2 %, residual < 5 %) pass with ≥ 6× margin on
every scenario. Full table in
`docs/diagnostics/m6-pr4-pre2-steady-state-check.md`.

### Pre-flight diagnostics archived

- `docs/diagnostics/m6-pr4-pre1-cmzt-audit.md` — `PtfmCMzt`
  convention audit (fix landed on `fix-pr2-cmzt`, see PR2
  retrospective addendum).
- `docs/diagnostics/m6-pr4-pre2-steady-state-check.md` — original
  pre-fix WaveMod=2 finding plus post-fix wave-generation
  verification table.
- `scripts/m6_pr4_pre2_smoke_wavetp_10.py` — single-frequency
  smoke test confirming WaveMod=1 generates clean Airy waves.
- `scripts/m6_pr4_pre2_steady_state.py` — full 14-frequency
  wave-generation verification (lstsq amp + residual).

PR4 implementation pending Pre-3 (RAO definition lock-down).

### PR4 implementation (closed 2026-05-09 with G3 narrowing + H1 marker refinement)

**Scope landed**: `tests/validation/test_m6_openfast_regular_wave.py`
validates FloatSim's **impedance-domain** RAO computation against
OpenFAST across 14 wave periods × 3 DOFs (heave, roll, pitch) ×
2 metrics (amp, phase) = 84 assertions. Time-domain Path A
preserved as xfail-strict pending two named follow-ups
(F-WAVE-FORCE-CONV + F-DAMP-MATCH; see below).

**Narrowing path**: PR4 was originally planned as time-domain
validation (Decision A locked at PR4 scoping). The Decision A
structural sub-check (time-domain ≅ impedance at WaveTp = 10 s,
25 s) revealed two distinct issues at first run:

1. Pitch phase mirrored vs OpenFAST (`FS_lag ≈ -OF_lag`,
   ~163° rotation) — sign-flip signature on Im(F_exc).
2. Heave amp/phase contamination from un-decayed free-decay
   transient (ζ_heave_radiation_only = 0.057 %; e-folding time
   81 min; OpenFAST's clean fit traced to MoorDyn dynamic
   damping that FloatSim's analytic catenary does not capture).

After confirming the heave damping ratio from first principles
and verifying the convention mechanism via
`scripts/m6_pr4_im_ratio_diagnostic.py`, PR4 was narrowed to
impedance-only Path A (G3) and time-domain split per metric (H1).

### Quantitative results (impedance Path A)

| outcome | count | comment |
|---|---:|---|
| passed | 23 | clean cross-check (heave 5/7/8/10/12/20/22/25/30 s; pitch 8/10/12/22 s amp + 8/10/12/14/18/22 s phase) |
| skipped (heading 0 not-excited) | 28 | roll at all 14 periods × 2 metrics |
| skipped (F-LOW-SNR, resp_resid > 0.10) | 22 | heave 4/5/6 s, pitch 4-8 s + 30 s |
| xfailed (F1-revised) | 12 | pitch amp at 14/16/18/20/22/25 s, pitch phase at 16/20/22/25 s |
| xfailed (F-RESONANCE-PEAK-FRAGILITY) | 5 | heave amp at 16/18 s, heave phase at 14/16/18 s |
| xfailed (F-WAVE-FORCE-CONV + F-DAMP-MATCH, time-domain) | 4 | TD pitch amp at 10/25 s, TD pitch phase + heave phase at 10/25 s |
| xpassed (TD heave amp at Pre-3 freqs, non-strict expected) | 2 | heave amp passes accidentally at WaveTp = 10/25 s (small Im/F + transient orthogonality); marker is non-strict |

Full per-period diagnostic table at
`docs/diagnostics/m6-pr4-rao-sweep-results.md`.

### Named follow-ups added at PR4

**F-WAVE-FORCE-CONV** -- `make_regular_wave_force` consumes its
F_exc input under `exp(-i*omega*t)` convention while the WAMIT
reader stores F_exc under `exp(+i*omega*t)` (Item 24). Result:
the time-domain force has the wrong sign on its sin component,
producing a conjugated motion. Phase prediction `2 * |arg(F_exc)|`
matches the empirical pattern (`scripts/m6_pr4_im_ratio_diagnostic.py`).
Investigation will land on `fix-make-regular-wave-force-convention`
branch off main after PR4 merges. The fix is well-scoped: either
(a) conjugate Im(F_exc) on read in the WAMIT reader, OR
(b) change `make_regular_wave_force` to use `+i` convention and
update M3/M5 callers. Choice depends on which side fewer downstream
consumers are touched. Synthetic unit test pinning
`make_regular_wave_force`'s convention by predicting the
time-domain force from first principles will land with the fix.

**F-DAMP-MATCH** -- forced-response time-domain cross-checks of
lightly-damped DOFs require the dominant damping mechanism to be
matched in both tools. OC4 unmoored (PR4's setup) has
ζ_heave_radiation_only = 0.057 %; OpenFAST gets a clean fit only
because MoorDyn provides dynamic mooring damping. Future
forced-response time-domain validation should be moved to S5
(drag-on heave decay; Morison drag dominates radiation in both
tools), or wait until FloatSim's analytic catenary is augmented
with MoorDyn-equivalent dynamic damping (out of Phase 1 scope).

**KD-2-revised** -- already tracked from fix-wamit-dimensionalisation;
PR4 confirmed empirically: pitch RAO disagrees in the resonance band
(WaveTp 14-30 s) consistent with the +20.54 % T_n_pitch shift.

**TODO-FRAGILITY-BAND-CRITERION** -- the empirical ±25 % omega_n
F-RESONANCE-PEAK-FRAGILITY band (Item 28) should be replaced by a
principled impedance-magnitude criterion (`|Z(omega)| < K * |Z(omega_n)|`
for chosen K) when a future cross-check at a different platform
makes the empirical band miscalibrated.

### Convention adds at PR4

- Item 26: MoorDyn dynamic damping vs analytic catenary
- Item 27: Free-decay vs forced-response damping tolerance
- Item 28: F-RESONANCE-PEAK-FRAGILITY (±25 % omega_n band, empirical)
- Item 29: F-LOW-SNR skip threshold (resp_resid > 0.10)

### Pre-step diagnostics archived

- `scripts/m6_pr4_resonance_fragility.py` -- 9.3 % interpolation-
  scheme span at omega_n_heave (Item 28 calibration)
- `scripts/m6_pr4_im_ratio_diagnostic.py` -- |Im/F| ratio sweep
  predicting F-WAVE-FORCE-CONV phase pattern; matches empirical

### What this PR did NOT do

- No time-domain forced-response validation (deferred per
  F-WAVE-FORCE-CONV + F-DAMP-MATCH). Time-domain pipeline
  remains exercised by M3 (synthetic) + M5 (drag) + M6 PR3
  (free decay) + the xfail-strict dual-path test in PR4.
- No KD-2-revised fix (deferred).
- No fix to `make_regular_wave_force` (deferred).

---

## PR6 -- S5 heave drag decay (closed 2026-05-10)

### Scope at PR6 (P2 narrowing per the locked plan)

Validates FloatSim's Morison-drag time-domain pipeline against
OpenFAST's S5 heave free-decay reference using the **hyperbolic-
envelope** signature characteristic of drag-dominated quadratic
damping (Faltinsen 1990, Ch. 4 + Item 16 regime classification).

Three options were on the table at PR6 scoping. **P2** (equivalent
Morison aggregate) was locked: rather than wire 25 Morison elements
+ 3 axial-drag joints + 3 MoorDyn catenary lines into FloatSim, use
a single calibrated equivalent Morison element whose lumped
``Cd · D · L`` matches the aggregated heave drag from the full OC4
deck. The full deck-identity test is deferred to a future PR.

### Pre-flight diagnostics + factor-of-2 finding

`scripts/m6_pr6_drag_aggregation.py` parses the S5 HydroDyn deck
and aggregates the per-member cylindrical Morison + per-joint
axial drag with appropriate heave-direction projections (sin³θ for
cylindrical, cos³θ for axial). First-principles hyperbolic-decay
derivation gives ``δ = (8/3) · R / m_eff``.

First pass: predicted δ = 0.620 1/m vs OpenFAST measured 0.309 1/m
(2.005× off, suspiciously clean factor of 2). Investigated per
the locked Q2 protocol (Outcome (a), Phase 1):

Source read of `OpenFAST/modules/hydrodyn/src/Morison.f90` lines
3085 + 4742 confirmed that HydroDyn's joint axial drag formula
applies the coefficient with a **1/4 factor**, not the standard
Morison 1/2:

```
F_z = -(1/4) · ρ · A_x · JAxCd · v_z · |v_z|       (HydroDyn)
F_z = -(1/2) · ρ · A_x · Cd_axial · v_z · |v_z|    (standard Morison)
```

`JAxCd` is implicitly a "two-face combined disc" coefficient;
per-face Morison-equivalent is `JAxCd / 2`. The HydroDyn User's
Guide does NOT state this — only the source does. Codified as
**Item 30** in conventions doc.

After applying the 1/4 correction, the aggregated prediction is:

```
δ_predicted = 0.3130 1/m  (1.28 % rel-err vs OpenFAST 0.3090)
```

### Equivalent Morison element (calibrated aggregate)

```
R_total = 3.4073e+06 kg/m
  cylindrical contribution: 6.87e+04 (2 %)
  axial contribution:       3.34e+06 (98 %, from 3 heave plates)

Equivalent: D = 24 m, L = 24 m, Cd = 11.54  (Cd·D·L = 6648 m²)
```

The high Cd reflects 3 heave plates with HydroDyn AxCd = 9.6
each aggregated into a single Morison-equivalent element.

### Implementation

`tests/validation/test_m6_openfast_drag_decay.py` builds a single
horizontal Morison element at the body reference, runs FloatSim's
Cummins + Morison pipeline at 600 s, and extracts the heave
hyperbolic envelope over peaks 0-15 (drag-dominated regime; below
~0.14 m the envelope flattens as radiation + mooring linear damping
take over — F-DAMP-MATCH regime crossover).

### Quantitative results

| metric | FloatSim | OpenFAST | rel-err | gate | status |
|---|---:|---:|---:|---:|---|
| Hyperbolic δ (peaks 0-1) | 0.3213 1/m | 0.3093 1/m | +3.88 % | rtol=5e-2 | **PASS** |
| Hyperbolic envelope RMS (peaks 0-15) | < 5 % | < 5 % | — | < 5 % | **PASS** (both) |
| Exponential RMS vs hyperbolic RMS | > 5x | > 5x | — | ≥ 5x discrim | **PASS** (both) |

Per the test architecture, three assertions land:

- `test_openfast_envelope_is_hyperbolic_drag_dominated` — pins OF
  reference regime classification (Item 16).
- `test_floatsim_delta_matches_openfast` — primary cross-check
  assertion.
- `test_floatsim_envelope_is_hyperbolic_not_exponential` — pins FS
  regime classification on the OC4-equivalent setup (M5 PR5's
  discriminator extended from synthetic to real-deck-aggregate).

### Convention adds at PR6

- **Item 30**: HydroDyn joint axial drag uses 1/4 factor, not
  standard Morison 1/2. Pinned by `scripts/m6_pr6_drag_aggregation.py`
  + this report.

### What this PR did NOT do

- No full 25-Morison + 3-joint + 3-MoorDyn deck-identity test.
  PR6 validates the drag MECHANISM via the equivalent aggregate;
  the deck-identity exercise is a future PR if PR6's regime-level
  agreement is judged insufficient for downstream needs.
- No mooring (test isolates the drag; equilibrium offset matched
  by IC, see test docstring).
- No coupling with waves or surge/pitch DOF (heave-only).
- No new named follow-ups required: the factor-of-2 mystery
  resolved cleanly within the Q2 budget (Outcome (a)).

---

(Subsequent PRs append below as they land.)
