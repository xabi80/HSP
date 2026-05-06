# OpenFAST Cross-Check Report (M6)

Living document, updated per scenario PR. Captures the design
decisions, quantitative results, and known discrepancies for the
M6 OpenFAST/HydroDyn cross-check. The companion conventions doc
(`docs/openfast-cross-check-conventions.md`) is the rules of the
game; this report is the play-by-play.

| Scenario | PR | Status | Date |
| -------- | -- | ------ | ---- |
| S1 -- unmoored static equilibrium | PR2 | ✅ landed | 2026-05-04 |
| S2 -- pitch free decay (radiation-only, post-Option-A) | PR3 | ✅ landed (period xfail-strict under F1-residual) | 2026-05-05 |
| S3 -- regular-wave RAO sweep | PR4 | ⏭ pending | -- |
| S4 -- moored static equilibrium | PR5 | ⏭ pending | -- |
| S5 -- drag-on heave free decay | PR6 | ⏭ pending | -- |

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
| Pitch period | 25.67 s | -4.29 % | rtol = 2e-2 | **xfail-strict** (F1-residual) |
| Pitch ζ (first 5 cycles) | ~1 × 10⁻⁹ | n/a | ζ ≥ -1e-6 | **PASS** (kernel-fix validation) |
| Envelope trend (any growth) | none | n/a | ≤ 1 % per triple | **PASS** |
| IC application | exact | n/a | abs = 1e-6° | **PASS** |

### Known discrepancies

**KD-2: F1-residual.** Combined-deck FloatSim period is 4.29 %
short of OpenFAST. The dominant hypothesis (per the Pre-step
diagnostic doc) is the platform-with-ballast distributed-inertia
treatment: Setup B uses a single point-mass at Robertson's
CoG of -13.46 m, but the actual ballast water is distributed in
column-fill members with their own volume distribution. A
proper F1-residual fix would parse HydroDyn's `FillGroups` and
member geometry to compute the actual ballast inertia. Out of
PR3 scope; xfail-strict will catch the day this is closed.

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

- No F1 / F1-residual fix (Phase 2 follow-up).
- No tolerance widening to mask the period mismatch (xfail-strict
  is the disposition; ``rtol = 2e-2`` stays as the M6 plan
  target).
- No heave-decay variant (separate fixture; future PR if S5 needs
  per-DOF cross-checks).
- No drag-on damping cross-check (deferred to S5 / PR6 by
  design).

---

(Subsequent PRs append below as they land.)
