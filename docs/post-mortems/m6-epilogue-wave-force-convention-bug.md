# Post-mortem -- wave-force phase-convention bug (F-WAVE-FORCE-CONV)

**Discovered:** 2026-05-06 during M6 PR4 Pre-3 dual-path RAO
verification. Diagnosis locked at M6 PR4 G3 narrowing
(`scripts/m6_pr4_im_ratio_diagnostic.py`).
**Fixed in:** branch `fix-make-regular-wave-force-convention`, the M6
epilogue. Merged to `main` 2026-05-11.
**Severity:** Phase 1 latent bug. The function consumed BEM-reader
output under the wrong sign convention from M3 onward; surfaced when
PR4 first cross-checked a time-domain wave-forced motion against an
independent (OpenFAST) ground-truth.

## TL;DR

`floatsim.hydro.excitation.make_regular_wave_force` realised the
time-domain wave-excitation force as

    F(t) = Re{ X * eta_hat * exp(-i * omega * t) }     # bug

where `X = HydroDatabase.RAO[dof, omega, heading]` is the complex
first-order excitation force per unit wave amplitude as loaded from
the BEM reader. But every BEM reader in FloatSim (WAMIT, OrcaFlex
VesselType YAML, Capytaine) stores `X` under the **WAMIT / HydroDyn /
OrcaFlex ecosystem convention**, which is

    F(t) = Re{ X * eta_hat * exp(+i * omega * t) }     # ecosystem +i

Feeding ecosystem-stored `X` into the bug's `-i` time evolution
conjugates the imaginary part. The motion produced is the **conjugate**
of the physical motion: same amplitude, mirror-reflected phase about
the real axis. The signature is `FloatSim_lag ≈ -OpenFAST_lag` on
every DOF with non-trivial `Im(X)`, with the discrepancy scaling
as `2 * |arg(X)|`.

The fix flips `make_regular_wave_force` (and the supporting `eta_hat`
spatial phase) to the `+i` convention, matching the BEM-reader
ecosystem. The choice (Option (b)) follows from a downstream-consumer
audit (§ Audit, below); Option (a) -- conjugate on read in the WAMIT
reader -- would have required flipping the impedance-domain code in
`scripts/m6_pr4_pre3_rao_verification.py` (Pre-3-validated), Item 24
LEAD/LAG codification, and the central `HydroDatabase.RAO` convention.

## Why the bug stayed latent through M3-M6 PR3

The convention is invisible in any test where the BEM-stored RAO has
**zero imaginary part**, where the body has no offset, or where the
test author defines the RAO and the analytical prediction in the same
internal convention (the two cancel).

- **M3 (synthetic RAOs)**: `tests/validation/test_m3_regular_wave_steady_state.py`
  builds RAOs by hand and predicts the steady-state response from a
  self-consistent `-i`-convention impedance assembled in the same
  test file. The analytical prediction and the time-domain code both
  used `-i` internally; the comparison was invariant under the
  convention flip.
- **M5 drag (calm sea)**: `tests/validation/test_m5_drag_free_decay.py`
  is a free-decay test in still water; `make_regular_wave_force` is
  not called.
- **M5 reader tests**: `tests/unit/test_wamit_reader.py` and
  `tests/unit/test_orcaflex_vessel_yaml.py` verify the file-parsing
  contract but do not exercise the reader → `make_regular_wave_force`
  composition.
- **M6 PR2/PR3 (static eq + free decay)**: no wave forcing on the
  scenario, so the bug is unreachable.
- **M6 PR4 Pre-3 (the surface)**: this is the first PR where (a) the
  BEM reader provides a complex-valued RAO, (b) that RAO is fed into
  `make_regular_wave_force` to drive a time-domain simulation, and
  (c) the result is cross-checked against an independent
  ground-truth (OpenFAST). The two paths' phase residuals at the
  Pre-3 frequencies (WaveTp = 10 s and 25 s) were the diagnostic
  signature.

This is the same shape as the WAMIT-dim and radiation-kernel latent
bugs: **a code path correct in synthetic / unit / partial-scenario
tests was silently wrong under production-quality inputs and
full-scenario activation** -- conventions doc Item 19 (the code-path
exercise principle).

## How the bug surfaced

At M6 PR4 Pre-3 the dual-path RAO verification compared:

- **Path A (FloatSim impedance)**: `xi_hat = Z(omega)^{-1} F_exc`,
  with `Z = -omega^2 (M+A) + i*omega*B + C` (`+i` convention).
- **Path B (OpenFAST lstsq)**: lstsq fit of heave/roll/pitch and
  wave elevation on the OpenFAST CSV, atan2(B,A) phase reporting
  (`-i` convention internally).

Path A and Path B converged at the Pre-3 frequencies after the
LEAD-vs-LAG fix in `fix-wamit-dimensionalisation` (Item 24). The
remaining mystery was: when the **time-domain** FloatSim path
(`make_regular_wave_force` → `integrate_cummins` → lstsq fit) was run
through the same dual-path, its phase residual was systematically
mirror-reflected about Path A:

    pitch at WaveTp = 10 s : FS_TD = -163° lag vs OF = +163° lag (off by ~326°)
    pitch at WaveTp = 25 s : FS_TD = -163° vs OF = +163° (same signature)
    heave at WaveTp = 10 s : FS_TD = -1.4° vs OF = +1.4°
    heave at WaveTp = 25 s : FS_TD = -16° vs OF = +16°

The predicted error magnitude is `2 * |arg(X_loaded)|`:

- Pitch X at the Pre-3 frequencies: `|Im(X)/|X|| ≈ 1.0` (nearly
  pure imaginary in OC4 pitch RAO), so `|arg(X)| ≈ 90°`, predicting
  ~180° phase error. Observed: 163° (within the small-offset of
  arg(X) ≠ exactly π/2).
- Heave X at the Pre-3 frequencies: `|Im(X)/|X|| ≈ 0.02`, predicting
  ~2-3° phase error from F-WAVE-FORCE-CONV alone. Observed gap was
  ~16/1.4° -- F-WAVE-FORCE-CONV contributes ~2-3° and F-DAMP-MATCH
  (un-decayed free-decay transient at zeta_heave = 0.057 %) accounts
  for the rest.

The structural prediction "phase residual scales with `arg(X_loaded)`"
locked the mechanism at M6 PR4 G3 narrowing. PR4 was scoped to the
impedance path (Path A only) and the time-domain dual-path test was
marked xfail-strict pending this epilogue.

## Fix

`make_regular_wave_force` was changed to consume `HydroDatabase.RAO`
under the `+i` convention. The two-line change is:

    # before (-i convention -- bug):
    eta_hat = wave.amplitude * np.exp(+1j * (k_dot_x + wave.phase))
    phasor = F_hat * np.exp(-1j * omega * t)

    # after (+i convention -- matches BEM-reader ecosystem):
    eta_hat = wave.amplitude * np.exp(-1j * (k_dot_x + wave.phase))
    phasor = F_hat * np.exp(+1j * omega * t)

The spatial phase `exp(-i*k_dot_x)` keeps the underlying physical
wave traveling in `+X` for heading 0 (cross-checked against
`RegularWave.elevation(x,y,t) = A cos(omega*t - k*(x cos beta + y sin beta) - phi)`).

The fix is pinned by `tests/unit/test_excitation_wamit_convention.py`,
which:

- Loads the committed `tests/fixtures/bem/wamit/synthetic_simple` WAMIT
  fixture (which carries the canonical `+i` convention header in its
  `.3` file).
- Predicts the time-domain force from first principles under +i.
- Asserts pointwise agreement between the prediction and
  `make_regular_wave_force` output at a discriminator time
  (`t = T/4`, where the `+i` and `-i` predictions differ in sign on
  the contribution from `Im(X)`).
- Covers the surge (real-valued RAO), heave (45° RAO), and pitch
  (90° RAO) cases, plus body-offset and non-zero wave-phase
  variations.

## Audit -- why option (b), not option (a)

Two fix options existed at M6 PR4 G3 narrowing:

**Option (a)** -- conjugate `Im(F_exc)` on read in the WAMIT reader.
After the change, `HydroDatabase.RAO` would carry the `-i`-convention
value (matching the existing `make_regular_wave_force` code) but
diverging from the WAMIT-file value and from every other BEM reader
in the ecosystem.

**Option (b)** -- flip `make_regular_wave_force` to use `+i`, leaving
`HydroDatabase.RAO` aligned with the ecosystem.

The downstream-consumer audit (Step 1 of the fix-branch workflow)
showed that option (b) is cheaper and correct:

| factor                                          | (a) cost                                      | (b) cost                                           |
|---|---|---|
| WAMIT / HydroDyn / OrcaFlex ecosystem alignment | breaks (HydroDatabase.RAO conjugated)         | preserved                                          |
| Impedance Path A (`scripts/m6_pr4_pre3_*.py`)   | needs `+iωB → -iωB` + LEAD/LAG sign flip       | unchanged (already Pre-3-validated)               |
| Conventions doc Item 24 (LEAD vs LAG)           | invalidates current text                       | preserves current text                             |
| Capytaine reader                                | needs to stop conjugating Capytaine's native -i| keeps existing conjugation                         |
| M3 test analytical pipeline                     | unchanged                                      | flips `exp(+iωτ)→exp(-iωτ)`, impedance `-iω→+iω`, lstsq `complex(α,β)→complex(α,-β)` |
| `tests/unit/test_excitation.py`                 | unchanged                                      | one test's `exp(-1j*w*t) → exp(+1j*w*t)`           |

Option (b) touches one production module, two test files, and three
docstrings. Option (a) would have required flipping the
already-validated impedance path and invalidating two convention
items. The audit produced a single clear decision before any
implementation work began.

## Pattern lock -- six findings, same shape

This is the **sixth** Phase-1 latent bug fitting CLAUDE.md §13's
pattern:

1. Hydrostatic-gravity (M5 → caught at M6 PR1).
2. Asymmetric-CoG factor (M6 PR1 convention audit).
3. Radiation kernel -- truncation + Nyquist (M6 PR3 pre-step).
4. WaveMod misconfiguration (M6 PR4 Pre-2).
5. WAMIT dimensionalisation (M6 PR4 Pre-3).
6. **Wave-force phase convention (M6 epilogue, this post-mortem).**

The mechanism is the same as #5: a code-path correct in synthetic /
unit / partial-scenario tests was silently wrong under
production-quality inputs and full-scenario activation. Specifically:

- The function `make_regular_wave_force` was written in M3 with the
  RAO convention left implicit. The M3 test builds synthetic RAOs
  and analytical predictions in matching internal conventions, so
  the test passed regardless of which convention was actually
  encoded.
- Through M3, M4, M5, M5 PR2 (Capytaine), the function was
  never paired with a BEM-reader-loaded RAO in a way that would
  surface the convention mismatch -- because no test exercised
  reader → make_regular_wave_force → independent-ground-truth.
- PR4 Pre-3 was the first place that composition fired against
  OpenFAST. The mirror-reflection signature surfaced immediately;
  the structural prediction `phase error ≈ 2·arg(X)` confirmed the
  mechanism in 30 minutes of diagnostic-script work
  (`scripts/m6_pr4_im_ratio_diagnostic.py`).

This is conventions doc Item 19 (the code-path exercise principle)
in action: every PR's pre-flight should ask **what code paths does
this PR newly activate, and does each have a real-data exerciser
somewhere in the suite?** PR4 newly activated the reader →
make_regular_wave_force pipeline against OpenFAST; that surfaced the
bug. Had PR4 been content with the M3-style synthetic test pattern,
the bug would still be latent.

## Convention pin -- documentation cascade

After the fix, the `+i` convention is documented in five places, in
priority order:

1. **`floatsim/hydro/database.py`** -- `HydroDatabase` module docstring
   states `+i` convention as a mandatory invariant for `RAO`, with a
   cross-reference to this post-mortem.
2. **`floatsim/hydro/excitation.py`** -- `make_regular_wave_force`
   module + function docstrings give the +i derivation and cite this
   post-mortem as the mechanism-of-fix record.
3. **`floatsim/waves/regular.py`** -- module docstring updated from
   `-i` (which was wrong, pre-fix) to `+i`.
4. **`tests/unit/test_excitation_wamit_convention.py`** -- the
   convention-pinning test file. Its presence in the suite is the
   regression backstop.
5. **`ARCHITECTURE.md` §8 M1.5** -- updated to reference `+i`
   convention with a pointer to this post-mortem.

The WAMIT reader (`floatsim/hydro/readers/wamit.py`), the OrcaFlex
VesselType YAML reader, and the Capytaine reader all already
document and produce `+i`-convention RAO; they did not need code
changes for this fix.

## What this implies for milestone-6 PR4

PR4's xfail-strict dual-path test
(`tests/validation/test_m6_openfast_regular_wave.py`,
`test_time_domain_phase_agrees_with_impedance`) cited two named
follow-ups in its reason string:

- F-WAVE-FORCE-CONV (this fix) -- predicted contribution `~2*|arg(X)|`.
- F-DAMP-MATCH (still open) -- transient bias on lightly-damped
  DOFs (heave ζ = 0.057 %) where the un-decayed free-decay mode
  contaminates the lstsq fit.

With F-WAVE-FORCE-CONV closed, the predicted residual is reduced
to the F-DAMP-MATCH-only contribution:

- Pitch: dominant ζ source is radiation damping which both tools
  capture; expected to flip from xfail to expected-pass.
- Heave: F-DAMP-MATCH is structural (radiation-only ζ = 0.057 %
  with no MoorDyn-equivalent dynamic mooring damping in FloatSim);
  expected to remain xfail, with the xfail reason now citing only
  F-DAMP-MATCH.

The empirical outcome is recorded in Step 5 of the fix branch's PR
description and the xfail markers updated accordingly.

## Convention-cascade audit summary (post-fix)

For any future code that consumes `HydroDatabase.RAO`, the rule is:

    F(t) = Re{ X * eta_hat * exp(+i * omega * t) }

with `eta_hat` the body-position-shifted complex wave-elevation
phasor under the same `+i` convention:

    eta_hat_at_body = A * exp(-i * (k * (x_b cos beta + y_b sin beta) + phi))

For any code reporting a time-domain phase shift relative to an
elevation reference, the LEAD-vs-LAG discipline of conventions doc
Item 24 applies: `arg(X)` and `arg(xi_hat)` under `+i` are LEADs;
`atan2(B, A)` from a `cos+sin` lstsq fit is the LAG. The two
differ in sign and must be reconciled at the call site.
