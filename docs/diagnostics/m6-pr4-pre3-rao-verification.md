# M6 PR4 Pre-3 — RAO definition lock-down

**Status**: 🔴 **PR4 blocked — latent FloatSim WAMIT reader bug
surfaced**. The marin_semi.{1,3,hst} files are non-dimensional;
FloatSim's `read_added_mass_and_damping` /
`read_excitation_force` / `read_hydrostatic_stiffness` return them
as-is, treating them as dimensional. OpenFAST/HydroDyn applies
the proper `ρ·g·ULEN^k` scaling internally; FloatSim does not.

The bug is **latent through M1–M6 PR3** because pitch / heave
free-decay periods are dominated by rigid-body mass `M` and
hydrostatic stiffness `C` (Robertson values, dimensional).
Frequency-dependent added mass `A(ω)` from marin_semi is ≤ 0.1 %
of `M` for OC4 in the natural-period band, so the missing 1000×
factor on `A` doesn't perturb `T = 2π·√((M+A)/C)` measurably.

The bug **surfaces immediately at PR4** because RAO at the
long-wave limit goes as `F_exc / C`, and `F_exc` is the whole
right-hand side. The missing `ρ·g·ULEN²` factor on `F_exc` makes
the heave RAO come out ~10⁴× too small.

CLAUDE.md `tests/validation/test_oc4_pitch_period_buoyancy_only_c.py`
docstring already flagged this as a known latent issue:

> "the WAMIT reader does NOT currently apply ULEN-based dimensional
>  rescaling — that's a separate latent bug, out of scope for this
>  fix."

PR4 is the fix-time scenario per Item 19 (code-path exercise
principle): RAO extraction is the first scenario PR that exercises
the F_exc-dominated regime.

## Numerical evidence

Pre-3 dual-path verification at WaveTp = 25 s on the regenerated
S3 fixture:

| Path | Method | Heave RAO amp (m/m) | Heave RAO phase (deg) |
|------|--------|---------------------:|----------------------:|
| A | WAMIT marin_semi.{1,3} → impedance | 7.4 × 10⁻⁵ | +0.29 |
| B | OpenFAST .outb → lstsq fit | 1.0918 | +0.40 |
| Δ rel-err | (A − B) / B | **−99.99 %** | − |

Path B is physically correct: at T = 25 s ≫ T_n_heave ≈ 17 s,
heave RAO → 1.0 (body follows wave like a cork; long-wave limit).

Path A is essentially zero. Ratio: A / B = 7.4 × 10⁻⁵ / 1.09 ≈
6.8 × 10⁻⁵ ≈ 1 / 14 700.

The expected non-dim → dim factor for translational `F_exc`:
`ρ · g · ULEN² = 1025 · 9.81 · 1² ≈ 10 055`. Multiplied by Path A
amp gives `7.4 × 10⁻⁵ × 10 055 = 0.74 m/m` — close to but not
exactly Path B's 1.09. The remaining factor of ~ 1.5 comes from
the `A(ω)` / `C` non-dim factors (`ρ` / `ρ·g` respectively) that
also need rescaling.

## What needs to change

`floatsim/hydro/readers/wamit.py` must apply the dimensionalisation
factors per the WAMIT manual (HydroDyn user manual §6 reference):

| File | Coefficient | Non-dim → dim factor (mode i, mode j) |
|------|-------------|---------------------------------------|
| `.1` | `A(ω)`, `A_inf` | `ρ · ULEN^k` where k = 3 for trans-trans, 4 for trans-rot or rot-trans, 5 for rot-rot |
| `.1` | `B(ω)` | `ρ · ω · ULEN^k` (same `k` table) |
| `.3` | `F_exc` (per unit wave amplitude) | `ρ · g · ULEN²` for translational mode (i = 1..3); `ρ · g · ULEN³` for rotational mode (i = 4..6) |
| `.hst` | `C` | `ρ · g · ULEN²` for trans-trans; `ρ · g · ULEN³` for trans-rot or rot-trans; `ρ · g · ULEN⁴` for rot-rot |

`ULEN`, `ρ_water`, and `g` come from the OpenFAST HydroDyn input
(or as parameters to the reader for non-OpenFAST contexts).

For OC4 marin_semi: `ULEN = 1.0`, `ρ_water = 1025 kg/m³`,
`g = 9.80665 m/s²`.

## Verification plan after the reader fix

1. Re-run M6 PR3 free-decay tests. Period should remain near 25.67 s
   for Setup B (the BEM correction is small relative to combined
   mass at pitch). If it shifts substantially, the original PR3
   F1-residual classification will need to be revisited.
2. Re-run Pre-3 dual-path verification at WaveTp = 10 s and 25 s.
   Path A and Path B must agree at `rtol < 1e-2` amp and
   `atol < 1°` phase at both frequencies.
3. Add a regression test in `tests/unit/test_wamit_reader.py`
   asserting that the dimensional output matches OpenFAST/HydroDyn's
   internal interpretation for at least one diagonal entry per file
   type at one frequency (the published OC4 numbers from Robertson
   give A_inf_heave ≈ 1.5 × 10⁷ kg, which is the post-rescale
   target).

## Disposition

This is a separate fix branch. Same precedent as `fix-pr2-cmzt`,
`fix-radiation-kernel`, `fix-s3-wavemod`: a Pre-step audit
surfaced a latent bug that's out of scope for the parent PR but
must close before the parent can land. Suggested branch name:
**`fix-wamit-dimensionalisation`** off main.

The fix is non-trivial: ~50-100 lines in
`floatsim/hydro/readers/wamit.py` plus regression tests. Unlike
the previous "small fix" branches, this one touches a code path
already in production use (M5 PR1 reader unit tests, M6 PR3
free-decay validation). The free-decay tests should still pass
after the fix because their physics is mass-dominated, but the
period values *will* shift slightly when `A(ω_n)` becomes 1000×
larger.

Anticipating the shift on M6 PR3 Setup B: pitch period ~ 25.67 s
becomes... let me compute. At ω_n = 2π/25.67 = 0.2447 rad/s,
A_55(ω_n) from marin_semi.1 is ~7.5e6 (interpolated). After
dimensionalisation: 1025 × 7.5e6 = 7.7e9 kg·m². Original Setup B
I_55 = 1.25e10. New M+A = 1.25e10 + 7.7e9 = 2.02e10. New T:
`2π·√(2.02e10 / 6.67e8) = 34.6 s`. **That's bigger than OpenFAST's
26.83 s.** So the dimensional fix shifts F1-residual from -4.29 %
to **+29 %** — a much larger mismatch in the OPPOSITE direction.

This means F1 (and F1-residual) classification in PR3 was based
on wrong physics. The combined-deck correction from Setup A
(18.43 s) to Setup B (25.67 s) was largely the gravity-stiffness
shift, not the inertia + added-mass shift. With dimensional
A(ω), the picture changes.

**Pause for Xabier review**: this finding has implications that
go beyond Pre-3. PR3's F1-residual disposition was correct at
the time but may need re-classification once the WAMIT reader
fix lands. PR4 cannot proceed without the fix. The fix itself
is a self-contained ~100-line change but its downstream effect
on PR3 needs to be quantified before it lands.

## Files produced

- `tests/support/rao_extraction.py` — sinusoidal lstsq RAO
  extractor (independent of the WAMIT-reader bug; usable
  unchanged after the fix).
- `scripts/m6_pr4_pre3_rao_verification.py` — dual-path
  verification (will pass after the WAMIT reader fix).
- This document.
