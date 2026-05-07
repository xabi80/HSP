# M6 PR4 Pre-2 — S3 wave-generation verification

**Status**: ✅ **closed 2026-05-06 on `fix-s3-wavemod` branch**. All 14
S3 RAO-sweep scenarios pass the wave-generation gate after the
`WaveMod = 1` fix and the IFFT-bin-quantisation fit correction.
PR4 may use these fixtures.

This document originally captured the pre-fix `WaveMod = 2`
finding; the historical analysis is preserved below for the
audit trail. The closure summary at the top reflects the current
state of the regenerated fixtures.

## Closure summary (post-fix, 2026-05-06)

After fixing `seastate_edits["WaveMod"] = 1` in
`openfast_setup/scenario_config.py` and regenerating all 14
sweep variants, the wave-elevation channel was fit at the
OpenFAST IFFT-quantised wave frequency (per conventions doc
Item 21) over the last 5 wave periods of each scenario:

| WaveTp [s] | fitted amp [m] | amp rel-err | fit residual / signal | verdict |
|-----------:|---------------:|------------:|----------------------:|---------|
|  4 | 0.49846 | -0.31 % | 0.0017 | PASS |
|  5 | 0.49901 | -0.20 % | 0.0011 | PASS |
|  6 | 0.49932 | -0.14 % | 0.0008 | PASS |
|  7 | 0.49949 | -0.10 % | 0.0006 | PASS |
|  8 | 0.49961 | -0.08 % | 0.0004 | PASS |
| 10 | 0.49975 | -0.05 % | 0.0003 | PASS |
| 12 | 0.49983 | -0.03 % | 0.0002 | PASS |
| 14 | 0.49987 | -0.03 % | 0.0001 | PASS |
| 16 | 0.49990 | -0.02 % | 0.0001 | PASS |
| 18 | 0.49993 | -0.02 % | 0.0001 | PASS |
| 20 | 0.49994 | -0.01 % | 0.0001 | PASS |
| 22 | 0.49995 | -0.01 % | 0.0001 | PASS |
| 25 | 0.49996 | -0.01 % | 0.0000 | PASS |
| 30 | 0.49997 | -0.01 % | 0.0000 | PASS |

Gates: amp rel-err < 2 % AND fit residual / signal < 5 %. All 14
PASS. The expected wave amplitude is `WaveHs / 2 = 0.5 m`; the
worst rel-err is 0.31 % on WaveTp = 4 s (a half-IFFT-bin amp
discretisation effect, well inside tolerance).

## Pre-fix → post-fix history (audit trail)

The original Pre-2 audit found two issues in sequence:

1. **`WaveMod = 2` (JONSWAP irregular)** instead of `WaveMod = 1`
   (regular Airy). Caught by FFT analysis of the wave_elev
   channel showing broadband content unrelated to the labelled
   `WaveTp`. Fixed by one-line change in
   `openfast_setup/scenario_config.py`. See "Root cause" section
   below for the original finding.
2. **IFFT-bin quantisation** of `WaveTp` to the nearest
   `omega_k = k * WaveDOmega` where `WaveDOmega = 2π / WaveTMax`.
   Surfaced after the `WaveMod` fix when the lstsq residual
   stayed at 9-12 % for `WaveTp ∈ {16, 18, 22}` even with clean
   wave generation. Fixed by computing the quantised period at
   fit time (the wave is at `T_actual = WaveTMax / round(WaveTMax / WaveTp)`).
   Documented as conventions doc Item 21.

The deck-generation regression test
(`openfast_setup/tests/test_scenario_decks.py`) pins issue 1 at
generator time. Issue 2 is a fit-time convention; pinned in PR4's
RAO extractor.

---

## Original finding (pre-fix; preserved for audit trail)

**Status (pre-fix)**: 🔴 the S3 RAO-sweep reference is generating
**JONSWAP irregular waves**, not regular Airy waves.

## Root cause

Both copies of `scenario_config.py` (the HSP-vendored
`tests/fixtures/openfast/oc4_deepcwind/baseline/case/scenario_config.py`
and the user's working `openfast_setup/scenario_config.py`) declare
the S3 wave kinematics as:

```python
seastate_edits={
    "WaveMod": 2,        # regular Airy   <-- COMMENT WRONG
    "WaveHs": 1.0,
    "WaveDir": 0.0,
    ...
},
```

Per OpenFAST's SeaState convention (visible directly in the
generated `*_SeaState.dat`):

```
WaveMod = 0 : still water
WaveMod = 1 : regular Airy waves (used for RAO sweeps)
WaveMod = 2 : JONSWAP / Pierson-Moskowitz irregular spectrum
WaveMod = 3 : white-noise irregular spectrum
...
```

The commit set `WaveMod = 2` (JONSWAP spectrum) but documented the
intent as "regular Airy". The downstream effect: every S3 scenario
runs an irregular sea state with `WaveTp` interpreted as the
*peak-spectral period*, not the regular-wave period. The actual
wave train is broadband around Tp, modulated by `WaveHs` /
`WvLowCOff` / `WvHiCOff`.

## Numerical evidence

### WaveTp = 30 s — entire spectrum below the low cutoff

The committed SeaState file has `WvLowCOff = 0.314159 rad/s` (= T = 20 s)
and `WvHiCOff = 1.570796 rad/s` (= T = 4 s). For a JONSWAP at
`Tp = 30 s`, the peak is at `ω_p ≈ 0.209 rad/s` — below the low
cutoff. The entire spectral peak gets zeroed; only the tail
remains.

Pitch FFT, last 200 s of `WaveTp_030p0/s3_rao_sweep_WaveTp_030p0.csv`:

| Peak T [s] | Spectral mag (rad) |
|-----------:|-------------------:|
| 14.29 | 1.23 × 10⁻⁴ |
| 10.53 | 1.13 × 10⁻⁴ |
| 15.39 | 8.97 × 10⁻⁵ |
| 11.77 | 6.38 × 10⁻⁵ |

No peak anywhere near 30 s. The pitch response oscillates at
12-15 s — entirely set by the truncated JONSWAP tail, not by the
labeled `WaveTp = 30 s`.

Wave-elevation FFT (same window):

| Peak T [s] | Spectral mag (m) |
|-----------:|-----------------:|
| 15.39 | 4.00 × 10⁻² |
| 13.34 | 3.30 × 10⁻² |
| 14.29 | 2.79 × 10⁻² |
| 16.67 | 2.55 × 10⁻² |
| 12.50 | 2.47 × 10⁻² |

The wave train itself has no spectral content at 30 s — the
"regular wave at Tp = 30 s" is a label mismatch with reality.

### WaveTp = 10 s — broadband response even when Tp is in band

For shorter Tp the JONSWAP peak does fall inside the cutoff window,
so a peak appears near the labeled period. But the response is
still broadband:

Wave-elevation FFT, last 200 s of `WaveTp_010p0`:

| Peak T [s] | Spectral mag (m) |
|-----------:|-----------------:|
| 10.53 | 7.26 × 10⁻² |
| 12.50 | 6.17 × 10⁻² |
|  6.90 | 4.71 × 10⁻² |
|  7.15 | 4.68 × 10⁻² |
|  7.69 | 4.19 × 10⁻² |

The 10.53 s peak is "near" the labeled 10 s, but the second peak
at 12.5 s carries 85 % of the dominant peak's amplitude — this is
NOT a regular wave. It's irregular waves with a JONSWAP envelope.

A regular wave at `WaveTp = 10 s` would show one peak at 10.0 s
with all other spectral bins essentially zero. The response would
be a clean sinusoid at exactly the wave frequency, with constant
per-cycle amplitude.

### Per-cycle amplitude spread (the "is steady state reached" check)

The `m6_pr4_pre2_steady_state.py` diagnostic looks for upward
zero crossings spaced at `~ WaveTp` and computes per-cycle peak
amplitudes. For `WaveTp = 30 s` the inter-crossing intervals
on the pitch signal are:

```
[11.93, 10.41, 8.95, 13.20, 11.93, 16.25, 12.45, 13.94, 12.68,
 16.40, 11.46, 8.83, 12.99, 13.40, 8.70]
```

— wildly varying, no consistent 30 s period. The "0 cycles
within 5 % of the labeled period" output of the diagnostic
script correctly flags that the response is not at the labeled
frequency.

## Why Pre-2 caught this and PR2 / PR3 didn't

S1 and S2 both run with `WaveMod = 0` (still water; `seastate_edits =
{"WaveMod": 0}` in the openfast_setup config), so neither scenario
exercised the wave-generation code path. PR4 is the first PR to
need wave forcing; the S3 misconfiguration was latent until now.

## Fix and cost

**Fix**: change `WaveMod: 2` → `WaveMod: 1` in the S3 entry of
both `scenario_config.py` copies (HSP-vendored + openfast_setup).
The comment "regular Airy" then matches the value.

**Other parameters**: with `WaveMod = 1` the JONSWAP fields
(`WvLowCOff`, `WvHiCOff`, `WavePkShp`, `WaveSeed*`,
`WaveDirSpread`, `WaveDirMod`, `WaveNDir`) become unused — the
deck schema retains them but OpenFAST ignores them. `WaveTp` is
re-interpreted as the regular-wave period (which is what the
sweep was intending). `WaveHs` becomes the regular-wave height.

**Regeneration cost**: per CLAUDE.md §14 "Full 18-scenario
regenerations triggered casually (S3 RAO sweep alone is ~30min;
flag in PR descriptions before committing fixture changes that
touch S3)". The S3 sweep alone is 14 OpenFAST runs at ~150 s each
~ 35 min, plus ~5 min for deck regeneration and CSV extraction =
**~40 min wall-clock**.

Per the conventions doc, this falls under "Out of scope without
explicit approval". **Awaiting Xabier's go-ahead** before
regenerating.

## Recommended downstream changes

1. **Fix `WaveMod` in both `scenario_config.py` copies** (1 line each).
2. **Remove the misleading "# regular Airy" comment** or correct it
   to "# regular Airy (WaveMod=1)" — keeping the *intent* visible
   in the source.
3. **Regenerate all 14 S3 scenarios** (the only practical option;
   no per-scenario fix exists since the same misconfig is in the
   shared S3 entry).
4. **Re-extract all 14 CSVs** via `extract_openfast_fixtures.py
   --mode read-only --scenario s3_rao_sweep`.
5. **Add a unit test** that asserts on the generated SeaState file:
   for any S3 scenario, the SeaState `WaveMod` written to disk
   must be `1` (or whatever the `seastate_edits` declared). Catches
   future scenario_config.py drift at deck-generation time, not at
   PR4 RAO-extraction time.
6. **Re-run Pre-2** on the regenerated fixtures: confirm pitch /
   surge response in the last 200 s is sinusoidal with constant
   per-cycle amplitude (per-cycle spread < 5 %) at all 14 wave
   periods.
7. **Conventions doc Item 18**: "Wave-mode label vs value: WaveMod
   integer values must match the comment / intent. WaveMod=1 is
   regular Airy; WaveMod=2 is JONSWAP. RAO cross-checks require
   WaveMod=1." Pinned by the unit test from step 5.

## Files produced

- `scripts/m6_pr4_pre2_steady_state.py` — Pre-2 runner (caught
  the issue by failing to find consistent inter-crossing intervals
  at the labeled wave period; FFT analysis revealed the cause).
- This document.
- (Pending Xabier's approval) Regenerated S3 fixtures and CSVs.

The diagnostic plots referenced by the script
(`docs/diagnostics/m6-pr4-steady-state-WaveTp_*.png`) are written
but show the broadband response; they will be regenerated against
the corrected fixtures once they exist.
