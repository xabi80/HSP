# M6 PR4 Pre-3 — WaveTp=10s phase-residual diagnosis

**Status**: ✓ Resolved. Bug was in **phase reporting convention**
(LEAD vs LAG), not in the WAMIT reader's data convention. Pre-3
dual-path verification now passes at both WaveTp = 10 s and 25 s
with phase gaps of 0.38° and 0.39° respectively.

## Summary

After the WAMIT dimensionalisation fix, the Pre-3 dual-path
verification passed amplitude (rtol < 1e-2) at both WaveTp = 10 s
and 25 s, but failed phase at 10 s by 12.7°. This document
diagnoses the residual.

## Diagnostic — sliding-window lstsq fit on the OpenFAST CSV

`scripts/m6_pr4_pre3_phase_residual_diagnosis.py` slides a
50 s (5-quantised-period) window through the 1200 s S3
WaveTp=10s simulation and reports heave-RAO phase per window.
Both effects show up in the data:

| t_center (s) | RAO amp (m/m) | RAO phase (deg) | resp_resid |
|---:|---:|---:|---:|
| 85.0   | 0.24740 | **+11.5318** | 0.88285 |
| 135.0  | 0.22950 | +11.3992 | 0.81717 |
| 285.0  | 0.20829 | +5.9160  | 0.52479 |
| 535.0  | 0.21775 | +6.8964  | 0.15921 |
| 1135.0 | 0.21563 | **+6.5368**  | 0.00745 |

Two effects are present:

1. **Theory (a) — transient bleed (real, but small)**: phase
   drifts from +11.53° (early, transient-contaminated) to
   +6.54° (late, transient-decayed). Drift = 5°. Caused by the
   heave free-decay mode at T_n = 17 s, ζ ≈ 1.2 %, which has
   not fully damped out by t = 85 s but is essentially gone
   by t = 1135 s. Late-window (last 50 s of 1200 s sim) is
   the right place to fit RAO.

2. **Theory (b) — phase-reporting convention (the dominant
   residual)**: even at late window, Path A (FloatSim
   impedance) reported −6.17° while Path B (OpenFAST lstsq
   fit) reported +6.54°. Mirror reflection around 0°.

## Root cause — LEAD vs LAG phase convention

Under the `exp(+i*omega*t)` convention used by FloatSim:

    x(t) = Re[x_hat * exp(+i*omega*t)]
         = Re(x_hat) * cos(omega*t) - Im(x_hat) * sin(omega*t)
         = |x_hat| * cos(omega*t + arg(x_hat))

`arg(x_hat)` is therefore the **LEAD** of the response: the
response peaks when `omega*t + arg(x_hat) = 0`, i.e., at
`t = -arg(x_hat)/omega`. Positive `arg(x_hat)` ⇒ negative t ⇒
peak BEFORE t=0 ⇒ response leads.

A `cos + sin` lstsq fit (`tests/support/rao_extraction.py`)
returns `atan2(B, A)` where the signal is fitted as
`A * cos(omega*t) + B * sin(omega*t)`. This factors as
`R * cos(omega*t - atan2(B, A))`, i.e., `atan2(B, A)` is the
**LAG** of the signal: the signal peaks when
`omega*t - atan2(B, A) = 0`, i.e., at `t = atan2(B, A)/omega > 0`.
Positive `atan2(B, A)` ⇒ peak AFTER t=0 ⇒ response lags.

Therefore: under the +i convention, **LEAD = `arg(x_hat)`** and
**LAG = `-arg(x_hat)` = `atan2(B, A)`**. Path A reported the
LEAD (`np.angle(xi_hat[2])`); Path B reported the LAG
(`atan2(B, A)`). They are negatives of each other for the same
physical motion.

For the same physical situation:
- Path A = +arg(xi_FS) (LEAD)
- Path B = atan2(B, A) (LAG) = -arg(xi_FS)

So Path A (raw) ≈ -Path B. The pre-fix gap of 12.7° at 10 s and
0.42° at 25 s reflected this exactly:
- 10 s: −6.17° vs +6.55° → reverse-sign agreement at 0.38°.
- 25 s: −0.0154° vs +0.40° → reverse-sign agreement at 0.39°.

## Fix

**`scripts/m6_pr4_pre3_rao_verification.py`** (Path A): negate
`arg(xi_hat)` to convert LEAD to LAG, matching Path B:

```python
# Was: np.angle(xi_hat[HEAVE_DOF])
# Now: -np.angle(xi_hat[HEAVE_DOF])
return float(np.abs(xi_hat[HEAVE_DOF])), -float(np.angle(xi_hat[HEAVE_DOF]))
```

Documentation update: `floatsim/hydro/readers/wamit.py` module
docstring now flags the LEAD-vs-LAG distinction explicitly so
future RAO consumers don't fall into the same trap.

The WAMIT reader itself is **unchanged** — `Re + i*Im` is the
correct +i-convention representation of the data, and the file's
`Mod * exp(+i*Pha_rad)` cross-check is a valid file-level
consistency check that does not need to flip sign.

## Why latent through M2-M5 + M6 PR1-PR3 pre-dim-fix

This is the THIRD distinct latent bug surfaced by the M6 PR4 RAO
work (after WaveMod, then dimensionalisation). The pattern is
the same: phase reporting is exercised end-to-end for the first
time at PR4. Earlier work (free-decay, static eq) does not
report RAO phase — it asserts on time-series amplitude or
period only. M6 PR4 Pre-3 is the first PR that compares an
impedance-domain phase against a time-domain lstsq phase, so it
is the first PR that needed the LEAD-vs-LAG discipline.

This is conventions Item 19 ("the code-path exercise principle")
in action for the third time on this fix branch alone.

## Verification

Re-run `scripts/m6_pr4_pre3_rao_verification.py` after the fix:

```
WaveTp  Path A amp  Path B amp  amp rel-err  Path A phase  Path B phase  phase err  verdict
  10.0  0.215553    0.215774   -0.00102     +6.1735       +6.5522       -0.3787    PASS
  25.0  1.087903    1.091798   -0.00357     +0.0154       +0.4038       -0.3884    PASS
```

Both gates met: amp rtol < 1e-2 and phase atol < 1° at both
frequencies. PR4 may proceed.

## Files updated by this fix branch (Pre-3 phase resolution)

- `scripts/m6_pr4_pre3_phase_residual_diagnosis.py` — sliding-window
  diagnostic
- `docs/diagnostics/m6-pr4-pre3-phase-residual-diagnosis.png` — plot
- `docs/diagnostics/m6-pr4-pre3-phase-residual-diagnosis.md` — this
  report
- `scripts/m6_pr4_pre3_rao_verification.py` — Path A returns LAG
  (negated `arg(xi_hat)`)
- `floatsim/hydro/readers/wamit.py` module docstring — LEAD vs LAG
  note
- `docs/openfast-cross-check-conventions.md` Item 24 — codifies the
  LEAD-vs-LAG discipline for future RAO consumers
