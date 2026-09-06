# OSU Test Buoy — non-official field heave-decay test

A handheld phone video of a **real heave-decay test** of the physical OSU Test
Buoy (not the official instrumented tank test — an informal field/lake release).
This note records how the period and damping were extracted from it, and how they
compare to the FloatSim / Capytaine model prediction.

`OSU_field_heave_decay.png` (embedded in `Capytaine_analysis_OSU_buoy.pptx`) is the
result figure.

## Method — video motion tracking

Period and damping are **scale-free** — they need no metric calibration, only the
buoy's vertical position over time in *any* consistent unit (pixels).

1. **Read the MP4 frame-by-frame** (`imageio` + `imageio-ffmpeg`; contributor
   scripting tools, not FloatSim runtime deps).
2. **Segment the buoy** by its teal freeboard: `teal = clip(min(G,B) − R, 0, 255)`,
   and track the topmost contiguous teal run in the buoy column each frame.
3. **Remove the handheld camera pan.** The camera pans ~95 px — far more than the
   heave itself — so the raw buoy track is dominated by camera motion. Each frame is
   referenced against the **fixed horizon** (the water/shore brightness-gradient
   line): `heave = −(buoy_top − horizon)`. A median filter on the horizon trace
   (the camera moves slowly) rejects detection glitches.
4. **Read the decay** about the settled equilibrium. The protocol visible in the
   trace: the buoy is pushed down, held, **released at t ≈ 9.3 s**, and rings down.
   It is heavily damped — only ~1½ clean cycles resolve — so the robust tools are
   direct **peak-to-peak spacing** (period) and **log-decrement** of successive
   extrema (damping), not a full decaying-sinusoid fit (underdetermined here).

## Result

Three cleanly alternating extrema about the settled equilibrium:

| Extremum | t (s) | amplitude about eq. |
|---|---|---|
| max | 10.73 | +10.8 px (≈ +14 cm) |
| min | 11.97 | −7.4 px |
| max | 13.23 | +4.7 px |

- **Period** — max→max = 2.50 s; the two half-periods ×2 = 2.47 and 2.53 s → **T ≈ 2.5 s**
- **Damping** — log-dec of the two maxima → ζ = 13%; of the max→min half-cycle → ζ = 12%
  → **ζ ≈ 12–13%**

Pixel scale (buoy OD 0.159 m ≈ 12 px → ~13 mm/px) is used only to report the ~14 cm
first-swing amplitude; **T and ζ do not depend on it**.

## Comparison to the model

| Quantity | FloatSim / Capytaine model | Field video |
|---|---|---|
| Heave period T | **2.52 s** (full decay sim; 2.3–2.4 s analytic M+C₃₃ bracket) | **≈ 2.50 s** |
| Heave damping ζ | **8–15%** (pre-test prediction at 100 mm release) | **≈ 12–13%** |

**Both match.** The period lands within ~1 % of the full simulation; the damping sits
squarely in the predicted band. This is a first independent, real-world corroboration
of the buoy's heave dynamics.

## Caveats

- **Non-official, handheld.** Camera pan removed via horizon referencing; a slow
  residual baseline drift remains, which is why ζ is a range, not a point. (A naive
  linear detrend *corrupts* the result — it tilts the baseline and inflates the first
  swing to a spurious T = 2.33 s / ζ = 16 %; the raw horizon-referenced signal about
  the settled equilibrium is the trustworthy one.)
- **Large release.** The first resolved swing is ~14 cm (the initial push-down was
  deeper but clips out of frame). At that amplitude Morison **quadratic drag**
  dominates, so this ζ is a *first-cycle, large-amplitude* value — expected at the
  **upper** end of the small-amplitude 8–15 % band. Damping should fall on smaller
  cycles; too few cycles resolve here to trace that curve.
- **Does not replace the tank test.** The official instrumented test is still what
  pins down the perforated heave-plate's added-mass / drag split (see
  `OSU-TEST-BUOY-GEOMETRY.md`); this field video corroborates the overall heave
  period and damping.

Source video: a user-provided upload (not committed). The analysis lived in a
scratchpad; only the result figure `OSU_field_heave_decay.png` is retained here.
