# Flume sidewall effect — Phase-3 12-buoy platform in the OSU LWF

Quantifies the **sidewall (blockage) + finite-depth effect** on the Phase-3 platform when
tested in the OSU Hinsdale **Large Wave Flume** (LWF, 3.66 m wide × 2.7 m deep), and provides
the rebuttal to a TEAMER reviewer comment:

> *"Phase 3 platform (2.50 m) inside HWRL's 3.67 m flume leaves 0.6 m side clearance, inducing
> severe sidewall reflections that will corrupt dynamic data."*

**Verdict: the sidewalls do not corrupt the dynamic data.** The rebuttal write-up is
[`REBUTTAL-sidewall.md`](./REBUTTAL-sidewall.md).

## Result at a glance (walls-in vs walls-out, same 2.7 m depth)

| Quantity | Heave | Pitch (=roll) | Surge |
|---|---|---|---|
| Free-decay natural-period shift | −0.25% | <0.2% | <0.7% |
| Wave-sweep excitation shift | ≤9% (at resonance) | ≤0.9% | ≤1.4% |

- The platform is a **weak wavemaker**: heave radiation damping is only ~3.5% of the total
  heave damping (rest is viscous), so there is almost no radiated wave for the walls to
  reflect. Pitch/roll radiate even less (out-of-phase buoy motion cancels).
- The operating band (heave ≈ 2.5 s) is **sub-cutoff**: the flume's transverse sloshing modes
  cut on at **T ≈ 2.19 / 1.53 / 1.25 s**, so below 2.2 s reflections are evanescent and cannot
  form a corrupting cross-flume standing wave. Avoid parking sweep periods on those cut-ons.
- Finite depth (2.7 m floor) is negligible for the periods themselves; it does reshape long-wave
  RAOs, but that is a depth effect (common to walls-in and walls-out), not a wall effect.

## Geometry

12 buoys (4 clusters × 3), buoy **centres on a 2.5 m circle** (Phase-3 spec; the
`../platform_common.py` layout scaled ×0.833). Each buoy is the **Phase-1 decay-correlated
hull** (`../../osu-test-buoy/osu_buoy_common.py`: 0.159 m spar, equal-area heave-plate disc
r = 0.1437 m) — the geometry that reproduced the Phase-1 free-decay (T ≈ 2.5 s, ζ ≈ 13%). With
the plates, the outer extent is 2.79 m → **~44 cm/side** clearance (the reviewer's 0.6 m
assumes 2.50 m is the outer extent; this study uses the tighter, as-built ~0.44 m).

## Method

Potential-flow BEM (Capytaine 2.3.1). Finite depth is native; the side walls (W = 3.66 m) are
imposed by the **method of images** (mirror the buoy centres across the walls, drive images with
the same in-phase DOF field, integrate the force over the real panels via a two-DOF trick). The
two-wall image series alternates and converges over ~3 reflection levels.

## Files

| File | What |
|---|---|
| `flume_wall_effect.py` | Geometry, image walls, and the BEM sweep (heave/pitch/surge, walls-in vs walls-out). `python flume_wall_effect.py` reproduces the numbers (~10–15 min). |
| `flume_wall_plots.py` | Geometry-only figures (blockage plan, regime map). |
| `REBUTTAL-sidewall.md` | The 2-page rebuttal memo for the TEAMER response. |
| `flume_blockage.png` | Plan view to scale — platform in the flume, clearance. |
| `flume_regime_map.png` | Transverse-mode map: where the walls matter across wave period. |
| `rebuttal_summary.png` | Free-decay period shifts + wave-sweep load shifts, all modes. |
| `rebuttal_heave_rao.png` | Heave RAO, walls-in vs walls-out. |

## Caveats

Coarse per-buoy mesh (the walls-in/out **ratio** is robust; absolute added mass runs ~8% high
vs the validated single-buoy value); linear ζ = 13% viscous damping applied to set RAO peak
heights (identical across cases, so the comparison is clean); the wave analysis is head-seas
(along-flume) — the only wave direction the flume supports — with roll/sway/yaw not excited by
symmetry.
