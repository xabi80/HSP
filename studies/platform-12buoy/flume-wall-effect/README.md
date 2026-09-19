# Flume sidewall effect — Phase-3 16-buoy platform in the OSU LWF

Quantifies the **sidewall (blockage) effect** — and separates it from the **finite-depth
effect** — on the Phase-3 platform when tested in the OSU Hinsdale **Large Wave Flume** (LWF,
3.66 m wide × 2.7 m deep), and provides the rebuttal to a TEAMER reviewer comment:

> *"Phase 3 platform (2.50 m) inside HWRL's 3.67 m flume leaves 0.6 m side clearance, inducing
> severe sidewall reflections that will corrupt dynamic data."*

**Verdict: the sidewalls do not corrupt the dynamic data.** The rebuttal write-up is
[`REBUTTAL-sidewall.md`](./REBUTTAL-sidewall.md).

## Test article

**16 buoys** (4 clusters × 4, **square** layout at 0/90/180/270°), buoy **centres on a 2.5 m
circle** (Phase-3 spec; `../platform-16buoy/platform16_common.py` layout scaled ×0.833). Each
buoy is the **Phase-1 decay-correlated hull** (`../../osu-test-buoy/osu_buoy_common.py`: 0.159 m
spar, equal-area heave-plate disc r = 0.1437 m) — the geometry that reproduced the Phase-1
free-decay (T ≈ 2.6 s, ζ ≈ 13 %). Outer span 2.79 m → **~44 cm/side** clearance (the 0° square is
the *widest* orientation; see the 45°-rotation companion study for the diagonal).

## Result at a glance (walls-in vs walls-out, same depth)

| Quantity | value | wall effect |
|---|---|---|
| Free-decay heave period (articulated, depth-robust) | 2.607 → 2.593 s | **−0.55 %** |
| Free-decay damping ζ | 6.4 % → 6.4 % | unchanged |
| Heave **response** through resonance (1.5–2.9 s), at 2.7 m | — | **≤ 2.6 %** |
| Accelerations (deck + 4 hubs), all excited DOFs, near resonance | — | **≤ 4 %** |
| Heave response long-period tail (T > 3 s), at 2.7 m | — | up to ~20 %* |

\* Off-resonance (small motion) and **depth-dominated** — see below.

- The platform is a **weak wavemaker**: heave radiation damping is only ~3.5 % of the total heave
  damping (rest is viscous), so there is almost no radiated wave for the walls to reflect. This
  makes the free-decay and near-resonance response insensitive to the walls (and depth-robust).
- The operating band (heave ≈ 2.6 s) is **sub-cut-on**: the flume's transverse sloshing modes cut
  on at **T ≈ 2.19 / 1.53 / 1.25 s**, so below 2.2 s reflections are evanescent and cannot form a
  corrupting cross-flume standing wave. Avoid parking sweep periods on those cut-ons.

## The dominant flume effect is depth, not the walls

Isolating the two effects on the heave wave excitation (`wall_vs_depth.png`):

| Wave period | Sidewall effect (2.7 m) | Finite-depth effect (2.7 m vs deep) |
|---|---|---|
| 2.52 s | −5.4 % | −19 % |
| 3.00 s | −14 % | −27 % |
| 3.50 s | −17 % | −33 % |

At every period the finite-depth effect exceeds the sidewall effect. Finite depth is a known
facility property corrected by depth-scaling; the walls add a smaller term, negligible through
the resonance where the platform's dynamics live.

## Orientation robustness (0° vs 45°)

A 90° rotation is a symmetry no-op (4-fold layout), so the study also re-runs the whole pipeline
at **45°** (`PLAT_ROT_DEG=45`), where the platform is corner-on and the clearance nearly doubles:

| Orientation | Clearance | Free-decay ΔT | Response wall effect (1.5–2.9 s) |
|---|---|---|---|
| 0° (flat-on) | 0.44 m/side | −0.55 % | ≤ 2.6 % |
| 45° (corner-on) | 0.80 m/side | −0.55 % | ≤ 3.0 % |

The wall effect is **unchanged** despite doubling the clearance (`orientation_compare.png`,
`compare_orientations.py`) — it is set by the bulk channel blockage, not the nearest-buoy
clearance. Every orientation-tagged output carries a `_rot45` suffix; regenerate with
`PLAT_ROT_DEG=45 python <script>.py`.

## Three complementary models

The wall effect is depth-sensitive for the *wave response* (a narrow shallow channel blocks long
waves far more than deep water), so the study uses three models, each in its valid regime:

| Model | Script | What / depth |
|---|---|---|
| Frequency-domain image-wall BEM | `flume_wall_effect.py` | isolated wall effect on coefficients per DOF + the depth effect; **native 2.7 m** |
| Single-DOF impedance | `floatsim_wall_1dof.py` | platform-heave **response** wall effect; **native 2.7 m** |
| Articulated 21-body FloatSim | `coupled_bem_osu.py` → `psd_project.py` → `articulated_wall.py` | free-decay + all-DOF near-resonance response, real quadratic drag, KKT gimbal joints; **deep water** (finite depth is impractically slow at the coupled panel count; free-decay is depth-robust) |

Figures are regenerated from the run outputs by `articulated_plots.py` (no BEM), closing the
earlier gap where the summary figures had no committed generator.

## Reproduce

```
# frequency-domain sweep (native 2.7 m; wall effect + depth effect) + single-DOF response
python flume_wall_effect.py
python floatsim_wall_1dof.py
# coupled BEM (deep) -> PSD-project -> articulated 21-body decay / RAO / all-DOF accel
python coupled_bem_osu.py open full 48
python coupled_bem_osu.py walled full 48
python psd_project.py coupled_osu_open.nc coupled_osu_walled.nc
python articulated_wall.py decay
python articulated_wall.py rao
python accel_multidof.py
# regenerate all figures from the saved outputs
python articulated_plots.py
```

The coupled `.nc` files are large and regeneratable, so they are not committed. The buoy count is
parametric (`BUOY_ANGLES_DEG` in `coupled_bem_osu.py` / `articulated_wall.py`); the coupled BEM
also honours a `FLUME_DEPTH` env var (default deep) for the record, though finite depth there is
impractically slow.

## Presentations

- `Flume_wall_effect_explained.pptx` — plain-English deck (`make_flume_ppt.py`), incl. a
  "what is the 21-piece platform" explainer.
- `Flume_wall_effect_technical.pptx` — technical deck (`make_flume_technical_ppt.py`): config +
  method, weak-wavemaker / sub-cut-on physics, the transverse-cut-on ("ring") analysis, and the
  free-decay / wave-response / all-DOF results; no lay explainer.

## Files

| File | What |
|---|---|
| `flume_wall_effect.py` | Frequency-domain image-wall BEM at 2.7 m: isolated wall effect + depth effect. |
| `floatsim_wall_1dof.py` | Single-DOF platform-heave response wall effect at 2.7 m. |
| `coupled_bem_osu.py` | Coupled 16-buoy method-of-images BEM (96-DOF), open/walled. |
| `psd_project.py` | Symmetrise + PSD-project the walled radiation matrix. |
| `articulated_wall.py` | Articulated 21-body FloatSim decay / RAO / all-DOF accel. |
| `accel_multidof.py` | All-DOF accelerations at the 5 sensor points (extends the RAO run). |
| `articulated_plots.py` | Regenerates the summary / accel / wall-vs-depth figures. |
| `flume_wall_plots.py` | Geometry figures (blockage plan, regime map). |
| `REBUTTAL-sidewall.md` | The rebuttal memo for the TEAMER response. |
| `ring_modes.png` | Transverse cut-on ("ring") mode shapes + periods. |
| `wall_vs_depth.png` | Sidewall effect vs finite-depth effect on heave excitation. |
| `floatsim_wall_rao.png` | Platform-heave RAO at 2.7 m, walls in vs out. |
| `flume_blockage.png` | Plan view to scale — platform in the flume, clearance. |
| `articulated_summary.png`, `accel_multidof.png` | Free-decay + RAO; all-DOF accelerations. |

## Caveats

Coarse per-buoy mesh (the walls-in/out **ratio** is mesh-robust; absolute added mass runs a few %
high vs the validated single-buoy value); the two-DOF image trick makes the walled radiation-B
non-reciprocal, so it is PSD-projected (heave decay is A-driven and robust); the coupled model is
at deep water while the wave-response assessment is at 2.7 m (see the three-model split above);
head-seas (along-flume) only — the wave direction the flume supports.
