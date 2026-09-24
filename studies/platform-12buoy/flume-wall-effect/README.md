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
free-decay (T ≈ 2.6 s, ζ ≈ 13 %). **Tested flat-on (45°)**: span across the flume 2.06 m
(56 % of the width) → **~80 cm/side** clearance. The corner-on 0° orientation — the widest, at
44 cm/side — is run as the comparison case and gives the same result (see below).

## Result at a glance (walls-in vs walls-out, same depth)

| Quantity | value | wall effect |
|---|---|---|
| Free-decay heave period (coupled 21-body BEM) | 2.607 → 2.593 s | **−0.55 %** |
| Free-decay damping ζ | 6.4 % → 6.4 % | unchanged |
| Heave RAO wall effect (whole wave band, coupled BEM) | — | **≤ 4 %** (no growth at long T) |
| Excitation wall effect (whole band, coupled BEM) | — | **≤ 4 %** |
| Accelerations (deck + 4 hubs), all excited DOFs | — | **≤ 4.5 %** |

**Fidelity note:** the sidewall effect is read from the coupled 21-body BEM. The single-array
frequency-domain method (`flume_wall_effect.py`) under-converges in image count at long periods
(2 vs 3 reflections differ ~10× and disagree with the coupled solve even in sign — it over-states
the wall effect there), so it is used only for the image-free finite-depth effect below.

- The platform is a **weak wavemaker**: heave radiation damping is only ~3.5 % of the total heave
  damping (rest is viscous), so there is almost no radiated wave for the walls to reflect. This
  makes the free-decay and near-resonance response insensitive to the walls (and depth-robust).
- The operating band (heave ≈ 2.6 s) is **sub-cut-on**: the flume's transverse sloshing modes cut
  on at **T ≈ 2.19 / 1.53 / 1.25 s**, so below 2.2 s reflections are evanescent and cannot form a
  corrupting cross-flume standing wave. Avoid parking sweep periods on those cut-ons.

## The dominant flume effect is depth, not the walls

Isolating the two effects on the heave wave excitation (`wall_vs_depth.png`); the sidewall column
is the coupled BEM, the depth column is the image-free (Airy-corroborated) sweep:

| Wave period | Sidewall effect (coupled BEM) | Finite-depth effect (2.7 m vs deep) |
|---|---|---|
| 2.52 s | +3.0 % | −19 % |
| 3.00 s | +1.7 % | −27 % |
| 3.50 s | +0.7 % | −33 % |
| 4.00 s | +0.1 % | −37 % |

The sidewall effect stays small and flat (and shrinks toward zero at long periods), while the
finite-depth effect grows to −37 %. Finite depth is a known facility property corrected by
depth-scaling; the walls are a minor term at every period.

**Why the depth effect grows with period** (`depth_effect_explained.png`, `depth_effect_plots.py`):
a wave only feels the bottom once its wavelength is long against the 2.7 m depth. Short waves
(h/λ = 0.88 at 1.4 s) are confined near the surface and behave as in deep water; at the 2.6 s
natural period h/λ = 0.27 and at 4 s it is 0.15, so the orbital motion reaches the floor. The floor
forces the vertical particle velocity to zero, flattening the circular orbits into ellipses and
shrinking the vertical motion that lifts the heave plate at −1.38 m. This is plain Airy wave
theory, and it reproduces the BEM depth effect to within a few percent at every period — the
independent corroboration that the depth effect is real and method-independent.

## Orientation robustness (0° vs 45°)

The test orientation is **45° (flat-on)**, run with `PLAT_ROT_DEG=45` (outputs carry a `_rot45`
suffix). A 90° rotation is a symmetry no-op (4-fold layout), so the study also runs the **0°
corner-on** case — the widest orientation, with roughly half the clearance — as the comparison:

| Orientation | Clearance | Free-decay ΔT | Excitation wall effect (coupled BEM) |
|---|---|---|---|
| 0° (corner-on) | 0.44 m/side | −0.55 % | ≤ 3.5 % |
| 45° (flat-on) | 0.80 m/side | −0.55 % | ≤ 3.6 % |

The wall effect is **unchanged** despite doubling the clearance (`orientation_compare.png`,
`compare_orientations.py`) — it is set by the bulk channel blockage, not the nearest-buoy
clearance. Every orientation-tagged output carries a `_rot45` suffix; regenerate with
`PLAT_ROT_DEG=45 python <script>.py`.

## Models and fidelity

| Model | Script | Role |
|---|---|---|
| **Coupled 21-body BEM** (authoritative) | `coupled_bem_osu.py` → `psd_project.py` → `articulated_wall.py` | the sidewall effect: free-decay + all-DOF response, real quadratic drag, KKT gimbal joints. Deep water (finite depth is impractically slow at the coupled panel count; the wall effect is depth-robust in this weak-radiator regime). |
| Single-array frequency-domain BEM | `flume_wall_effect.py` | the image-free **finite-depth effect** only. Its wall effect is **not** used — it under-converges in image count at long periods (verified: 2 vs 3 reflections give −14 % → −5.7 % at 3 s, vs the coupled BEM's +1.7 %). |

The **finite-depth effect** is corroborated independently by textbook Airy wave kinematics (the
orbital-motion attenuation with depth), so it does not depend on the frequency-domain method's
image machinery. Figures are regenerated from the run outputs by `articulated_plots.py`.

## Reproduce

```
# coupled BEM (deep) -> PSD-project -> articulated 21-body decay / RAO / all-DOF accel
python coupled_bem_osu.py open full 48
python coupled_bem_osu.py walled full 48
python psd_project.py coupled_osu_open.nc coupled_osu_walled.nc
python articulated_wall.py decay
python articulated_wall.py rao
python accel_multidof.py
# single-array sweep for the finite-depth effect (its wall effect is not used)
python flume_wall_effect.py
# regenerate all figures (coupled wall effect + Airy-corroborated depth effect)
python articulated_plots.py
python compare_orientations.py
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
| `flume_wall_effect.py` | Single-array BEM at 2.7 m: the finite-depth effect (its wall effect under-converges and is not used). |
| `coupled_bem_osu.py` | Coupled 16-buoy method-of-images BEM (96-DOF), open/walled — the authoritative sidewall effect. |
| `compare_orientations.py` | 0° vs 45° sidewall comparison (coupled BEM). |
| `depth_effect_plots.py` | Finite-depth explainer figure from Airy theory, with the BEM points overlaid. |
| `psd_project.py` | Symmetrise + PSD-project the walled radiation matrix. |
| `articulated_wall.py` | Articulated 21-body FloatSim decay / RAO / all-DOF accel. |
| `accel_multidof.py` | All-DOF accelerations at the 5 sensor points (extends the RAO run). |
| `articulated_plots.py` | Regenerates the summary / accel / wall-vs-depth figures. |
| `flume_wall_plots.py` | Geometry figures (blockage plan, regime map). |
| `REBUTTAL-sidewall.md` | The rebuttal memo for the TEAMER response. |
| `ring_modes.png` | Transverse cut-on ("ring") mode shapes + periods. |
| `wall_vs_depth.png` | Sidewall effect (coupled BEM) vs finite-depth effect on heave excitation. |
| `orientation_compare.png` | 0° vs 45° sidewall effect. |
| `depth_effect_explained.png` | Why long waves feel the 2.7 m bottom: orbits deep vs flume, and the effect vs period. |
| `flume_blockage.png` | Plan view to scale — platform in the flume, clearance. |
| `articulated_summary.png`, `accel_multidof.png` | Free-decay + RAO; all-DOF accelerations. |

## Caveats

Coarse per-buoy mesh (the walls-in/out **ratio** is mesh-robust; absolute added mass runs a few %
high vs the validated single-buoy value); the two-DOF image trick makes the walled radiation-B
non-reciprocal, so it is PSD-projected (heave decay is A-driven and robust); the coupled model is
at deep water (the wall effect is depth-robust for this weak radiator; the separate depth effect
is assessed at 2.7 m); the natural period (~2.6 s) uses the **unloaded** OSU-buoy mass — a
deck-loaded platform (~33 kg/buoy) would sit at ~3 s (see `platform-16buoy` fin study), which
does not change the small walls-in/out ratios; head-seas (along-flume) only.
