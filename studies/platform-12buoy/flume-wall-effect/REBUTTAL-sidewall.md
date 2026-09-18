# Response to reviewer comment — sidewall effects on the Phase-3 platform test

**Reviewer comment:** *"Phase 3 platform (2.50 m) inside HWRL's 3.67 m flume leaves 0.6 m side
clearance, inducing severe sidewall reflections that will corrupt dynamic data."*

## Summary

We assessed the concern quantitatively with potential-flow boundary-element simulations
(Capytaine) in which HWRL's Large Wave Flume side walls are modelled **explicitly**. The
result: **sidewall reflections do not corrupt the Phase-3 dynamic data.** Modelling the walls
in versus out (at the same 2.7 m operating depth) shifts every measured **natural period by
< 0.7 %** and every **wave-frequency load by ≤ 9 %** (≤ 2 % for pitch and surge) — inside normal
model-test uncertainty — with no spurious resonances in the operating band. The physical reason
is that the platform is a very weak wavemaker in the modes being measured, so there is little
radiated wave energy for the walls to reflect.

## Why the standard "sidewall reflection" concern does not apply here

"Severe reflections corrupt data" is the correct concern for bodies that are **strong
wavemakers** or that **span the flume**. The Phase-3 platform is neither:

1. **It barely radiates.** The platform is a sparse array of 0.16 m spars on perforated heave
   plates. Its **heave radiation damping is only 3.5 % of the total heave damping** (the rest is
   viscous); pitch and roll radiate far less still, because the buoys move out of phase and their
   radiated waves largely cancel. With so little wave energy leaving the platform, there is
   almost nothing to reflect.
2. **The operating band is sub-cutoff.** Reflections only build a coherent, corrupting
   cross-flume standing wave at the flume's transverse cut-on periods, **T ≈ 2.19 / 1.53 / 1.25 s**
   (k = nπ/W). The platform's dynamics of interest (heave resonance ≈ 2.5 s) lie **below** the first
   cut-on, where the transverse field is *evanescent* — it decays away from the platform and cannot
   organise into a standing wave.
3. **It is porous and does not span the flume.** Water passes between the 12 buoys and escapes
   freely along the 104 m flume length; only the cross-flume direction is bounded, at several
   plate-radii from each outer buoy.

## Method

Capytaine 2.3.1 potential-flow BEM. Platform = 12 spar-plate buoys (Phase-1 decay-correlated
geometry: 0.159 m spar, equal-area heave plate r = 0.144 m; the geometry that reproduced the
Phase-1 free-decay at T ≈ 2.5 s, ζ ≈ 13 %) at the Phase-3 layout (buoy centres on a 2.5 m
circle). Flume side walls (W = 3.66 m) imposed by the **method of images** (converged over
reflection levels); finite depth h = 2.7 m native. All comparisons are **walls-in vs walls-out
at the same 2.7 m depth**, isolating the sidewall effect the reviewer raised.

## Findings

**Free-decay tests (natural periods and damping):**

| Mode | Wall shift in natural period | Wall shift in added mass |
|---|---|---|
| Heave | −0.25 % | −1.5 % |
| Pitch (= roll) | < 0.2 % | < 0.3 % |
| Surge (= sway) | < 0.7 % | ≤ 2 % |

All shifts are **well inside the ±(3–5) % uncertainty of a physical decay test**. Because heave
radiation is only 3.5 % of the damping, even a hypothetical perfect wall reflection could perturb
the measured damping by at most a few percent — and the explicit walls-in simulation shows less.

**Wave-sweep tests (RAOs):**

- Sidewall effect on wave-frequency loads: **≤ 9 % (heave, at resonance), ≤ 2 % (pitch, surge)**.
- Heave RAO peak: 0.394 (walls out) → 0.359 (walls in), **−9 %** — a smooth, predictable offset,
  not "corruption."
- No spurious resonances in the operating band; the only wall-sensitive periods are the
  transverse cut-ons above — and even there the effect is bounded (next section).

**Confirmed in the time-domain simulator, including the full articulated model.** The findings
above are frequency-domain. We reproduced them in FloatSim's validated time-domain solver (real
quadratic Morison drag), twice — a single-buoy model and the **full 17-body articulated platform**
(12 buoys + 4 hubs + 1 platform, gimbal joints, driven by a fresh method-of-images *coupled*
BEM). All three approaches agree:

| Method | heave-decay period shift | wave-RAO wall effect |
|---|---|---|
| Frequency-domain | −0.25 % | ≤ 9 % |
| Single-buoy FloatSim (time domain) | −0.45 % | ≤ 7.4 % |
| Full articulated 17-body FloatSim | −0.38 % | ≤ 2.9 % |

In the articulated model every one of the platform's ~30 free DOFs — the six rigid-body modes and
the 24 buoy gimbal tilts — moves by less than the platform heave, the damping is unchanged, and
the accelerations at the deck centre and the four cluster centres are essentially identical with
the walls in or out.

## What we concede, and how we handle it

- **Transverse cut-ons (T ≈ 2.19 / 1.53 / 1.25 s):** refining the wall model to three image
  reflections shows the wall effect on the loads **stays within ~±6 % even at these periods** —
  the platform's weak scattering prevents the sharp cross-flume resonance one would see with a
  strong wavemaker. As good practice we still avoid parking sweep periods exactly on them; they
  are narrow, known, and easily skipped in the test matrix.
- **Clearance number:** the reviewer's 0.6 m assumes 2.50 m is the outer extent. Our 2.50 m is the
  **buoy-centre** circle; with the heave plates the outer clearance is ~0.44 m. We analysed the
  as-built ~0.44 m clearance; the conclusions above already reflect it.
- The separate, larger effect at long waves is the **finite 2.7 m depth** (not the walls) — a known
  property of the facility that we account for when relating flume RAOs to the target environment.

## Conclusion

Explicit BEM modelling of HWRL's side walls shows the Phase-3 platform's measured natural periods
change by < 0.7 % and its wave-frequency loads by ≤ 9 %, with no in-band spurious resonances. The
platform's very low wave radiation and the sub-cutoff operating band mean sidewall reflections are
negligible for both the free-decay and wave-sweep campaigns. The 2.50 m platform is compatible
with the HWRL Large Wave Flume.
