# Response to reviewer comment — sidewall effects on the Phase-3 platform test

**Reviewer comment:** *"Phase 3 platform (2.50 m) inside HWRL's 3.67 m flume leaves 0.6 m side
clearance, inducing severe sidewall reflections that will corrupt dynamic data."*

## Summary

We assessed the concern quantitatively with potential-flow boundary-element simulations
(Capytaine) in which HWRL's Large Wave Flume side walls are modelled **explicitly** (method of
images). The result: **sidewall reflections do not corrupt the Phase-3 dynamic data.** With the
walls modelled in versus out (at the same water depth):

- **Free-decay** natural periods shift **< 0.6 %** and the damping is unchanged;
- the **heave response through resonance** — the dynamically important band — shifts **≤ 3 %** at
  the 2.7 m operating depth;
- accelerations at the deck centre and the four cluster hubs change by **≤ 4 %** near resonance.

All are inside normal model-test uncertainty. The physical reason is that the platform is a very
weak wavemaker in the modes being measured, so there is little radiated wave energy for the walls
to reflect. **The larger flume effect is the finite 2.7 m depth — a known, routinely corrected
facility property, not a sidewall artifact.**

## Test article and method

Platform = **16 spar-plate buoys** (4 clusters × 4, square layout) at the Phase-3 layout (buoy
centres on a 2.5 m circle), each the Phase-1 decay-correlated hull (0.159 m spar, equal-area
heave plate r = 0.144 m; the geometry that reproduced the Phase-1 free-decay at T ≈ 2.6 s,
ζ ≈ 13 %). With the plates the outer span is 2.79 m (76 % of the 3.66 m width), giving ~0.44 m
side clearance. (The reviewer's 0.6 m assumes 2.50 m is the outer extent; our 2.50 m is the
buoy-centre circle, so we analysed the tighter as-built ~0.44 m.)

Side walls (W = 3.66 m) are imposed by the **method of images**. Every comparison is
**walls-in vs walls-out at the same depth**, which isolates the sidewall effect the reviewer
raised from the separate (and larger) finite-depth effect. Three complementary models are used,
each in its valid regime:

| Model | What it gives | Depth |
|---|---|---|
| Frequency-domain image-wall BEM | isolated wall effect on the hydrodynamic coefficients, per DOF | native 2.7 m |
| Single-DOF impedance model | platform-heave **response** wall effect | native 2.7 m |
| Articulated 21-body FloatSim (coupled BEM) | free-decay + all-DOF near-resonance response, real quadratic drag, gimbal joints | deep water* |

\* The coupled/articulated BEM runs at deep water because finite depth is impractically slow at
the coupled panel count. Free-decay is radiation-driven and **depth-robust**, so deep-water
modelling is valid there; the depth-sensitive wave response is carried by the two native-2.7 m
models.

## Why the standard "sidewall reflection" concern does not apply here

1. **The platform barely radiates.** Its heave radiation damping is only **~3.5 % of the total
   heave damping** (the rest is viscous drag); pitch and roll radiate less still because the
   buoys move out of phase and their radiated waves largely cancel. With so little wave energy
   leaving the platform, there is almost nothing to reflect — this is what makes the free-decay
   and near-resonance response so insensitive to the walls.
2. **The operating band is sub-cut-on.** A corrupting cross-flume standing wave can build only at
   the flume's transverse cut-on periods, **T ≈ 2.19 / 1.53 / 1.25 s** (kₙ = nπ/W). The dynamics
   of interest (heave resonance ≈ 2.6 s) lie **below** the first cut-on, where the transverse
   field is evanescent — it decays away from the platform and cannot organise into a standing
   wave. Even *at* the cut-ons the weak scattering keeps the effect bounded (~±6 %).
3. **It is porous and does not span the flume.** Water passes between the 16 buoys and escapes
   freely along the 104 m length; only the cross-flume direction is bounded.

## Findings

**Free-decay (natural period and damping), articulated 21-body, walls-in vs walls-out:**

| Quantity | walls out → walls in | wall effect |
|---|---|---|
| Heave natural period | 2.607 → 2.593 s | **−0.55 %** |
| Heave damping ζ | 6.4 % → 6.4 % | unchanged |
| Buoy gimbal tilt | 0.0007 → 0.0009 rad | ~0 (negligible) |

The period shift is ~20× smaller than the ±(3–5) % scatter of a physical free-decay test, and
the damping (viscous-dominated) is untouched.

**Wave response at the 2.7 m operating depth (single-DOF platform-heave):**

- **Through resonance (1.5–2.9 s)** — the dynamically important band: heave response wall effect
  **≤ 2.6 %**.
- **Long-period tail (T > 3 s):** grows to ~20 %, but this is *off-resonance* (small motion), and
  there the **finite-depth effect is far larger** (see below). Pitch/surge loads ≤ 2.6 %
  throughout.

**All-DOF accelerations near resonance (articulated 21-body), at the deck centre + 4 cluster
hubs:** surge ≤ 0.9 %, heave ≤ 4.0 %, pitch ≤ 2.2 %; sway/roll/yaw unexcited in head seas. Peaks
occur at the 2.19 s cut-on.

## The dominant flume effect is depth, not the walls

Isolating the two effects on the heave wave excitation (both at head seas):

| Wave period | Sidewall effect (walls in vs out, 2.7 m) | Finite-depth effect (2.7 m vs deep) |
|---|---|---|
| 2.52 s | −5.4 % | −19 % |
| 3.00 s | −14 % | −27 % |
| 3.50 s | −17 % | −33 % |
| 4.00 s | −15 % | −37 % |

At every period the finite-depth effect exceeds the sidewall effect. Finite depth is a known
facility property that any flume campaign corrects for by depth-scaling; the sidewalls add a
smaller term on top, negligible through the resonance where the platform's dynamics live.

## What we concede, and how we handle it

- **Transverse cut-ons (T ≈ 2.19 / 1.53 / 1.25 s):** narrow, known, and skipped in the sweep
  matrix; the wall effect stays within ~±6 % even there.
- **Long-period tests (T > 3 s):** apply the standard finite-depth correction (and the smaller
  blockage correction) — the response there is depth-dominated, not a sidewall artifact.
- **Clearance number:** analysed at the as-built ~0.44 m (buoy-centre circle + plates), tighter
  than the reviewer's 0.6 m.

## Conclusion

Explicit BEM modelling of HWRL's side walls shows the Phase-3 platform's free-decay periods
change by < 0.6 % and its near-resonance response by ≤ 3 %, with no in-band spurious resonance.
The platform's very low wave radiation and the sub-cut-on operating band make sidewall
reflections negligible for both the free-decay and the wave-sweep campaigns. The primary flume
consideration is the finite 2.7 m depth — larger than the walls and handled by standard
depth-scaling. The 2.50 m platform is compatible with the HWRL Large Wave Flume.
