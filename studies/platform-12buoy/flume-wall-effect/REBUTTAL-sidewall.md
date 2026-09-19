# Response to reviewer comment — sidewall effects on the Phase-3 platform test

**Reviewer comment:** *"Phase 3 platform (2.50 m) inside HWRL's 3.67 m flume leaves 0.6 m side
clearance, inducing severe sidewall reflections that will corrupt dynamic data."*

## Summary

We assessed the concern quantitatively with potential-flow boundary-element simulations
(Capytaine) in which HWRL's Large Wave Flume side walls are modelled **explicitly** (method of
images). The result: **sidewall reflections do not corrupt the Phase-3 dynamic data.** With the
walls modelled in versus out (at the same water depth):

- **Free-decay** natural periods shift **< 0.6 %** and the damping is unchanged;
- the **heave RAO wall effect is ≤ 4 %** across the whole wave band, and does not grow at long
  periods;
- accelerations at the deck centre and the four cluster hubs change by **≤ 4.5 %** in every DOF.

All are inside normal model-test uncertainty. The physical reason is that the platform is a very
weak wavemaker in the modes being measured, so there is little radiated wave energy for the walls
to reflect. **The larger flume effect is the finite 2.7 m depth — a known, routinely corrected
facility property, not a sidewall artifact.**

## Test article and method

Platform = **16 spar-plate buoys** (4 clusters × 4, square layout) at the Phase-3 layout (buoy
centres on a 2.5 m circle), each the Phase-1 decay-correlated hull (0.159 m spar, equal-area
heave plate r = 0.144 m; the geometry that reproduced the Phase-1 free-decay at T ≈ 2.6 s,
ζ ≈ 13 %). The platform is tested **corner-on (45°)**: with the plates, its span across the
3.66 m flume is 2.06 m (56 % of the width), giving **~0.80 m side clearance** — more than the
reviewer's 0.6 m. We also analysed the flat-on (0°) orientation, the widest at ~0.44 m/side, and
it gives the same result (see "Orientation independence" below).

Side walls (W = 3.66 m) are imposed by the **method of images**. Every comparison is
**walls-in vs walls-out at the same depth**, which isolates the sidewall effect the reviewer
raised from the separate (and larger) finite-depth effect.

The sidewall effect is taken from the **coupled 21-body BEM** (the full 96-DOF open-vs-walled
solve — free-decay, RAO, and all-DOF accelerations, with real quadratic drag and gimbal joints).
A simpler single-array frequency-domain model was also run, but it under-converges in image count
at long periods (2 vs 3 reflections change the answer ~10× and disagree with the coupled solve
even in sign), so it is used only for the separate **finite-depth effect**, which is image-free
and corroborated independently by textbook Airy wave kinematics. The coupled BEM runs at deep
water (finite depth is impractically slow at its panel count); for this weak radiator the wall
effect is depth-robust, and the depth effect itself is assessed separately at 2.7 m.

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

**Wave-frequency response (coupled 21-body BEM, walls-in vs walls-out):** the heave RAO wall
effect is **≤ 4 % across the whole wave band** and does **not** grow at long periods — the
excitation wall effect stays ≤ 4 % and shrinks toward zero at 4 s. (The simpler single-array
model over-states this at long periods, but it under-converges in image count there; the coupled
solve is authoritative.)

**All-DOF accelerations near resonance (articulated 21-body), at the deck centre + 4 cluster
hubs:** surge ≤ 0.6 %, heave ≤ 4.4 %, pitch ≤ 2.1 %; sway/roll/yaw unexcited in head seas. Peaks
occur at the 2.19 s cut-on.

## The dominant flume effect is depth, not the walls

Isolating the two effects on the heave wave excitation (both at head seas):

| Wave period | Sidewall effect (coupled BEM) | Finite-depth effect (2.7 m vs deep) |
|---|---|---|
| 2.52 s | +3.0 % | −19 % |
| 3.00 s | +1.7 % | −27 % |
| 3.50 s | +0.7 % | −33 % |
| 4.00 s | +0.1 % | −37 % |

The sidewall effect stays small (≤ 4 %) and shrinks toward zero at long periods, while the
finite-depth effect grows to −37 %. Finite depth is a known facility property that any flume
campaign corrects for by depth-scaling; the sidewalls are a minor term at every period.

**Why the depth effect grows with period.** A wave only feels the bottom once its wavelength is
long compared with the 2.7 m depth. Short waves (h/λ = 0.88 at 1.4 s) are confined near the
surface and behave exactly as in deep water; at the 2.6 s natural period h/λ = 0.27, and at 4 s
it is 0.15, so the orbital motion reaches the floor. The floor forces the vertical particle
velocity to zero, which flattens the circular orbits into ellipses and shrinks the vertical
motion that lifts the heave plate (at −1.38 m, a good part of the way to the 2.7 m floor). This
is plain Airy wave theory, and it reproduces the BEM depth effect at every period — so the depth
effect is real, method-independent, and, like any finite-depth correction, routinely applied.

## Orientation independence — clearance is not the controlling parameter

The reviewer's argument rests on the 0.6 m clearance. We tested whether clearance is actually
what drives the effect by re-running the entire study with the platform **rotated 45°** (a 90°
rotation is a symmetry no-op for the 4-fold-symmetric layout). At 45° the platform is corner-on
to the walls and the side clearance **nearly doubles, 0.44 → 0.80 m**:

| Orientation | Clearance | Free-decay ΔT | Excitation wall effect (coupled BEM) |
|---|---|---|---|
| 0° (flat-on) | 0.44 m/side | −0.55 % | ≤ 3.5 % |
| 45° (corner-on) | 0.80 m/side | −0.55 % | ≤ 3.6 % |

The wall effect is **unchanged** despite doubling the clearance. It is set by the **bulk channel
blockage** — the total array volume relative to the flume cross-section, which is
orientation-independent — not by the nearest-buoy clearance. So the 0.6 m clearance the comment
is built on is not the parameter that controls sidewall reflections here; the (small) effect is
robust to how the platform is oriented in the flume.

## What we concede, and how we handle it

- **Transverse cut-ons (T ≈ 2.19 / 1.53 / 1.25 s):** narrow, known, and skipped in the sweep
  matrix; the wall effect stays within ~±6 % even there.
- **Long-period tests (T > 3 s):** apply the standard finite-depth correction — the response
  there is depth-dominated (the sidewall effect stays ≤ 4 % at every period).
- **Clearance number:** at the 45° test orientation the as-built clearance is ~0.80 m (more than
  the reviewer's 0.6 m); the widest, flat-on orientation (~0.44 m) was analysed too and gives the
  same result.
- **Method fidelity:** the sidewall effect is taken from the coupled 21-body BEM; the simpler
  single-array frequency-domain model under-converges in image count at long periods, so its
  wall effect is not used (only its image-free, Airy-corroborated depth effect).

## Conclusion

Explicit BEM modelling of HWRL's side walls shows the Phase-3 platform's free-decay periods
change by < 0.6 % and its wave-frequency response by ≤ 4 % across the band, with no in-band
spurious resonance.
The platform's very low wave radiation and the sub-cut-on operating band make sidewall
reflections negligible for both the free-decay and the wave-sweep campaigns. The primary flume
consideration is the finite 2.7 m depth — larger than the walls and handled by standard
depth-scaling. The 2.50 m platform is compatible with the HWRL Large Wave Flume.
