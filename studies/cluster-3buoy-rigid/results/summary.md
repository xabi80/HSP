# 3-buoy rigid-cluster heave decay -- summary

## Headline: hydrodynamic interaction

**Interaction ratio R(inf) = A33_composite / (3 x A33_single) =
1.0113** (+1.13% on heave added mass). R(omega_n) =
1.0106. The radiation-damping interaction is
slightly destructive: B33_composite / (3 x B33_single) at omega_n =
2.8935 (< 3, i.e. -3.5%).
The three hulls sit ~0.87 m apart (0.5 m cluster radius) -- far
relative to the 0.215 m plate radius -- so hydrodynamic coupling is
weak but measurable and, per the prior band [1.00, 1.20], at the low
end.

**Period effect of the interaction:**

| quantity | value |
|---|---|
| T_n WITH interaction (measured A33_composite) | 3.10609 s |
| T_n WITHOUT (3 x single values) | 3.09924 s |
| interaction delta | +6.85 ms (+0.221%) |

The +1.1% added-mass interaction lengthens the heave
period by 6.8 ms.

## Period validation

| quantity | value |
|---|---|
| M + A33_composite | 162.0838 kg |
| C33_composite | 663.2420 N/m (= 3 x 221.0807) |
| T_n analytical (with interaction) | 3.10609 s |
| T_n FloatSim (zero-crossing) | 3.10533 s |
| rel-err | 0.024% (gate 1e-2: PASS) |

## Radiation damping (BEM-only)

| quantity | value |
|---|---|
| omega_n | 2.0229 rad/s |
| B33_composite(omega_n) | 1.8914e-01 |
| zeta_rad predicted | 2.8844e-04 (0.029% crit) |
| zeta_rad FloatSim | 2.8855e-04 (0.029% crit) |
| rel-err | 0.039% (gate 5e-2: PASS) |

Radiation damping is very light (< 0.1% critical), as for the single
buoy -- a low-radiation heave geometry.

## Heave-plate drag (BEM + Morison)

Single degenerate horizontal-cylinder element, projected area
D*L = 3 x 0.1452 = 0.4356 m^2, Cd = 5.0 (equivalent to three offset
elements for pure heave of a rigid symmetric body). KC per plate =
2*pi*0.10/0.43 = 1.46 (unchanged from the single-buoy
study; Cd=5.0 valid per Tao & Cai).

| quantity | value |
|---|---|
| effective zeta (first peaks, amplitude-dependent) | 2.5079e-02 (2.51% crit) |
| period | 3.1067 s |

## Cross-DOF

Symmetric 3-fold cluster -> heave decouples. Measured max |xi_k|
(k != heave) = 2.639e-10:

- surge: 1.896e-10
- sway: 1.222e-13
- roll: 2.680e-14
- pitch: 2.639e-10
- yaw: 1.707e-13

Band asserted from the measurement (< 2.64e-09, 10x). Three-hull
BEM panel asymmetries are real but here remain at numerical-noise
level.

## Figures

- `figures/heave_decay.png` -- both decays + envelope
- `figures/decay_envelope_log.png` -- log-decrement
- `figures/cross_dof.png` -- off-heave DOF
- `figures/interaction_A33.png` -- A33 composite vs 3 x single
