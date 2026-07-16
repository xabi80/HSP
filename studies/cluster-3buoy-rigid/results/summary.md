# 3-buoy rigid-cluster heave decay -- summary

## Headline: hydrodynamic interaction

**Interaction ratio R(inf) = A33_composite / (3 x A33_single) =
1.0113** (+1.13% on heave added mass). R(omega_n) = 1.0106. Added
mass is a near-field effect, so it barely interacts at this spacing.

**Radiation damping, by contrast, interacts STRONGLY and
CONSTRUCTIVELY.** B33_composite / (3 x B33_single) at omega_n =
**2.894** (>> 1), i.e. B33_composite / B33_single = **8.68x** --
approaching the coherent-radiation ceiling N^2 = 9 for N = 3 in-phase
sources. A rigid cluster heaves in phase; at sub-wavelength hull
spacing (~0.87 m, far below the ~60 m wavelength at omega_n) the three
radiated wave fields add coherently, so radiated power scales toward
N^2 and the heave radiation damping rises ~8.7x per unit velocity.
(An earlier draft of this summary mislabelled the 2.894 ratio as
"slightly destructive -3.5%"; that was a misread -- 2.894 is the
composite / 3x-single ratio itself, i.e. strongly constructive.)

The three hulls sit ~0.87 m apart (0.5 m cluster radius) -- far
relative to the 0.215 m plate radius (near-field / added mass sees
them as independent) but far BELOW the radiated wavelength (far-field
/ radiation damping sees them as one coherent source). Per the prior
band [1.00, 1.20] the added-mass ratio is at the low end; the
radiation-damping enhancement was not in that band's scope.

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

Radiation damping is still very light in absolute terms (0.029%
critical) but it ROSE 2.58x relative to the single buoy (single
zeta_rad 0.0112% -> cluster 0.0288%; measured at each body's own
omega_n), driven by the ~8.7x constructive rise in B33 partly offset
by the 3.1x rise in sqrt((M+A)*C). This is the decay-time signature of
the coherent-radiation enhancement above.

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

## Cross-DOF (closeout diagnostic, crossdof_diagnostic.py)

The placement -- one buoy on +x, two mirrored across the x-axis -- has
an exact y-mirror (x-z) symmetry plane (mirror check:
max |y_i + y_j| over the two off-axis hulls = 0.000, exact mirror
copies). Under y -> -y, heave is EVEN; it may couple only to the other
EVEN DOFs (surge, pitch) and is forbidden to couple to the ODD DOFs
(sway, roll, yaw). The measured decay traces obey this exactly:

| DOF | parity under y-mirror | max \|xi_k\| |
|---|---|---|
| surge | even (allowed) | 4.196e-11 |
| pitch | even (allowed) | 2.639e-10 |
| sway  | odd (forbidden) | 1.222e-13 |
| roll  | odd (forbidden) | 2.680e-14 |
| yaw   | odd (forbidden) | 1.707e-13 |

max allowed (surge/pitch) = 2.64e-10; max forbidden (sway/roll/yaw)
= 1.71e-13; ratio **1546x**. The tiny residual coupling therefore is
NOT random panel noise -- it is confined to the symmetry-allowed
channels (surge and pitch, not surge alone), consistent with the
y-mirror argument. The forbidden channels sit ~1500x lower at the
numerical-noise floor. (The single-buoy study's on-axis geometry gave
~1e-13 across all off-heave DOFs; the cluster's allowed channels rise
to ~1e-10 because the composite panel discretisation breaks exact
3-fold symmetry while preserving the y-mirror.)

Band asserted from the measurement: max |xi_k| < 2.64e-09 (10x).

## Conclusions

1. **Added mass barely interacts (near-field): R = 1.011.** The +1.1%
   lengthens the heave period by +6.85 ms.
2. **Radiation damping interacts strongly and constructively
   (far-field coherent): B33 up ~8.7x (toward the N^2=9 ceiling),
   decay zeta_rad up 2.58x.** The two ratios differ because added mass
   is a local near-field quantity while radiation damping is set by
   the coherent far-field radiated power of the in-phase rigid cluster.
3. **Consequence for the 12-buoy model.** Even after rising 2.58x, the
   cluster's radiation damping (0.029% critical) is ~86x smaller than
   the Morison heave-plate viscous damping (2.5% critical). So the
   decay rate is overwhelmingly set by the plate drag: **the tank-test
   Cd calibration is the single most important empirical input the
   12-buoy model will have.** (Note this conclusion is reached via the
   OPPOSITE mechanism to the pre-diagnostic expectation: radiation
   damping rose, it did not drop -- but it remains negligible against
   the viscous term either way, so the practical priority on Cd
   stands. The viscous/radiation ratio is ~86, not ~2.5e3.)

## Figures

- `figures/heave_decay.png` -- both decays + envelope
- `figures/decay_envelope_log.png` -- log-decrement
- `figures/cross_dof.png` -- off-heave DOF
- `figures/interaction_A33.png` -- A33 composite vs 3 x single
