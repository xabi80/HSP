# Spar-fin free-decay study -- Step F summary

Regenerated at the M7.5 resumption on the **eqdraft** mesh (buoy
translated down dz = 0.1846 m to the true free-floating equilibrium
waterline; see waterline_balance.py and the STEP-A-FINDING addendum).

## Heave natural period

| quantity | value |
|---|---|
| M + A_inf(heave) | 49.7891 kg (A_inf = 21.1191) |
| C33 | 221.0807 N/m |
| T_n analytical (2*pi*sqrt((M+A_inf)/C33)) | 2.9818 s |
| T_n FloatSim (zero-crossing) | 2.9820 s |
| rel-err | 0.008% (gate 1e-2: PASS) |

## Radiation damping (BEM-only)

| quantity | value |
|---|---|
| omega_n | 2.1072 rad/s |
| B_heave(omega_n) | 2.3452e-02 N.s/m |
| zeta_rad analytical | 1.1177e-04 (0.011% crit) |
| zeta_rad FloatSim (log-decrement) | 1.1320e-04 (0.011% crit) |
| rel-err | 1.281% (gate 5e-2: PASS) |

The radiation-only damping is very light (< 0.1% critical): the
spar+fin is a low-radiation heave geometry, so BEM-only decay persists
tens of periods.

## Heave-plate drag (BEM + Morison)

Modelled with the degenerate horizontal-cylinder approximation
(Pre-flight 3 audit): a single Morison member, axis horizontal, with
projected area D*L = A_plate = 0.1452 m^2 and Cd = 5.0,
reproduces the plate's vertical drag F_z = 0.5*rho*Cd*A*|v_z|*v_z
exactly for pure heave.

| quantity | value |
|---|---|
| effective zeta (first peaks, amplitude-dependent) | 2.5225e-02 (2.52% crit) |
| period | 2.9827 s |

Quadratic drag is amplitude-dependent, so this "effective zeta" is not
a constant modal damping; it is reported without a quantitative gate
(tank data is the validator). The plate drag dominates the radiation
damping by ~2 orders of magnitude and kills the decay within a few
periods.

## Cross-DOF magnitudes

Measured max |xi_k| for k != heave (both runs):

| DOF | max |xi_k| |
|---|---|
| surge | 6.922e-18 |
| sway | 9.877e-18 |
| roll | 8.089e-18 |
| pitch | 1.065e-17 |
| yaw | 9.441e-13 |

cross_max = 9.441e-13, entirely at numerical-noise level:
surge/sway/roll/pitch sit at ~1e-17 (machine epsilon), and yaw at
~9e-13 is the largest only because yaw has zero hydrostatic restoring
(C[5,5]=0), so machine-epsilon forces integrate into a negligible
drift rather than being restored. The body behaves as effectively
axisymmetric in heave. Per the resumption instruction the assertion
band is derived from the measurement (cross_max < 9.44e-12,
10x the measured value) rather than inherited blindly; note the
measured coupling is in fact well below the README's 1e-11 figure, so
that gate would also have passed here. The band is kept
measurement-derived so a future geometry with genuine fin-offset
coupling is not silently over-constrained.

## Figures

- `figures/heave_decay.png` -- both decays + analytical radiation envelope
- `figures/decay_envelope_log.png` -- log-decrement (BEM-only) vs analytical
- `figures/cross_dof_silence.png` -- off-heave DOF traces
