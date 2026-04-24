# Elastic Catenary — Derivation

Self-contained derivation of the closed-form elastic catenary with
frictionless-seabed contact, used by `floatsim/mooring/catenary_analytic.py`
to supply analytical mooring reaction forces for Phase 1 (ARCHITECTURE.md
§4, §7). This document is the sign-convention reference — review this
before touching the code.

References:

* Irvine, H.M. 1981. *Cable Structures*. MIT Press, Chs. 1–3.
* Faltinsen, O.M. 1990. *Sea Loads on Ships and Offshore Structures*,
  Ch. 8 (compact version with seabed contact handled explicitly).

## 1. Geometry and sign conventions

* Inertial frame: `x` horizontal, `z` vertical, `z = 0` at mean water
  level, `z = -h` at seabed (so `h > 0` is the water depth).
* Anchor at `(x_A, z_A)`; fairlead at `(x_F, z_F)`. We require
  `x_F > x_A` (the anchor is to the left of the fairlead by convention;
  sign flips are handled by the caller via a reflection).
* Unstretched arc length `s` runs from `0` at the anchor to `L` at the
  fairlead. Each element `ds` has weight per unit unstretched length `w`
  (submerged, N/m).
* Axial stiffness `EA` (N). Linear elasticity; a unit length under
  tension `T` stretches by `T/EA`.

Let `T(s)` denote the line tension at arc position `s`, and `theta(s)`
the angle of the line from horizontal (measured in the vertical plane
containing the anchor and fairlead — the line lies in a single vertical
plane because weight is the only external load). Then
`H = T cos(theta)` is the **horizontal tension**, conserved along the
line (no horizontal force per element), and `V(s) = T sin(theta) =
V_A + w s` is the **vertical tension**, increasing with arc length due
to the weight per unit length.

At the fairlead `V_F = V_A + w L`. The total line tension at the
fairlead is `T_F = sqrt(H^2 + V_F^2)`.

## 2. Fully suspended regime

The entire line hangs between anchor and fairlead; `V_A > 0` (the line
makes a strictly positive angle with the horizontal at the anchor).
From the tangent equations with stretch correction `(1 + T/EA)`:

```
dx/ds = (1 + T/EA) cos(theta) = H/T + H/EA
dz/ds = (1 + T/EA) sin(theta) = V/T + V/EA
```

Integrating `0 -> L`:

```
x_F - x_A = integral_0^L [H/T(s) + H/EA] ds
          = (H/w) [asinh(V_F/H) - asinh(V_A/H)] + H L / EA              (S1)

z_F - z_A = integral_0^L [V(s)/T(s) + V(s)/EA] ds
          = (1/w) [sqrt(H^2 + V_F^2) - sqrt(H^2 + V_A^2)]
            + (V_A + V_F) L / (2 EA)                                     (S2)

V_F = V_A + w L                                                          (S3)
```

With `V_F` eliminated via (S3), (S1) and (S2) are two nonlinear equations
in the two unknowns `(H, V_A)`. Solve numerically.

## 3. Touchdown regime

Part of the line rests on the seabed. The anchor is on the seabed at
`z_A = -h`. Let `L_s` denote the **unstretched** arc length resting on
the seabed. The suspended part spans `s in [L_s, L]`; the touchdown
point (where the resting part lifts off) is at position `(x_TD, -h)`.

Frictionless seabed: the line exerts no tangential force on the bottom,
so horizontal tension `H` is constant throughout, including the resting
portion. At the touchdown point the line is tangent to horizontal
(`V = 0`), so `V_A = 0` as well; the seabed supports any vertical load
the line tries to apply (i.e. the reaction is normal to the seabed).

Under tension `H`, the resting part stretches uniformly, so its
projected horizontal extent is `L_s (1 + H/EA)`. On the suspended part,
run the same integrals as §2 but from `s = L_s` (at touchdown, `V = 0`,
`z = -h`) to `s = L` (at fairlead):

```
x_F - x_A = L_s (1 + H/EA) + (H/w) asinh(V_F/H) + H (L - L_s) / EA
          = L_s + H L / EA + (H/w) asinh(V_F/H)                          (T1)

z_F - z_A = (1/w) [sqrt(H^2 + V_F^2) - H] + V_F (L - L_s) / (2 EA)       (T2)

V_F = w (L - L_s)                                                        (T3)
```

Note that `z_F - z_A = z_F - (-h) = z_F + h`, matching the height rise
over the suspended portion. After substituting (T3), (T1) and (T2) are
two nonlinear equations in `(H, L_s)`. Physical solutions satisfy
`L_s in [0, L]`; `L_s = 0` is the geometric boundary with the suspended
regime, `L_s = L` corresponds to the entire line lying on the seabed
(degenerate: `V_F = 0`, and then (T1) reduces to
`x_F = L + H L/EA` — a straight elastic rod — valid only when
`x_F >= L` and `z_F = z_A`).

**Touchdown horizontal coordinate** (for diagnostic / post-processing):

```
x_TD = x_A + L_s (1 + H/EA)                                              (T4)
```

## 4. Regime selection

Given `(L, w, EA, (x_A, z_A), (x_F, z_F), h)`:

1. If `z_A > -h`, the anchor is above the seabed: only the suspended
   regime is meaningful (§2). Solve (S1)-(S3).
2. If `z_A = -h` (anchor on seabed), first attempt the touchdown regime
   (§3). Solve (T1)-(T3) for `(H, L_s)`:
   * if `L_s in (0, L)`, accept.
   * if `L_s <= 0`, no portion rests — fall back to (S1)-(S3).
   * if `L_s >= L`, the line is entirely flat (degenerate). The caller
     should be alerted (rare in practice — would require `x_F >= L`).
3. If `z_A < -h` (anchor below seabed) is a caller error (undefined).

In this repository the anchor is always specified explicitly; we do not
assume `z_A = -h`. The caller signals seabed contact is allowed by
supplying `seabed_depth=h >= 0` to the solver. Without it, the regime
is forced to §2.

## 5. Limiting cases (sanity checks)

* **Inextensible limit** (`EA -> infty`): the `H L / EA`, `V_A L / EA`,
  and `V_F (L - L_s) / EA` stretch terms drop out. Equations reduce to
  the classical Irvine (1981) inextensible form. A test with
  `EA = 1e20` and a short span verifies convergence.
* **Vertical hanging line** (`x_F = x_A`): the only solution is
  `theta = 90 deg` uniformly, so `H -> 0` and `V_F = w L` (weight).
  The line stretches by `w L^2 / (2 EA)` (cumulative weight under each
  element), so `z_F - z_A = -L - w L^2 / (2 EA)` when hanging free.
* **Taut elastic rod** (short `L`, large `EA`, `w` small): the line
  approximates a straight rod of stretched length
  `L (1 + T_avg/EA)` with `T_avg ~ sqrt((x_F-x_A)^2 + (z_F-z_A)^2) w / 2`
  for small sag.

## 6. Outputs

The solver returns:

| symbol                 | description                               | units |
| ---------------------- | ----------------------------------------- | ----- |
| `H`                    | horizontal tension (constant along line)  | N     |
| `V_F`                  | vertical tension at fairlead              | N     |
| `V_A`                  | vertical tension at anchor                | N     |
| `T_F = sqrt(H^2+V_F^2)`| total tension at fairlead                 | N     |
| `top_angle = atan(V_F/H)`| angle at fairlead from horizontal       | rad   |
| `touchdown_length L_s` | unstretched length on seabed (0 suspended)| m     |
| `touchdown_x`          | `x_A + L_s (1 + H/EA)` (NaN if suspended) | m     |
| `regime`               | `"suspended"` or `"touchdown"`            | —     |

## 7. Numerical solution

The 2x2 nonlinear system is solved with `scipy.optimize.root`
(hybrid Powell, `method="hybr"`). Initial guesses:

* **Touchdown**: `H_0 = w * (x_F - x_A) / 2` (rough, scales like the
  average horizontal reach), `L_s_0 = L - sqrt((z_F - z_A)^2
  + 2 (x_F - x_A) (z_F - z_A))` clamped to `(0, L)` — the inextensible
  touchdown length for a parabolic approximation.
* **Suspended**: `H_0` as above, `V_A_0 = w * L / 4` (rough, between 0
  and `w L`).

An analytical Jacobian is supplied (small 2x2 matrix, closed-form from
the partial derivatives of (S1)-(S2) or (T1)-(T2)) for robustness when
the problem is near-degenerate.

## 8. Validation (M4 gate 2)

The independent cross-check is a **shooting method**: given candidate
`(H, V_A)`, integrate

```
dx/ds = H/T + H/EA,    dz/ds = V/T + V/EA,    V(s) = V_A + w s
```

with `scipy.integrate.solve_ivp` from `s = 0` at the anchor to
`s = L`, then solve for `(H, V_A)` such that `(x(L), z(L)) = (x_F, z_F)`.
The closed-form and shooting-method solutions must agree on `H` and
`V_F` to `rtol = 1e-4` (docs/milestone-4-plan.md Q2).

For the touchdown regime, the shooting method integrates only the
suspended portion (arc length `L - L_s`) starting at `(x_TD, -h)` with
`V_A = 0`; the resting portion's length is determined by the closed-form
`L_s`.
