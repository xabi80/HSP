# M6 PR6 — Drag aggregation derivation + HydroDyn AxCd factor

## Summary

Pre-flight to PR6 / S5 heave drag decay validation. Aggregated the
OC4 marin_semi 25 Morison members + 3 axial-drag heave-plate joints
into a single heave-equivalent quadratic-drag coefficient `R`, then
predicted the hyperbolic-envelope per-cycle decay constant `δ` from
first principles and compared against OpenFAST's S5 measurement.

**Step A (aggregation)** + **Step B (δ prediction)** + **Step C
(validate against OF)** is implemented in
`scripts/m6_pr6_drag_aggregation.py`. The script reads the S5
HydroDyn deck and produces the per-member / per-joint contribution
table reproduced below.

## Hyperbolic decay derivation

For an oscillator `m ξ̈ + R ξ̇|ξ̇| + C ξ = 0` (pure quadratic damping):

- Energy per cycle lost = `R · ∫₀^T |ẋ|³ dt = (8/3) · R · A³ · ω_n²`
  using `∫₀^(2π) |sin u|³ du = 8/3`.
- `E = (1/2) C A²`, so `dE/dn = C · A · dA/dn`.
- Equating and solving: `dA/dn = -(8/3) · R · A² / m`
  (using `C = m ω_n²`).
- Integrating: `1/A(n) = 1/A(0) + (8/3) · (R/m) · n`.
- Defining `δ` such that `A(n) = A(0) / (1 + n · A(0) · δ)`:
  **`δ = (8/3) · R / m_eff`**, units `1/m`.

Consistent with the M5 PR5 reference test
(`tests/validation/test_m5_drag_free_decay.py`) which uses
equivalent `δ = (4/3) · ρ · Cd · D · L / m_eff` once
`R = 0.5 ρ Cd D L` is substituted.

## HydroDyn axial drag formula — first crucial finding

**A naive "standard Morison" axial drag formula
`F = 0.5 ρ A_x Cd v|v|` is off by a factor of 2 from HydroDyn's
actual code.** Read of `OpenFAST/modules/hydrodyn/src/Morison.f90`:

```fortran
! init (line 3079-3085):
p%An_End(:,i) = An_drag      ! Σ_members sgn · k · π · R²   (full area vector)
Amag_drag = Dot_Product(An_drag, An_drag)
p%DragConst_End(i) = JAxCd · ρ / (4 · Amag_drag)

! runtime (line 4729 + 4742):
vmag = vrel · An_End                                 ! scalar
F_D_End(i, j) = An_End(i) · DragConst_End(j) · |vmag| · vmag
```

For a single attached vertical member of diameter D
(`A_x = πR² = πD²/4`, `An_End = ±A_x ẑ`):

```
F_z = -(1/4) · ρ · A_x · JAxCd · v_z · |v_z|
```

The `(1/4)` is **half** of "standard Morison" `(1/2)`. HydroDyn's
`JAxCd` is implicitly a **two-face combined disc coefficient**;
per-face Morison equivalent is `JAxCd / 2`. The HydroDyn User's
Guide does NOT state this — only the source does. Codified as
conventions doc Item 30.

## OC4 marin_semi aggregation

```
Cylindrical Morison contribution to heave drag
(only members with at least one underwater node contribute -- skip those entirely above SWL)

memb   j1   j2        L      D     Cd  theta_deg   sin^3 th    R_cyl [kg/m]  note
-----------------------------------------------------------------------------------------------
   1    1    2    30.00    6.5   0.56       0.00    0.00000       0.000e+00  partial L_sub=20.00
   2    3    4    26.00   12.0   0.61       0.00    0.00000       0.000e+00  partial L_sub=14.00
   3    5    6    26.00   12.0   0.61       0.00    0.00000       0.000e+00  partial L_sub=14.00
   4    7    8    26.00   12.0   0.61       0.00    0.00000       0.000e+00  partial L_sub=14.00
   5   42    3     5.94   24.0   0.68       0.00    0.00000       0.000e+00
   6   43    5     5.94   24.0   0.68       0.00    0.00000       0.000e+00
   7   44    7     5.94   24.0   0.68       0.00    0.00000       0.000e+00
  23    9   42     0.06   24.0   0.68       0.01    0.00000       2.323e-09
  24   10   43     0.06   24.0   0.68       0.01    0.00000       2.323e-09
  25   11   44     0.06   24.0   0.68       0.01    0.00000       2.323e-09
   8   12   13    38.00    1.6   0.63      90.00    1.00000       0.000e+00  above SWL
   9   14   15    38.00    1.6   0.63      90.00    1.00000       0.000e+00  above SWL
  10   16   17    38.00    1.6   0.63      90.00    1.00000       0.000e+00  above SWL
  11   18   19    26.00    1.6   0.63      90.00    1.00000       1.343e+04
  12   20   21    26.00    1.6   0.63      90.00    1.00000       1.343e+04
  13   22   23    26.00    1.6   0.63      90.00    1.00000       1.343e+04
  14   24   25    19.62    1.6   0.63      90.00    1.00000       0.000e+00  above SWL
  15   26   27    19.62    1.6   0.63      90.00    1.00000       0.000e+00  above SWL
  16   28   29    19.62    1.6   0.63      90.00    1.00000       0.000e+00  above SWL
  17   30   31    13.62    1.6   0.63      90.00    1.00000       7.035e+03
  18   32   33    13.62    1.6   0.63      90.00    1.00000       7.036e+03
  19   34   35    13.62    1.6   0.63      90.00    1.00000       7.035e+03
  20   36   37    32.04    1.6   0.63      37.76    0.22957       2.430e+03  partial L_sub=20.49
  21   38   39    32.04    1.6   0.63      37.76    0.22963       2.431e+03  partial L_sub=20.49
  22   40   41    32.04    1.6   0.63      37.76    0.22957       2.430e+03  partial L_sub=20.49

  Total cylindrical Morison R: 6.8691e+04 kg/m
```

```
Axial drag contribution at joints (1/4 factor, HydroDyn convention per Item 30)
joint                 (x,y,z)  D_attached        A_x   AxCd  theta_deg   cos^3    R_ax [kg/m]
-----------------------------------------------------------------------------------------------
    9  (+14.43,+25.00,-20.00)       24.00     452.39   9.60       0.01   1.000     1.1129e+06
   10  (-28.87, +0.00,-20.00)       24.00     452.39   9.60       0.01   1.000     1.1129e+06
   11  (+14.43,-25.00,-20.00)       24.00     452.39   9.60       0.01   1.000     1.1129e+06

  Total axial R: 3.3386e+06 kg/m
```

```
GRAND TOTAL R = 3.4073e+06 kg/m
  cylindrical: 6.8691e+04 (2.0%)
  axial:       3.3386e+06 (98.0%)
```

**Heave-plate axial drag dominates by ~50×.** The cylindrical
cross-braces (members 11/12/13 at z=−17, members 17/18/19 at
z=−17, members 20/21/22 partially submerged) contribute the
remaining 2 %.

## Step C — δ prediction vs OpenFAST measurement

```
M_combined (platform+tower+RNA) = 1.4074e+07 kg
A_inf_33 (marin_semi)            = 1.4960e+07 kg
m_eff = M + A_inf                = 2.9034e+07 kg

δ_predicted = (8/3) · R / m_eff = 0.3130 1/m
δ_OF (S5 measured)              = 0.3090 1/m
rel-err |pred - OF| / OF        = 1.28 %     ✓ within 5 % gate
```

The S5 reference time series at
`tests/fixtures/openfast/oc4_deepcwind/inputs/s5_drag_decay/s5_drag_decay.csv`
shows clean hyperbolic decay over peaks 0-15 (within 1 % of the
hyperbolic prediction). The envelope flattens at amplitude < 0.5 m
where the residual radiation + mooring linear damping crosses over
to dominate the v² drag (regime-transition caveat documented for
PR6's fit window).

## Outcome / proceeding to Step D

Aggregation validated (Outcome (a) per the locked Q2 investigation
protocol). PR6 Step D builds a single FloatSim Morison element
with the validated aggregate (`R_eff = 3.407e6 kg/m`) and asserts:

1. FloatSim's hyperbolic δ agrees with OpenFAST's within rtol = 5e-2.
2. FloatSim's envelope is hyperbolic, not exponential (M5-style
   discriminator).
3. OpenFAST's reference itself passes the regime classification
   (hyperbolic over peaks 0-15 within 5 % rel-err; exponential
   shows > 50 % deviation by peak 10).

Per Decision B discipline: any failure in Step D pauses for
diagnosis, not silent xfail.
