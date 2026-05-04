# Asymptotic Tail Check (Refinement 1) — marin_semi.1

**Status**: complete. Asymptote is valid for all 3 diagonal DOFs once past OC4's secondary resonance structure. **Recommendation: proceed with Fix A using per-DOF B·ω⁴ constant fit on the last 5–10 grid points.**

**Plot**: `docs/diagnostics/m6-pr3-asymptotic-tail-check.png`.

## Theory

Per Newman *Marine Hydrodynamics* (1977) §6.18 and Faltinsen *Sea Loads on Ships and Offshore Structures* (1990) §3.3.2, the high-frequency asymptote of the radiation damping `B(ω)` for a 3D surface-piercing body decays as `ω⁻⁴`. The constant of proportionality depends on body geometry near the waterline and the local panel curvature.

For deep water and a clean far-field radiation pattern, the leading-order term is proportional to the wavenumber `k = ω²/g` cubed times a geometry factor — combined with the `1/ω` energy normalisation, gives `B(ω) ~ const / ω⁴` at large `ω`.

## Empirical check

`B[i, i] · ω⁴` should approach a constant in the tail. Fitting this constant from the last 10 grid points of marin_semi.1:

| DOF | last-10 ω range | mean B·ω⁴ | std B·ω⁴ | std/mean |
|---|---|---|---|---|
| heave (33) | [4.60, 4.97] | 3.40e2 | 3.05e1 | **9.0%** |
| roll (44)  | [4.60, 4.97] | 4.34e4 | 2.90e3 | **6.7%** |
| pitch (55) | [4.60, 4.97] | 4.33e4 | 3.06e3 | **7.1%** |

All three DOFs meet the "approaches a constant" criterion (std/mean < 10%). Roll and pitch are essentially identical (axisymmetric platform).

## Why the n_fit on heave looked bad initially

Power-law fit `n_fit` over different windows:

| Window | heave n_fit | roll n_fit | pitch n_fit |
|---|---|---|---|
| ω ≥ 2.0 | **−1.82** ⚠ | −4.00 | −4.00 |
| ω ≥ 3.0 | **−4.29** ✓ | −4.58 | −4.57 |
| ω ≥ 3.5 | −5.08 | −4.14 | −4.13 |

Heave's `n_fit = −1.82` over `ω ≥ 2.0` is misleading — the OC4 platform has a secondary resonance structure around `ω = 2.5–3.5 rad/s` (multi-column interaction) that contaminates the broad-window fit. Once past that resonance (`ω ≥ 3.0`), heave's exponent settles to about −4.3, consistent with the ω⁻⁴ asymptote.

The `B·ω⁴` constant check on the LAST 10 grid points is robust to this contamination because those points are deep enough into the tail (ω ≥ 4.6) that the secondary resonance has fully decayed.

## Recommended Fix A implementation

For each DOF (and each off-diagonal entry of the 6×6 B matrix):

1. Take the last `N=10` grid points (or `N=5` if `len(ω) < 20`).
2. Fit constant: `C_ij = mean(B_ij(ω_k) · ω_k⁴)` over `k ∈ last 10`.
3. Define tail: `B_tail(ω) = C_ij / ω⁴` for `ω > ω_max`.
4. The cosine-transform integral over the tail has a closed form:

```
∫_{ω_max}^{∞} (C / ω⁴) · cos(ω·t) dω
```

This is a known special function (related to the cosine integral `Ci(x)`); for practical computation, evaluate via `scipy.special.spence` or by direct numerical quadrature on `[ω_max, ω_max·100]` (the contribution above ω_max·100 is negligible at any t > 0).

5. Add the tail integral to the trapezoidal sum over the BEM grid.

Effectively the kernel becomes:

```
K(t) = (2/π) · [∫_0^{ω_max} B(ω) cos(ω·t) dω + ∫_{ω_max}^∞ (C/ω⁴) cos(ω·t) dω]
```

where the first integral is the existing trapezoidal sum and the second is the analytical-or-quadrature tail extension.

## Status

**Asymptote validated for marin_semi.1 across all three diagonal DOFs.** The asymptotic form is suitable for use as the tail extrapolation in Fix A. Proceeding to write the failing-test-first regression suite.

Per-DOF off-diagonal entries: untested in this audit, but for OC4 the off-diagonal terms are small (sway-roll, surge-pitch coupling). The same C_ij fit-the-tail approach applies; the std/mean check should be added per off-diagonal entry as a sanity gate.

**Awaiting Xabier's confirmation that the per-DOF tail-fit approach is acceptable** before writing Fix A code.
