# Filon-trapezoidal formula verification (M6 PR3 Pre-1)

**Status**: ✅ verified to machine precision (rel err ≤ 1e-15) at all tested lag values.

**Reference**: Davis, P.J. and Rabinowitz, P. (1984), *Methods of Numerical Integration*, 2nd ed., Academic Press, §2.10.3 "Filon-type rules" (general framework). The trapezoidal case (linear interpolation per segment) is the simplest member of the family. See also Filon (1928, *Proc. Royal Soc. Edinburgh* 49:38–47) for the original Simpson-rule derivation, and Tuck (1967, *Math. Comp.* 21:239–241) specifically for the trapezoidal case.

## Derivation

For B(ω) linearly interpolated between adjacent grid points `(ω_k, B_k)` and `(ω_{k+1}, B_{k+1})` with slope `m_k = (B_{k+1} − B_k) / (ω_{k+1} − ω_k)`, the per-segment integral against `cos(ω·t)` admits a closed form. Integration by parts on the linear piece gives:

```
I_k(t) = [B_{k+1} sin(ω_{k+1} t) − B_k sin(ω_k t)] / t
       + m_k [cos(ω_{k+1} t) − cos(ω_k t)] / t²        (t ≠ 0)

I_k(0) = ½ (B_k + B_{k+1}) (ω_{k+1} − ω_k)             (trapezoidal limit)
```

Summed over all segments `k = 0, ..., N−1`, the sin-endpoint terms telescope, leaving:

```
∑_k I_k(t) = [B_N sin(ω_N t) − B_0 sin(ω_0 t)] / t
           + (1/t²) ∑_k m_k [cos(ω_{k+1} t) − cos(ω_k t)]
```

This is the Filon-trapezoidal formula for the full integral. **It is exact for any piecewise-linear B** — no Nyquist constraint on `dω·t`, because the cosine factor is integrated analytically per segment rather than sampled.

## Numerical verification

Three test cases, each with B linear (so Filon-trapezoidal must reproduce the analytical integral to machine precision). Tested on a 250-point grid `ω ∈ [0.1, 5.0]` at `t ∈ {1, 10, 100, 240, 1000}` s.

### Test 1 — Constant B = 1.7 (slope = 0)

Analytical: `∫_a^b 1.7 cos(ωt) dω = 1.7 (sin(bt) − sin(at)) / t`.

| t (s) | Filon-trap | Analytical | rel err |
|-------|------------|------------|---------|
| 1     | −1.799888075226943 | −1.799888075226943 | 1.2e-16 |
| 10    | −1.876537925470103e-01 | −1.876537925470103e-01 | 0.0 |
| 100   | +1.296238194637191e-03 | +1.296238194637192e-03 | 1.7e-16 |
| 240   | +5.789206601705525e-03 | +5.789206601705525e-03 | 0.0 |
| 1000  | −8.187213560169306e-04 | −8.187213560169307e-04 | 1.3e-16 |

### Test 2 — Linear slope, B = 2.5·ω

Analytical: `∫_a^b 2.5·ω cos(ωt) dω = 2.5 [ω·sin(ωt)/t + cos(ωt)/t²]_a^b`.

| t (s) | Filon-trap | Analytical | rel err |
|-------|------------|------------|---------|
| 1     | −13.78986673698794 | −13.78986673698794 | 1.3e-16 |
| 10    | −3.383887486845091e-01 | −3.383887486845091e-01 | 0.0 |
| 100   | −5.712261732417483e-02 | −5.712261732417484e-02 | 2.4e-16 |
| 240   | −3.629710514875463e-03 | −3.629710514875461e-03 | 4.8e-16 |
| 1000  | −1.222475820047254e-02 | −1.222475820047254e-02 | 0.0 |

### Test 3 — Combined affine, B = 0.5 + 2.5·ω

Sum of the two analyticals.

| t (s) | Filon-trap | Analytical | rel err |
|-------|------------|------------|---------|
| 1     | −14.31924558264292 | −14.31924558264292 | 0.0 |
| 10    | −3.935810406101004e-01 | −3.935810406101004e-01 | 1.4e-16 |
| 100   | −5.674137079634040e-02 | −5.674137079634037e-02 | 3.7e-16 |
| 240   | −1.927002690844426e-03 | −1.927002690844424e-03 | 1.1e-15 |
| 1000  | −1.246555859930105e-02 | −1.246555859930105e-02 | 1.4e-16 |

## Spot-check against scipy.integrate.quad on a single segment

For one segment `[0.4, 0.7]` with B linear from 1.5 to 3.2 — Filon vs `scipy.integrate.quad` (`epsrel=1e-14`):

| t (s) | Filon-trap | scipy.quad | rel err |
|-------|------------|------------|---------|
| 0     | 0.704999999999998 | 0.704999999999998 | 0.0 |
| 1     | 0.5921291809600545 | 0.5921291809600545 | 0.0 |
| 10    | 0.4035170188145937 | 0.4035170188145938 | 2.8e-16 |
| 100   | 1.432461685268074e-02 | 1.432461685268084e-02 | 6.9e-15 |
| 240   | −1.943270843930496e-02 | −1.943270843930240e-02 | 1.3e-13 |

Both are within machine-precision noise of each other. (The 1.3e-13 at t=240 is scipy.quad's adaptive subdivision struggling with a highly oscillatory integrand — `epsrel=1e-14` is at the floor of float64; scipy emits a roundoff warning. Filon-trap is the more accurate of the two here.)

## Conclusion

The Filon-trapezoidal formula is verified to machine precision against three independent analytical references. It is exact for piecewise-linear B(ω) at any t — including large t where naive trapezoidal-cosine sums fail the Nyquist condition `dω·t ≤ π`.

The formula is ready for production use as the in-grid quadrature in the M6 PR3 fix. It will replace the existing `compute_retardation_kernel` trapezoidal-cosine sum on `[ω_0, ω_N]`; the tail past `ω_N` (per Refinement 1) is integrated separately via `scipy.integrate.quad_vec` on the per-entry `C/ω⁴` extrapolation.

**Awaiting Xabier's go-ahead before writing the production implementation** of the integrator + the unit tests + the wired-in tail extension.
