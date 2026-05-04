# M6 PR3 Audit — Radiation kernel bug (Phase 1 findings)

**Status**: Phase 1 audit complete. Awaiting Xabier's fix-scope decision (A / B / C) before any fix code lands.

**Branch**: `fix-radiation-kernel` off `main`.

**Surfaced by**: M6 PR3 pre-flight diagnostic. OC4 pitch free decay produced ζ=0.014 at FloatSim's M2 default `t_max=60s`, but this turns out to be a transient artifact — at converged `t_max≥200s`, the same kernel produces ζ≈0, with a sign-flip excursion to ζ=-0.006 at intermediate `t_max=120s`.

---

## 1 -- Synthetic-input verification (audit step 1.a)

Three analytical-kernel test cases, run on a 60-sample uniform grid `ω ∈ [0.05, 3.0]` (much finer than platform_small.yml's 10-point grid):

- **Case A** — `B(ω) = exp(-ω/τ)` with τ=2. Analytical: `K(t) = (2/π) · a / (a² + t²)`, a = 1/τ. Smooth Lorentzian decay, ~1/t².
- **Case B** — `B(ω) = ω·exp(-ω/τ)` with τ=2. Analytical: `K(t) = (2/π) · (a²-t²)/(a²+t²)²`. Crosses zero at t=a, then negative, decays.
- **Case C** — `B(ω) = 1` over the band, zero outside. Analytical: `K(t) = (2/(πt))·(sin(ω_max·t) - sin(ω_min·t))`. 1/t decay with sinusoidal modulation.

### Quantitative results (relative error vs analytical)

| Case | t=0    | t=1    | t=5    | t=30   | t=60   | t=120  | t=240  |
|------|--------|--------|--------|--------|--------|--------|--------|
| A    | 23.6%  | 22.2%  | 35.6%  | **35×** | **180×** | **1316×** | **1136×** |
| B    | 55.8%  | 53.0%  | 233%   | 14×    | 4.5×   | 2050×  | 1389×  |
| C    | 0.85%  | 27.4%  | 29.5%  | 71%    | 68%    | 1972%  | 696×   |

**Findings**:

- The ~25% short-lag error for cases A and B is **truncation of the high-frequency tail** of the analytic integrand: B(ω=3) is still ~22% of B(0) for Case A, contributing a missed integral that explains the K(0) deficit.
- Errors **grow without bound** at long lag — by t=120s, all three cases have errors of >1000× the analytical magnitude. This is NOT physical: K_floatsim is producing oscillatory artifacts at amplitude similar to K_max-near-t=0, while K_analytical decays toward zero.
- **Case C is the most diagnostic**: K(0) error is only 0.85% (constant B integrates exactly across the grid), but long-lag error explodes — confirming the long-lag pathology is not from short-lag truncation but from the **discrete cosine sum sustaining oscillations indefinitely**.

### Bug confirmed in the cosine-transform implementation, not in BEM-data preprocessing

Even with smooth analytic B(ω) on a clean uniform grid, the FloatSim kernel exhibits long-lag artifacts >1000× the true kernel magnitude. Marin_semi BEM data with its peaks and edge structure makes this worse but is not the root cause.

---

## 2 -- B(ω) inspection (audit step 1.b)

Comparison of two BEM data sources for the same OC4 platform:

| Source | n_samples | ω range (rad/s) | B_55 peak | B_55(ω_max) | (B at ω_max) / peak |
|--------|-----------|------------------|-----------|--------------|---------------------|
| `platform_small.yml` (M2 fixture) | 10 | 0.1 -- 1.0 | 7.26e8 @ ω=0.8 | 3.59e8 | **49.5%** |
| `marin_semi.1` (M5 PR1 reference) | 498 | 0.01 -- 4.98 | 9.47e5 @ ω=0.74 | 7.45e1 | **8e-5** |

(The 3-order-of-magnitude absolute difference between the two is OrcaFlex's tonnes-vs-kg unit scaling; relative shapes are what matter for kernel computation.)

`platform_small.yml`'s grid is **massively under-resolved at the high frequency end** — B(ω_max) is still 50% of peak, meaning the cosine-transform integration cuts off mid-curve. `marin_semi.1` is properly resolved (B(ω_max) is 8e-5 of peak — converged).

**Yet the synthetic-input results above demonstrate the bug surfaces even on a 60-point grid where the integrand is smooth.** Resolution alone does not fix it; the implementation is missing a grid-edge handling step.

---

## 3 -- Kramers-Kronig consistency check (audit step 1.d)

For the marin_semi data:

| DOF | A_inf | A(ω_max=4.98) | (A_inf - A(ω_max)) / A_inf |
|-----|-------|---------------|----------------------------|
| 22 (sway) | 1.45e7 | 1.47e7 | -1.6% |
| 33 (heave) | 7.27e9 | 7.31e9 | -0.6% |
| 44 (roll) | 7.27e9 | 7.31e9 | -0.6% |

A(ω_max) is within 1.6% of A_inf for all DOFs — Kramers-Kronig consistency is GOOD on marin_semi. The high-frequency limit in the BEM data IS reaching a steady value that matches the reported A_inf.

**K-K inconsistency is NOT the root cause.**

---

## 4 -- Implementation inspection (audit step 1.e + code review)

`floatsim/hydro/retardation.py:compute_retardation_kernel`:

```python
omega = hdb.omega           # B's frequency grid, e.g. [0.05, 0.10, ..., 3.0]
b_stack = hdb.B             # (6, 6, n_omega)
if omega[0] > _FLOAT_EPS:
    omega = np.concatenate([[0.0], omega])
    b_stack = np.concatenate([np.zeros((6,6,1)), b_stack], axis=2)
weights = _trapezoidal_weights(omega)
cos_mat = np.cos(np.outer(omega, t_arr))
weighted_cos = weights[:, None] * cos_mat
K = (2.0 / np.pi) * np.einsum("ijk,kn->ijn", b_stack, weighted_cos)
```

What the implementation does:
- ✅ Prepends `(ω=0, B=0)` if the grid doesn't start at zero (correct: B is exactly zero at ω=0 in linear potential flow).
- ✅ Uses non-uniform trapezoidal weights for the integral over ω.
- ✅ Computes `K(t) = (2/π) · Σ_k w_k · B(ω_k) · cos(ω_k · t)`.
- ❌ **Does NOTHING at the high-frequency endpoint.** The integral effectively cuts off at `ω_max`, treating `B(ω) = 0` for `ω > ω_max`. For B that doesn't smoothly decay to zero at the grid edge, this discontinuity drives Gibbs-type ringing in the cosine transform.
- ❌ **No windowing function** to taper B(ω) smoothly toward zero at the grid edges.
- ❌ **No tail extrapolation** beyond ω_max (e.g. continuing the local slope, or extending with a 1/ω⁴ tail per the long-wavelength radiation theory).

The decay diagnostic warning fires post-hoc but is passive — it tells the caller something is wrong without correcting the kernel.

### Why the artifacts grow with t_max

At small t, the trapezoidal cosine sum approximates the true integral well — the cosines vary slowly and don't probe the grid-edge truncation. At large t, the cosines oscillate rapidly and **the discrete sum is dominated by the highest-frequency term that's still significantly non-zero**, which is exactly the grid-edge value. This produces a sustained oscillation at frequency ω_max with amplitude proportional to `B(ω_max) · weight[-1]` — never decaying because the discrete cosine doesn't decay.

For marin_semi the artifact is small because B(ω_max) ≈ 0. For platform_small.yml the artifact is huge because B(ω_max) ≈ 50% of peak. For the synthetics A and B, B(ω_max) is ~22% of B(0), giving large-amplitude artifacts.

---

## 5 -- Why M2 didn't catch this

Two M2 validation tests exercise the kernel-convolution path:

1. **`test_oc4_heave_free_decay.py`** — uses `platform_small.yml`, `t_max=60s`. Asserts only **period** (rtol=3e-2), not damping. Period is determined by C and M+A_inf and is **independent of kernel artifacts** (the kernel only contributes the damping term). The test passes because period agreement is a separate validation channel.

2. **`test_cummins_free_decay_analytical.py`** — uses a **synthetic constant-B(ω) fixture** over a fine grid (`omega = [0.05, ..., 3.0] step 0.05`). Asserts damping with rtol=5e-2 (loose tolerance, justified in the test docstring as accommodating the explicit-mu integration error). This test is closest to a damping cross-check, but constant-B has special properties:
   - The local slope at the grid edges is zero — minimising Gibbs amplitude.
   - The kernel artifacts integrate roughly cancellingly against `ξ̇` over many cycles.
   - The 5% tolerance absorbs the residual error.

Pattern: **bug invisible through synthetic constant-B tests, surfaces against real BEM data with grid-edge structure in cross-check**. Same shape as the hydrostatic-gravity bug from M5.

---

## 6 -- Proposed fix scope

Based on the diagnosis: **Fix A (grid extension / windowing)**.

### Recommended implementation (~80-150 lines)

Two complementary preprocessing steps before the trapezoidal cosine integration:

1. **High-frequency tail extrapolation.** Extend `B(ω)` past `ω_max` with a smooth analytic tail. Two options:
   - **1/ω⁴ tail** per long-wavelength radiation theory: `B(ω) ≈ B(ω_max) · (ω_max/ω)⁴` for ω > ω_max. Physically motivated (radiation damping at high frequency falls off as a fourth power for surface-piercing bodies). Integrate analytically out to ω→∞.
   - **Cosine taper (Hann window) at the grid edges**: multiply `B(ω)` by a window that smoothly tapers to zero at ω=0 and ω=ω_max. Suppresses Gibbs but distorts the integrand near the edges (~5% systematic error in K(0) for moderately-edged B).

2. **Refine the trapezoidal sum to a Filon-type quadrature** for `cos(ωt)` with `t` large. The standard trapezoidal rule isn't designed for highly oscillatory integrands; Filon's method weights the integrand against `cos(ωt)` exactly over each segment and is well-conditioned at large t.

I recommend **option 1 first** — extend with a 1/ω⁴ tail. It's the smaller change and addresses the root cause (truncation discontinuity) directly. Filon quadrature is a follow-up if a residual artifact remains.

### Fix B not applicable

Kramers-Kronig consistency check (step 3) showed marin_semi's A_inf vs A(ω_max) agrees to within 1.6% — K-K is not the issue.

### Fix C deferred

State-space approximation of K(t) is a Phase 2 deliverable that would replace the convolution entirely. This is overkill for the current bug and is a much larger change.

---

## 7 -- What Phase 2 (the fix) needs

When Xabier scopes the fix, the test list to pin the bug:

1. **Synthetic kernel regression test** — for each of cases A, B, C above, assert `|K_floatsim(t) - K_analytical(t)| < rtol_per_t` at sample lags. Different rtol per regime: tight (1%) at short lag, looser (5%) at long lag. This is the failing-test-first.

2. **Damping non-positivity test** — for any kernel produced by `compute_retardation_kernel`, simulate a free decay and assert the log-decrement damping is non-negative (within some integrator-noise floor). This is the smoking-gun test from the OC4 pitch case.

3. **t_max convergence test** — for marin_semi, assert that damping stops changing as t_max grows past some convergence threshold (e.g. ζ at t_max=200s and t_max=400s agree to 2%). This pins the kernel-artifact issue at the boundary between "converged enough" and "truncated".

4. **M2 free-decay regression** — confirm period-fitting on `test_oc4_heave_free_decay` is unchanged after the fix (any fix that breaks this is suspect).

These are the regression suite for Phase 3 (implement the fix).

---

## 8 -- Recommendation

**Proceed with Fix A — high-frequency tail extension via 1/ω⁴ extrapolation.** Estimated scope: ~80 lines of new code in `compute_retardation_kernel`, ~150 lines of regression tests across the four categories above. Plus the post-mortem doc per Xabier's Phase 4 spec.

**Awaiting Xabier's confirmation of the fix path** before writing any fix code.
