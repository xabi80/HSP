# Post-mortem — radiation retardation kernel: truncation discontinuity + Nyquist aliasing

**Discovered:** 2026-05-04 during M6 PR3 pre-flight diagnostic on the
OC4 OpenFAST damping cross-check.
**Fixed in:** branch `fix-radiation-kernel` (Filon-trapezoidal in-grid
quadrature + 1/ω⁴ high-frequency tail extension + Refinement-2 input
gates), merged to `main`.
**Severity:** Phase 1 latent bug. Visibly wrong on free-decay damping at
non-trivial `t_max`; silently wrong on period (period is dominated by
``M+A_inf+C`` and barely sees the kernel). All M2-M5 milestone gates
passed because none isolated the kernel's long-lag fidelity.

## TL;DR

`compute_retardation_kernel` evaluated
``K(t) = (2/π) ∫_0^∞ B(ω) cos(ωt) dω`` as a discrete trapezoidal
sum on the BEM grid. Two distinct, compounding pathologies:

1. **Truncation discontinuity at ω_max.** The integral was implicitly
   cut off at the highest BEM grid sample. If `B(ω_max)` was not
   already at the noise floor (the OrcaFlex `platform_small.yml`
   fixture had `B(ω_max)/peak ≈ 50 %`), the missing tail produced a
   frequency-domain step discontinuity. Its time-domain image is
   sustained Gibbs ringing at `ω_max`.
2. **Nyquist aliasing in the discrete cosine sum.** Even on a grid
   where the truncation is benign, the *discrete* trapezoidal sum
   ``Σ_k w_k B(ω_k) cos(ω_k t) dω`` requires `dω·t < π` to resolve
   the cosine on the grid. For `dω = 0.05 rad/s` (typical BEM
   resolution), Nyquist hits at `t = 63 s`. Beyond that the discrete
   sum produces sustained oscillations at amplitude `~ K_max`,
   regardless of the analytical `K(t)`'s decay.

The audit case that caught it: free-decay damping ratio versus `t_max`,
on otherwise well-resolved marin_semi.1 BEM data (M5 reference
fixture). At `t_max = 60 s` the integrator extracted ζ ≈ 0.014. At
`t_max = 120 s` it flipped to ζ ≈ -0.006 (negative damping is
physically impossible — radiation is dissipative). At `t_max ≥ 200 s`
ζ → 0. The **converged** answer was wrong (zero damping where
OpenFAST reports finite), and the path to the converged answer was
non-monotone. Both were symptoms of long-lag artifacts in `K(t)`.

## How the bug stayed invisible through M2-M5

No existing test exercised the combination
**(realistic-resolution BEM grid) ∧ (converged `t_max`) ∧ (damping
extracted from a free-decay envelope)**:

- **M2** `test_oc4_heave_free_decay` — read the OrcaFlex `platform_small.yml`
  fixture (10-point grid, `ω` up to 1.0 rad/s) and asserted **period
  only**, not damping. The docstring noted "we don't assert damping
  because the YAML's `B_33(ω)` has a near-zero dip at `ω = 0.4`
  rad/s." That dodged the issue. Period is governed by `M+A_inf+C`;
  the kernel has only a sub-percent secondary effect.
- **M2** `test_cummins_free_decay_analytical` — synthetic constant
  `B(ω) = B_0` on a 60-point grid, ran `t_max = 200 s`. Damping ratio
  was checked at `rtol = 5e-2`. The constant-B case is the *worst*
  for the truncation discontinuity (B at ω_max is 100 % of peak), and
  yet the test passed because at `dt = 0.01 s` and `dω = 0.05 rad/s`
  the Nyquist horizon is `t = π/dω = 63 s` — so by `t_max = 200 s`
  the kernel was already in the artifact regime, but the artifact
  amplitude was small enough versus the dominant `B_0` plateau that
  the log-decrement fit happened to land within 5 % of analytical.
- **M3, M4, M5** — used either the same `platform_small.yml` fixture
  or synthetic constant-B/gaussian-B fixtures. All asserted period,
  amplitude envelope, or steady-state response — never damping
  extracted from a long-time free-decay envelope on a realistic-resolution
  grid.
- **M5 PR1** introduced the `marin_semi.1` fixture (498-point grid,
  `ω` up to 4.98 rad/s with B at the noise floor). Its tests
  exercised reader unit-tests only — no kernel computation through
  free-decay damping.

The bug surfaced in **M6 PR3** because that PR was the first to
combine: marin_semi.1 → kernel → free-decay → damping ratio — exactly
the combination that none of M2-M5 covered.

## Root cause and fix

The discrete trapezoidal cosine sum is the wrong tool. Replaced by:

1. **Filon-trapezoidal quadrature** (Davis & Rabinowitz 1984 §2.10.3,
   Tuck 1967, *Math. Comp.* 21:239) on the BEM grid `[ω_0, ω_N]`. For
   piecewise-linear `B(ω)` the per-segment integral
   ``∫_{ω_k}^{ω_{k+1}} (B_k + (ω−ω_k)·m_k) cos(ωt) dω`` admits a
   closed form. Summed across segments the sin-endpoints telescope to
   leave a single-cosine-difference sum. This is **exact** for
   piecewise-linear B at any t — no Nyquist constraint on `dω·t`,
   because the cosine factor is integrated analytically per segment
   rather than sampled. Verified to machine precision in
   `docs/diagnostics/m6-pr3-filon-formula-check.md`.
2. **High-frequency tail extension** on `[ω_N, 5·ω_N]`. Per Newman
   (1977) §6.18 / Faltinsen (1990) §3.3.2, 3D surface-piercing bodies
   in deep water have `B(ω) ~ C/ω⁴` at high frequency. The constant
   `C_ij` is fitted per entry as `mean(B_ij(ω_k)·ω_k⁴)` over the last
   10 grid samples, and the tail is integrated via
   `scipy.integrate.quad_vec` per `(i, j)` entry. Verified valid for
   marin_semi.1 in
   `docs/audits/m6-pr3-asymptotic-tail-check.md`.
3. **Refinement-2 input gates** — hard-fail when the BEM grid does
   not reach the asymptotic regime, instead of silently producing a
   garbage kernel:
   - **Check 1** (amplitude): for any diagonal entry,
     `|B_ii(ω_max)| < 0.01·max|B_ii|` must hold. Diagonals dominate
     the kernel and their high-frequency truncation cannot be
     papered over.
   - **Check 2** (asymptote consistency): for any diagonal entry,
     `std(B_ii·ω⁴) / mean(B_ii·ω⁴) < 0.10` over the last 10 grid
     samples. The `C/ω⁴` fit is meaningful only in the asymptotic
     regime. Off-diagonal failures fall back to zeroing the tail
     contribution rather than erroring (a calibrated deviation from
     the locked review spec — marin_semi.1's surge-pitch and
     sway-roll couplings reach the BEM solver's noise floor at the
     highest frequencies, where the asymptote check legitimately
     fails because `B*ω⁴` is dominated by random noise; the tail
     contribution from such entries is a small fraction of the
     diagonal-driven kernel anyway).

## Why Fix A (in-grid Filon only) was insufficient

The first scoping call ("Fix A") was Filon-trapezoidal on `[ω_0, ω_N]`
without the high-frequency tail extension. Pre-implementation
diagnostic on the audit case showed the fix recovered the correct
short-lag behaviour but still produced wrong long-lag behaviour: with
the Filon formula, the integral over `[ω_0, ω_N]` is exact, but
`B(ω_N) ≠ 0` still drives a sustained `(B(ω_N)/t)` oscillation
because the analytical `K(t)` includes the tail integral
`∫_{ω_N}^∞ B(ω)cos(ωt)dω` which *Fix A omits*. On marin_semi.1
this was a small effect (`B(ω_N)` at noise floor); on under-resolved
grids like `platform_small.yml` it was large. Adding the per-entry
`C/ω⁴` tail (Refinement 1) closed the gap.

## Pattern lock — what to never do again

1. **Never assume a discrete sum approximates a continuous integral
   for arbitrary integrand and parameters.** A trapezoidal sum of
   `f(ω)cos(ωt)` is an *interpolatory* quadrature — it integrates the
   piecewise-linear interpolant of `f(ω)cos(ωt)`. At large `t` the
   cosine is *not* piecewise-linear on a grid where `dω·t > π/2`. The
   resulting integral is dominated by aliasing of the cosine onto the
   grid, not by the true integrand. **For oscillatory integrals at
   variable frequency, use Filon-type rules** that integrate the
   oscillation analytically per segment.
2. **Never assume the BEM grid extends to the asymptotic regime.**
   Add an explicit gate. Hard-fail when the assumption breaks — a
   silently-wrong kernel is worse than a noisy one. `platform_small.yml`
   was the primary M2 fixture for two milestones; nothing in the code
   path warned that its `B(ω_max)/peak ≈ 50 %` invalidated the kernel
   integration.
3. **Whenever a kernel computation produces a `t_max`-dependent
   answer, the kernel is wrong, not the integration.** A pure
   function of `B(ω)` cannot have its value at fixed `t` change with
   the truncation `t_max`. The
   `test_synthetic_kernel_value_is_independent_of_t_max` test was
   added during the fix to make this assertion explicit on synthetic
   inputs with known closed-form `K(t)`.
4. **Damping ratios extracted from free-decay envelopes are the
   strictest validation lever for kernel correctness.** Period is
   forgiving (dominated by `M+A_inf+C`); damping is unforgiving
   (entirely from the kernel). Any future kernel-touching change
   needs to ship a damping-from-free-decay test on a well-resolved
   real BEM fixture.

## Linked artefacts

- `docs/audits/m6-pr3-radiation-kernel-bug.md` — Phase 1 audit.
- `docs/diagnostics/m6-pr3-filon-formula-check.md` —
  machine-precision verification of the Filon-trapezoidal closed form
  against three analytical references and `scipy.integrate.quad`.
- `docs/audits/m6-pr3-asymptotic-tail-check.md` — verification of the
  `B(ω) ~ C/ω⁴` asymptote on marin_semi.1.
- `floatsim/hydro/_filon.py` — Filon-trapezoidal integrator and
  per-entry tail-constant fit.
- `floatsim/hydro/retardation.py` — wired-up `compute_retardation_kernel`
  with the input gates.
- `tests/unit/test_filon_quadrature.py` — Filon utility tests.
- `tests/unit/test_retardation_kernel_extension.py` —
  kernel-correctness regression suite (synthetic Lorentzian, peaked,
  Hann-windowed-box, marin_semi convergence-rate, OC4 period
  unchanged, t_max-independence).
