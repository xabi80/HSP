"""Pitch/surge radiation-damping gate — the DOF the trapezoid-convolution fix protects.

Heave-based validation is BLIND to the radiation-convolution rule: `K(0)[heave]` is
tiny, so the rectangular-rule excess `dt*K(0)/2` is negligible there (~0.01x B). The
excess bites DOFs with a broadband `B(w)` — pitch, surge, roll — where `K(0) =
(2/pi) INT B dw` is large. A defect that inflated pitch radiation damping ~7x survived
the full suite and was found three codebases downstream precisely because nothing
exercised the affected DOF (FloatFEA note AG1). This is that missing exercise.

Construction: a broadband `B(w)` bump at high frequency gives a large `K(0)`, and a
mode placed BELOW the bump sees `B(w_n) ~ 0` — so its physical radiation damping is
~0. The trapezoidal convolution must leave such a mode **~undamped**; the (fixed)
rectangular rule injects a spurious, near-frequency-independent `dt*K(0)/2` damping.
"Undamped must stay undamped" is an unambiguous physical expectation, and the
rect-vs-trap contrast is the DOF-selective fingerprint (AG5): huge on this large-`K(0)`
DOF, ~1 on a small-`K(0)` (heave-like) DOF.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from floatsim.hydro.radiation import CumminsLHS
from floatsim.hydro.retardation import RadiationConvolution, RetardationKernel
from floatsim.solver.newmark import integrate_cummins

_DT = 0.01
_DOF = 4  # pitch (the fingerprint holds identically for surge/roll)


def _rectangular_evaluate(self):  # type: ignore[no-untyped-def]
    """The OLD left-rectangle rule (full dt on every lag), for the contrast only.
    `evaluate` itself is now trapezoidal; this reintroduces the defect in-test so the
    gate proves both that the current rule is right AND that the wrong rule is caught."""
    return self._dt * np.einsum("ijk,kj->i", self._K, self._buffer)


def _broadband_kernel(k0_dof: int, b_peak: float) -> RetardationKernel:
    """6-DOF diagonal kernel whose DOF `k0_dof` carries a broadband B(w) bump
    (Gaussian at w=12, sigma=2) -> large K(0); all other DOFs zero."""
    w = np.linspace(0.0, 40.0, 4000)
    bw = b_peak * np.exp(-((w - 12.0) / 2.0) ** 2)
    t = np.arange(0.0, 12.0, _DT)
    k = np.array([(2.0 / np.pi) * np.trapezoid(bw * np.cos(w * ti), w) for ti in t])
    big_k = np.zeros((6, 6, t.size), dtype=np.float64)
    big_k[k0_dof, k0_dof, :] = k
    return RetardationKernel(K=big_k, t=t, dt=_DT)


def _decay_zeta(kernel: RetardationKernel, m_ainf: float, wn: float, patched) -> float:
    """Radiation-only free-decay damping ratio of a single-DOF mode at ~`wn`."""
    m = np.eye(6)
    m[_DOF, _DOF] = m_ainf
    c = np.zeros((6, 6))
    c[_DOF, _DOF] = wn**2 * m_ainf
    lhs = CumminsLHS(M_plus_Ainf=m, C=c)
    original = RadiationConvolution.evaluate
    RadiationConvolution.evaluate = patched if patched else original
    try:
        xi0 = np.zeros(6)
        xi0[_DOF] = 0.05
        r = integrate_cummins(
            lhs=lhs, kernel=kernel, xi0=xi0, xi_dot0=np.zeros(6),
            duration=60.0, dt=_DT, rho_inf=1.0,  # rho_inf=1 -> no numerical damping
        )
        x = r.xi[:, _DOF]
        pk, _ = find_peaks(x, height=1e-6)
        d = np.log(x[pk][:-1] / x[pk][1:])
        d = d[np.isfinite(d) & (d > 0)]
        return float(np.mean(d[:6]) / (2 * np.pi))
    finally:
        RadiationConvolution.evaluate = original


def test_pitch_radiation_damping_not_inflated_and_defect_is_caught() -> None:
    """The trapezoidal convolution leaves a B(w_n)~0 pitch mode ~undamped; the
    rectangular defect would inflate it several-fold. Guards the DOF heave cannot see."""
    kernel = _broadband_kernel(_DOF, b_peak=50.0)  # K(0) ~ 113 (pitch-scale)
    zeta_trap = _decay_zeta(kernel, m_ainf=10.0, wn=3.0, patched=None)
    zeta_rect = _decay_zeta(kernel, m_ainf=10.0, wn=3.0, patched=_rectangular_evaluate)

    # Physical: B(w_n) ~ 0 -> radiation damping ~ 0. Trapezoid respects it.
    assert zeta_trap < 2.0e-3, f"trapezoid over-damps a B~0 mode: zeta={zeta_trap:.2e}"
    # The rectangular defect injects spurious dt*K(0)/2 damping.
    assert zeta_rect > 4.0e-3, f"rectangular rule should inflate: zeta={zeta_rect:.2e}"
    # ...and the inflation is large on this broadband-B (large K(0)) DOF.
    assert zeta_rect / zeta_trap > 4.0, f"ratio {zeta_rect / zeta_trap:.1f} too small"


def test_convolution_defect_is_dof_selective_tracking_k0() -> None:
    """The fingerprint (AG5): the rect-vs-trap inflation tracks K(0). On a SMALL-K(0)
    (heave-like) mode the two rules agree; only the large-K(0) DOF collapses."""
    small = _broadband_kernel(_DOF, b_peak=0.4)  # K(0) ~ 0.9 (heave-scale)
    zeta_trap = _decay_zeta(small, m_ainf=10.0, wn=3.0, patched=None)
    zeta_rect = _decay_zeta(small, m_ainf=10.0, wn=3.0, patched=_rectangular_evaluate)
    # small K(0) -> the endpoint correction is negligible -> both rules ~agree
    assert abs(zeta_rect - zeta_trap) < 2.0e-4, (zeta_trap, zeta_rect)
