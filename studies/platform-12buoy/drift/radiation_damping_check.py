"""Resolve the radiation-brake magnitude: does kappa = sum_surge integral K ds
equal the zero-frequency radiation damping B(0)? Compare against the BEM B at
its lowest omega. Decides whether radiation (~+0.014 N needed) is the closer or
my kernel contraction is buggy."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

_STUDY = Path(__file__).resolve().parent.parent  # studies/platform-12buoy (this file: drift/)
sys.path.insert(0, str(_STUDY / "fin_study"))
sys.path.insert(0, str(_STUDY))
sys.path.insert(0, str(_STUDY.parent / "cluster-3buoy-rigid"))

import platform_fin_fan as pff  # noqa: E402
import platform_rao_pilot as prp  # noqa: E402

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
_SIDX = [6 * b for b in range(17)]


def main() -> None:
    hdb = pff._hdb("0215")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        setup = pff._build(0.215, 5.0, hdb)
    K = setup.kernel.K
    print(f"kernel K shape {K.shape}  dt {setup.kernel.dt}  t[-1] {setup.kernel.t[-1]:.1f}s  "
          f"max|K| {np.abs(K).max():.3e}")
    Kint = _trapz(K, dx=setup.kernel.dt, axis=2)  # (102,102) = B(0) if identity holds
    kappa_surge = float(Kint[np.ix_(_SIDX, _SIDX)].sum())
    print(f"kappa_surge = sum_surge integral K ds = {kappa_surge:.4f}  N*s/m")

    # BEM damping B at low omega, surge-block system sum. hdb.B is (ndof,ndof,nomega)
    # over the HYDRO dofs; map to global via hydro_dof.
    B = np.asarray(hdb.B)
    w = np.asarray(hdb.omega)
    hydro_dof = prp._hydro_dof(prp._deck_with_drag())
    # surge hydro-dof positions within the hydro block: those global surge dofs that are hydro
    surge_global = set(_SIDX)
    hy = list(hydro_dof)
    surge_local = [i for i, g in enumerate(hy) if g in surge_global]
    heave_global = {6 * b + 2 for b in range(17)}
    heave_local = [i for i, g in enumerate(hy) if g in heave_global]
    print(f"\nmapping: {len(surge_local)} surge hydro-dofs, {len(heave_local)} heave hydro-dofs "
          f"(expect 12 each)")
    kw = int(np.argmin(np.abs(w - 2.0)))  # near the wave frequency
    print(f"omega grid: {w.min():.3f} .. {w.max():.3f} rad/s ({w.size} pts)")
    for k in [0, 1, kw]:
        bs = float(B[np.ix_(surge_local, surge_local, [k])].sum())
        bh = float(B[np.ix_(heave_local, heave_local, [k])].sum())
        print(f"  omega={w[k]:6.3f}:  B_surge_system={bs:10.4f}   B_heave_system={bh:10.4f} N*s/m")
    print(f"\nintegral-K (=B(0) if identity holds): {kappa_surge:.4f}")
    print("If B_surge(low omega) ~ integral-K -> kernel fine, radiation truly small.")
    print("If B_surge(low omega) >> integral-K -> kernel contraction underestimates B(0).")


if __name__ == "__main__":
    main()
