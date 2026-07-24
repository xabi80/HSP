"""M8 Phase-1 Measurement B: cross-term damping-kernel decay.

Decides program-plan Q3's t_max sub-item: does the heave cross-kernel
K_ij(t) decay on the same timescale as the diagonal K_ii(t)? On the
18-DOF fixture, compute B_ij(omega) and the Filon kernel
K(t) = (2/pi) * integral B(omega) cos(omega t) domega (Item-25 override
= zero-fill tail beyond omega_max, so just the grid integral). Report
K(t)/K(0) at t = 5, 10, 20, 30 s. Measurement, not a gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from floatsim.hydro._filon import filon_trap_cosine

_HERE = Path(__file__).resolve().parent
_NC = _HERE / "capytaine_multibody_18dof.nc"

# Override rationale (same small hulls as the single-buoy study).
_RATIONALE = "M8 cross-kernel diagnostic: small hulls; see ITEM25-SMALL-BODY-APPLICABILITY"

# 18-DOF index of each hull's Heave (order Surge,Sway,Heave,Roll,Pitch,Yaw).
H1, H2 = 2, 8   # buoy1__Heave, buoy2__Heave


def _kernel(omega, B_row, t):
    """(2/pi) * Filon cosine transform, omega=0/B=0 prepended (override tail=0)."""
    if omega[0] > 1e-9:
        omega = np.concatenate([[0.0], omega])
        B_row = np.concatenate([[0.0], B_row])
    return (2.0 / np.pi) * filon_trap_cosine(omega, B_row, np.asarray(t, float))


def main() -> None:
    print("=" * 70)
    print("M8 Measurement B: cross-term damping-kernel decay")
    print(f"  ({_RATIONALE})")
    print("=" * 70)
    with xr.open_dataset(_NC) as ds:
        w = ds["omega"].values
        B = ds["radiation_damping"].values  # (n_omega, 18, 18)
    fin = np.isfinite(w)
    order = np.argsort(w[fin])
    wf = w[fin][order]
    Bf = B[fin][:, :, :][order]
    print(f"  omega grid: {wf.round(4).tolist()} rad/s (from the 18-DOF fixture)")

    B_diag = Bf[:, H1, H1]     # buoy1 heave-heave (diagonal)
    B_cross = Bf[:, H1, H2]    # buoy1 heave - buoy2 heave (cross)
    print(f"\n  B_diag(omega)  range: [{B_diag.min():.4e}, {B_diag.max():.4e}]")
    print(f"  B_cross(omega) range: [{B_cross.min():.4e}, {B_cross.max():.4e}]")
    print(f"  |B_cross|/|B_diag| peak = {np.abs(B_cross).max()/np.abs(B_diag).max():.3f}")

    tq = np.array([0.0, 5.0, 10.0, 20.0, 30.0])
    Kd = _kernel(wf, B_diag, tq)
    Kc = _kernel(wf, B_cross, tq)
    print(f"\n  K(0):  diagonal = {Kd[0]:.4e};  cross = {Kc[0]:.4e}")
    print(f"\n  {'t (s)':>6} {'K_diag/K_diag(0)':>18} {'K_cross/K_cross(0)':>20}")
    for i, t in enumerate(tq):
        rd = Kd[i] / Kd[0] if Kd[0] != 0 else float("nan")
        rc = Kc[i] / Kc[0] if Kc[0] != 0 else float("nan")
        print(f"  {t:6.0f} {rd:18.4f} {rc:20.4f}")

    # Envelope decay timescale: last t where |K|/|K(0)| > 0.1 on a dense grid.
    td = np.arange(0.0, 30.0 + 1e-9, 0.01)
    Kdd = np.abs(_kernel(wf, B_diag, td))
    Kcc = np.abs(_kernel(wf, B_cross, td))
    def t10(K):
        below = np.where(K / K[0] < 0.1)[0] if K[0] != 0 else []
        return td[below[0]] if len(below) else float("nan")
    print("\n  timescale (first t with |K|/|K(0)| < 0.1):")
    print(f"    diagonal: {t10(Kdd):.2f} s;  cross: {t10(Kcc):.2f} s")
    print("\n  FINDING (report, not a gate): compare the two decay columns.")


if __name__ == "__main__":
    main()
