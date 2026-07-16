"""Step 3 result: interaction ratio R + PF1/PF2 smoke check on composite.

R = A33_composite / (3 * A33_single_at_cluster_draft). Prior band
[1.00, 1.20] is informational only -- ANY value is a finding (R<1 =
destructive interference is physically allowed). Also reports the B33
ratio at omega_n and the A33(omega) overlay data.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.hydro.retardation import compute_retardation_kernel

import cluster_common as cc

_HERE = Path(__file__).resolve().parent
_REF_NC = _HERE / "reference_single_bem.nc"
_COMP_NC = _HERE / "composite_bem.nc"

_OVERRIDE = (
    "3-buoy cluster study: small-body hulls (L~1.85 m); 1/omega^4 regime "
    "not reached at omega_max=30; see ITEM25-SMALL-BODY-APPLICABILITY"
)


def _a33(nc: Path):
    with xr.open_dataset(nc) as ds:
        omega = ds["omega"].values
        A = ds["added_mass"].sel(radiating_dof="Heave", influenced_dof="Heave").values
        B = ds["radiation_damping"].sel(radiating_dof="Heave", influenced_dof="Heave").values
    fin = np.isfinite(omega)
    inf = np.isposinf(omega)
    return omega[fin], A[fin], B[fin], float(A[inf][0])


def main() -> None:
    print("=" * 70)
    print("Step 3 result: 3-buoy heave interaction ratio")
    print("=" * 70)
    w_s, A_s, B_s, Ainf_s = _a33(_REF_NC)
    w_c, A_c, B_c, Ainf_c = _a33(_COMP_NC)
    os_ = np.argsort(w_s)
    w_s, A_s, B_s = w_s[os_], A_s[os_], B_s[os_]
    oc = np.argsort(w_c)
    w_c, A_c, B_c = w_c[oc], A_c[oc], B_c[oc]

    # C33 for omega_n.
    with xr.open_dataset(_COMP_NC) as ds:
        C33_c = float(ds["hydrostatic_stiffness"].sel(
            radiating_dof="Heave", influenced_dof="Heave"))
    M_A = cc.M_CLUSTER + Ainf_c
    Tn = 2 * np.pi * np.sqrt(M_A / C33_c)
    wn = 2 * np.pi / Tn
    A_s_wn = float(np.interp(wn, w_s, A_s))
    A_c_wn = float(np.interp(wn, w_c, A_c))
    B_s_wn = float(np.interp(wn, w_s, B_s))
    B_c_wn = float(np.interp(wn, w_c, B_c))

    R_inf = Ainf_c / (3.0 * Ainf_s)
    R_wn = A_c_wn / (3.0 * A_s_wn)
    B_ratio_wn = B_c_wn / (3.0 * B_s_wn)

    print(f"  A33_single(inf)      = {Ainf_s:.4f} kg (at cluster draft)")
    print(f"  A33_composite(inf)   = {Ainf_c:.4f} kg (raw)")
    print(f"  3 x A33_single(inf)  = {3*Ainf_s:.4f} kg")
    print(f"  >>> R(inf) = A33_comp / (3 A33_single) = {R_inf:.4f}")
    print()
    print(f"  omega_n = {wn:.4f} rad/s (T_n = {Tn:.4f} s)")
    print(f"  A33_single(omega_n)    = {A_s_wn:.4f} kg")
    print(f"  A33_composite(omega_n) = {A_c_wn:.4f} kg")
    print(f"  >>> R(omega_n) = {R_wn:.4f}")
    print(f"  B33_single(omega_n)    = {B_s_wn:.4e}")
    print(f"  B33_composite(omega_n) = {B_c_wn:.4e}")
    print(f"  >>> B33 ratio (comp / 3 single) at omega_n = {B_ratio_wn:.4f}")
    print()
    print(f"  Prior band [1.00, 1.20] is informational; R = {R_inf:.4f} is "
          f"the measurement.")

    # --- PF1: reader ingestion of composite NetCDF ---
    print("\n[PF1] read_capytaine(composite) ...")
    hdb = read_capytaine(_COMP_NC)
    assert isinstance(hdb, HydroDatabase)
    n_w = hdb.A.shape[-1]
    max_asym = max(float(np.max(np.abs(hdb.A[..., k] - hdb.A[..., k].T)))
                   for k in range(n_w))
    print(f"  ingested; resid_A={hdb.metadata['symmetrization_max_residual_A']}, "
          f"post-construction max|A-A.T|={max_asym:.2e}")
    assert max_asym < 1e-12

    # --- PF2: kernel with override ---
    print("[PF2] compute_retardation_kernel(composite, override) ...")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        kernel = compute_retardation_kernel(
            hdb, t_max=30.0, dt=0.01, asymptote_check_override=_OVERRIDE)
    fired = any("Item 25 asymptote check bypassed" in str(w.message) for w in rec)
    print(f"  override warning fired: {fired}; kernel shape {kernel.K.shape}; "
          f"Check 3 passed.")

    out = {
        "R_inf": R_inf, "R_omega_n": R_wn, "B33_ratio_omega_n": B_ratio_wn,
        "A33_single_inf": Ainf_s, "A33_composite_inf": Ainf_c,
        "C33_composite": C33_c, "omega_n": wn, "T_n_with_interaction": Tn,
        "A33_single_curve": [w_s.tolist(), A_s.tolist()],
        "A33_composite_curve": [w_c.tolist(), A_c.tolist()],
    }
    (_HERE / "results" / "interaction.json").write_text(json.dumps(out, indent=2))
    print("\n  Wrote results/interaction.json")


if __name__ == "__main__":
    main()
