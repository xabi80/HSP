"""Study pre-flight re-verification on the eqdraft BEM (PF1/PF2 pattern).

PF1 (reader chain, retired): load the regenerated raw asymmetric NetCDF
via the FloatSim Capytaine reader; expect clean ingestion + symmetric
post-construction.
PF2 (kernel chain, retired): compute_retardation_kernel with the
small-body override; expect warning + kernel + Check 3 pass.

Also inspects the eqdraft A(omega) heave curve so the A_inf-vs-omega_n
distinction is explicit.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.hydro.retardation import compute_retardation_kernel

_HERE = Path(__file__).resolve().parent
_NC = _HERE / "capytaine_bem.nc"

_RATIONALE = (
    "spar-fin study resumption: small-body L~1.85 m, 1/omega^4 regime "
    "not reached at omega_max=30; see ITEM25-SMALL-BODY-APPLICABILITY"
)


def main() -> None:
    print("=" * 70)
    print("Study pre-flight re-verification (eqdraft BEM)")
    print("=" * 70)

    # --- inspect raw asymmetry + A(omega) heave curve ---
    with xr.open_dataset(_NC) as ds:
        omega = ds["omega"].values
        A = ds["added_mass"].sel(radiating_dof="Heave", influenced_dof="Heave").values
        A_full = ds["added_mass"].values
        a_inf = A_full[np.isposinf(omega)][0]  # (6,6) at omega=inf
    fin = np.isfinite(omega)
    w = omega[fin]
    A_h = A[fin]
    order = np.argsort(w)
    w, A_h = w[order], A_h[order]
    print("\nA_heave(omega) on eqdraft:")
    print(f"  A(omega->0)   = {A_h[0]:.4f} kg  (omega={w[0]:.3f})")
    print(f"  A(omega=inf)  = {float(a_inf[2,2]):.4f} kg   <- study A_inf")
    print(f"  A(omega_max)  = {A_h[-1]:.4f} kg  (omega={w[-1]:.1f})")
    print(f"  max A over finite omega = {A_h.max():.4f} kg at omega={w[A_h.argmax()]:.3f}")
    # A at the expected omega_n ~ 2 rad/s
    for wn in (1.8, 2.1, 2.4):
        i = int(np.argmin(np.abs(w - wn)))
        print(f"  A(omega~{wn}) = {A_h[i]:.4f} kg (nearest omega={w[i]:.3f})")

    # --- PF1: reader ingestion ---
    # The regenerated NetCDF already carries an omega=inf sample
    # (capytaine_run.py adds it), so the reader extracts A_inf itself;
    # passing a_inf would raise (double-source).
    print("\n[PF1] FloatSim Capytaine reader ingestion (no pre-symmetrization) ...")
    hdb = read_capytaine(_NC)
    assert isinstance(hdb, HydroDatabase)
    rA = float(hdb.metadata["symmetrization_max_residual_A"])
    rB = float(hdb.metadata["symmetrization_max_residual_B"])
    print(f"  ingested OK. symmetrization residual A={rA:.3e}, B={rB:.3e}")
    n_w = hdb.A.shape[-1]
    max_asym = max(
        float(np.max(np.abs(hdb.A[..., k] - hdb.A[..., k].T))) for k in range(n_w)
    )
    print(f"  post-construction max |A-A.T| over omega = {max_asym:.3e} (expect <1e-12)")
    assert max_asym < 1e-12
    print(f"  A_inf(heave) via reader = {float(hdb.A_inf[2,2]):.4f} kg")
    print(f"  C33 = {float(hdb.C[2,2]):.4f} N/m")

    # --- PF2: kernel with override ---
    print("\n[PF2] compute_retardation_kernel with small-body override ...")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        kernel = compute_retardation_kernel(
            hdb, t_max=30.0, dt=0.01, asymptote_check_override=_RATIONALE
        )
    fired = [str(x.message) for x in rec if "Item 25 asymptote check bypassed" in str(x.message)]
    print(f"  override warning fired: {bool(fired)}")
    print(f"  kernel shape: {kernel.K.shape}, t: [{kernel.t[0]}, {kernel.t[-1]}]")
    K33 = kernel.K[2, 2, :]
    print(f"  K33(0) = {K33[0]:.4e}; |K33(t_max)| = {abs(K33[-1]):.4e} "
          f"(ratio {abs(K33[-1])/max(abs(K33).max(),1e-30):.3e})")
    print("  Check 3 (kernel decay) passed inside compute_retardation_kernel.")

    print("\nPre-flight re-verification complete: PF1 + PF2 pass on eqdraft BEM.")


if __name__ == "__main__":
    main()
