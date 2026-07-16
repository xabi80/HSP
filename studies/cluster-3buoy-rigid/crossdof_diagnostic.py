"""Closeout diagnostic: cross-DOF direction + y-mirror check + radiation
damping direction re-verification.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

import cluster_common as cc

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"
_DOF = ["surge", "sway", "heave", "roll", "pitch", "yaw"]


def main() -> None:
    print("=" * 70)
    print("Cross-DOF direction + mirror + radiation re-check")
    print("=" * 70)

    # --- (1) per-DOF max from BEM-only decay ---
    d = np.loadtxt(_RESULTS / "decay_bem_only.csv", delimiter=",", skiprows=1)
    xi = d[:, 1:7]
    print("\nPer-DOF max |xi_k| over the BEM-only decay:")
    maxes = {}
    for k, n in enumerate(_DOF):
        maxes[n] = float(np.max(np.abs(xi[:, k])))
        print(f"  {n:6s} (DOF {k}): {maxes[n]:.4e}")
    print("\n  y-mirror symmetry (one buoy on +x, two mirrored across x-axis):")
    print("    EVEN under y->-y (allowed to couple to heave): surge, pitch")
    print("    ODD  under y->-y (forbidden):                   sway, roll, yaw")
    allowed = max(maxes["surge"], maxes["pitch"])
    forbidden = max(maxes["sway"], maxes["roll"], maxes["yaw"])
    print(f"    max allowed (surge/pitch)  = {allowed:.4e}")
    print(f"    max forbidden (sway/roll/yaw) = {forbidden:.4e}")
    print(f"    allowed / forbidden = {allowed/forbidden:.1f}x")

    # --- (2) mirror placement check on the two off-axis hulls ---
    from floatsim.hydro.mesh_hygiene import load_gdf_panels
    panels = load_gdf_panels(cc.CLUSTER_MESH).panels  # (4464, 4, 3)
    h1 = panels[1488:2976].reshape(-1, 3)   # buoy 1 at (-0.25, +0.433)
    h2 = panels[2976:4464].reshape(-1, 3)   # buoy 2 at (-0.25, -0.433)
    # h2 should equal the y-reflection of h1. Reflect h1, nearest-neighbour.
    h1m = h1.copy()
    h1m[:, 1] *= -1.0
    # Match by sorting on rounded coords (exact copies => exact match).
    key1 = np.round(h1m, 9)
    key2 = np.round(h2, 9)
    order1 = np.lexsort(key1.T)
    order2 = np.lexsort(key2.T)
    resid = np.abs(key1[order1] - key2[order2]).max()
    # Also the raw |y_i + y_j| over the sorted-matched pairs.
    y_sum = np.abs(h1[order1][:, 1] + h2[order2][:, 1]).max()
    print(f"\n  Mirror check (buoy1 reflected-in-y vs buoy2):")
    print(f"    max |vertex mismatch| = {resid:.3e} (expect ~1e-9 round tol)")
    print(f"    max |y_i + y_j| paired = {y_sum:.3e} (expect ~1e-15 if exact mirror)")

    # --- (3) radiation damping direction re-verification ---
    print("\n  Radiation damping interaction (re-verify direction):")
    inter = json.loads((_RESULTS / "interaction.json").read_text())
    wn = inter["omega_n"]

    def b33(nc):
        with xr.open_dataset(_HERE / nc) as ds:
            w = ds["omega"].values
            B = ds["radiation_damping"].sel(radiating_dof="Heave",
                                            influenced_dof="Heave").values
        fin = np.isfinite(w)
        o = np.argsort(w[fin])
        return float(np.interp(wn, w[fin][o], B[fin][o]))

    B_s = b33("reference_single_bem.nc")
    B_c = b33("composite_bem.nc")
    print(f"    B33_single(omega_n)    = {B_s:.4e}")
    print(f"    B33_composite(omega_n) = {B_c:.4e}")
    print(f"    B33_composite / B33_single         = {B_c/B_s:.3f} "
          f"(N=3; coherent limit N^2=9)")
    print(f"    B33_composite / (3 x B33_single)   = {B_c/(3*B_s):.3f} "
          f"(>1 constructive, <1 destructive)")

    # zeta comparison (cluster vs single-buoy study).
    M_A_c = cc.M_CLUSTER + inter["A33_composite_inf"]
    C_c = inter["C33_composite"]
    zeta_c = B_c / (2 * np.sqrt(M_A_c * C_c))
    # Single-buoy study numbers (spar-fin, isolated draft).
    B_s_iso, MA_s, C_s = 2.3452e-2, 49.789, 221.081
    zeta_s = B_s_iso / (2 * np.sqrt(MA_s * C_s))
    print(f"\n    zeta_rad single (spar-fin study) = {zeta_s:.4e} "
          f"({zeta_s*100:.4f}% crit)")
    print(f"    zeta_rad cluster                 = {zeta_c:.4e} "
          f"({zeta_c*100:.4f}% crit)")
    print(f"    cluster / single = {zeta_c/zeta_s:.2f}x  "
          f"({'ROSE' if zeta_c > zeta_s else 'DROPPED'})")


if __name__ == "__main__":
    main()
