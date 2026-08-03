"""Validate the single-buoy heave added mass A33 from Capytaine against the
analytical broadside-disc result, and split fin vs spar. See
ADDED-MASS-VALIDATION.md for the write-up.

A rigid circular disc of radius a translating normal to its face in infinite
fluid has added mass m_a = (8/3) rho a^3. Writes added_mass_single.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from floatsim.hydro.mesh_hygiene import load_gdf_panels
from floatsim.hydro.readers.capytaine import read_capytaine

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "cluster-3buoy-rigid"))
import cluster_common as cc  # noqa: E402

RHO = 1025.0
A_FIN = 0.215  # heave-plate (fin) radius
R_SPAR = 0.0841  # spar-column radius
TN = 2.982


def disc(a: float) -> float:
    """Broadside circular-disc added mass in infinite fluid."""
    return (8.0 / 3.0) * RHO * a**3


def main() -> None:
    # --- geometry from the mesh ---
    v = load_gdf_panels(cc.SINGLE_EQDRAFT_MESH).panels.reshape(-1, 3)
    r, z = np.hypot(v[:, 0], v[:, 1]), v[:, 2]
    zfin = z[np.abs(r - r.max()) < 0.005]
    print(f"mesh: spar R~{np.median(r[z > z.min() + 0.3]):.4f}, fin R={r.max():.4f} "
          f"at z={np.median(zfin):.3f} m; z range {z.min():.3f}..{z.max():.3f}")

    # --- Capytaine A33(omega), B33(omega) ---
    h = read_capytaine(_HERE / "capytaine_bem.nc")
    w = np.asarray(h.omega)
    A33, B33 = np.asarray(h.A)[2, 2, :], np.asarray(h.B)[2, 2, :]
    Ainf = float(np.asarray(h.A_inf)[2, 2])
    wn = 2 * np.pi / TN
    A_wn = float(np.interp(wn, w, A33))
    m_disc = disc(A_FIN)

    print(f"\nA33: flat {A33.min():.2f}..{A33.max():.2f} kg over omega; "
          f"A33(omega_n)={A_wn:.2f}, A_inf={Ainf:.2f}")
    print(f"B33: ~0 (peak {B33.max():.3f}, at omega_n {float(np.interp(wn, w, B33)):.3f} kg/s)")
    print(f"solid disc (8/3)rho a^3 (a={A_FIN}) = {m_disc:.2f} kg -> A_inf is {Ainf/m_disc:.0%}")
    print(f"fin vs spar-bottom by radius^3: {disc(A_FIN):.1f} vs {disc(R_SPAR):.1f} kg "
          f"-> spar is {disc(R_SPAR)/disc(A_FIN):.0%} of the fin")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(w, A33, "-o", ms=3, color="#1f77b4", label="Capytaine A33")
    ax[0].axhline(m_disc, color="crimson", ls="--", lw=1.2,
                  label=f"(8/3)rho a^3 solid disc = {m_disc:.1f}")
    ax[0].axhline(Ainf, color="gray", ls=":", lw=1.2, label=f"A_inf = {Ainf:.1f}")
    ax[0].axvline(wn, color="green", ls=":", lw=1.0, alpha=0.7, label=f"omega_n={wn:.2f}")
    ax[0].set_xlabel("omega (rad/s)")
    ax[0].set_ylabel("A33 (kg)")
    ax[0].set_title("single-buoy heave added mass")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    ax[0].set_xlim(0, 12)
    ax[1].plot(w, B33, "-o", ms=3, color="#d62728")
    ax[1].axvline(wn, color="green", ls=":", lw=1.0, alpha=0.7)
    ax[1].set_xlabel("omega (rad/s)")
    ax[1].set_ylabel("B33 (kg/s)")
    ax[1].set_title("single-buoy heave radiation damping (~0)")
    ax[1].grid(alpha=0.3)
    ax[1].set_xlim(0, 12)
    fig.tight_layout()
    out = _HERE / "added_mass_single.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
