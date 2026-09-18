"""Symmetrize + PSD-project the coupled-BEM radiation matrices.

The method-of-images two-DOF (_all/_real) trick in coupled_bem_osu.py does not preserve
the discrete reciprocity of the full BEM matrix, so the walled radiation-damping matrix
B(omega) comes out slightly non-symmetric and non-positive-semidefinite (min eigenvalue
~ -max|B|), which FloatSim's multi-body PSD gate rejects. A(omega) and B(omega) are
symmetric by reciprocity, so we symmetrize both and eigenvalue-clip B to PSD.

Heave free-decay (A-driven, viscous-damped) is robust to this; the walled B stays the
least-precise input, so the articulated wave-sweep RAO is read as "<= a few %" rather than
a precise number (see REBUTTAL-sidewall.md / README).

Usage: python psd_project.py coupled_osu_open.nc coupled_osu_walled.nc   (writes *_psd.nc)
"""
from __future__ import annotations

import sys

import numpy as np
import xarray as xr


def project(nc: str) -> None:
    d = xr.load_dataset(nc)
    A = d["added_mass"].values.copy()
    B = d["radiation_damping"].values.copy()
    min_eig = np.inf
    for k in range(A.shape[0]):
        A[k] = 0.5 * (A[k] + A[k].T)
        B[k] = 0.5 * (B[k] + B[k].T)
        ev, vecs = np.linalg.eigh(B[k])
        min_eig = min(min_eig, float(ev.min()))
        B[k] = (vecs * np.clip(ev, 0.0, None)) @ vecs.T
    d["added_mass"].values[:] = A
    d["radiation_damping"].values[:] = B
    out = nc.replace(".nc", "_psd.nc")
    d.to_netcdf(out)
    print(f"{nc}: min B eigenvalue {min_eig:.2e} -> symmetrized + PSD-projected, wrote {out}")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        project(path)
