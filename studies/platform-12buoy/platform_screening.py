"""M11b PR7 -- the embedded two-detector BEM screening (OPTION 2).

Two COMPLEMENTARY detectors, validated on the committed cluster-draft cases
(they catch DIFFERENT phenomena; STEP 1 established neither sees the other's):

1. cond(K) -- 1/rcond of the BEM system matrix K (capytaine solver.py:174),
   via LAPACK zgecon on the LU the solver ALREADY computes (no re-factorize);
   neighbour-trend z on log10(cond). Catches genuinely ILL-CONDITIONED solves
   (irregular frequencies): 16.837 at z=11.7 (Capytaine warns independently);
   FLAT at 4.934 (z=0.06).
2. symmetrized-B min-eigenvalue -- the M8 PSD gate (magnitude floor) as the
   "fires" test, then neighbour-trend z as the ISOLATION discriminator.
   Catches OUTPUT anomalies behind well-conditioned solves: 4.934 at z=1027,
   20.909 at z=88045. The z only splits PSD-firing slices: isolated ->
   contamination (exclude); SMOOTH -> physical near-singularity (tolerate, the
   PR3 F2 deepening-coherent-coupling trend). Sub-magnitude isolated spikes
   (3.2, 27.9, 9.45) do NOT trip the PSD magnitude floor, so they never reach
   the z test.

Thresholds are set from the measured separation (18-DOF fixture), stated with
their margin below. Disposition of a flagged slice is EXCLUSION (grid
selection, M8 PR3 pattern) -- NEVER value modification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from floatsim.hydro.retardation import _PSD_REL_TOL

# --- Thresholds (measured separation, 18-DOF fixture; STEP 1 validation) ------
# cond(K) neighbour-z: ill-conditioned 16.837 at z=11.7; clean/physical <= 1.3.
COND_Z_THRESHOLD = 5.0  # margin: 11.7/5 = 2.3x above; 5/1.3 = 3.8x above clean
# B-min-eig neighbour-z (only tested on PSD-firing slices): contamination
# 4.934 z=1027, 20.909 z=88045; physical 2-3 band z=0.5-3.0.
BMINEIG_Z_THRESHOLD = 50.0  # margin: 1027/50 = 20x above; 50/3 = 17x above physical


def neighbour_trend_z(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-point z vs the median/MAD of its two neighbours (Measurement E idea).

    Endpoints get z=0 (no two-sided neighbourhood). Robust (median/MAD) so a
    single isolated spike does not inflate its own baseline.
    """
    v = np.asarray(values, dtype=np.float64)
    z = np.zeros_like(v)
    for i in range(1, len(v) - 1):
        nb = np.array([v[i - 1], v[i + 1]])
        med = float(np.median(nb))
        mad = float(np.median(np.abs(nb - med)))
        z[i] = abs(v[i] - med) / (1.4826 * mad + 1e-12)
    return z


def significant_negative(B: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Per omega: min-eig(symB) below the M8 magnitude floor -_PSD_REL_TOL*gmax.

    Reuses the committed M8 magnitude constant but DROPS the M8 significance
    pre-filter (max|B(omega)| >= 5% of global). That pre-filter exists to keep
    the KERNEL gate from firing on tail *noise*, but it also skips tail
    *contaminations* -- e.g. omega=20.909, min-eig=-3.29 in the high-omega tail,
    which the significance-skip hides. The magnitude floor itself already
    rejects tail noise (3.2 at -0.014 and 27.9 at -0.0045 stay ABOVE the
    -0.048 floor; 4.934 at -0.12 and 20.909 at -3.29 fall below it), so the
    screening uses magnitude-only. ``_PSD_SIGNIFICANCE_FLOOR`` is imported only
    to document this deliberate departure.
    """
    gmax = float(np.max(np.abs(B)))
    return np.array(
        [
            bool(np.linalg.eigvalsh(0.5 * (B[:, :, k] + B[:, :, k].T)).min() < -_PSD_REL_TOL * gmax)
            for k in range(B.shape[2])
        ]
    )


def b_min_eig(B: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetrized-B smallest eigenvalue per omega."""
    return np.array(
        [
            float(np.linalg.eigvalsh(0.5 * (B[:, :, k] + B[:, :, k].T)).min())
            for k in range(B.shape[2])
        ]
    )


@dataclass(frozen=True)
class SliceVerdict:
    omega: float
    cond_k: float
    cond_z: float
    bmineig: float
    bmineig_z: float
    psd_fires: bool
    verdict: str  # clean | ill_conditioned | output_contam | both | physical
    exclude: bool


def screen(
    omega: NDArray[np.float64], cond_k: NDArray[np.float64], B: NDArray[np.float64]
) -> list[SliceVerdict]:
    """The four-way (+physical) verdict per frequency (STEP C).

    - neither detector fires        -> clean            (retain)
    - cond(K) only                  -> ill_conditioned  (exclude; irregular-freq)
    - B-min-eig only (PSD+isolated) -> output_contam    (exclude)
    - BOTH                          -> both             (exclude + REPORT: new class)
    - PSD fires but SMOOTH          -> physical         (tolerate/retain)
    Exclusion = grid selection (M8 PR3), never value modification.
    """
    cond_z = neighbour_trend_z(np.log10(np.asarray(cond_k, dtype=np.float64)))
    fires = significant_negative(B)
    mineig = b_min_eig(B)
    mineig_z = neighbour_trend_z(mineig)

    out: list[SliceVerdict] = []
    for k in range(len(omega)):
        cond_flag = bool(cond_z[k] > COND_Z_THRESHOLD)
        isolated = bool(fires[k] and mineig_z[k] > BMINEIG_Z_THRESHOLD)
        physical = bool(fires[k] and not isolated)
        if cond_flag and isolated:
            verdict, exclude = "both", True
        elif cond_flag:
            verdict, exclude = "ill_conditioned", True
        elif isolated:
            verdict, exclude = "output_contam", True
        elif physical:
            verdict, exclude = "physical", False
        else:
            verdict, exclude = "clean", False
        out.append(
            SliceVerdict(
                omega=float(omega[k]),
                cond_k=float(cond_k[k]),
                cond_z=float(cond_z[k]),
                bmineig=float(mineig[k]),
                bmineig_z=float(mineig_z[k]),
                psd_fires=bool(fires[k]),
                verdict=verdict,
                exclude=exclude,
            )
        )
    return out
