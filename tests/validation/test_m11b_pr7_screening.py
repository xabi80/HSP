"""M11b PR7 -- the two-detector BEM conditioning screening (OPTION 2).

Permanent re-validation of BOTH detectors on the committed cluster-draft cases
(STEP D promoted to a gate). They catch DIFFERENT phenomena:
  - cond(K)-z (solve conditioning): flags the ill-conditioned solve at 16.837,
    FLAT at 4.934 (an output anomaly behind a well-conditioned solve);
  - B-min-eig (M8 magnitude floor, significance-skip DROPPED, + neighbour-z
    isolation): excludes exactly {4.934, 20.909}, tolerates the smooth physical
    2-3 band, retains the benign sub-magnitude isolated spikes (3.2, 27.9).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
_PLAT = REPO / "studies" / "platform-12buoy"
sys.path.insert(0, str(_PLAT))
sys.path.insert(0, str(REPO / "studies" / "cluster-3buoy-rigid"))

import platform_screening as ps  # noqa: E402

from floatsim.hydro.readers.capytaine import read_capytaine  # noqa: E402

_NC18 = REPO / "studies" / "cluster-3buoy-rigid" / "capytaine_multibody_18dof.nc"
_CLUSTER_MESH = REPO / "studies" / "cluster-3buoy-rigid" / "mesh" / "cluster3_fullfix.gdf"


def _fixture_B():  # type: ignore[no-untyped-def]
    h = read_capytaine(_NC18)
    return np.asarray(h.omega), np.asarray(h.B)


# ---------------------------------------------------------------------------
# B-min-eig detector (fast): excludes exactly {4.934, 20.909}
# ---------------------------------------------------------------------------


def test_b_min_eig_detector_excludes_known_contaminations() -> None:
    omega, B = _fixture_B()
    # flat cond -> isolate the B detector
    verdicts = ps.screen(omega, np.ones(omega.size), B)
    excluded = sorted(round(v.omega, 3) for v in verdicts if v.exclude)
    k49 = int(np.argmin(np.abs(omega - 4.934)))
    k209 = int(np.argmin(np.abs(omega - 20.909)))
    assert excluded == sorted([round(float(omega[k49]), 3), round(float(omega[k209]), 3)])
    assert verdicts[k49].verdict == "output_contam"
    assert verdicts[k209].verdict == "output_contam"


def test_b_detector_tolerates_physical_and_retains_benign() -> None:
    """The smooth physical 2-3 band and the benign sub-magnitude isolated spikes
    (3.2 at -0.014, 27.9 at -0.0045 -- ABOVE the M8 magnitude floor) are
    retained, not excluded."""
    omega, B = _fixture_B()
    verdicts = ps.screen(omega, np.ones(omega.size), B)
    for target in (2.230, 2.769, 3.200, 9.450, 27.910):
        k = int(np.argmin(np.abs(omega - target)))
        assert not verdicts[k].exclude, f"omega={omega[k]:.3f} wrongly excluded"


def test_four_way_verdict_logic() -> None:
    """The verdict table on synthetic inputs (cond high/low x B isolated/not)."""
    omega = np.array([1.0, 2.0, 3.0])
    # build B: middle omega an isolated deep-negative (contamination), edges clean-PSD
    n = 6
    B = np.zeros((n, n, 3))
    for k in range(3):
        B[:, :, k] = np.eye(n) * 1.0  # clean positive
    B[0, 0, 1] = -10.0  # middle: hugely negative -> sig-neg + isolated
    # cond flat -> only B fires at middle -> output_contam
    v = ps.screen(omega, np.array([1.0, 1.0, 1.0]), B)
    assert v[1].verdict == "output_contam" and v[1].exclude
    assert v[0].verdict == "clean" and not v[0].exclude
    # cond spike at middle too -> both
    v2 = ps.screen(omega, np.array([1.0, 1e6, 1.0]), B)
    assert v2[1].verdict == "both" and v2[1].exclude


def test_neighbour_trend_z_isolated_vs_smooth() -> None:
    z_iso = ps.neighbour_trend_z(np.array([1.0, 1.0, 100.0, 1.0, 1.0]))
    z_smooth = ps.neighbour_trend_z(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert z_iso[2] > 50.0  # isolated spike -> high z
    assert z_smooth[2] < 5.0  # linear ramp -> low z


# ---------------------------------------------------------------------------
# cond(K) detector (slow: re-runs BEM on the cluster mesh)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cond_k_detector_flags_irregular_flat_on_output_anomaly() -> None:
    """cond(K)-z fires at the ill-conditioned solve 16.837 and stays FLAT at the
    output-anomaly 4.934 (the two detectors see different phenomena)."""
    import platform_bem as pb

    grid = np.geomspace(0.1, 30.0, 80)
    k4 = int(np.argmin(np.abs(grid - 4.934)))
    k16 = int(np.argmin(np.abs(grid - 16.837)))
    band = sorted({*range(k4 - 3, k4 + 4), *range(k16 - 3, k16 + 4)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cond = pb.cond_k_sweep(_CLUSTER_MESH, grid[band])
    cz = ps.neighbour_trend_z(np.log10(cond))
    i4, i16 = band.index(k4), band.index(k16)
    assert cz[i16] > ps.COND_Z_THRESHOLD  # 16.837 flagged (~11.7)
    assert cz[i4] < ps.COND_Z_THRESHOLD  # 4.934 flat (~0.06)
