"""Weight preload, the geometric stiffness for modal tools (docs/weight-preload-plan.md §2.4, PR2).

``SimulationSetup.K_geo = -d(G(xi)^T lambda_eq)/dxi`` at the preloaded equilibrium: the stiffness
the time domain gets from ``G``'s configuration dependence, made available to matrix tools.

On the pinned float (tests/support/pinned_float.py), the pin force ``lambda = (0, 0, -P)`` acts on
the body at the arm ``a = (0, 0, L)``:
- **closed form:** the body's rotational block is ``(a . F) I - a F^T = diag(-P L, -P L, 0)``,
  and the translational blocks are zero (``G``'s translational columns are constant);
- **modal check:** on the feasible subspace, ``C + K_geo`` gives
  ``omega^2 = (C55 - P L) / (I_yy + m L^2)`` exactly;
- **stability gate:** a preload past the stability limit (``P L > C55``) is a negative-stiffness
  mode, which the preloaded restoring gate must reject at build time.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from tests.support import pinned_float as pf


@pytest.mark.parametrize("net_up", [30.0, -30.0])
def test_kgeo_rotational_block_is_minus_pl(net_up: float) -> None:
    s = pf.setup(net_up)
    assert s.K_geo is not None
    expected = np.zeros((6, 6))
    expected[3, 3] = expected[4, 4] = -net_up * pf.L_PIN
    np.testing.assert_allclose(s.K_geo, expected, atol=1e-6)


@pytest.mark.parametrize("net_up", [30.0, -30.0])
def test_modal_frequency_with_kgeo_is_the_closed_form(net_up: float) -> None:
    s = pf.setup(net_up)
    assert s.K_geo is not None and s.constraints is not None
    n = sla.null_space(s.constraints.jacobian(s.xi0))
    k = n.T @ (np.asarray(s.lhs.C) + s.K_geo) @ n
    m = n.T @ np.asarray(s.lhs.M_plus_Ainf) @ n
    w2 = np.sort(sla.eigvals(k, m).real)
    expected = (pf.C55 - net_up * pf.L_PIN) / (pf.I_YY + pf.MASS * pf.L_PIN**2)
    np.testing.assert_allclose(w2[-2:], [expected, expected], rtol=1e-6)


def test_kgeo_absent_without_preload() -> None:
    assert pf.setup(30.0, preload=False).K_geo is None


def test_preload_past_the_stability_limit_fails_fast() -> None:
    """P = 50 N: P L = 81.2 > C55 = 75.28, so the pinned float has no restoring left."""
    with pytest.raises(ValueError, match=r"preload"):
        pf.setup(50.0)
