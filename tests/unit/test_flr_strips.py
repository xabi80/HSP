"""Re-evaluated strip and patch loads must sum to what the solver applied.

:mod:`floatsim.io.flr_strips` recovers per-element and per-patch drag by
re-evaluating pure functions over the stored kinematics, rather than by hooking
the solve loop. That keeps the export additive -- but it shifts the burden:
"same function, same arguments" is an argument, and it has to become a
measurement.

Here the measurement is direct and needs no instrumentation, because
``make_morison_state_force`` is public. The closure it returns is exactly what
the integrator calls, so evaluating it and comparing against the sum of the
re-evaluated parts tests the reconstruction against the real thing.

This is the local half of FloatFEA gate **G1.6**: a strip set can be internally
consistent and still not correspond to the loads the simulator applied. Summing
back is the only check that catches it.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.hydro.morison import (
    MorisonElement,
    PlateDragElement,
    make_morison_state_force,
)
from floatsim.io.flr_strips import (
    STRIP_ROTATION_PARAMETERISATION,
    STRIP_TIME_ALIGNMENT,
    recompute_patch_loads,
    recompute_strip_loads,
    verify_against_summed,
)

_RHO = 1025.0
_N_DOF = 12  # two bodies
_SPAR_D = 0.1682


def _wave_field(point: np.ndarray, t: float) -> np.ndarray:
    """A depth-decaying oscillatory field, so the fluid sample point matters.

    A calm field (all zeros) would pass even if the reconstruction sampled the
    fluid at the wrong location, which is one of the mistakes this test exists
    to catch.
    """
    k, omega, amp = 0.41, 2.0, 0.05
    decay = float(np.exp(k * min(point[2], 0.0)))
    psi = omega * t - k * point[0]
    return np.array([amp * omega * decay * np.cos(psi), 0.0, amp * omega * decay * np.sin(psi)])


def _elements() -> tuple[list, PlateDragElement]:
    """Ten stacked spar segments on body 0 plus a plate, mirroring the deck."""
    z = np.linspace(-0.2617, 1.1957, 11)
    spars = [
        MorisonElement(
            body_index=0,
            node_a_body=np.array([0.0, 0.0, z[i]]),
            node_b_body=np.array([0.0, 0.0, z[i + 1]]),
            diameter=_SPAR_D,
            Cd=1.2,
        )
        for i in range(10)
    ]
    plate = PlateDragElement(
        body_index=1,
        center_body=np.array([0.0, 0.0, -0.2617]),
        normal_body=np.array([0.0, 0.0, 1.0]),
        radius=0.215,
        thickness=0.0039,
        Cd_n=5.0,
        Cd_t=1.5,
    )
    return spars, plate


def _state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A pose with genuine rotation, so the pose convention is exercised."""
    t = np.array([0.0, 0.37, 0.94])
    xi = np.zeros((3, _N_DOF))
    xi_dot = np.zeros((3, _N_DOF))
    for i, ti in enumerate(t):
        xi[i, 0:3] = [0.03 * np.cos(ti), 0.01 * np.sin(ti), 0.02]
        xi[i, 3:6] = [0.08 * np.sin(ti), 0.06 * np.cos(ti), 0.03]
        xi[i, 6:9] = [0.02, -0.01 * np.cos(ti), 0.015]
        xi[i, 9:12] = [0.05 * np.cos(ti), -0.04 * np.sin(ti), 0.02]
        xi_dot[i, 0:3] = [-0.03 * np.sin(ti), 0.01 * np.cos(ti), 0.004]
        xi_dot[i, 3:6] = [0.08 * np.cos(ti), -0.06 * np.sin(ti), 0.0]
        xi_dot[i, 6:9] = [0.01, 0.01 * np.sin(ti), -0.006]
        xi_dot[i, 9:12] = [-0.05 * np.sin(ti), -0.04 * np.cos(ti), 0.0]
    return t, xi, xi_dot


def test_strip_loads_sum_to_the_applied_body_force() -> None:
    """Per-element re-evaluation sums to ``make_morison_state_force``'s output."""
    spars, _ = _elements()
    t, xi, xi_dot = _state()
    closure = make_morison_state_force(
        spars, n_dof=_N_DOF, fluid_velocity_fn=_wave_field, rho=_RHO
    )
    applied = np.array([closure(float(t[i]), xi[i], xi_dot[i])[0:6] for i in range(t.size)])

    parts = [
        recompute_strip_loads(
            e, xi=xi, xi_dot=xi_dot, t=t, fluid_velocity_fn=_wave_field, rho=_RHO
        )[0]
        for e in spars
    ]
    dev = verify_against_summed(parts, applied)
    assert dev == 0.0, f"strips do not sum to the applied force: max deviation {dev:.3e}"


def test_patch_loads_sum_to_the_applied_plate_force() -> None:
    """Per-patch normal force plus the lumped tangential term reproduce the whole."""
    _, plate = _elements()
    t, xi, xi_dot = _state()
    closure = make_morison_state_force(
        [plate], n_dof=_N_DOF, fluid_velocity_fn=_wave_field, rho=_RHO
    )
    applied = np.array([closure(float(t[i]), xi[i], xi_dot[i])[6:12] for i in range(t.size)])

    df_n, f_tan, _ = recompute_patch_loads(
        plate, xi=xi, xi_dot=xi_dot, t=t, fluid_velocity_fn=_wave_field, rho=_RHO
    )
    from floatsim.hydro.morison import _body_pose_from_xi

    for i in range(t.size):
        _, R = _body_pose_from_xi(xi[i, 6:12])
        n_hat = R @ plate._n_hat_body
        reconstructed = n_hat * float(df_n[i].sum()) + f_tan[i]
        np.testing.assert_allclose(reconstructed, applied[i, 0:3], rtol=0.0, atol=1e-12)


def test_a_wrong_fluid_sample_point_is_detectable() -> None:
    """The fixture's field varies with depth and position, so a mis-sampled
    reconstruction fails. A calm field would pass regardless, which is why the
    test does not use one."""
    spars, _ = _elements()
    t, xi, xi_dot = _state()
    origin_only = lambda _p, tt: _wave_field(np.zeros(3), tt)  # noqa: E731
    closure = make_morison_state_force(
        spars, n_dof=_N_DOF, fluid_velocity_fn=_wave_field, rho=_RHO
    )
    applied = np.array([closure(float(t[i]), xi[i], xi_dot[i])[0:6] for i in range(t.size)])
    wrong = [
        recompute_strip_loads(
            e, xi=xi, xi_dot=xi_dot, t=t, fluid_velocity_fn=origin_only, rho=_RHO
        )[0]
        for e in spars
    ]
    assert verify_against_summed(wrong, applied) > 1e-6


def test_declared_conventions_match_the_producing_module() -> None:
    """Strips carry the producing module's conventions, not a global default."""
    assert STRIP_ROTATION_PARAMETERISATION == "zyx_intrinsic_euler"
    assert STRIP_TIME_ALIGNMENT == "state_n"


def test_history_shape_mismatch_is_rejected() -> None:
    spars, _ = _elements()
    t, xi, xi_dot = _state()
    with pytest.raises(ValueError, match="same shape"):
        recompute_strip_loads(
            spars[0], xi=xi, xi_dot=xi_dot[:, :6], t=t,
            fluid_velocity_fn=_wave_field, rho=_RHO,
        )
