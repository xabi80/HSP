"""STEP 5 PR1 commit (c), integration tier: ``build_system`` plumbs ``drag_wave`` /
``drag_wave_ramp`` into the setup's state force (two synthetic-BEM builds, ~25 s).
The drag physics itself is unit-tested in tests/unit/test_drag_wave_wiring.py."""

from __future__ import annotations

import numpy as np

from floatsim.driver import _build_drag_state_force, build_system
from scripts.m7_pr4_driver_prediction import _single_body_hdb
from tests.unit.test_drag_wave_wiring import _RAMP, _RHO, _WAVE, _deck, _states

_DT_COARSE = 0.05  # Nyquist-safe for the synthetic BEM (omega_max = 20 rad/s -> dt < 0.157 s)


def test_build_system_wires_the_drag_wave() -> None:
    """build_system(drag_wave=, drag_wave_ramp=) produces exactly the wired drag; without them
    its state force is the calm-water one."""
    dk = _deck()
    bem = {"b0": _single_body_hdb(), "b1": _single_body_hdb()}
    calm = build_system(
        dk, bem_databases=bem, dt=_DT_COARSE, t_max_kernel=120.0, solve_equilibrium=False
    )
    wave = build_system(
        dk,
        bem_databases=bem,
        dt=_DT_COARSE,
        t_max_kernel=120.0,
        solve_equilibrium=False,
        drag_wave=_WAVE,
        drag_wave_ramp=_RAMP,
    )
    ref_calm = _build_drag_state_force(dk, 12, rho=_RHO)
    ref_wave = _build_drag_state_force(dk, 12, rho=_RHO, wave=_WAVE, ramp=_RAMP)
    assert ref_calm is not None and ref_wave is not None
    for t, xi, xd in _states(8):
        assert calm.state_force(t, xi, xd).tobytes() == ref_calm(t, xi, xd).tobytes()
        assert wave.state_force(t, xi, xd).tobytes() == ref_wave(t, xi, xd).tobytes()
    assert not np.array_equal(
        calm.state_force(30.0, np.zeros(12), np.zeros(12)),
        wave.state_force(30.0, np.zeros(12), np.zeros(12)),
    )
