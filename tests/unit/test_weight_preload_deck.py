"""Weight preload, deck schema (docs/weight-preload-plan.md §6, Q1/Q2 locks).

- ``simulation.weight_preload`` is opt-in (Q1): default ``False``.
- ``Body.displaced_volume`` is optional (Q2): default ``None``, meaning ``m / rho`` (the body
  floats freely at its BEM draft). It is meaningful only for a hydro body; a structural body has
  no buoyancy, so declaring one is rejected rather than silently ignored.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from floatsim.io.deck import Body, HydroDatabaseRef, Inertia, Simulation


def _inertia() -> Inertia:
    return Inertia(Ixx=1.0, Iyy=1.0, Izz=1.0)


def test_weight_preload_is_opt_in() -> None:
    assert Simulation(duration=1.0, dt=0.01).weight_preload is False


def test_weight_preload_can_be_switched_on() -> None:
    assert Simulation(duration=1.0, dt=0.01, weight_preload=True).weight_preload is True


def test_displaced_volume_defaults_to_none() -> None:
    b = Body(
        name="b",
        reference_point=[0.0, 0.0, 0.0],
        mass=2.0,
        inertia=_inertia(),
        hydro_body_label="b",
    )
    assert b.displaced_volume is None


def test_displaced_volume_accepted_on_hydro_body() -> None:
    b = Body(
        name="b",
        reference_point=[0.0, 0.0, 0.0],
        mass=2.0,
        inertia=_inertia(),
        hydro_database=HydroDatabaseRef(format="wamit", path="x"),
        displaced_volume=0.003,
    )
    assert b.displaced_volume == pytest.approx(0.003)


def test_displaced_volume_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Body(
            name="b",
            reference_point=[0.0, 0.0, 0.0],
            mass=2.0,
            inertia=_inertia(),
            hydro_body_label="b",
            displaced_volume=0.0,
        )


def test_displaced_volume_rejected_on_structural_body() -> None:
    with pytest.raises(ValidationError, match=r"displaced_volume.*structural"):
        Body(
            name="hub",
            reference_point=[0.0, 0.0, 0.0],
            mass=2.0,
            inertia=_inertia(),
            structural=True,
            displaced_volume=0.001,
        )
