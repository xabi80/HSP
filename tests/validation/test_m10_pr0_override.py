"""M10 PR0 -- Item-25 asymptote_check_override threaded through the coupled
``build_system`` path (Q4-a; plan Amendment A1).

The override lets a small-body coupled BEM (e.g. the cluster hulls,
L~1.85 m, whose ``1/omega^4`` tail is not reached by ``omega_max``)
bypass the M7.5 PR1 asymptote gate (Check 1 advisory + Check 2 hard
error) with an explicit rationale. PR0 threads it
``build_system -> _build_coupled_lhs_kernel -> compute_retardation_kernel``
(``driver.py:498``), purely additively.

Gates here cover the NEW behaviour (override plumbing):
* GATE 2 -- a valid rationale reaches ``compute_retardation_kernel``
  INTACT (asserted via the Item-25 bypass ``UserWarning`` that echoes
  the rationale) and the override path is taken.
* GATE 3 -- an empty / whitespace-only rationale still RAISES through
  the coupled path (M7.5 PR1 forcing-function contract preserved).

GATE 1 (byte-identity of every existing deck path) is the additive
change's guarantee: the coupled kernel call now passes
``asymptote_check_override=None`` -- identical to the pre-PR0 default --
and the per-body path is untouched. It is enforced by the unchanged
pass of ``test_deck.py``, ``test_driver.py`` (per-body round-trip,
rtol 1e-12) and ``test_m9_coupled_build.py`` (coupled) in the full
suite.
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pytest

from floatsim.driver import build_system
from floatsim.hydro.database import HydroDatabase
from floatsim.io.deck import (
    Body,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    Output,
    RegularWave,
    Simulation,
)


def _labelled_2body_hdb(labels: tuple[str, str]) -> HydroDatabase:
    """Small 12x12 labelled coupled database (zero B). The override tests
    reach the kernel only after the rationale validation + bypass warning,
    so the kernel VALUES are irrelevant here."""
    nd = 12
    omega = np.linspace(0.1, 3.0, 6)
    nw = omega.size
    a_inf = np.eye(nd, dtype=np.float64) * 1.0e6
    c = np.eye(nd, dtype=np.float64) * 1.0e5
    a = np.stack([a_inf for _ in range(nw)], axis=-1)
    b = np.zeros((nd, nd, nw), dtype=np.float64)
    rao = np.zeros((nd, nw, 2), dtype=np.complex128)
    return HydroDatabase(
        omega=omega,
        heading_deg=np.array([0.0, 90.0]),
        A=a,
        B=b,
        A_inf=a_inf,
        C=c,
        RAO=rao,
        reference_point=np.zeros(3),
        C_source="full",
        metadata={"src": "m10-pr0-override-fixture"},
        body_labels=labels,
    )


def _coupled_deck(labels: tuple[str, str]) -> Deck:
    def _body(name: str, label: str) -> Body:
        return Body(
            name=name,
            reference_point=[0.0, 0.0, 0.0],
            mass=1.0e6,
            inertia=Inertia(Ixx=1.0e8, Iyy=1.0e8, Izz=1.0e8),
            hydro_body_label=label,
        )

    return Deck(
        simulation=Simulation(duration=10.0, dt=0.1),
        environment=Environment(water_depth=200.0, water_density=1025.0),
        waves=RegularWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[_body("body_0", labels[0]), _body("body_1", labels[1])],
        shared_hydro_database=HydroDatabaseRef(format="capytaine", path="shared.nc"),
        output=Output(file="out.h5", channels=["heave"], sample_rate=10.0),
    )


def _build(override: str | None):
    return build_system(
        _coupled_deck(("alpha", "beta")),
        bem_databases={},
        dt=0.1,
        t_max_kernel=2.0,
        solve_equilibrium=False,
        shared_hydro_database=_labelled_2body_hdb(("alpha", "beta")),
        asymptote_check_override=override,
    )


# --- GATE 3: empty / whitespace rationale still raises through the coupled path ---


def test_override_empty_rationale_raises_through_coupled_path() -> None:
    with pytest.raises(ValueError, match=r"empty or whitespace"):
        _build("")


def test_override_whitespace_rationale_raises_through_coupled_path() -> None:
    with pytest.raises(ValueError, match=r"empty or whitespace"):
        _build("   \t ")


# --- GATE 2: a valid rationale reaches the kernel INTACT (echoed in the warning) ---


def test_override_rationale_reaches_kernel_intact() -> None:
    rationale = "M10 cluster small-body hulls L~1.85 m; ITEM25-SMALL-BODY-APPLICABILITY"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # A tiny zero-B fixture may not satisfy the un-bypassable Check 3; the
        # bypass warning is emitted BEFORE that, so the plumbing is still
        # proven. (Check 3 is orthogonal to the override plumbing.)
        with contextlib.suppress(Exception):
            _build(rationale)
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any(rationale in m for m in msgs), (
        "override rationale did not reach compute_retardation_kernel intact",
        msgs,
    )
    assert any("Item 25 asymptote check bypassed" in m for m in msgs)


def test_no_override_emits_no_bypass_warning() -> None:
    """Control: with override=None (the default, byte-identity path), no
    Item-25 bypass warning is emitted -- the gate stays active."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.suppress(Exception):
            _build(None)
    msgs = [str(w.message) for w in caught]
    assert not any("Item 25 asymptote check bypassed" in m for m in msgs)
