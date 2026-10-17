"""Weight-preload Gate A fixture: a floating body pinned to earth at its top
(docs/weight-preload-plan.md §3, Gate A).

A hydro body with a synthetic database (A = 0, a small smooth B, diagonal C about its CoG,
``C_source = "full"``) hangs from a yaw-locked joint to earth at ``L`` above its CoG. Its
displaced volume is set so the net static force is ``P = rho g V - m g`` (up). Heave is held by
the pin. Small rotation theta about the pin:
- the CoG swings horizontally by ``L theta`` and rises by ``L theta^2 / 2``;
- raising the body against its net upward force P stores ``-P L theta^2 / 2``;
- so ``(I_yy + m L^2) theta'' = -(C55 - P L) theta`` and
  ``T = 2 pi sqrt((I_yy + m L^2) / (C55 - P L))``.

Numbers: the OSU buoy (studies/payload-stability, 12-panel table). m = 21.52 kg,
I_yy = 10.2 kg m^2, L = 1.624 m, C55 = 75.28 N m/rad. T = 9.976 s (P = +30 N),
4.617 s (P = -30 N), 5.926 s (P = 0).

Finite amplitude. The pin holds the CoG at ``L`` below it, so the body rises by
``L (1 - cos theta)`` against its heave stiffness C33, and the potential is
``U = C55 theta^2 / 2 - P L (1 - cos theta) + C33 (L (1 - cos theta))^2 / 2``. Its quartic
coefficient is ``eps = P L / 24 + C33 L^2 / 8``. The kinetic energy is exactly
``(I_yy + m L^2) theta'^2 / 2`` (pure rotation about the fixed pin). By Duffing's first-order
result the period shortens by ``3 eps theta0^2 / (2 k)``, ``k = C55 - P L``: 3.58 theta0^2 at
P = +30 N.
"""

from __future__ import annotations

import numpy as np

from floatsim.driver import SimulationSetup, build_system
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
    YawLockedJoint,
)
from tests.support.synthetic_bem import make_diagonal_hdb, well_behaved_b

RHO, G = 998.0, 9.806
MASS, I_XX, I_YY, I_ZZ = 21.52, 10.2, 10.2, 0.063
L_PIN = 1.624
C33, C55 = 186.26, 75.28
Z_REF = -0.907
DT, T_MAX_KERNEL = 0.01, 30.0
_CUTOFF = 5.0


def analytic_period(net_up_n: float) -> float:
    """Gate A closed form: pitch period about the pin for a net upward static force P."""
    return float(2.0 * np.pi * np.sqrt((I_YY + MASS * L_PIN**2) / (C55 - net_up_n * L_PIN)))


def duffing_shift(net_up_n: float, theta0: float) -> float:
    """Relative period change at amplitude theta0 (first order; see the module docstring)."""
    eps = net_up_n * L_PIN / 24.0 + C33 * L_PIN**2 / 8.0
    return -3.0 * eps * theta0**2 / (2.0 * (C55 - net_up_n * L_PIN))


def volume_for(net_up_n: float) -> float:
    """Displaced volume giving a net upward static force P."""
    return (MASS * G + net_up_n) / (RHO * G)


def hdb() -> HydroDatabase:
    """Synthetic database with ``A = 0``. ``B`` must stay tiny for that to be consistent: the
    radiation convolution of a non-zero ``B(omega)`` implies, by Kramers-Kronig, an added mass
    ``A(omega) - A_inf = (2/pi) PV int B(w') / (w'^2 - omega^2) dw'``. For a flat ``B = 1`` up to
    5 rad/s that is about -0.13 kg (kg m^2) at 1 rad/s: about -0.47 kg m^2 on this pendulum's
    67 kg m^2, a period 0.35 % short, which was measured. ``B = 1e-3`` makes it ~4e-6 relative."""
    omega = np.linspace(0.0, 4.0 * _CUTOFF, 401)
    b = well_behaved_b(omega, band_value=1.0e-3, cutoff_omega=_CUTOFF)
    return make_diagonal_hdb(
        A_inf_diag=[0.0] * 6,
        C_diag=[0.0, 0.0, C33, C55, C55, 0.0],
        A_diag_per_omega=[[0.0] * 6 for _ in omega],
        B_diag_per_omega=[[float(v)] * 6 for v in b],
        omega=omega.tolist(),
        reference_point=(0.0, 0.0, Z_REF),
        C_source="full",
    )


def deck(net_up_n: float | None, *, preload: bool = True, duration: float = 40.0) -> Deck:
    """The pinned float; ``net_up_n = None`` leaves ``displaced_volume`` at its default."""
    body = Body(
        name="float",
        reference_point=[0.0, 0.0, Z_REF],
        mass=MASS,
        inertia=Inertia(Ixx=I_XX, Iyy=I_YY, Izz=I_ZZ),
        hydro_database=HydroDatabaseRef(format="capytaine", path="synthetic"),
        displaced_volume=None if net_up_n is None else volume_for(net_up_n),
    )
    pin = YawLockedJoint(
        type="yaw_locked",
        body_a="float",
        body_b="earth",
        attach_a_body=[0.0, 0.0, L_PIN],
        attach_b_body=[0.0, 0.0, Z_REF + L_PIN],  # earth side: the fixed world anchor
        axis=[0.0, 0.0, 1.0],
    )
    return Deck(
        simulation=Simulation(duration=duration, dt=DT, weight_preload=preload),
        environment=Environment(water_depth=200.0, water_density=RHO, gravity=G),
        waves=RegularWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[body],
        joints=[pin],
        output=Output(file="o.h5", channels=["pitch"], sample_rate=10.0),
    )


def setup(net_up_n: float | None, *, preload: bool = True) -> SimulationSetup:
    return build_system(
        deck(net_up_n, preload=preload),
        bem_databases={"float": hdb()},
        dt=DT,
        t_max_kernel=T_MAX_KERNEL,
        solve_equilibrium=True,
    )
