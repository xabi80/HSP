"""FloatSim single-buoy setup for the OSU Test Buoy (measured geometry + spreadsheet mass).

Assembled with FloatSim's own per-body functions from a deck ``Body`` whose reference point is
the CoG: ``floatsim.driver._per_body_lhs`` (M + A_inf, C) and ``_build_drag_state_force`` (the
deck's Morison drag, calm water). That is the deck convention (reference point = body origin =
CoG = BEM origin), and it is how ``bem_database.py`` built ``capytaine_osu_buoy.nc``
(``rotation_center`` = CoG). The retardation kernel keeps the small-body override: the per-body
``build_system`` path has none (tracker ITEM25-SMALL-BODY-APPLICABILITY).

FIX (2026-09-23). ``build_lhs`` used to assemble about the WATERLINE, with
``cog_offset (0, 0, -0.907)`` AND the gravity term, although the BEM is already about the CoG.
That gave C55 = 265 N·m/rad instead of 73.9 and a 2.11 s pitch period; heave was unaffected
(C33, M33 and every heave coupling are identical under both assemblies). The drag geometry below
is given in the waterline frame and converted to the CoG (body) frame.

PLACEHOLDER: the heave-plate hydro (added mass in the .nc, drag Cd here) is a solid
equal-area-disc stand-in; the real perforated/webbed plate needs the tank test.
"""
from __future__ import annotations

from pathlib import Path

import floatsim.driver as fsd
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.hydro.retardation import compute_retardation_kernel
from floatsim.io.deck import (
    Body,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    Output,
    PlateMember,
    Simulation,
    distributed_cylinder_drag,
)
from floatsim.io.deck import RegularWave as DeckWave

_HERE = Path(__file__).resolve().parent
_NC = _HERE / "capytaine_osu_buoy.nc"

RHO, G = 998.0, 9.806                     # fresh water (OSU Hinsdale lab)
M_BODY = 21.52                            # total floating mass (unloaded, from spreadsheet + WL)
CoG_Z = -0.907                            # CoG in the waterline frame
I_XX = I_YY = 10.2                        # pitch/roll inertia about CoG (kg·m²), from gmsh
I_ZZ = 0.063                              # per-part inertia + lead-at-plate (uniform eff. density)
DT, DURATION, KERNEL_TMAX = 0.01, 60.0, 30.0   # matches the validated single-buoy grid/kernel

# --- drag geometry (waterline frame; converted to the CoG frame in drag_elements) ---
_SPAR_D, _SPAR_CD = 0.1593, 1.2
_SPAR_BOT, _WL = -0.967, 0.0
_PLATE_Z, _PLATE_R = -1.383, 0.1437       # equal-area disc (placeholder)
_PLATE_CD_N, _PLATE_CD_T, _PLATE_T = 5.0, 1.5, 0.0039
_OVR = "OSU test buoy: spar + placeholder plate, small body, B not fully asymptotic at omega_max"


def load_hdb():  # type: ignore[no-untyped-def]
    return read_capytaine(_NC)


def drag_elements(n_seg: int = 10, *, plate_z: float = _PLATE_Z) -> list:
    """Spar (distributed, transverse) + heave plate, as deck elements in the CoG frame."""
    spar = distributed_cylinder_drag(z_bottom=_SPAR_BOT - CoG_Z, z_top=_WL - CoG_Z,
                                     diameter=_SPAR_D, cd=_SPAR_CD, n_segments=n_seg)
    plate = PlateMember(type="plate", center=[0.0, 0.0, plate_z - CoG_Z], normal=[0.0, 0.0, 1.0],
                        radius=_PLATE_R, thickness=_PLATE_T, Cd_n=_PLATE_CD_N, Cd_t=_PLATE_CD_T)
    return [*spar, plate]


def body(*, drag: list | None = None, nc: Path = _NC) -> Body:
    """The buoy as a FloatSim deck body: reference point = CoG = BEM origin."""
    return Body(name="osu_buoy", reference_point=[0.0, 0.0, CoG_Z], mass=M_BODY,
                inertia=Inertia(Ixx=I_XX, Iyy=I_YY, Izz=I_ZZ),
                hydro_database=HydroDatabaseRef(format="capytaine", path=str(nc)),
                drag_elements=drag or [])


def deck(*, drag: list | None = None, nc: Path = _NC) -> Deck:
    return Deck(simulation=Simulation(duration=DURATION, dt=DT),
                environment=Environment(water_depth=200.0, water_density=RHO, gravity=G),
                waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
                bodies=[body(drag=drag, nc=nc)],
                output=Output(file="o.h5", channels=["heave"], sample_rate=10.0))


def build_lhs(hdb, nc: Path = _NC):  # type: ignore[no-untyped-def]
    """M + A_inf and C from FloatSim's per-body assembly (reference point = CoG)."""
    return fsd._per_body_lhs(body(nc=nc), hdb, gravity=G)


def build_kernel(hdb):  # type: ignore[no-untyped-def]
    return compute_retardation_kernel(hdb, t_max=KERNEL_TMAX, dt=DT,
                                      asymptote_check_override=_OVR,
                                      kernel_decay_floor_override=_OVR)


def make_drag(n_seg: int = 10, *, plate_z: float = _PLATE_Z):  # type: ignore[no-untyped-def]
    """The deck's calm-water Morison drag, through FloatSim's drag wiring."""
    return fsd._build_drag_state_force(deck(drag=drag_elements(n_seg, plate_z=plate_z)),
                                       n_dof=6, rho=RHO)
