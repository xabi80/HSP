"""Shared setup for the spar-fin free-decay study (Steps C-G).

Hand-assembles the single-body Cummins system because the deck-driven
`floatsim.driver.build_system` calls `compute_retardation_kernel`
WITHOUT the Item-25 override, which the small-body spar-fin BEM
(std/mean of B*omega^4 = 0.60 > 0.10 gate) cannot pass. Study-side
assembly only; no floatsim/ modification.

Locked inputs (studies/spar-fin-decay/README.md), updated at the M7.5
resumption to the true equilibrium draft:
  - eqdraft mesh + BEM (test2_spar_fin_fullfix_eqdraft.gdf), waterline
    at z=0, buoy sinks dz=0.1846 m from design draft.
  - CoG z in eqdraft frame = -0.8317 - 0.1846 = -1.0163 m.

Reference point (FIX 2026-09-24). ``build_lhs`` used to assemble about the
mesh origin (the WATERLINE), with ``cog_offset (0, 0, CoG_Z)`` AND the
gravity term, although capytaine_bem.nc (and every sparfin_fin_bem.py
database) is solved about the CoG (``rotation_center`` = CoG). That gave
C44 = C55 = 393.39 instead of 107.56 N*m/rad (28.67 * 9.81 * 1.0163 = 285.8
added) and the wrong M55 / M15. It now uses FloatSim's own per-body assembly
(``floatsim.driver._per_body_lhs``) on a deck body whose reference point is
the CoG: the deck convention (reference point = body origin = CoG = BEM
origin). Heave is exactly unaffected (identical C33, M33 and heave
couplings). Drag geometry that acts in pitch/surge must be given in the
body (CoG) frame: use ``PLATE_Z_B`` / ``WL_Z_B``. ``make_morison_force`` is
the heave-only plate surrogate and keeps its mesh-frame z (irrelevant in
heave; pinned by tests/validation/test_m11a_pr1_drag_wiring.py). Tracker
STUDY-HYDROSTATIC-REFERENCE-POINT.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import floatsim.driver as fsd
from floatsim.hydro.morison import MorisonElement, make_morison_state_force
from floatsim.hydro.radiation import CumminsLHS
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.hydro.retardation import RetardationKernel, compute_retardation_kernel
from floatsim.io.deck import Body, HydroDatabaseRef, Inertia

_HERE = Path(__file__).resolve().parent
_NC = _HERE / "capytaine_bem.nc"

# --- Locked physical inputs ---
RHO = 1025.0
G = 9.81
M_BODY = 28.67
CoG_Z = -1.0163  # eqdraft frame (mesh origin at waterline)
I_XX = I_YY = 24.0    # kg*m^2 at CoM
I_ZZ = 0.114
IC_HEAVE = 0.10      # m, initial heave displacement for decay

# --- Simulation ---
DT = 0.01
DURATION = 50.0
KERNEL_TMAX = 30.0

# --- Morison heave plate (degenerate horizontal-cylinder approximation) ---
# See STEP-A-FINDING addendum / Pre-flight 3 audit: FloatSim's Morison
# element is a slender cylinder that drags on the velocity component
# NORMAL to its axis. A heave plate's vertical drag is reproduced by a
# HORIZONTAL cylinder (axis in the horizontal plane) with projected
# area D*L = A_plate: for pure heave the normal velocity is v_z and
# F_z = 0.5*rho*Cd*(D*L)*|v_z|*v_z, matching the plate drag exactly.
PLATE_CD = 5.0
PLATE_AREA = 0.1452          # m^2 = pi*(0.215)^2
PLATE_RADIUS = 0.215
PLATE_L = 2.0 * PLATE_RADIUS  # 0.43 m (cylinder spans the plate diameter)
PLATE_D = PLATE_AREA / PLATE_L  # so D*L = A_plate
PLATE_Z = -1.278             # eqdraft plate z (mesh bottom); heave-irrelevant
# Body (CoG) frame positions for drag that acts in pitch/surge (reference = CoG).
PLATE_Z_B = PLATE_Z - CoG_Z  # -0.2617 (the M11a PR4 committed plate depth)
WL_Z_B = 0.0 - CoG_Z         # +1.0163, the waterline

_OVERRIDE = (
    "spar-fin study resumption: small-body L~1.85 m, 1/omega^4 regime "
    "not reached at omega_max=30; see ITEM25-SMALL-BODY-APPLICABILITY"
)


def load_hdb():
    """Read the eqdraft BEM database (reader symmetrizes A/B internally)."""
    return read_capytaine(_NC)


def body() -> Body:
    """The buoy as a FloatSim deck body: reference point = CoG = BEM origin."""
    return Body(
        name="spar_fin",
        reference_point=[0.0, 0.0, CoG_Z],
        mass=M_BODY,
        inertia=Inertia(Ixx=I_XX, Iyy=I_YY, Izz=I_ZZ),
        hydro_database=HydroDatabaseRef(format="capytaine", path=str(_NC)),
    )


def build_lhs(hdb) -> CumminsLHS:
    """Single-body M + A_inf and C from FloatSim's per-body assembly
    (reference point = CoG; gravity term with zero CoG offset)."""
    return fsd._per_body_lhs(body(), hdb, gravity=G)


def build_kernel(hdb) -> RetardationKernel:
    """Retardation kernel with the small-body Item-25 override."""
    return compute_retardation_kernel(
        hdb, t_max=KERNEL_TMAX, dt=DT, asymptote_check_override=_OVERRIDE
    )


def make_morison_force():
    """Calm-sea Morison state-force closure for the heave plate."""
    elem = MorisonElement(
        body_index=0,
        node_a_body=np.array([-PLATE_L / 2.0, 0.0, PLATE_Z]),
        node_b_body=np.array([+PLATE_L / 2.0, 0.0, PLATE_Z]),
        diameter=PLATE_D,
        Cd=PLATE_CD,
        Ca=0.0,
        include_inertia=False,
    )

    def calm(_point: NDArray[np.float64], _t: float) -> NDArray[np.float64]:
        return np.zeros(3, dtype=np.float64)

    return make_morison_state_force([elem], n_dof=6, fluid_velocity_fn=calm, rho=RHO)
