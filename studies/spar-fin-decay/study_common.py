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
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from floatsim.bodies.mass_properties import rigid_body_mass_matrix
from floatsim.hydro.morison import MorisonElement, make_morison_state_force
from floatsim.hydro.radiation import CumminsLHS, assemble_cummins_lhs
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.hydro.retardation import RetardationKernel, compute_retardation_kernel

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

_OVERRIDE = (
    "spar-fin study resumption: small-body L~1.85 m, 1/omega^4 regime "
    "not reached at omega_max=30; see ITEM25-SMALL-BODY-APPLICABILITY"
)


def load_hdb():
    """Read the eqdraft BEM database (reader symmetrizes A/B internally)."""
    return read_capytaine(_NC)


def build_lhs(hdb) -> CumminsLHS:
    """Single-body M + A_inf and C (buoyancy_only C -> add gravity term)."""
    r = np.array([0.0, 0.0, CoG_Z], dtype=np.float64)
    # Inertia at the reference point (mesh origin) via parallel axis.
    i_cog = np.diag([I_XX, I_YY, I_ZZ]).astype(np.float64)
    r2 = float(r @ r)
    i_ref = i_cog + M_BODY * (r2 * np.eye(3) - np.outer(r, r))
    M = rigid_body_mass_matrix(
        mass=M_BODY, inertia_at_reference=i_ref, cog_offset_body=r
    )
    return assemble_cummins_lhs(
        rigid_body_mass=M,
        hdb=hdb,
        mass=M_BODY,
        cog_offset_from_bem_origin=r,
        gravity=G,
    )


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
