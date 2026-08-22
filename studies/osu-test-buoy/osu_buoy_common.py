"""FloatSim single-buoy setup for the OSU Test Buoy (measured geometry + spreadsheet mass).

Mirrors studies/spar-fin-decay/study_common.py (hand-assembled Cummins system + kernel
override for a small body) but with the OSU constants and the placeholder BEM database
(capytaine_osu_buoy.nc). Frame: still water = z=0, z up.

PLACEHOLDER: the heave-plate hydro (added mass in the .nc, drag Cd here) is a solid
equal-area-disc stand-in; the real perforated/webbed plate needs the tank test.
Pitch/roll inertia is a rough estimate (structure-as-rod + lead-at-plate).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from floatsim.bodies.mass_properties import rigid_body_mass_matrix
from floatsim.hydro.morison import MorisonElement, PlateDragElement, make_morison_state_force
from floatsim.hydro.radiation import assemble_cummins_lhs
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.hydro.retardation import compute_retardation_kernel

_HERE = Path(__file__).resolve().parent
_NC = _HERE / "capytaine_osu_buoy.nc"

RHO, G = 998.0, 9.806                     # fresh water (OSU Hinsdale lab)
M_BODY = 21.52                            # total floating mass (unloaded, from spreadsheet + WL)
CoG_Z = -0.907                            # CoG in the waterline frame
I_XX = I_YY = 10.0                        # pitch/roll inertia estimate (kg·m²) -- PLACEHOLDER
I_ZZ = 0.12
DT, DURATION, KERNEL_TMAX = 0.01, 60.0, 20.0   # t_max <= Nyquist (π/dω≈21 s) for this ω grid

# --- drag geometry (waterline frame) ---
_SPAR_D, _SPAR_CD = 0.1593, 1.2
_SPAR_BOT, _WL = -0.967, 0.0
_PLATE_Z, _PLATE_R = -1.383, 0.1437       # equal-area disc (placeholder)
_PLATE_CD_N, _PLATE_CD_T, _PLATE_T = 5.0, 1.5, 0.0039
_OVR = "OSU test buoy: spar + placeholder plate, small body, B not fully asymptotic at omega_max"


def load_hdb():  # type: ignore[no-untyped-def]
    return read_capytaine(_NC)


def build_lhs(hdb):  # type: ignore[no-untyped-def]
    r = np.array([0.0, 0.0, CoG_Z])
    i_ref = np.diag([I_XX, I_YY, I_ZZ]) + M_BODY * ((r @ r) * np.eye(3) - np.outer(r, r))
    M = rigid_body_mass_matrix(mass=M_BODY, inertia_at_reference=i_ref, cog_offset_body=r)
    return assemble_cummins_lhs(rigid_body_mass=M, hdb=hdb, mass=M_BODY,
                                cog_offset_from_bem_origin=r, gravity=G)


def build_kernel(hdb):  # type: ignore[no-untyped-def]
    return compute_retardation_kernel(hdb, t_max=KERNEL_TMAX, dt=DT,
                                      asymptote_check_override=_OVR, kernel_decay_floor_override=_OVR)


def make_drag(n_seg: int = 10):  # type: ignore[no-untyped-def]
    """Calm-water Morison drag: distributed spar (transverse) + heave-plate (placeholder)."""
    elems: list = []
    edges = np.linspace(_SPAR_BOT, _WL, n_seg + 1)
    for i in range(n_seg):
        elems.append(MorisonElement(body_index=0, node_a_body=np.array([0.0, 0.0, edges[i]]),
                                     node_b_body=np.array([0.0, 0.0, edges[i + 1]]),
                                     diameter=_SPAR_D, Cd=_SPAR_CD))
    elems.append(PlateDragElement(body_index=0, center_body=np.array([0.0, 0.0, _PLATE_Z]),
                                  normal_body=np.array([0.0, 0.0, 1.0]), radius=_PLATE_R,
                                  thickness=_PLATE_T, Cd_n=_PLATE_CD_N, Cd_t=_PLATE_CD_T))
    return make_morison_state_force(elems, n_dof=6, fluid_velocity_fn=lambda p, t: np.zeros(3), rho=RHO)
