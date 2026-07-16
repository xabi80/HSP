"""Hand-assembled Cummins system for the 3-buoy cluster decay.

Like the spar-fin study, `floatsim.driver.build_system` is unusable:
it calls `compute_retardation_kernel` without the Item-25 override the
small-body composite BEM needs (std/mean of B*omega^4 fails the 0.10
gate). Study-side assembly only; no `floatsim/` change. The deck YAMLs
(deck_bem_only.yaml, deck_bem_morison.yaml) are the faithful config
record; this module mirrors them for the actual run.

Reference point = composite CoG (heave is reference-independent, so
this is exact for the heave-only study; pitch/roll are not exercised).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from floatsim.bodies.mass_properties import rigid_body_mass_matrix
from floatsim.hydro.morison import MorisonElement, make_morison_state_force
from floatsim.hydro.radiation import CumminsLHS, assemble_cummins_lhs
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.hydro.retardation import RetardationKernel, compute_retardation_kernel

import cluster_common as cc

_HERE = Path(__file__).resolve().parent
_COMP_NC = _HERE / "composite_bem.nc"

DT = 0.01
DURATION = 50.0
KERNEL_TMAX = 30.0
IC_HEAVE = 0.10

# Morison heave plate for the whole cluster: a single degenerate
# horizontal cylinder with projected area D*L = 3 * A_plate = 0.4356
# m^2, Cd = 5.0, on the cluster axis at the plate depth. For pure heave
# of a rigid symmetric body this is equivalent to three offset elements
# (each 0.1452 m^2) -- the three vertical drag forces sum and, being on
# a symmetric ring, produce zero net moment. Single-element form chosen
# for simplicity; the equivalence is exact for pure heave.
PLATE_CD = 5.0
PLATE_AREA_CLUSTER = 3.0 * 0.1452   # 0.4356 m^2
PLATE_L = 0.86                       # spans the cluster (~2 x radius)
PLATE_D = PLATE_AREA_CLUSTER / PLATE_L
PLATE_Z = -1.45                      # cluster-frame plate depth (near mesh bottom)

_OVERRIDE = (
    "3-buoy cluster study: small-body hulls (L~1.85 m); 1/omega^4 regime "
    "not reached at omega_max=30; see ITEM25-SMALL-BODY-APPLICABILITY"
)


def _props() -> dict:
    return json.loads((_HERE / "results" / "mass_properties.json").read_text())


def load_hdb():
    return read_capytaine(_COMP_NC)


def build_lhs(hdb) -> CumminsLHS:
    """Composite M + A_inf and C, reference = composite CoG."""
    p = _props()
    I_cog = np.array(p["inertia_about_cog"], dtype=np.float64)
    M = rigid_body_mass_matrix(
        mass=cc.M_CLUSTER, inertia_at_reference=I_cog, cog_offset_body=None
    )
    # Reference = CoG => cog_offset_from_bem_origin = 0. Heave restoring
    # C[2,2] = rho*g*A_wp is reference-independent; the (unexercised)
    # pitch/roll gravity term is therefore zero here by construction.
    return assemble_cummins_lhs(
        rigid_body_mass=M, hdb=hdb, mass=cc.M_CLUSTER,
        cog_offset_from_bem_origin=np.zeros(3), gravity=cc.G,
    )


def build_kernel(hdb) -> RetardationKernel:
    return compute_retardation_kernel(
        hdb, t_max=KERNEL_TMAX, dt=DT, asymptote_check_override=_OVERRIDE
    )


def make_morison_force():
    elem = MorisonElement(
        body_index=0,
        node_a_body=np.array([-PLATE_L / 2.0, 0.0, PLATE_Z]),
        node_b_body=np.array([+PLATE_L / 2.0, 0.0, PLATE_Z]),
        diameter=PLATE_D, Cd=PLATE_CD, Ca=0.0, include_inertia=False,
    )

    def calm(_pt: NDArray[np.float64], _t: float) -> NDArray[np.float64]:
        return np.zeros(3, dtype=np.float64)

    return make_morison_state_force([elem], n_dof=6, fluid_velocity_fn=calm, rho=cc.RHO)
