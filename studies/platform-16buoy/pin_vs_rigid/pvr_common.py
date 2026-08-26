"""Shared builders for the pin-vs-rigid 16-buoy study.

Two configurations off the SAME deck (same buoys, hydro, drag, geometry) -- only
the joint type changes:
  * articulated -- every joint ``yaw_locked`` (roll/pitch free): the current model.
  * rigid       -- every joint ``rigid`` (weld): the whole 21-body assembly is one
                   rigid raft.

The rigid deck is the articulated deck with its joints swapped, so the two are
byte-identical everywhere else. Reuses ``platform16_rao._deck_with_drag`` (drag +
geometry) and ``build_system`` (the coupled Cummins + KKT assembly). See
STUDY-PLAN.md.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # platform-16buoy (platform16_rao, platform16_common)
sys.path.insert(0, str(_HERE.parent.parent / "cluster-3buoy-rigid"))

import platform16_rao as prp16  # noqa: E402

from floatsim.driver import build_system  # noqa: E402
from floatsim.hydro.readers.capytaine import read_capytaine  # noqa: E402
from floatsim.io.deck import RigidJoint  # noqa: E402

_STUDY16 = prp16._HERE / "fin_study"
_REF = prp16._HERE.parent / "cluster-3buoy-rigid" / "reference_single_bem.nc"
_ASYMPTOTE_OVR = prp16._ASYMPTOTE_OVR
_KERNEL_OVR = "platform16 pin-vs-rigid small-body hulls"


def load_hdb(tag: str = "0215"):  # type: ignore[no-untyped-def]
    """Coupled 16-buoy array BEM for a fin variant (default 0.215 m fin)."""
    return read_capytaine(_STUDY16 / f"capytaine_platform16_fin{tag}.nc")


def _to_rigid(j):  # type: ignore[no-untyped-def]
    return RigidJoint(
        type="rigid", body_a=j.body_a, body_b=j.body_b,
        attach_a_body=list(j.attach_a_body), attach_b_body=list(j.attach_b_body),
        axis=list(j.axis),
    )


def build_deck(rigid: bool):  # type: ignore[no-untyped-def]
    """The 21-body / 20-joint 16-buoy deck; joints ``rigid`` if ``rigid`` else
    ``yaw_locked`` (everything else identical)."""
    deck = prp16._deck_with_drag()
    if not rigid:
        return deck
    return deck.model_copy(update={"joints": [_to_rigid(j) for j in deck.joints]})


def build_setup(rigid: bool, hdb):  # type: ignore[no-untyped-def]
    """Coupled Cummins + KKT setup (lhs/kernel/constraints/state_force/xi0) for the
    articulated or whole-chain-rigid platform. ``solve_equilibrium=False`` (RAOs
    about the reference), matching the fin fan."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_system(
            build_deck(rigid), bem_databases={}, dt=0.01, t_max_kernel=60.0,
            solve_equilibrium=False, shared_hydro_database=hdb,
            hydrostatic_database=read_capytaine(_REF),
            asymptote_check_override=_ASYMPTOTE_OVR,
            kernel_decay_floor_override=_KERNEL_OVR,
        )
