"""Milestone 1.5 validation — OrcaFlex YAML fixture round-trip.

Load the committed OrcaFlex VesselType YAML fixture (an OC4 DeepCwind-shaped
semi-submersible exported by OrcaFlex 11.2c), assemble the Cummins LHS with
an OC4-shaped rigid-body mass matrix, and verify that the uncoupled natural
periods computed via :func:`natural_periods_uncoupled` fall inside the same
physical ranges used by the pure-synthetic OC4 validation
(:mod:`tests.validation.test_oc4_natural_periods`).

This exercises the full M1.5 wiring: YAML parse, unit conversion
(OrcaFlex "SI" tonnes/kN -> pure SI), HydroDatabase invariants, and the
downstream Cummins assembly. If the reader's unit scale or DOF mapping
drifts, one of the period checks will trip long before a subtle
downstream regression can hide it.

Why these ranges
----------------
The fixture's diagonal ``C`` and ``A_inf`` entries are close to the
published OC4 DeepCwind values (heave C_33 ~ 3.65e6 N/m vs 3.836e6;
pitch C_55 ~ 9.97e8 N*m/rad vs 1.078e9; A_inf heave ~ 1.45e7 kg,
A_inf pitch ~ 7.27e9 kg*m^2). Using the OC4 platform mass and
parallel-axis-corrected inertia, the uncoupled-at-A_inf formula
yields ~17 s heave and ~25 s pitch — well inside the documented
14-20 s / 22-32 s bands from
:mod:`tests.validation.test_oc4_natural_periods`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.radiation import (
    assemble_cummins_lhs,
    natural_periods_uncoupled,
)
from floatsim.hydro.readers.orcaflex_vessel_yaml import read_orcaflex_vessel_yaml
from tests.validation.test_oc4_natural_periods import (
    HEAVE_PERIOD_RANGE_S,
    PITCH_PERIOD_RANGE_S,
    _oc4_rigid_body_mass_matrix,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "bem" / "orcaflex" / "platform_small.yml"
)


def _load_fixture_hdb() -> HydroDatabase:
    return read_orcaflex_vessel_yaml(FIXTURE_PATH)


def test_roundtrip_heave_period_in_physical_range() -> None:
    """Fixture + OC4 mass -> heave period inside 14-20 s band.

    Confirms unit conversion (tonnes -> kg, kN/m -> N/m) and DOF indexing
    of the reader, by funnelling the fixture through assemble_cummins_lhs
    and verifying the resulting heave period matches the OC4 physical range.
    """
    hdb = _load_fixture_hdb()
    lhs = assemble_cummins_lhs(rigid_body_mass=_oc4_rigid_body_mass_matrix(), hdb=hdb)
    periods = natural_periods_uncoupled(lhs)

    t_heave = float(periods[2])
    lo, hi = HEAVE_PERIOD_RANGE_S
    assert lo <= t_heave <= hi, (
        f"heave period {t_heave:.2f} s from fixture outside [{lo}, {hi}] s; "
        "likely a unit-scale or DOF-mapping regression in the YAML reader."
    )


def test_roundtrip_pitch_period_in_physical_range() -> None:
    """Fixture + OC4 mass -> pitch period inside 22-32 s band."""
    hdb = _load_fixture_hdb()
    lhs = assemble_cummins_lhs(rigid_body_mass=_oc4_rigid_body_mass_matrix(), hdb=hdb)
    periods = natural_periods_uncoupled(lhs)

    t_pitch = float(periods[4])
    lo, hi = PITCH_PERIOD_RANGE_S
    assert lo <= t_pitch <= hi, (
        f"pitch period {t_pitch:.2f} s from fixture outside [{lo}, {hi}] s; "
        "likely a unit-scale or DOF-mapping regression in the YAML reader."
    )


def test_roundtrip_horizontal_and_yaw_periods_are_nan() -> None:
    """Fixture has C[i,i] == 0 for surge, sway, yaw — periods must be NaN."""
    hdb = _load_fixture_hdb()
    lhs = assemble_cummins_lhs(rigid_body_mass=_oc4_rigid_body_mass_matrix(), hdb=hdb)
    periods = natural_periods_uncoupled(lhs)
    for i in (0, 1, 5):
        assert np.isnan(periods[i])
