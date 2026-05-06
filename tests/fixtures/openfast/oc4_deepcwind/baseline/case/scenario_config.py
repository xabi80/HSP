"""Scenario definitions for the M6 OpenFAST cross-check.

Each scenario is a frozen dataclass describing what differs from the OC4
DeepCwind baseline. The deck generator consumes this and produces per-
scenario OpenFAST input files.

Convention: edits are sparse — only fields that differ from baseline are
listed. The generator preserves everything else verbatim from the
vendored baseline.

Reference: docs/milestone-6-plan.md (M6 plan v2, OpenFAST/HydroDyn).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Turbine structural DOFs to lock for every scenario. These are all
# ElastoDyn flags. Platform DOFs (PtfmSgDOF, PtfmSwDOF, PtfmHvDOF,
# PtfmRDOF, PtfmPDOF, PtfmYDOF) are scenario-specific and set below.
_LOCKED_TURBINE_DOFS: dict[str, bool] = {
    "FlapDOF1": False,
    "FlapDOF2": False,
    "EdgeDOF": False,
    "TeetDOF": False,
    "DrTrDOF": False,
    "GenDOF": False,
    "YawDOF": False,
    "TwFADOF1": False,
    "TwFADOF2": False,
    "TwSSDOF1": False,
    "TwSSDOF2": False,
}


# Module-level disables. Wind turbine machinery is off for every M6
# scenario — FloatSim does not model aero/inflow/control.
_DISABLE_TURBINE_MODULES: dict[str, int] = {
    "CompAero": 0,
    "CompInflow": 0,
    "CompServo": 0,
}


# QTF disables. FloatSim is first-order only in Phase 1; the OC4
# baseline deck has full QTFs active. Disable for fair cross-check.
_DISABLE_QTFS: dict[str, int] = {
    "MnDrift": 0,
    "NewmanApp": 0,
    "DiffQTF": 0,
    "SumQTF": 0,
}


@dataclass(frozen=True)
class Scenario:
    """One M6 cross-check scenario.

    Attributes
    ----------
    name : str
        Short identifier, used as directory name (e.g. 's1_static_eq').
    purpose : str
        One-line description for logs and the M6 report.
    fst_edits : dict
        Top-level .fst overrides (CompXxx flags, TMax, DT, ...).
    elastodyn_edits : dict
        ElastoDyn .dat overrides (DOFs, initial conditions).
    hydrodyn_edits : dict
        HydroDyn .dat overrides (WaveMod, WaveHs, WaveTp, ...).
    moordyn_active : bool
        Whether to enable MoorDyn (CompMooring=1) for this scenario.
        OC4 baseline has 3 catenary lines; mooring scenarios use them
        as-is, non-mooring scenarios disable to isolate hydrodynamics.
    sweep_param : tuple[str, list] | None
        If set, generate one deck per value of the swept parameter.
        Used by S3 for the RAO frequency sweep.
        Tuple form: (parameter_name, [value1, value2, ...]).
    output_channels : tuple[str, ...]
        OpenFAST output channels that must be written for this
        scenario. Channels are configured per-module (in HydroDyn.dat
        and ElastoDyn.dat OutList sections).
    """

    name: str
    purpose: str
    fst_edits: dict[str, float | int | str] = field(default_factory=dict)
    elastodyn_edits: dict[str, float | int | bool | str] = field(default_factory=dict)
    hydrodyn_edits: dict[str, float | int | str] = field(default_factory=dict)
    moordyn_active: bool = False
    # When True, generate_scenario_decks.py post-processes the HydroDyn
    # fst_vt to zero ALL Morison drag coefficients (CylMemberCd1/2,
    # CylMemberCdMG1/2, CylMemberAxCd1/2, CylMemberAxCdMG1/2, AxCd,
    # CylSimplCd*). The members and joints stay in place (kinematics
    # still computed; member buoyancy via PropPot still runs through
    # the BEM), but quadratic drag forces vanish. Used by S2 to
    # isolate radiation-only physics for the M6 PR3 cross-check
    # (see docs/diagnostics/m6-pr3-damping-stability.md).
    morison_drag_disabled: bool = False
    sweep_param: tuple[str, list[float]] | None = None
    output_channels: tuple[str, ...] = ()


# ---------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------

# Baseline channel set every scenario writes. Per-scenario channels add
# to this.
_BASE_CHANNELS: tuple[str, ...] = (
    "Time",
    "PtfmSurge", "PtfmSway", "PtfmHeave",
    "PtfmRoll", "PtfmPitch", "PtfmYaw",
    "PtfmTVxt", "PtfmTVyt", "PtfmTVzt",
    "PtfmRVxt", "PtfmRVyt", "PtfmRVzt",
)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="s1_static_eq",
        purpose=(
            "Static equilibrium under gravity + buoyancy, no waves, "
            "no mooring. Cross-checks hydrostatic decomposition "
            "(conventions doc Item 5)."
        ),
        fst_edits={
            **_DISABLE_TURBINE_MODULES,
            "CompMooring": 0,
            "TMax": 200.0,
        },
        elastodyn_edits={
            **_LOCKED_TURBINE_DOFS,
            "PtfmSgDOF": True,
            "PtfmSwDOF": True,
            "PtfmHvDOF": True,
            "PtfmRDOF": True,
            "PtfmPDOF": True,
            "PtfmYDOF": True,
            # Initial conditions: zero offset; let solver settle.
            "PtfmSurge": 0.0,
            "PtfmSway": 0.0,
            "PtfmHeave": 0.0,
            "PtfmRoll": 0.0,
            "PtfmPitch": 0.0,
            "PtfmYaw": 0.0,
        },
        hydrodyn_edits={
            "WaveMod": 0,  # still water
            **_DISABLE_QTFS,
        },
        moordyn_active=False,
        output_channels=_BASE_CHANNELS,
    ),
    Scenario(
        name="s2_pitch_decay",
        purpose=(
            "Free-decay from 5deg pitch offset, no waves, no mooring, "
            "Morison drag disabled. Cross-checks pitch natural period "
            "and radiation-only response (M6 PR3, post-Option-A)."
        ),
        fst_edits={
            **_DISABLE_TURBINE_MODULES,
            "CompMooring": 0,
            "TMax": 600.0,
        },
        elastodyn_edits={
            **_LOCKED_TURBINE_DOFS,
            "PtfmSgDOF": True,
            "PtfmSwDOF": True,
            "PtfmHvDOF": True,
            "PtfmRDOF": True,
            "PtfmPDOF": True,
            "PtfmYDOF": True,
            # Override the baseline ElastoDyn template (which carries
            # PtfmSurge=5.0 m as a non-zero default). Surge has zero
            # hydrostatic stiffness in unmoored OC4, so a non-zero IC
            # would drift indefinitely and contaminate the pitch
            # response through cross-coupling. Mod 1 of M6 PR3 plan
            # locks zero IC on unrestored DOFs in free-decay tests
            # (analogous to Item 14 for static eq).
            "PtfmSurge": 0.0,
            "PtfmPitch": 5.0,  # initial pitch offset, degrees
        },
        hydrodyn_edits={
            "WaveMod": 0,
            **_DISABLE_QTFS,
        },
        # Morison drag disabled: Mod 2 of the M6 PR3 plan classified
        # S2's quadratic-drag damping as amplitude-dependent (hyperbolic
        # envelope; docs/diagnostics/m6-pr3-damping-stability.md),
        # invalidating a tight zeta cross-check against FloatSim's
        # radiation-only setup. Disabling drag here makes S2 a clean
        # radiation-only cross-check on both sides. Quantitative drag
        # damping cross-check moves to S5 (M6 PR6).
        morison_drag_disabled=True,
        moordyn_active=False,
        output_channels=_BASE_CHANNELS,
    ),
    Scenario(
        name="s3_rao_sweep",
        purpose=(
            "Regular-wave RAO at 14 wave periods, no mooring. Cross-"
            "checks first-order excitation and steady-state response."
        ),
        fst_edits={
            **_DISABLE_TURBINE_MODULES,
            "CompMooring": 0,
            "TMax": 1200.0,
        },
        elastodyn_edits={
            **_LOCKED_TURBINE_DOFS,
            "PtfmSgDOF": True,
            "PtfmSwDOF": True,
            "PtfmHvDOF": True,
            "PtfmRDOF": True,
            "PtfmPDOF": True,
            "PtfmYDOF": True,
        },
        hydrodyn_edits={
            "WaveMod": 2,        # regular Airy
            "WaveHs": 1.0,       # 1 m wave height — small-amplitude regime
            "WaveDir": 0.0,      # heading 0 deg
            **_DISABLE_QTFS,
        },
        moordyn_active=False,
        # Sweep wave period across 14 values covering OC4's natural
        # periods (heave ~17s, pitch ~26s) and short waves.
        sweep_param=("WaveTp", [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0,
                                14.0, 16.0, 18.0, 20.0, 22.0, 25.0, 30.0]),
        output_channels=_BASE_CHANNELS + (
            "Wave1Elev",
            "B1WvsF1xi", "B1WvsF1yi", "B1WvsF1zi",
            "B1WvsM1xi", "B1WvsM1yi", "B1WvsM1zi",
        ),
    ),
    Scenario(
        name="s4_moored_eq",
        purpose=(
            "Moored static equilibrium. Read MoorDyn's converged "
            "steady state at t~50s as quasi-static reference. Cross-"
            "checks FloatSim's analytic catenary against MoorDyn."
        ),
        fst_edits={
            **_DISABLE_TURBINE_MODULES,
            "CompMooring": 1,  # MoorDyn ON
            "TMax": 200.0,
        },
        elastodyn_edits={
            **_LOCKED_TURBINE_DOFS,
            "PtfmSgDOF": True,
            "PtfmSwDOF": True,
            "PtfmHvDOF": True,
            "PtfmRDOF": True,
            "PtfmPDOF": True,
            "PtfmYDOF": True,
        },
        hydrodyn_edits={
            "WaveMod": 0,
            **_DISABLE_QTFS,
        },
        moordyn_active=True,
        output_channels=_BASE_CHANNELS + (
            "FairTen1", "FairTen2", "FairTen3",
            "AnchTen1", "AnchTen2", "AnchTen3",
        ),
    ),
    Scenario(
        name="s5_drag_decay",
        purpose=(
            "Heave free-decay with viscous drag elements active. "
            "Cross-checks Morison drag formulation via hyperbolic "
            "envelope of decay peaks."
        ),
        fst_edits={
            **_DISABLE_TURBINE_MODULES,
            "CompMooring": 0,
            "TMax": 600.0,
        },
        elastodyn_edits={
            **_LOCKED_TURBINE_DOFS,
            "PtfmSgDOF": False,  # lock surge/sway/yaw to isolate heave
            "PtfmSwDOF": False,
            "PtfmHvDOF": True,
            "PtfmRDOF": False,
            "PtfmPDOF": False,
            "PtfmYDOF": False,
            "PtfmHeave": 1.0,  # initial heave offset, meters
        },
        hydrodyn_edits={
            "WaveMod": 0,
            **_DISABLE_QTFS,
            # Note: OC4 baseline already has Morison/strip-theory members
            # configured for the columns. Drag coefficients are kept at
            # baseline values; do not modify here.
        },
        moordyn_active=False,
        output_channels=_BASE_CHANNELS,
    ),
)
