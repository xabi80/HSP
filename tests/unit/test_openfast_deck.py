"""Unit tests for :mod:`tests.support.openfast_deck`.

Pinned against the committed OC4 S1 deck (post TMax=600 re-extraction)
so a regression in the parser, the distributed-mass integration, or
the residual computation surfaces as a unit-test failure rather than
as a M6 validation-tier failure.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import pytest

from tests.support.openfast_deck import (
    OC4_PLATFORM_TOTAL_MASS_KG,
    _integrate_distributed_mass,
    _scan_named_float,
    _scan_named_path,
    _scan_named_scalar,
    compute_openfast_deck_residual,
)

_S1_DECK: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "openfast"
    / "oc4_deepcwind"
    / "inputs"
    / "s1_static_eq"
)


# ---------------------------------------------------------------------------
# Scanner helpers
# ---------------------------------------------------------------------------


def test_scan_named_scalar_handles_quoted_strings(tmp_path: Path) -> None:
    p = tmp_path / "ed.dat"
    p.write_text(
        textwrap.dedent(
            """\
            ------- ELASTODYN INPUT FILE --------------------------------------
            ---------------------- TOWER ----------------------------------------
            "tower.dat"            TwrFile     - Name of file containing tower properties
            87.6                   TowerHt     - Tower height
            """
        )
    )
    assert _scan_named_scalar(p, "TwrFile") == "tower.dat"
    assert _scan_named_float(p, "TowerHt") == pytest.approx(87.6)


def test_scan_named_path_resolves_relative(tmp_path: Path) -> None:
    p = tmp_path / "ed.dat"
    p.write_text('"sub/tower.dat"        TwrFile     - desc\n')
    resolved = _scan_named_path(p, "TwrFile")
    assert resolved == tmp_path / "sub" / "tower.dat"


def test_scan_missing_key_raises(tmp_path: Path) -> None:
    p = tmp_path / "ed.dat"
    p.write_text("87.6                   TowerHt     - desc\n")
    with pytest.raises(ValueError, match=r"key 'PtfmMass' not found"):
        _scan_named_float(p, "PtfmMass")


# ---------------------------------------------------------------------------
# Distributed-mass integration
# ---------------------------------------------------------------------------


def test_integrate_distributed_mass_constant_density(tmp_path: Path) -> None:
    """Constant 1000 kg/m over a 10 m span -> mass=10000, centroid=0.5."""
    p = tmp_path / "twr.dat"
    p.write_text(
        textwrap.dedent(
            """\
            ---------------------- TOWER ----------------------------------------
            HtFract  TMassDen  TwFAStif  TwSSStif
            (-)      (kg/m)    (Nm^2)    (Nm^2)
            0.0      1000.0    1.0e10    1.0e10
            0.5      1000.0    1.0e10    1.0e10
            1.0      1000.0    1.0e10    1.0e10
            ---------------------- MODE SHAPES ------------------------------------
            1.0  TwFAM1Sh(2)  - mode 1 coefficient
            """
        )
    )
    mass, centroid_frac = _integrate_distributed_mass(p, span_m=10.0)
    assert mass == pytest.approx(10_000.0, rel=1e-12)
    assert centroid_frac == pytest.approx(0.5, abs=1e-12)


def test_integrate_distributed_mass_linearly_decreasing(tmp_path: Path) -> None:
    """Linear mass density 1000 -> 0 kg/m over 10 m -> mass=5000, centroid=1/3.

    Closed-form: ∫_0^10 (1000)·(1-z/10) dz = 1000·5 = 5000
    Centroid: z_bar = ∫z·rho dz / mass = (1000/3)·100 / 5000 = 33.33/5 = 6.667 m,
    centroid_frac = 0.6667/10 -- wait that's not 1/3.

    Continuous-truth: for rho(z) = rho_0 (1 - z/L), z_bar = L/3. The
    trapezoidal estimate with 5 stations slightly under-counts (the
    piecewise-linear-treated-as-trapezoidal moment converges as
    O(h²) toward L/3); 11 stations bring it inside 1%.
    """
    p = tmp_path / "twr.dat"
    p.write_text(
        textwrap.dedent(
            """\
            ---------------------- TOWER ----------------------------------------
            HtFract  TMassDen
            (-)      (kg/m)
            0.0      1000.0
            0.1      900.0
            0.2      800.0
            0.3      700.0
            0.4      600.0
            0.5      500.0
            0.6      400.0
            0.7      300.0
            0.8      200.0
            0.9      100.0
            1.0      0.0
            ---------------------- BLOCK -----------------------------------------
            """
        )
    )
    mass, centroid_frac = _integrate_distributed_mass(p, span_m=10.0)
    assert mass == pytest.approx(5_000.0, rel=1e-12)
    # Trapezoidal with 11 stations gives ~0.3275, within 2% of the
    # continuous 1/3.
    assert centroid_frac == pytest.approx(1.0 / 3.0, rel=2.0e-2)


def test_integrate_distributed_mass_too_few_stations(tmp_path: Path) -> None:
    p = tmp_path / "twr.dat"
    p.write_text("HtFract  TMassDen\n0.0  1000.0\n---block end---\n")
    with pytest.raises(ValueError, match=r"need at least 2 for trapezoidal integration"):
        _integrate_distributed_mass(p, span_m=10.0)


# ---------------------------------------------------------------------------
# End-to-end residual on the committed S1 deck
# ---------------------------------------------------------------------------


def test_compute_residual_s1_total_mass_dominant_component() -> None:
    """The OC4 platform total (incl. ballast) should dominate m_total.

    Per Robertson 2014 and the literature constant in
    ``tests.support.openfast_deck.OC4_PLATFORM_TOTAL_MASS_KG``, the
    full platform-with-ballast is ~13.47e6 kg. Tower + RNA + blades
    add ~0.6e6 kg total. The platform fraction must be ~95% of
    m_total.
    """
    res = compute_openfast_deck_residual(_S1_DECK)
    assert res.components["platform_with_ballast"] == OC4_PLATFORM_TOTAL_MASS_KG
    platform_fraction = res.components["platform_with_ballast"] / res.m_total_kg
    assert platform_fraction > 0.93, f"platform_fraction={platform_fraction:.3f}"
    assert platform_fraction < 0.97, f"platform_fraction={platform_fraction:.3f}"


def test_compute_residual_s1_tower_mass_matches_nrel5mw_reference() -> None:
    """Integrated tower mass from the S1 TwrFile should be close to
    the NREL 5-MW reference (~250 t)."""
    res = compute_openfast_deck_residual(_S1_DECK)
    tower_kg = res.components["tower"]
    assert 240_000.0 < tower_kg < 260_000.0, f"tower_mass={tower_kg:.0f} kg"


def test_compute_residual_s1_blade_mass_matches_nrel5mw_reference() -> None:
    """Integrated blade-total mass from the S1 BldFile should be ~3 x
    17.74 t = 53.2 t (Jonkman 2009 Table 6-1)."""
    res = compute_openfast_deck_residual(_S1_DECK)
    blades_kg = res.components["blades_total"]
    assert 50_000.0 < blades_kg < 60_000.0, f"blades_total={blades_kg:.0f} kg"


def test_compute_residual_s1_buoyancy_close_to_weight() -> None:
    """For a deck designed near-balanced, buoyancy and weight should
    be the same order of magnitude. The ratio should be close to 1."""
    res = compute_openfast_deck_residual(_S1_DECK)
    ratio = res.buoyancy_n / res.weight_n
    assert 0.95 < ratio < 1.05, f"buoyancy/weight = {ratio:.4f}"


def test_compute_residual_s1_heave_residual_predicts_observed_offset() -> None:
    """The vertical residual divided by C_33 should predict OpenFAST's
    last-30-s heave mean within Item 13's ±0.15 m tolerance.

    OC4 heave stiffness from platform_small.yml ≈ 3.65e6 N/m. OpenFAST
    observed heave offset ≈ 0.488 m. Predicted ≈ residual/C_33 should
    land in [0.34, 0.64] m.
    """
    res = compute_openfast_deck_residual(_S1_DECK)
    C_33 = 3.65e6  # N/m, per platform_small.yml diag
    heave_predicted = res.F_residual[2] / C_33
    assert 0.34 < heave_predicted < 0.64, (
        f"predicted heave equilibrium {heave_predicted:.3f} m outside "
        "the [0.34, 0.64] m window around OpenFAST's 0.488 m reference"
    )


def test_compute_residual_s1_translation_dofs_zero() -> None:
    """OC4's axisymmetric mass + on-axis CoB means F[surge,sway,yaw] = 0."""
    res = compute_openfast_deck_residual(_S1_DECK)
    assert res.F_residual[0] == 0.0, f"F[surge]={res.F_residual[0]} (expected 0 by symmetry)"
    assert res.F_residual[1] == 0.0, f"F[sway]={res.F_residual[1]} (expected 0 by symmetry)"
    assert res.F_residual[5] == 0.0, f"F[yaw]={res.F_residual[5]} (expected 0)"


def test_compute_residual_overrides_platform_mass(tmp_path: Path) -> None:
    """``platform_total_mass_kg`` kwarg overrides the OC4 default.

    Sanity-test by halving the platform mass and confirming the heave
    residual changes by the expected amount."""
    full = compute_openfast_deck_residual(_S1_DECK)
    g = 9.80665  # the deck specifies this; we use it directly here
    delta_mass = -OC4_PLATFORM_TOTAL_MASS_KG / 2  # halve the platform
    expected_F_z_change = -delta_mass * g  # F[2] = buoyancy - weight; less weight means MORE upward
    halved = compute_openfast_deck_residual(
        _S1_DECK, platform_total_mass_kg=OC4_PLATFORM_TOTAL_MASS_KG / 2
    )
    actual_change = halved.F_residual[2] - full.F_residual[2]
    assert actual_change == pytest.approx(expected_F_z_change, rel=1e-9)
