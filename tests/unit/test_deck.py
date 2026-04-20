"""Pydantic deck schema — ARCHITECTURE.md §5.

Structural validation only in M0: we validate shape, types, ranges, and the
closed set of string enums. We do NOT check that referenced file paths exist,
parse BEM files, or cross-reference body names against connectors yet.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from floatsim.io.deck import Deck, load_deck

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DECK = REPO_ROOT / "examples" / "two_body_semisub_barge.yml"


# ---------- happy path ----------


def test_sample_deck_parses() -> None:
    deck = load_deck(SAMPLE_DECK)
    assert isinstance(deck, Deck)
    assert deck.simulation.duration == pytest.approx(600.0)
    assert deck.simulation.integrator == "generalized_alpha"
    assert len(deck.bodies) == 2
    assert deck.bodies[0].name == "semisub"
    assert deck.bodies[1].name == "barge"
    assert len(deck.connections) == 2


def test_sample_deck_yaml_roundtrip_is_stable() -> None:
    """Model → YAML → model is a fixed point (preserves structure and values)."""
    deck_a = load_deck(SAMPLE_DECK)
    dumped = yaml.safe_dump(deck_a.model_dump(mode="python", by_alias=True), sort_keys=False)
    deck_b = Deck.model_validate(yaml.safe_load(dumped))
    assert deck_a == deck_b


def test_waves_heading_parses_as_degrees_at_deck_boundary() -> None:
    """Per CLAUDE.md §3, degrees only at deck I/O. Schema must not silently convert."""
    deck = load_deck(SAMPLE_DECK)
    assert deck.waves.heading == pytest.approx(0.0)
    # Sanity: valid range for heading in degrees.
    assert -360.0 <= deck.waves.heading <= 360.0


# ---------- required fields ----------


def _minimal_deck_dict() -> dict:
    return yaml.safe_load(SAMPLE_DECK.read_text())


def test_missing_required_simulation_field_rejected() -> None:
    raw = _minimal_deck_dict()
    del raw["simulation"]["dt"]
    with pytest.raises(ValidationError):
        Deck.model_validate(raw)


def test_missing_bodies_rejected() -> None:
    raw = _minimal_deck_dict()
    raw["bodies"] = []
    with pytest.raises(ValidationError):
        Deck.model_validate(raw)


# ---------- range / sign checks ----------


@pytest.mark.parametrize("field", ["duration", "dt", "retardation_memory", "ramp_duration"])
def test_simulation_times_must_be_positive(field: str) -> None:
    raw = _minimal_deck_dict()
    raw["simulation"][field] = -1.0
    with pytest.raises(ValidationError):
        Deck.model_validate(raw)


def test_spectral_radius_inf_must_be_in_unit_interval() -> None:
    raw = _minimal_deck_dict()
    raw["simulation"]["spectral_radius_inf"] = 1.5
    with pytest.raises(ValidationError):
        Deck.model_validate(raw)


def test_water_depth_must_be_positive() -> None:
    raw = _minimal_deck_dict()
    raw["environment"]["water_depth"] = 0.0
    with pytest.raises(ValidationError):
        Deck.model_validate(raw)


def test_water_density_must_be_positive() -> None:
    raw = _minimal_deck_dict()
    raw["environment"]["water_density"] = -1025.0
    with pytest.raises(ValidationError):
        Deck.model_validate(raw)


def test_regular_wave_height_must_be_positive() -> None:
    raw = _minimal_deck_dict()
    raw["waves"]["height"] = 0.0
    with pytest.raises(ValidationError):
        Deck.model_validate(raw)


def test_body_mass_must_be_positive() -> None:
    raw = _minimal_deck_dict()
    raw["bodies"][0]["mass"] = 0.0
    with pytest.raises(ValidationError):
        Deck.model_validate(raw)


def test_inertia_diagonal_must_be_positive() -> None:
    raw = _minimal_deck_dict()
    raw["bodies"][0]["inertia"]["Ixx"] = -1.0
    with pytest.raises(ValidationError):
        Deck.model_validate(raw)


def test_initial_conditions_length_must_be_six() -> None:
    raw = _minimal_deck_dict()
    raw["bodies"][0]["initial_conditions"]["position"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValidationError):
        Deck.model_validate(raw)


def test_reference_point_length_must_be_three() -> None:
    raw = _minimal_deck_dict()
    raw["bodies"][0]["reference_point"] = [0.0, 0.0]
    with pytest.raises(ValidationError):
        Deck.model_validate(raw)


# ---------- enum / discriminator checks ----------


def test_unknown_integrator_rejected() -> None:
    raw = _minimal_deck_dict()
    raw["simulation"]["integrator"] = "leapfrog"
    with pytest.raises(ValidationError):
        Deck.model_validate(raw)


def test_unknown_wave_type_rejected() -> None:
    raw = _minimal_deck_dict()
    raw["waves"]["type"] = "jonswap"  # Phase 2
    with pytest.raises(ValidationError):
        Deck.model_validate(raw)


def test_unknown_bem_format_rejected() -> None:
    raw = _minimal_deck_dict()
    raw["bodies"][0]["hydro_database"]["format"] = "aqwa"
    with pytest.raises(ValidationError):
        Deck.model_validate(raw)


def test_unknown_connection_type_rejected() -> None:
    raw = _minimal_deck_dict()
    raw["connections"].append({"type": "magnet", "body_a": "semisub", "body_b": "barge"})
    with pytest.raises(ValidationError):
        Deck.model_validate(raw)


# ---------- defaults from §9 ----------


def test_section9_defaults_applied_when_omitted() -> None:
    raw = _minimal_deck_dict()
    for k in ("retardation_memory", "ramp_duration", "skip_static_equilibrium"):
        raw["simulation"].pop(k, None)
    deck = Deck.model_validate(raw)
    # §9.1 default
    assert deck.simulation.retardation_memory == pytest.approx(60.0)
    # §9.3 default
    assert deck.simulation.ramp_duration == pytest.approx(20.0)
    # §9.4 default
    assert deck.simulation.skip_static_equilibrium is False


def test_nan_values_rejected() -> None:
    """Defence against silent NaN contamination."""
    raw = _minimal_deck_dict()
    raw["simulation"]["dt"] = math.nan
    with pytest.raises(ValidationError):
        Deck.model_validate(raw)
