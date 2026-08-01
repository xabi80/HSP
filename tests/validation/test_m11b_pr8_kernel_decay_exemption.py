"""M11b PR8 -- Check-3 noise-floor exemption (``kernel_decay_floor_override``).

The 12-buoy platform's rigid-yaw radiation has ``peak|K| / dominant ~ 4e-15``
(numerical noise; a rigid buoy radiates ~no yaw wave), so on the coarse
13-omega BEM grid its "kernel" fails the post-extension decay gate (Check 3)
as a false positive. The exemption -- modelled on M10 PR0's
``asymptote_check_override`` -- lets a DOF bypass Check 3 ONLY by the measured
criterion ``peak|K_ii| / max_j peak|K_jj| < _KERNEL_DECAY_NOISE_FLOOR``, and
ONLY when an explicit rationale string is supplied. It never fires
automatically, and a physical DOF (above the floor) that fails to decay still
raises.

Gates:
* Exemption FIRES on a noise-floor DOF with a rationale -> no raise, a
  ``UserWarning`` reports the DOF and its measured ratio.
* Same kernel WITHOUT the rationale -> Check 3 raises (the noise-floor DOF is
  an offender by default; the exemption is strictly opt-in).
* A PHYSICAL DOF (above the floor) that fails to decay -> raises EVEN WITH the
  rationale (the measured floor gates the exemption, not the rationale).
* Empty / whitespace rationale -> ``ValueError`` at
  ``compute_retardation_kernel`` (forcing-function contract, as PR0).
* The rationale threads ``build_system -> _build_coupled_lhs_kernel ->
  compute_retardation_kernel`` (an empty rationale raises through the coupled
  path -- proving the driver passes it, not silently drops it).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from floatsim.driver import build_system
from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.retardation import (
    _GATE_KERNEL_DECAY_RATIO,
    _KERNEL_DECAY_NOISE_FLOOR,
    _validate_kernel_decay,
    compute_retardation_kernel,
)
from floatsim.io.deck import (
    Body,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    Output,
    RegularWave,
    Simulation,
)

_RATIONALE = "M11b PR8: rigid-yaw radiation noise floor on the coarse 13-omega grid"


def _diagonal_kernel(peaks_and_end_ratios: list[tuple[float, float]]) -> np.ndarray:
    """Build a ``(n, n, 5)`` kernel whose diagonal ``i`` has ``max|K_ii| =
    peak`` (at lag 0) and ``|K_ii(t_max)| = end_ratio * peak`` (at the last
    lag). Off-diagonals are zero; only the diagonal drives Check 3."""
    n = len(peaks_and_end_ratios)
    nt = 5
    k = np.zeros((n, n, nt), dtype=np.float64)
    for i, (peak, end_ratio) in enumerate(peaks_and_end_ratios):
        k[i, i, 0] = peak  # the max over the lag axis
        k[i, i, -1] = end_ratio * peak  # the terminal value Check 3 measures
    return k


# --- direct _validate_kernel_decay behaviour --------------------------------


def test_noise_floor_dof_exempted_with_rationale() -> None:
    """A DOF at ``rel = 1e-12`` (far below the 1e-9 floor) that does NOT decay
    (end_ratio 0.5) is exempted when a rationale is given: no raise, and a
    warning names the DOF and its measured ratio."""
    # DOF 0 dominant + decaying; DOF 1 noise-floor + non-decaying. The
    # noise-floor peak (dominant * rel = 1e-10) is ABOVE _FLOAT_EPS (1e-12,
    # the "kernel is absent" skip) but BELOW the 1e-9 relative floor -- the
    # regime the platform's yaw DOFs sit in (abs ~1.6e-12, rel ~4e-15).
    rel = 1.0e-10
    k = _diagonal_kernel([(1.0, 1.0e-6), (rel, 0.5)])
    assert rel < _KERNEL_DECAY_NOISE_FLOOR  # premise: below the relative floor
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _validate_kernel_decay(k, floor_override=_RATIONALE)  # must NOT raise
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("EXEMPTED" in m and "DOF 1" in m for m in msgs), msgs
    assert any(_RATIONALE in m for m in msgs), msgs
    # The measured ratio (1.00e-10) is reported, not a fixed placeholder.
    assert any(f"{rel:.2e}" in m for m in msgs), msgs


def test_noise_floor_dof_raises_without_override() -> None:
    """The exemption is strictly opt-in: the SAME kernel with ``floor_override
    = None`` (the default) raises Check 3 on the non-decaying noise-floor DOF."""
    k = _diagonal_kernel([(1.0, 1.0e-6), (1.0e-10, 0.5)])
    with pytest.raises(ValueError, match=r"Check 3.*kernel decay"):
        _validate_kernel_decay(k, floor_override=None)


def test_physical_dof_nondecay_raises_even_with_override() -> None:
    """A DOF ABOVE the floor (``rel = 0.5``) that fails to decay still raises
    even with a rationale -- the measured floor gates the exemption, so the
    rationale cannot mask a real physical non-decay."""
    rel = 0.5
    k = _diagonal_kernel([(1.0, 1.0e-6), (rel, 0.5)])
    assert rel > _KERNEL_DECAY_NOISE_FLOOR  # premise: above the floor
    with pytest.raises(ValueError, match=r"Check 3.*kernel decay"):
        _validate_kernel_decay(k, floor_override=_RATIONALE)


def test_exemption_boundary_is_the_measured_floor() -> None:
    """A DOF exactly AT the floor is NOT exempted (strict ``<``); just below IS.
    Pins the boundary to ``_KERNEL_DECAY_NOISE_FLOOR`` so a future change to the
    constant moves the test with it."""
    at = _diagonal_kernel([(1.0, 1.0e-6), (_KERNEL_DECAY_NOISE_FLOOR, 0.5)])
    with pytest.raises(ValueError, match=r"Check 3.*kernel decay"):
        _validate_kernel_decay(at, floor_override=_RATIONALE)
    below = _diagonal_kernel([(1.0, 1.0e-6), (_KERNEL_DECAY_NOISE_FLOOR * 0.9, 0.5)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _validate_kernel_decay(below, floor_override=_RATIONALE)  # must NOT raise


def test_decaying_noise_floor_dof_needs_no_exemption() -> None:
    """Control: a noise-floor DOF that DOES decay (end_ratio below the gate) is
    never an offender, so it is neither exempted nor raised -- no warning."""
    k = _diagonal_kernel([(1.0, 1.0e-6), (1.0e-10, _GATE_KERNEL_DECAY_RATIO * 0.5)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _validate_kernel_decay(k, floor_override=_RATIONALE)
    assert not [w for w in caught if "EXEMPTED" in str(w.message)]


def test_absent_dof_below_float_eps_is_skipped_not_exempted() -> None:
    """A DOF whose ABSOLUTE peak is <= _FLOAT_EPS (1e-12) is treated as a
    structurally absent kernel and skipped by the pre-existing guard -- BEFORE
    the exemption logic. It neither raises nor emits an exemption warning, with
    or without a rationale. This documents the two-tier structure: _FLOAT_EPS is
    an absolute 'kernel is zero' skip; _KERNEL_DECAY_NOISE_FLOOR is a relative
    'kernel is noise vs dominant' exemption. (The platform's yaw peaks sit at
    ~1.6e-12, just ABOVE _FLOAT_EPS, so they take the exemption path, not this
    skip.)"""
    k = _diagonal_kernel([(1.0, 1.0e-6), (1.0e-13, 0.5)])  # abs peak 1e-13 < eps
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _validate_kernel_decay(k, floor_override=None)  # must NOT raise
        _validate_kernel_decay(k, floor_override=_RATIONALE)  # must NOT warn
    assert not [w for w in caught if "EXEMPTED" in str(w.message)]


# --- rationale validation at compute_retardation_kernel ---------------------


def _minimal_hdb() -> HydroDatabase:
    """A trivial single-body database; the rationale validation runs BEFORE any
    kernel work, so the values are irrelevant."""
    omega = np.linspace(0.1, 3.0, 6)
    nd, nw = 6, omega.size
    return HydroDatabase(
        omega=omega,
        heading_deg=np.array([0.0]),
        A=np.zeros((nd, nd, nw)),
        B=np.zeros((nd, nd, nw)),
        A_inf=np.eye(nd),
        C=np.eye(nd),
        RAO=np.zeros((nd, nw, 1), dtype=np.complex128),
        reference_point=np.zeros(3),
        C_source="full",
        metadata={"src": "pr8-exemption-fixture"},
    )


@pytest.mark.parametrize("bad", ["", "   \t "])
def test_empty_rationale_raises_at_kernel(bad: str) -> None:
    with pytest.raises(ValueError, match=r"empty or whitespace"):
        compute_retardation_kernel(
            _minimal_hdb(), t_max=2.0, dt=0.1, kernel_decay_floor_override=bad
        )


def test_non_string_rationale_raises_at_kernel() -> None:
    with pytest.raises(ValueError, match=r"non-empty rationale string"):
        compute_retardation_kernel(
            _minimal_hdb(),
            t_max=2.0,
            dt=0.1,
            kernel_decay_floor_override=1.0,  # type: ignore[arg-type]
        )


# --- driver threading through the coupled build path ------------------------


def _labelled_2body_hdb() -> HydroDatabase:
    nd = 12
    omega = np.linspace(0.1, 3.0, 6)
    nw = omega.size
    a_inf = np.eye(nd) * 1.0e6
    return HydroDatabase(
        omega=omega,
        heading_deg=np.array([0.0]),
        A=np.stack([a_inf for _ in range(nw)], axis=-1),
        B=np.zeros((nd, nd, nw)),
        A_inf=a_inf,
        C=np.eye(nd) * 1.0e5,
        RAO=np.zeros((nd, nw, 1), dtype=np.complex128),
        reference_point=np.zeros(3),
        C_source="full",
        metadata={"src": "pr8-thread-fixture"},
        body_labels=("alpha", "beta"),
    )


def _coupled_deck() -> Deck:
    def _body(name: str, label: str) -> Body:
        return Body(
            name=name,
            reference_point=[0.0, 0.0, 0.0],
            mass=1.0e6,
            inertia=Inertia(Ixx=1.0e8, Iyy=1.0e8, Izz=1.0e8),
            hydro_body_label=label,
        )

    return Deck(
        simulation=Simulation(duration=10.0, dt=0.1),
        environment=Environment(water_depth=200.0, water_density=1025.0),
        waves=RegularWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[_body("body_0", "alpha"), _body("body_1", "beta")],
        shared_hydro_database=HydroDatabaseRef(format="capytaine", path="shared.nc"),
        output=Output(file="out.h5", channels=["heave"], sample_rate=10.0),
    )


def test_empty_override_raises_through_coupled_build() -> None:
    """The empty-rationale ``ValueError`` lives in ``compute_retardation_kernel``;
    an empty ``kernel_decay_floor_override`` raising through ``build_system``
    proves the driver threads the argument rather than dropping it."""
    with pytest.raises(ValueError, match=r"empty or whitespace"):
        build_system(
            _coupled_deck(),
            bem_databases={},
            dt=0.1,
            t_max_kernel=2.0,
            solve_equilibrium=False,
            shared_hydro_database=_labelled_2body_hdb(),
            kernel_decay_floor_override="",
        )
