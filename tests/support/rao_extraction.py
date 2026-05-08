"""Sinusoidal least-squares RAO extraction at the wave frequency.

Used by both:

  - ``scripts/m6_pr4_pre3_rao_verification.py`` (the Pre-3 dual-path
    convention check), and
  - ``tests/validation/test_m6_openfast_regular_wave.py`` (M6 PR4's
    84-assertion sweep over heave / roll / pitch RAOs at 14 wave
    periods).

Choices locked in M6 PR4 Pre-3:

  - **lstsq fit at the OpenFAST IFFT-quantised wave frequency**, NOT
    at the labelled ``WaveTp`` — see conventions doc Item 21. The
    quantised frequency is computed from ``WaveTMax`` (read from the
    per-scenario ``*_SeaState.dat``) via
    ``T_actual = WaveTMax / round(WaveTMax / WaveTp)``.
  - **Single-harmonic basis** ``[cos(omega t), sin(omega t), 1]``
    over the steady-state window. Captures the wave-driven response
    and rejects body natural-frequency transients (Item 20). The
    constant column absorbs DC drift / equilibrium offset.
  - **Steady-state window**: the last ``N_FIT_PERIODS`` (default 5)
    quantised wave periods. Caller is responsible for ensuring the
    ramp-up and the first few transient periods have already fallen
    outside this window.
  - **RAO phase** is reported wrapped to ``(-π, π]``. Phase
    differences (response - wave_elev) are computed via circular
    subtraction so that, e.g., ``φ_FS = 3.10 rad`` and
    ``φ_OF = -3.18 rad`` report ~0.04 rad difference, not 6.28.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Default number of quantised wave periods used for the steady-state
# window. Skip the first 4 wave periods after ramp-up; fit on the
# next 5 (per Xabier's PR4 plan).
DEFAULT_N_FIT_PERIODS: int = 5
DEFAULT_N_SKIP_PERIODS: int = 4


@dataclass(frozen=True)
class HarmonicFit:
    """Result of a single-frequency lstsq fit."""

    omega: float  # rad/s (the basis frequency used in the fit)
    amplitude: float
    phase_rad: float  # wrapped to (-pi, pi]
    fit_residual_normalized: float  # ||signal - fit|| / ||signal - mean||
    dc_offset: float
    n_samples: int


def quantised_wave_period_s(wave_tp_label_s: float, wave_tmax_s: float) -> float:
    """Return the OpenFAST IFFT-quantised wave period.

    OpenFAST's regular-wave generator (``WaveMod = 1``) builds the
    wave train via IFFT on a frequency grid with spacing
    ``WaveDOmega = 2 pi / WaveTMax``. The labelled ``WaveTp`` is
    snapped to the nearest grid point ``omega_k = k * WaveDOmega``
    with ``k = round(WaveTMax / WaveTp)``. For ``WaveTp`` values that
    don't evenly divide ``WaveTMax``, the actual generated wave is
    at a slightly different period -- up to ~ 1.3 % off for OC4 S3
    with ``WaveTMax = 600`` s. See conventions doc Item 21.
    """
    if wave_tmax_s <= 0.0 or wave_tp_label_s <= 0.0:
        raise ValueError(
            f"wave_tmax_s ({wave_tmax_s}) and wave_tp_label_s "
            f"({wave_tp_label_s}) must be positive"
        )
    k = max(round(wave_tmax_s / wave_tp_label_s), 1)
    return wave_tmax_s / k


def read_wave_tmax_from_seastate(seastate_dat_path: Path) -> float:
    """Read ``WaveTMax`` (s) from an OpenFAST ``*_SeaState.dat`` file.

    Returns the OpenFAST default of ``600 s`` if the file is missing
    or does not contain the ``WaveTMax`` field (older OpenFAST
    versions had it in HydroDyn, not SeaState).
    """
    if not seastate_dat_path.is_file():
        return 600.0
    pattern = re.compile(r"^\s*([\d.eE+-]+)\s+WaveTMax\b")
    with seastate_dat_path.open() as fh:
        for line in fh:
            m = pattern.match(line)
            if m:
                return float(m.group(1))
    return 600.0


def lstsq_fit_at_omega(
    t: NDArray[np.floating],
    x: NDArray[np.floating],
    omega: float,
) -> HarmonicFit:
    """Single-harmonic lstsq fit of ``x(t)`` at ``omega``.

    Fits ``x(t) ≈ A cos(omega t) + B sin(omega t) + C`` and returns:

    - ``amplitude = sqrt(A^2 + B^2)``
    - ``phase = atan2(B, A)`` (signal lags ``cos`` reference by this)
    - ``fit_residual_normalized = ||x - fit|| / ||x - mean(x)||``
    - ``dc_offset = C``

    The fit is unique and unambiguous; no design-dependent phase
    shift (cf. band-pass filtering, Item 20). The residual is a
    useful diagnostic: a residual >> 0 means the chosen ``omega``
    is wrong (e.g., labelled vs quantised), or the signal has
    significant content at other frequencies.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    if t_arr.ndim != 1 or x_arr.shape != t_arr.shape:
        raise ValueError(f"t and x must be 1D matching shapes; got {t_arr.shape}, {x_arr.shape}")
    basis = np.column_stack([np.cos(omega * t_arr), np.sin(omega * t_arr), np.ones_like(t_arr)])
    coeffs, *_ = np.linalg.lstsq(basis, x_arr, rcond=None)
    A, B, C = coeffs
    fit = basis @ coeffs
    amp = float(np.hypot(A, B))
    phase = float(np.arctan2(B, A))  # in (-pi, pi]
    x_zm = x_arr - float(np.mean(x_arr))
    residual = float(np.linalg.norm(x_arr - fit) / max(np.linalg.norm(x_zm), 1.0e-12))
    return HarmonicFit(
        omega=omega,
        amplitude=amp,
        phase_rad=phase,
        fit_residual_normalized=residual,
        dc_offset=float(C),
        n_samples=int(t_arr.size),
    )


def steady_state_window(
    t: NDArray[np.floating],
    quantised_period_s: float,
    *,
    n_fit_periods: int = DEFAULT_N_FIT_PERIODS,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Return the last ``n_fit_periods`` quantised wave periods of ``t``.

    Returns ``(t_window, mask)`` where ``mask`` indexes into ``t``.
    Raises ``ValueError`` if the simulation is shorter than
    ``n_fit_periods * quantised_period_s``.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    if t_arr.size < 2:
        raise ValueError(f"t must have >= 2 samples; got {t_arr.size}")
    window_s = n_fit_periods * quantised_period_s
    if t_arr[-1] - t_arr[0] < window_s:
        raise ValueError(
            f"simulation duration {t_arr[-1] - t_arr[0]:.1f} s is shorter "
            f"than {n_fit_periods} wave periods of {quantised_period_s:.3f} s "
            f"(window {window_s:.1f} s)"
        )
    mask = t_arr >= t_arr[-1] - window_s
    return t_arr[mask], mask


def wrap_phase_diff_rad(phase_a: float, phase_b: float) -> float:
    """Circular phase subtraction wrapped to ``(-pi, pi]``.

    Catches the ``φ_a = 3.10 rad, φ_b = -3.18 rad → ~0.04 rad`` case
    that naive subtraction would render as 6.28. Use this for any
    phase comparison between two RAO computations.
    """
    diff = phase_a - phase_b
    return float(((diff + np.pi) % (2.0 * np.pi)) - np.pi)


def extract_rao_from_history(
    t: NDArray[np.floating],
    response: NDArray[np.floating],
    wave_elev: NDArray[np.floating],
    omega: float,
    *,
    n_fit_periods: int = DEFAULT_N_FIT_PERIODS,
) -> tuple[float, float, float, float]:
    """Extract RAO amp + phase from response and wave elevation channels.

    Both channels are fit at the same ``omega`` (the OpenFAST
    quantised wave frequency, per Item 21). RAO amplitude is
    ``amp(response) / amp(wave_elev)``; RAO phase is the circular
    difference ``phase(response) - phase(wave_elev)``.

    Returns
    -------
    rao_amplitude : float
        Response amplitude per unit wave amplitude (units depend on
        the response channel: m/m for translations, rad/m for
        rotations).
    rao_phase_rad : float
        Wrapped phase ``(-pi, pi]``. Sign convention: positive
        means response leads wave elevation.
    response_residual : float
        Normalized fit residual on the response channel. Diagnostic.
    wave_residual : float
        Normalized fit residual on the wave_elev channel. Should be
        ~ 0 for a clean Airy wave at the quantised frequency
        (Item 21 verifies this).
    """
    quantised_period = 2.0 * np.pi / omega
    t_w, mask = steady_state_window(t, quantised_period, n_fit_periods=n_fit_periods)
    fit_resp = lstsq_fit_at_omega(t_w, response[mask], omega)
    fit_wave = lstsq_fit_at_omega(t_w, wave_elev[mask], omega)
    if fit_wave.amplitude < 1.0e-12:
        raise ValueError(
            f"wave_elev fit amplitude {fit_wave.amplitude:.3e} too small to "
            "form a meaningful RAO denominator"
        )
    rao_amp = fit_resp.amplitude / fit_wave.amplitude
    rao_phase = wrap_phase_diff_rad(fit_resp.phase_rad, fit_wave.phase_rad)
    return rao_amp, rao_phase, fit_resp.fit_residual_normalized, fit_wave.fit_residual_normalized
