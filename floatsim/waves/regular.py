"""Regular (monochromatic) Airy waves — ARCHITECTURE.md §8 M3, §3.1.

A single-frequency, single-heading wave train with constant amplitude.
In the inertial frame the free-surface elevation is::

    eta(x, y, t) = A * cos(omega * t - k (x cos beta + y sin beta) - phi)

with ``A`` the wave amplitude (half the crest-to-trough height), ``omega``
the angular frequency, ``beta`` the propagation heading (0 deg = waves
travel toward +X per §3.1), and ``phi`` an arbitrary phase offset.

Phase convention
----------------
The amplitude above is the real part of a complex phasor under the
``exp(-i * omega * t)`` convention — the same convention carried by the
OrcaFlex VesselType YAML reader (ARCHITECTURE.md §M1.5), so a regular
wave paired with a BEM RAO composes by straight complex multiplication
in :mod:`floatsim.hydro.excitation`.

Dispersion
----------
Phase 1 assumes **deep water**: ``k = omega**2 / g``. Finite-depth
dispersion (``omega**2 = g k tanh(k h)``) is a trivial extension that
will land when a finite-depth validation case does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

# WGS-84 / deck-default gravity. Decks override this via the `environment`
# block; the default mirrors ARCHITECTURE.md §5 example decks.
_DEFAULT_GRAVITY: Final[float] = 9.80665


@dataclass(frozen=True)
class RegularWave:
    """Monochromatic Airy wave.

    Attributes
    ----------
    amplitude
        Wave amplitude ``A`` in metres (half the crest-to-trough wave
        height ``H``; ``A = H / 2``). Must be non-negative.
    omega
        Angular frequency ``omega = 2*pi / T`` in rad/s. Must be positive.
    heading_deg
        Propagation heading in degrees measured from the inertial ``+X``
        axis toward ``+Y`` (i.e. ``0 deg`` is a wave travelling in the
        ``+X`` direction). No range restriction — cosines/sines absorb
        any wraparound.
    phase
        Extra phase offset ``phi`` in radians (defaults to ``0``). Useful
        for aligning against measured time series or other references.
    gravity
        Gravitational acceleration in m/s^2 used in the dispersion
        relation. Defaults to the deck-standard ``9.80665``.
    """

    amplitude: float
    omega: float
    heading_deg: float = 0.0
    phase: float = 0.0
    gravity: float = _DEFAULT_GRAVITY

    def __post_init__(self) -> None:
        if self.amplitude < 0.0:
            raise ValueError(f"amplitude must be non-negative; got {self.amplitude}")
        if self.omega <= 0.0:
            raise ValueError(f"omega must be positive; got {self.omega}")
        if self.gravity <= 0.0:
            raise ValueError(f"gravity must be positive; got {self.gravity}")

    @property
    def period(self) -> float:
        """Wave period ``T = 2*pi / omega`` in seconds."""
        return 2.0 * np.pi / self.omega

    @property
    def wavenumber(self) -> float:
        """Deep-water wavenumber ``k = omega^2 / g`` in rad/m."""
        return (self.omega**2) / self.gravity

    @property
    def wavelength(self) -> float:
        """``lambda = 2*pi / k`` in metres."""
        return 2.0 * np.pi / self.wavenumber

    def elevation(
        self,
        t: NDArray[np.floating] | float,
        x: NDArray[np.floating] | float = 0.0,
        y: NDArray[np.floating] | float = 0.0,
    ) -> NDArray[np.float64] | float:
        """Free-surface elevation ``eta(x, y, t)`` in metres.

        Parameters
        ----------
        t
            Time in seconds; scalar or ndarray.
        x, y
            Inertial-frame horizontal coordinates in metres; scalars or
            ndarrays broadcastable with ``t``.

        Returns
        -------
        Scalar or ndarray matching the broadcast of ``(t, x, y)``.
        """
        beta = np.radians(self.heading_deg)
        k = self.wavenumber
        phase_total = self.omega * t - k * (x * np.cos(beta) + y * np.sin(beta)) - self.phase
        result = self.amplitude * np.cos(phase_total)
        if isinstance(result, np.ndarray):
            return result.astype(np.float64, copy=False)
        return float(result)
