"""Half-cosine excitation ramp — ARCHITECTURE.md §9.3.

All wave-excitation forces are multiplied by a smooth envelope ``r(t)``
that rises from 0 at ``t = 0`` to 1 at ``t = T_ramp`` to suppress the
non-physical transient that would otherwise be excited by a discontinuous
switch-on of harmonic forcing into a body with zero velocity history.

Formula (§9.3)::

    r(t) = 0.5 * (1 - cos(pi * t / T_ramp))   for 0 <= t < T_ramp
    r(t) = 1                                  for t >= T_ramp

This envelope has the properties

* ``r(0) = 0``, ``r(T_ramp) = 1`` (matches startup and steady-state),
* ``r'(0) = 0``, ``r'(T_ramp) = 0`` (no force-rate jumps at the
  boundaries — cleaner transient than a linear ramp),
* strictly monotone on ``[0, T_ramp]``,
* ``r(T_ramp/2) = 0.5`` (symmetric about the midpoint).

The default duration ``20 s`` is set in the deck schema; callers pass an
explicit :class:`HalfCosineRamp` so unit tests can override.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

_DEFAULT_DURATION_S: Final[float] = 20.0


@dataclass(frozen=True)
class HalfCosineRamp:
    """Half-cosine excitation envelope of configurable duration.

    Attributes
    ----------
    duration
        Ramp length in seconds. ``0`` disables the ramp entirely
        (``r(t) = 1`` for every ``t > 0``), intended only for unit
        tests that verify steady-state behavior without the ramp
        transient; production decks must use a positive duration per
        ARCHITECTURE.md §9.3.
    """

    duration: float = _DEFAULT_DURATION_S

    def __post_init__(self) -> None:
        if self.duration < 0.0:
            raise ValueError(f"duration must be non-negative; got {self.duration}")

    def value(self, t: float) -> float:
        """Return the scalar ramp factor ``r(t)`` at a single time."""
        if self.duration == 0.0:
            return 0.0 if t <= 0.0 else 1.0
        if t <= 0.0:
            return 0.0
        if t >= self.duration:
            return 1.0
        return 0.5 * (1.0 - float(np.cos(np.pi * t / self.duration)))

    def __call__(self, t: NDArray[np.floating] | float) -> NDArray[np.float64] | float:
        """Vectorized ramp: accepts a scalar or an ndarray of times."""
        t_arr = np.asarray(t, dtype=np.float64)
        if t_arr.ndim == 0:
            return self.value(float(t_arr))
        out = np.ones_like(t_arr)
        if self.duration == 0.0:
            out[t_arr <= 0.0] = 0.0
            return out
        in_ramp = (t_arr > 0.0) & (t_arr < self.duration)
        out[t_arr <= 0.0] = 0.0
        out[in_ramp] = 0.5 * (1.0 - np.cos(np.pi * t_arr[in_ramp] / self.duration))
        return out
