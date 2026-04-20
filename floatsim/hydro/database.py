"""Abstract BEM hydrodynamic database — ARCHITECTURE.md §2, §6.2.

Every reader (OrcaWave, WAMIT, Capytaine, or a synthetic test fixture)
must produce an instance of :class:`HydroDatabase`. Downstream code
(Cummins assembly, RAO evaluation, retardation kernel) is allowed to
assume the invariants enforced in :meth:`HydroDatabase.__post_init__`.

Shape conventions (single body, Phase 1):

    omega          (n_w,)           float64, strictly increasing, >= 0
    heading_deg    (n_h,)           float64, degrees at deck boundary
    A              (6, 6, n_w)      float64, added mass (symmetric at each omega)
    B              (6, 6, n_w)      float64, radiation damping (symmetric at each omega)
    A_inf          (6, 6)           float64, infinite-frequency added mass (symmetric)
    C              (6, 6)           float64, hydrostatic restoring (symmetric)
    RAO            (6, n_w, n_h)    complex128, first-order wave excitation force
                                    per unit wave amplitude
    reference_point (3,)            float64, point in inertial frame about which
                                    BEM coefficients are given

DOF order throughout is ``(surge, sway, heave, roll, pitch, yaw)`` — see
ARCHITECTURE.md §3.3. Multi-body extension (block-diagonal with off-diagonal
coupling when the BEM case was multi-body) is deferred to Milestone 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

DOF_ORDER: Final[tuple[str, ...]] = ("surge", "sway", "heave", "roll", "pitch", "yaw")

_SYMMETRY_RTOL: Final[float] = 1.0e-6
_SYMMETRY_ATOL: Final[float] = 1.0e-10


def _require_symmetric(m: NDArray[np.floating], label: str) -> None:
    if not np.allclose(m, m.T, rtol=_SYMMETRY_RTOL, atol=_SYMMETRY_ATOL):
        raise ValueError(f"{label} must be symmetric (within rtol={_SYMMETRY_RTOL:.0e})")


def _require_all_finite(arr: NDArray[Any], label: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be all-finite (no NaN or inf)")


@dataclass(frozen=True)
class HydroDatabase:
    """Frequency-domain hydrodynamic database for a single floating body.

    All arrays are stored as-passed — callers should treat them as read-only.
    Copy on ingestion if you need to mutate.
    """

    omega: NDArray[np.floating]
    heading_deg: NDArray[np.floating]
    A: NDArray[np.floating]
    B: NDArray[np.floating]
    A_inf: NDArray[np.floating]
    C: NDArray[np.floating]
    RAO: NDArray[np.complexfloating]
    reference_point: NDArray[np.floating]
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # --- omega grid ---
        if self.omega.ndim != 1:
            raise ValueError("omega must be 1-D")
        if self.omega.size < 2:
            raise ValueError("omega must have at least 2 samples")
        if np.any(self.omega < 0.0):
            raise ValueError("omega must be non-negative")
        if not np.all(np.diff(self.omega) > 0.0):
            raise ValueError("omega must be strictly monotonically increasing")
        _require_all_finite(self.omega, "omega")

        # --- heading grid ---
        if self.heading_deg.ndim != 1:
            raise ValueError("heading_deg must be 1-D")
        if self.heading_deg.size < 1:
            raise ValueError("heading_deg must have at least 1 sample")
        _require_all_finite(self.heading_deg, "heading_deg")

        n_w = self.omega.size
        n_h = self.heading_deg.size

        # --- matrix shapes ---
        if self.A.shape != (6, 6, n_w):
            raise ValueError(f"A must have shape (6, 6, {n_w}); got {self.A.shape}")
        if self.B.shape != (6, 6, n_w):
            raise ValueError(f"B must have shape (6, 6, {n_w}); got {self.B.shape}")
        if self.A_inf.shape != (6, 6):
            raise ValueError(f"A_inf must have shape (6, 6); got {self.A_inf.shape}")
        if self.C.shape != (6, 6):
            raise ValueError(f"C must have shape (6, 6); got {self.C.shape}")
        if self.RAO.shape != (6, n_w, n_h):
            raise ValueError(f"RAO must have shape (6, {n_w}, {n_h}); got {self.RAO.shape}")
        if self.reference_point.shape != (3,):
            raise ValueError(
                f"reference_point must have shape (3,); got {self.reference_point.shape}"
            )

        # --- finiteness ---
        for arr, label in [
            (self.A, "A"),
            (self.B, "B"),
            (self.A_inf, "A_inf"),
            (self.C, "C"),
            (self.RAO, "RAO"),
            (self.reference_point, "reference_point"),
        ]:
            _require_all_finite(arr, label)

        # --- dtype ---
        if not np.issubdtype(self.RAO.dtype, np.complexfloating):
            raise ValueError("RAO must be complex-valued")

        # --- symmetry ---
        _require_symmetric(self.A_inf, "A_inf")
        _require_symmetric(self.C, "C")
        for k in range(n_w):
            _require_symmetric(self.A[..., k], f"A[:, :, {k}]")
            _require_symmetric(self.B[..., k], f"B[:, :, {k}]")

    # --- convenience accessors -------------------------------------------------

    @property
    def n_frequencies(self) -> int:
        """Number of frequency samples in the BEM grid."""
        return int(self.omega.size)

    @property
    def n_headings(self) -> int:
        """Number of wave-heading samples in the RAO grid."""
        return int(self.heading_deg.size)

    @property
    def dof_order(self) -> tuple[str, ...]:
        """DOF ordering for all 6-dimensional axes (ARCHITECTURE.md §3.3)."""
        return DOF_ORDER
