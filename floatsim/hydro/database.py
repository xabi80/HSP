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
                                    per unit wave amplitude, under the +i phase
                                    convention (see below)
    reference_point (3,)            float64, point in inertial frame about which
                                    BEM coefficients are given

Phase convention -- the **+i convention** is mandatory for ``RAO``.
The stored complex coefficient ``X = RAO[dof, omega, heading]`` is such
that the time-domain wave excitation force at the body is::

    F(t) = Re[ X * A_wave * exp(+i * omega * t) ]

where ``A_wave`` is the complex wave-elevation phasor at the body
(under the same +i convention). This matches the WAMIT default ("leads"),
the OrcaFlex VesselType YAML serialisation, and the Capytaine reader
(which conjugates to translate Capytaine's native -i convention). Every
reader producing a ``HydroDatabase`` must honour this; downstream
consumers (``floatsim.hydro.excitation.make_regular_wave_force``, the
impedance-domain solver, etc.) consume it under +i. Mixing in
-i-convention RAOs is a silent bug class (F-WAVE-FORCE-CONV, M6 PR4); the
post-mortem at ``docs/post-mortems/m6-epilogue-wave-force-convention-bug.md``
records the audit-trail.

DOF order throughout is ``(surge, sway, heave, roll, pitch, yaw)`` — see
ARCHITECTURE.md §3.3. Multi-body extension (block-diagonal with off-diagonal
coupling when the BEM case was multi-body) is deferred to Milestone 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal, get_args

import numpy as np
from numpy.typing import NDArray

DOF_ORDER: Final[tuple[str, ...]] = ("surge", "sway", "heave", "roll", "pitch", "yaw")

CSourceLiteral = Literal["buoyancy_only", "full"]
"""Provenance flag for the hydrostatic restoring matrix ``C``.

- ``"buoyancy_only"`` -- ``C`` carries only the **buoyancy / waterplane**
  contribution (rho*g*I_wp + rho*g*V*z_B and the corresponding
  cross-couplings). The gravity restoring term ``-m*g*z_G`` is absent
  and must be added by the assembly step
  (:func:`floatsim.hydro.hydrostatics.gravity_restoring_contribution`,
  invoked via ``assemble_cummins_lhs``'s
  ``mass`` / ``cog_offset_from_bem_origin`` / ``gravity`` kwargs).
  The two raw-BEM readers (WAMIT, Capytaine) produce this form.
- ``"full"`` -- ``C`` already contains both buoyancy and gravity. Used
  by the OrcaFlex VesselType reader (OrcaFlex bundles mass into the
  VesselType and exports a full linearised stiffness) and by
  hand-authored synthetic test fixtures where the test author has
  pre-baked the desired total restoring. Passing
  ``mass`` / ``cog_offset_from_bem_origin`` / ``gravity`` to
  ``assemble_cummins_lhs`` alongside this source is a double-count
  (warned at assembly time).

The flag is **mandatory**: every reader must declare what its ``C``
contains. There is no default -- silence is not an option for this
field. See ``docs/post-mortems/hydrostatic-gravity-bug.md`` for the
class-of-bug that motivated making this explicit.
"""

_C_SOURCE_VALUES: Final[tuple[str, ...]] = get_args(CSourceLiteral)

# M7.5 PR2 (Q1 sub-decision, 2026-06-30): tightened from 1e-6 to 1e-12.
# Post-symmetrization residual is at float64 precision (~1e-15); 1e-12
# leaves three orders of margin and makes the gate a real invariant
# check. Any construction path that bypasses __post_init__'s
# symmetrization step (or a hand-authored fixture with subtle
# asymmetry) is caught. See docs/m7.5-reader-hygiene-plan.md §Q1.
_SYMMETRY_RTOL: Final[float] = 1.0e-12
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

    All arrays are stored as-passed except A, B, A_inf, and C, which are
    replaced with their symmetrized counterparts by ``__post_init__``
    (M7.5 PR2 Q1 lock). Callers' original arrays are NOT mutated in
    place; a new symmetrized array is created and stored on the
    frozen dataclass via ``object.__setattr__``. Other arrays (omega,
    heading_deg, RAO, reference_point) are stored as-passed. Callers
    should treat all stored arrays as read-only; copy on ingestion if
    you need to mutate.

    The ``C_source`` field is **mandatory** (no default). It declares
    whether ``C`` is buoyancy-only (the BEM-reader convention) or full
    (buoyancy + gravity, used by hand-authored test fixtures); see the
    :data:`CSourceLiteral` docstring for the rule. Callers must consult
    this flag before consuming ``C`` directly — most callers should
    instead route through :func:`floatsim.hydro.radiation.assemble_cummins_lhs`,
    which adds the gravity term when ``C_source == "buoyancy_only"`` and
    the body's mass/CoG/gravity are supplied.

    Symmetrization audit trail (M7.5 PR2 Q1 lock, see plan §I1)
    -----------------------------------------------------------
    ``__post_init__`` symmetrizes A, B, A_inf, and C via
    ``M_sym = 0.5 * (M + M.T)`` (per omega for A and B; block for
    A_inf and C). The pre-symmetrization asymmetry residual for each
    matrix is captured on ``self.metadata`` as four string-formatted
    keys (per plan §I1 specification, ``:.6e`` format)::

        metadata["symmetrization_max_residual_A"]      = "{max|A - A.T|:.6e}"
        metadata["symmetrization_max_residual_B"]      = "{max|B - B.T|:.6e}"
        metadata["symmetrization_max_residual_A_inf"]  = "{max|A_inf - A_inf.T|:.6e}"
        metadata["symmetrization_max_residual_C"]      = "{max|C - C.T|:.6e}"

    For A and B the residual is computed over the full (6, 6, n_w)
    array before per-omega symmetrization: ``max|A - A.swapaxes(0, 1)|``
    (a scalar summarizing worst-case asymmetry across ALL omega
    slices AND all off-diagonal 6x6 positions). This matches the
    plan §I1 "max over both omega slices AND the 6x6 block" spec.

    Post-symmetrization, ``_require_symmetric`` runs at
    ``rtol = 1e-12`` (Q1 sub-decision) on A_inf, C, and every
    per-omega A / B slice. Symmetrized-input residuals are at float64
    precision (~1e-15), so the tightened gate always passes on
    normal-construction paths; the gate catches any code path that
    constructs HydroDatabase bypassing ``__post_init__`` (e.g., via
    ``object.__setattr__`` from user code, or hand-authored fixtures
    with subtle asymmetry that somehow evade the symmetrization step).
    """

    omega: NDArray[np.floating]
    heading_deg: NDArray[np.floating]
    A: NDArray[np.floating]
    B: NDArray[np.floating]
    A_inf: NDArray[np.floating]
    C: NDArray[np.floating]
    RAO: NDArray[np.complexfloating]
    reference_point: NDArray[np.floating]
    C_source: CSourceLiteral
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

        # --- symmetrization (M7.5 PR2 Q1 lock, docstring "Symmetrization
        # audit trail" section). Apply BEFORE _require_symmetric so the
        # tightened rtol=1e-12 gate sees post-symmetrization residuals.
        # For A and B: shape (6, 6, n_w); DOF indices are axes 0/1,
        # omega is axis 2. Per-omega transpose = swapaxes(0, 1). The
        # residual is a scalar max across all omega slices and all 6x6
        # off-diagonals. Store residuals on metadata via direct dict
        # assignment (the metadata field is a mutable dict; frozen
        # only prevents rebinding self.metadata, not mutating it).
        # Store symmetrized arrays via object.__setattr__ (standard
        # frozen-dataclass idiom).

        A_transpose = self.A.swapaxes(0, 1)
        residual_A = float(np.max(np.abs(self.A - A_transpose)))
        object.__setattr__(self, "A", 0.5 * (self.A + A_transpose))
        self.metadata["symmetrization_max_residual_A"] = f"{residual_A:.6e}"

        B_transpose = self.B.swapaxes(0, 1)
        residual_B = float(np.max(np.abs(self.B - B_transpose)))
        object.__setattr__(self, "B", 0.5 * (self.B + B_transpose))
        self.metadata["symmetrization_max_residual_B"] = f"{residual_B:.6e}"

        residual_A_inf = float(np.max(np.abs(self.A_inf - self.A_inf.T)))
        object.__setattr__(self, "A_inf", 0.5 * (self.A_inf + self.A_inf.T))
        self.metadata["symmetrization_max_residual_A_inf"] = f"{residual_A_inf:.6e}"

        residual_C = float(np.max(np.abs(self.C - self.C.T)))
        object.__setattr__(self, "C", 0.5 * (self.C + self.C.T))
        self.metadata["symmetrization_max_residual_C"] = f"{residual_C:.6e}"

        # --- symmetry (post-symmetrization invariant check at rtol=1e-12) ---
        _require_symmetric(self.A_inf, "A_inf")
        _require_symmetric(self.C, "C")
        for k in range(n_w):
            _require_symmetric(self.A[..., k], f"A[:, :, {k}]")
            _require_symmetric(self.B[..., k], f"B[:, :, {k}]")

        # --- C_source flag ---
        if self.C_source not in _C_SOURCE_VALUES:
            raise ValueError(
                f"C_source must be one of {_C_SOURCE_VALUES}; got {self.C_source!r}. "
                f"See HydroDatabase docstring for the convention."
            )

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
