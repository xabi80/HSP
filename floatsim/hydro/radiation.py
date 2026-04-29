"""Cummins LHS assembly and natural-period helpers — ARCHITECTURE.md §2.1, §2.2.

The Cummins equation for a single rigid body in the inertial frame is::

    [M + A_inf] * xi_ddot(t)
      + integral_{0}^{t} K(t - tau) * xi_dot(tau) dtau
      + C * xi(t)
      = F_exc(t) + F_visc + F_moor + F_ext

In Milestone 1 we assemble only the frequency-domain left-hand-side pieces
that do not depend on time history: the generalized inertia ``M + A_inf``
and the hydrostatic restoring matrix ``C``. The retardation kernel ``K(t)``
and the convolution buffer arrive in Milestone 2.

DOF ordering matches :mod:`floatsim.hydro.database`: ``(surge, sway,
heave, roll, pitch, yaw)`` — see ARCHITECTURE.md §3.3.

Multi-body
----------
The same dataclass carries the N-body global LHS (§2.2). In that case
``M_plus_Ainf`` and ``C`` are ``6N x 6N`` block matrices — each 6x6
block is the per-body 6-DOF quantity, with off-block-diagonal entries
populated when the BEM run captured hydrodynamic interaction. For
independent bodies see :func:`floatsim.solver.state.assemble_global_lhs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from floatsim.hydro.database import HydroDatabase

_SYMMETRY_RTOL: Final[float] = 1.0e-6
_SYMMETRY_ATOL: Final[float] = 1.0e-10


def _require_symmetric(m: NDArray[np.floating], label: str) -> None:
    if not np.allclose(m, m.T, rtol=_SYMMETRY_RTOL, atol=_SYMMETRY_ATOL):
        raise ValueError(f"{label} must be symmetric (within rtol={_SYMMETRY_RTOL:.0e})")


@dataclass(frozen=True)
class CumminsLHS:
    """Time-independent left-hand-side matrices of the Cummins equation.

    Attributes
    ----------
    M_plus_Ainf
        Symmetric generalized inertia ``M + A_inf`` of shape
        ``(n_dof, n_dof)`` with ``n_dof = 6N`` for ``N >= 1`` bodies.
        Single-body (``n_dof = 6``) is the common case; larger sizes
        carry a multi-body system per ARCHITECTURE.md §2.2. Units are
        kg / kg*m / kg*m^2 per block.
    C
        Symmetric hydrostatic restoring matrix of the same shape, with
        units N/m / N / N*m per block.
    """

    M_plus_Ainf: NDArray[np.float64]
    C: NDArray[np.float64]

    def __post_init__(self) -> None:
        _validate_global_matrix(self.M_plus_Ainf, "M_plus_Ainf")
        _validate_global_matrix(self.C, "C")
        if self.M_plus_Ainf.shape != self.C.shape:
            raise ValueError(
                f"M_plus_Ainf shape {self.M_plus_Ainf.shape} must match " f"C shape {self.C.shape}"
            )
        _require_symmetric(self.M_plus_Ainf, "M_plus_Ainf")
        _require_symmetric(self.C, "C")

    @property
    def n_dof(self) -> int:
        """Total number of degrees of freedom (``6 * N`` for ``N`` bodies)."""
        return int(self.M_plus_Ainf.shape[0])

    @property
    def n_bodies(self) -> int:
        """Number of bodies ``N = n_dof / 6``."""
        return self.n_dof // 6


def _validate_global_matrix(m: NDArray[np.floating], label: str) -> None:
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"{label} must be square 2-D; got shape {m.shape}")
    n = int(m.shape[0])
    if n < 6 or n % 6 != 0:
        raise ValueError(f"{label} must have size 6N for some N >= 1; got {n} x {n}")


def assemble_cummins_lhs(
    *,
    rigid_body_mass: NDArray[np.floating],
    hdb: HydroDatabase,
    mass: float | None = None,
    cog_offset_from_bem_origin: NDArray[np.floating] | None = None,
    gravity: float | None = None,
) -> CumminsLHS:
    """Assemble ``M + A_inf`` and ``C`` for a single body — ARCHITECTURE.md §2.1.

    Parameters
    ----------
    rigid_body_mass
        6x6 rigid-body mass/inertia matrix from
        :func:`floatsim.bodies.mass_properties.rigid_body_mass_matrix`,
        expressed at the body reference point in the body frame.
    hdb
        Validated :class:`HydroDatabase` for this body. Must be referred
        to the same reference point as ``rigid_body_mass``.
    mass, cog_offset_from_bem_origin, gravity
        Optional gravity-restoring inputs. **Required** when
        ``hdb.C_source == "buoyancy_only"`` — without them the assembly
        cannot add the ``-m·g·z_G`` term and would produce a physically
        wrong (often statically unstable in pitch/roll) ``C``. See
        :func:`floatsim.hydro.hydrostatics.gravity_restoring_contribution`
        for the formula. Pass ``mass`` in kg, ``cog_offset_from_bem_origin``
        as a length-3 body-frame vector from the BEM hydrostatic
        reference origin to the centre of gravity, and ``gravity`` in
        m/s² (deck-supplied, typically 9.80665).

        When ``hdb.C_source == "full"``, supplying any of these triggers
        a :class:`UserWarning` for likely double-counting; the gravity
        term is not added in that case (the database already contains it).

    Returns
    -------
    CumminsLHS
        Frozen dataclass carrying ``M_plus_Ainf`` (``M + A_inf``) and ``C``
        (the full hydrostatic restoring matrix — buoyancy from the BEM
        plus gravity from this assembly when applicable).

    Raises
    ------
    ValueError
        If ``rigid_body_mass`` has the wrong shape or is non-symmetric;
        if ``hdb.C_source == "buoyancy_only"`` but any of ``mass``,
        ``cog_offset_from_bem_origin``, ``gravity`` is ``None``; or if
        the gravity-term inputs themselves fail their validation
        (delegated to :func:`gravity_restoring_contribution`).

    Notes
    -----
    No frame rotation is applied here: in Phase-1 Level-2 fidelity
    (ARCHITECTURE.md §9.2), BEM coefficients are evaluated in the body
    frame and rotated to inertial at each time step by the solver, not
    at assembly time. The quantities returned here are body-frame.

    The gravity-coupling responsibility was missing through M5 — see
    ``docs/post-mortems/hydrostatic-gravity-bug.md`` for the post-mortem.
    """
    m_body = np.asarray(rigid_body_mass, dtype=np.float64)
    if m_body.shape != (6, 6):
        raise ValueError(f"rigid_body_mass must have shape (6, 6); got {m_body.shape}")
    _require_symmetric(m_body, "rigid_body_mass")

    m_plus_ainf = m_body + np.asarray(hdb.A_inf, dtype=np.float64)
    c_matrix = np.asarray(hdb.C, dtype=np.float64).copy()
    c_matrix = _maybe_add_gravity_restoring(
        c_matrix=c_matrix,
        c_source=hdb.C_source,
        mass=mass,
        cog_offset_from_bem_origin=cog_offset_from_bem_origin,
        gravity=gravity,
    )
    return CumminsLHS(M_plus_Ainf=m_plus_ainf, C=c_matrix)


def _maybe_add_gravity_restoring(
    *,
    c_matrix: NDArray[np.float64],
    c_source: str,
    mass: float | None,
    cog_offset_from_bem_origin: NDArray[np.floating] | None,
    gravity: float | None,
) -> NDArray[np.float64]:
    """Apply or refuse the gravity restoring term per the C-source flag.

    Centralises the policy: ``"buoyancy_only"`` requires all three
    gravity inputs (raise if any missing); ``"full"`` warns if any are
    provided (likely double-count). The gravity contribution itself is
    delegated to
    :func:`floatsim.hydro.hydrostatics.gravity_restoring_contribution`.
    """
    # Local import to avoid an import cycle (hydrostatics imports nothing
    # from radiation, but radiation -> hydrostatics is fine).
    from floatsim.hydro.hydrostatics import gravity_restoring_contribution

    has_grav_inputs = (
        mass is not None or cog_offset_from_bem_origin is not None or gravity is not None
    )

    if c_source == "buoyancy_only":
        if mass is None or cog_offset_from_bem_origin is None or gravity is None:
            missing = [
                name
                for name, val in [
                    ("mass", mass),
                    ("cog_offset_from_bem_origin", cog_offset_from_bem_origin),
                    ("gravity", gravity),
                ]
                if val is None
            ]
            raise ValueError(
                "hdb.C_source == 'buoyancy_only' requires mass, "
                "cog_offset_from_bem_origin, and gravity to be passed to "
                "assemble_cummins_lhs (the gravity m*g*z_G term is added "
                "here from these inputs). Missing: " + ", ".join(missing) + ". "
                "If your BEM database stores the FULL restoring (buoyancy + "
                "gravity already combined), set C_source='full' on the "
                "HydroDatabase to skip this step."
            )
        dC = gravity_restoring_contribution(
            mass=float(mass),
            cog_offset_from_bem_origin=np.asarray(cog_offset_from_bem_origin, dtype=np.float64),
            gravity=float(gravity),
        )
        return c_matrix + dC

    if c_source == "full":
        if has_grav_inputs:
            import warnings

            warnings.warn(
                "hdb.C_source == 'full' but mass/cog_offset_from_bem_origin/gravity "
                "were supplied to assemble_cummins_lhs. The full restoring already "
                "contains the gravity term -- adding it again would double-count, "
                "so it is *not* added. Drop these kwargs to silence this warning, "
                "or change the HydroDatabase C_source to 'buoyancy_only' if the "
                "database genuinely lacks gravity.",
                UserWarning,
                stacklevel=3,
            )
        return c_matrix

    # _C_SOURCE_VALUES check in HydroDatabase.__post_init__ should catch
    # other values, but defend against future additions.
    raise ValueError(f"Unknown C_source: {c_source!r}")


def natural_periods_uncoupled(lhs: CumminsLHS) -> NDArray[np.float64]:
    """Uncoupled natural period per DOF: ``T_i = 2*pi * sqrt((M+A_inf)_ii / C_ii)``.

    This ignores off-diagonal coupling in ``M + A_inf`` and ``C`` — it is a
    sanity check, not the true coupled eigenperiod. DOFs whose diagonal
    restoring is non-positive (e.g. surge, sway, yaw for an unmoored body)
    have no natural period and are returned as NaN.

    Parameters
    ----------
    lhs
        Assembled Cummins LHS, single-body (6 DOF) or global (6N DOF).

    Returns
    -------
    ndarray of shape ``(n_dof,)``, float64
        Natural periods in seconds, per DOF. In the multi-body case
        body ``k`` occupies slots ``[6k, 6k+6)``. NaN where ``C_ii <= 0``
        or where ``(M+A_inf)_ii <= 0`` (the latter should never happen
        for physical inputs, but we guard against it to avoid silent
        bad output).
    """
    m_diag = np.diag(lhs.M_plus_Ainf).astype(np.float64)
    c_diag = np.diag(lhs.C).astype(np.float64)
    periods = np.full(lhs.n_dof, np.nan, dtype=np.float64)
    valid = (c_diag > 0.0) & (m_diag > 0.0)
    periods[valid] = 2.0 * np.pi * np.sqrt(m_diag[valid] / c_diag[valid])
    return periods
