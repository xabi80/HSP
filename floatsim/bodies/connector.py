"""Inter-body 6-DOF spring-damper connectors — ARCHITECTURE.md §4, §5.

A connector couples two bodies (or one body and earth) via a generalized
6-DOF linear spring-damper. The force on body A from a connector between
bodies A and B is::

    Delta       = xi_A - xi_B - rest_offset
    Delta_dot   = xi_dot_A - xi_dot_B
    F_A         = -(K @ Delta + B @ Delta_dot)
    F_B         = -F_A                                   (Newton III)

``K`` and ``B`` are 6x6 stiffness/damping matrices; ``rest_offset`` is a
length-6 generalized displacement at which the spring is unstretched.
When one end attaches to "earth" (a fixed inertial point), pass
``body_a=-1`` or ``body_b=-1`` — that end contributes ``xi = 0`` and
``xi_dot = 0`` and no force is deposited into a body slot.

Rigid-link penalty (M4)
-----------------------
A rigid constraint between two bodies along a subset of DOFs is modelled
as a linear spring with a very high stiffness, deck-configurable in the
range ``10^3 ... 10^4 * max(diag(C_global))``. See
``docs/milestone-4-plan.md`` for the rationale (Q1). The explicit
treatment of the connector force by
:func:`floatsim.solver.newmark.integrate_cummins` imposes a conditional
stability floor ``dt < 2 / omega_max``, with ``omega_max`` computed over
the antisymmetric modes of each connector. :func:`check_connector_stability`
returns a list of diagnostic messages that the caller (deck loader,
validation script) is expected to surface before the integration starts.

Scope
-----
M4 PR3 restricts the connector to **body-frame** small-angle linear
behaviour: ``xi`` is the 6-DOF generalized position of the body
reference point (see ARCHITECTURE.md §3.3), and ``K, B`` act on that
6-vector directly. Attachment offsets, rotational frame transformations,
and nonlinear stiffness curves (elastomer bumpers, catenary lookup) are
deferred. The analytical catenary lives in :mod:`floatsim.mooring`; drag
elements arrive in M5.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray

from floatsim.hydro.radiation import CumminsLHS

_EARTH: Final[int] = -1


@dataclass(frozen=True)
class LinearConnector:
    """6-DOF linear spring-damper between two bodies (or body-earth).

    Attributes
    ----------
    body_a, body_b
        Global body indices. ``-1`` designates an earth (fixed) attachment.
        Must differ; connecting a body to itself is not allowed.
    K
        6x6 symmetric positive-semidefinite stiffness matrix, in the
        generalized-DOF ordering ``(surge, sway, heave, roll, pitch, yaw)``.
        Zero rows/columns are permitted and indicate an unconstrained DOF
        along that direction.
    B
        6x6 symmetric positive-semidefinite damping matrix, same ordering.
        Zero for a pure elastic element.
    rest_offset
        Length-6 generalized displacement at which the spring is
        unstretched. Defaults to the zero 6-vector.
    """

    body_a: int
    body_b: int
    K: NDArray[np.float64]
    B: NDArray[np.float64]
    rest_offset: NDArray[np.float64] = field(default_factory=lambda: np.zeros(6, dtype=np.float64))

    def __post_init__(self) -> None:
        if self.body_a == self.body_b:
            raise ValueError(
                f"connector requires distinct endpoints; got body_a = body_b = {self.body_a}"
            )
        if self.body_a < _EARTH or self.body_b < _EARTH:
            raise ValueError(
                f"body indices must be >= -1 (earth); got ({self.body_a}, {self.body_b})"
            )
        for name, m in (("K", self.K), ("B", self.B)):
            if m.shape != (6, 6):
                raise ValueError(f"{name} must have shape (6, 6); got {m.shape}")
            if not np.allclose(m, m.T, rtol=1e-8, atol=1e-12):
                raise ValueError(f"{name} must be symmetric")
        if self.rest_offset.shape != (6,):
            raise ValueError(f"rest_offset must have shape (6,); got {self.rest_offset.shape}")


def heave_rigid_link(
    *,
    body_a: int,
    body_b: int,
    penalty_stiffness: float,
    penalty_damping: float = 0.0,
) -> LinearConnector:
    """Rigid-link penalty connector along the heave (DOF 2) axis only.

    Builds a :class:`LinearConnector` with ``K = diag(0, 0, k, 0, 0, 0)``
    and ``B = diag(0, 0, c, 0, 0, 0)``. For truly rigid behaviour, choose
    ``penalty_stiffness`` at the upper end of the 10^3 ... 10^4 * max
    diag(C_global) range (``docs/milestone-4-plan.md`` Q1) and run
    :func:`check_connector_stability` to confirm the integrator step size
    stays under the explicit-stability floor.

    Parameters
    ----------
    body_a, body_b
        Body indices (``-1`` for earth).
    penalty_stiffness
        Heave stiffness in N/m. Must be strictly positive.
    penalty_damping
        Heave damping in N*s/m. Defaults to 0 (pure elastic rigid link).
    """
    if not np.isfinite(penalty_stiffness) or penalty_stiffness <= 0.0:
        raise ValueError(f"penalty_stiffness must be finite and positive; got {penalty_stiffness}")
    if not np.isfinite(penalty_damping) or penalty_damping < 0.0:
        raise ValueError(f"penalty_damping must be finite and non-negative; got {penalty_damping}")
    K = np.zeros((6, 6), dtype=np.float64)
    K[2, 2] = float(penalty_stiffness)
    B = np.zeros((6, 6), dtype=np.float64)
    B[2, 2] = float(penalty_damping)
    return LinearConnector(body_a=body_a, body_b=body_b, K=K, B=B)


def _body_slice(
    xi: NDArray[np.float64], xi_dot: NDArray[np.float64], body_idx: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the 6-vector position and velocity for ``body_idx``.

    For ``body_idx == -1`` (earth) both are the zero 6-vector.
    """
    if body_idx == _EARTH:
        zeros = np.zeros(6, dtype=np.float64)
        return zeros, zeros
    slc = slice(6 * body_idx, 6 * body_idx + 6)
    return xi[slc], xi_dot[slc]


def make_connector_state_force(
    connectors: Sequence[LinearConnector], n_dof: int
) -> Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    """Build the ``(t, xi, xi_dot) -> F`` closure consumed by ``integrate_cummins``.

    Validates body indices against ``n_dof // 6`` up front so illegal
    configurations fail fast, not at step zero of a long run.

    Parameters
    ----------
    connectors
        Collection of :class:`LinearConnector` instances.
    n_dof
        Global DOF count ``6 * N`` for the system being integrated.

    Returns
    -------
    Callable
        ``state_force(t, xi, xi_dot)`` returning a length-``n_dof`` force
        vector. ``t`` is ignored (connectors are autonomous in M4). The
        returned function sums contributions across all connectors.

    Raises
    ------
    ValueError
        If ``n_dof`` is not a positive multiple of 6 or any connector
        references a body index outside ``[-1, n_dof // 6)``.
    """
    if n_dof <= 0 or n_dof % 6 != 0:
        raise ValueError(f"n_dof must be a positive multiple of 6; got {n_dof}")
    n_bodies = n_dof // 6
    for k, c in enumerate(connectors):
        for idx in (c.body_a, c.body_b):
            if idx != _EARTH and not (0 <= idx < n_bodies):
                raise ValueError(
                    f"connector {k}: body index {idx} outside valid range "
                    f"[-1, {n_bodies}) for n_dof = {n_dof}"
                )

    conn_list = list(connectors)

    def _state_force(
        _t: float, xi: NDArray[np.float64], xi_dot: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        F = np.zeros(n_dof, dtype=np.float64)
        for c in conn_list:
            xi_a, xi_dot_a = _body_slice(xi, xi_dot, c.body_a)
            xi_b, xi_dot_b = _body_slice(xi, xi_dot, c.body_b)
            delta = xi_a - xi_b - c.rest_offset
            delta_dot = xi_dot_a - xi_dot_b
            F_a = -(c.K @ delta + c.B @ delta_dot)
            if c.body_a != _EARTH:
                F[6 * c.body_a : 6 * c.body_a + 6] += F_a
            if c.body_b != _EARTH:
                F[6 * c.body_b : 6 * c.body_b + 6] -= F_a
        return F

    return _state_force


def connector_drift(
    xi_hist: NDArray[np.floating], connector: LinearConnector
) -> NDArray[np.float64]:
    """Return peak ``|xi_a - xi_b - rest_offset|`` per DOF over a run.

    Useful as a rigid-link fidelity diagnostic: for a well-chosen penalty
    stiffness the per-DOF drift should stay below 0.1% of the operating
    amplitude in the constrained directions. Directions where ``K`` is
    zero (unconstrained) are not expected to stay at zero — the drift
    readout is only meaningful where the connector is stiff.

    Parameters
    ----------
    xi_hist
        Shape ``(n_samples, 6N)`` generalized-position history from
        :class:`floatsim.solver.newmark.IntegrationResult`.
    connector
        The connector whose drift is of interest. Both endpoints must
        be real bodies (``!= -1``) — earth has no drift history.

    Returns
    -------
    ndarray of shape ``(6,)``, float64
        Per-DOF peak absolute drift over the run.
    """
    if connector.body_a == _EARTH or connector.body_b == _EARTH:
        raise ValueError(
            "connector_drift requires both endpoints to be real bodies; "
            "body-to-earth connectors have no drift history"
        )
    xi = np.asarray(xi_hist, dtype=np.float64)
    if xi.ndim != 2 or xi.shape[1] % 6 != 0:
        raise ValueError(f"xi_hist must be 2-D with 6N columns; got shape {xi.shape}")
    n_bodies = xi.shape[1] // 6
    for idx in (connector.body_a, connector.body_b):
        if not (0 <= idx < n_bodies):
            raise ValueError(
                f"body index {idx} outside valid range [0, {n_bodies}) for "
                f"xi_hist with {xi.shape[1]} columns"
            )
    xi_a = xi[:, 6 * connector.body_a : 6 * connector.body_a + 6]
    xi_b = xi[:, 6 * connector.body_b : 6 * connector.body_b + 6]
    delta = xi_a - xi_b - connector.rest_offset
    peak = np.max(np.abs(delta), axis=0)
    return np.asarray(peak, dtype=np.float64)


def check_connector_stability(
    *,
    lhs: CumminsLHS,
    connectors: Sequence[LinearConnector],
    dt: float,
    safety_factor: float = 0.8,
) -> list[str]:
    """Return diagnostic messages for connectors whose stiffness violates ``dt``.

    The generalized-alpha integrator treats the connector force
    explicitly (evaluated at the previous step's state), which introduces
    an explicit-stability floor on ``dt``. For each connector DOF with
    non-zero ``K[i, i]`` we estimate the antisymmetric-mode frequency::

        omega_i = sqrt(K_ii / mu_eff)

    where ``mu_eff`` is the two-body reduced mass along DOF ``i`` (``m_a
    * m_b / (m_a + m_b)`` for body-body connectors, or the single body's
    mass for body-earth). The run is stable when ``dt < safety_factor * 2
    / omega_i`` for every such mode. Connectors/DOFs that violate this
    bound are listed in the returned messages.

    Parameters
    ----------
    lhs
        Global :class:`CumminsLHS` for the full N-body system. Diagonal
        entries of ``M_plus_Ainf`` supply the per-DOF body masses /
        inertias.
    connectors
        Connector set to validate.
    dt
        Integrator step size in seconds.
    safety_factor
        Safety factor in ``(0, 1]`` applied to the theoretical stability
        bound ``2 / omega``. Defaults to ``0.8`` — i.e. require a 20 %
        margin. Lower values tighten the margin.

    Returns
    -------
    list of str
        One message per violating (connector, DOF) pair. Empty list
        means every connector mode is within the safety margin.
    """
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError(f"dt must be finite and positive; got {dt}")
    if safety_factor <= 0.0 or safety_factor > 1.0:
        raise ValueError(f"safety_factor must be in (0, 1]; got {safety_factor}")

    messages: list[str] = []
    M_diag = np.diag(lhs.M_plus_Ainf)
    n_bodies = lhs.n_bodies

    for k, c in enumerate(connectors):
        K_diag = np.diag(c.K)
        for dof in range(6):
            k_ii = float(K_diag[dof])
            if k_ii <= 0.0:
                continue

            if c.body_a != _EARTH and c.body_b != _EARTH:
                if not (0 <= c.body_a < n_bodies and 0 <= c.body_b < n_bodies):
                    raise ValueError(
                        f"connector {k}: body indices out of range for lhs with "
                        f"{n_bodies} bodies"
                    )
                m_a = float(M_diag[6 * c.body_a + dof])
                m_b = float(M_diag[6 * c.body_b + dof])
                if m_a <= 0.0 or m_b <= 0.0:
                    continue
                mu_eff = m_a * m_b / (m_a + m_b)
            else:
                idx = c.body_a if c.body_a != _EARTH else c.body_b
                if not (0 <= idx < n_bodies):
                    raise ValueError(
                        f"connector {k}: body index {idx} out of range for lhs "
                        f"with {n_bodies} bodies"
                    )
                mu_eff = float(M_diag[6 * idx + dof])
                if mu_eff <= 0.0:
                    continue

            omega = float(np.sqrt(k_ii / mu_eff))
            dt_stable = safety_factor * 2.0 / omega
            if dt > dt_stable:
                messages.append(
                    f"connector {k} dof {dof}: K={k_ii:.3e} N/m, "
                    f"mu_eff={mu_eff:.3e}, omega={omega:.3f} rad/s, "
                    f"dt_stable={dt_stable:.3e} s < dt={dt:.3e} s (safety={safety_factor})"
                )
    return messages
