"""Morison drag (and optional inertia) on slender members — ARCHITECTURE.md §4.

A ``MorisonElement`` is one straight cylindrical segment of an offshore
structure (a brace, a column, a heave plate's edge ring) attached to a
body in body-frame coordinates. Each element contributes a 6-DOF
generalized force on its parent body, computed from the local relative
flow at the member's midpoint.

Decisions locked in `docs/milestone-5-plan.md` Q1
-------------------------------------------------
- **Relative velocity**: ``u_rel = u_fluid − u_body``; standard Morison.
- **Member-normal projection**: only the component of ``u_rel`` in the
  plane orthogonal to the member axis contributes to drag::

      ê_axis = (r_b − r_a) / |r_b − r_a|
      u_n    = u_rel − (u_rel · ê_axis) · ê_axis

  Members aligned with the flow contribute negligible drag — the
  full-vector formula is wrong for arbitrary orientations and is the
  textbook bug-catcher.
- **Drag formula** (force per unit length)::

      dF_drag/dl = ½ · ρ · D · Cd · |u_n| · u_n

  evaluated at the midpoint and multiplied by the member length ``L``
  (midpoint quadrature; valid in the slender-body limit ``L ≪ λ``).
- **Drag-only by default** (``include_inertia=False``). Most FloatSim
  bodies have a BEM database that already accounts for inertia
  (added-mass and Froude-Krylov are in ``A(ω)`` and ``F_exc(ω)``).
  Adding the Morison inertia on top would double-count.
- **Inertia formula** (when ``include_inertia=True``)::

      dF_inertia/dl = ρ · A_x · (1 + Ca) · a_fluid_n − ρ · A_x · Ca · a_body_n

  with ``A_x = π·D²/4`` the cross-sectional area and ``a_fluid_n``,
  ``a_body_n`` the member-normal projections of the fluid and body
  accelerations at the midpoint.
- **Body-frame transforms**: each member's ``node_a``, ``node_b`` rotate
  with the body's quaternion. The integrator passes generalized state
  ``xi = (surge, sway, heave, roll, pitch, yaw)``; the (small-angle)
  rotation is built from ``xi[3:6]`` via ZYX-intrinsic Euler →
  quaternion → rotation matrix (see :func:`floatsim.bodies.rigid_body.
  quaternion_from_euler_zyx`). For ``|roll, pitch, yaw| << 1`` this is
  identical to the linear small-angle limit; for moderate angles
  (10°–30°) it preserves orthogonality, unlike a naive
  ``I + skew(θ)`` linearisation.

Generalized-force assembly
--------------------------
Inertial-frame force ``F_i`` applied at midpoint ``r_mid_inertial`` maps
to a 6-DOF generalized force on the body's reference point as::

    F_translation = F_i                                    (3,)
    F_rotation    = (r_mid_inertial − r_ref_inertial) × F_i (3,)

At small angles this is exact; for the moderate-angle regime FloatSim
targets, the linearization error in the moment arm (``r_mid`` rotates
with the body) is captured exactly by transforming the body-frame
``r_mid_body`` through the full rotation matrix before taking the cross
product. The resulting 6-vector is what the integrator's ``state_force``
callable consumes.

Integrator wiring
-----------------
:func:`make_morison_state_force` builds the closure consumed by
:func:`floatsim.solver.newmark.integrate_cummins`. The closure:

- decomposes the global state ``xi``, ``xi_dot`` per body;
- builds each body's pose (translation + rotation matrix) from its xi
  slice;
- evaluates ``u_fluid`` (and ``a_fluid`` if any element has
  ``include_inertia``) at each member's midpoint via the user-supplied
  callable;
- computes ``u_body`` at the midpoint as ``v_body + ω_body × Δr``;
- accumulates the 6-DOF generalized force into the global force vector.

Body acceleration in the inertia term
-------------------------------------
The integrator's ``state_force`` callable receives ``(t, xi, xi_dot)``
but **not** ``xi_ddot``. For ``include_inertia=True`` the
``ρ·V·Ca·a_body_n`` term therefore uses ``a_body = 0`` as the Phase-1
default. Two arguments justify this:

1. The dominant ``include_inertia`` use case in offshore practice is a
   non-BEM Morison-only body (the body's own inertia is already on the
   LHS via its rigid mass, separate from this drag-element module). In
   that regime ``a_body`` and ``a_fluid`` are similar in magnitude and
   the ``Ca·a_body_n`` term partially cancels with the body's mass on
   the LHS — properly accounted for by an *implicit* drag treatment
   that Phase 1 explicitly defers (CLAUDE.md §6 ``state_force`` is
   evaluated at the previous step's state).
2. Mixing this term with a BEM-equipped body double-counts; the
   startup warning emitted by :func:`startup_inertia_double_count_warnings`
   is the user-facing signal that ``include_inertia=True`` on a
   BEM-equipped body is suspicious.

This module never assumes BEM. The double-count check is the caller's
responsibility (it has access to each body's ``HydroDatabase``).

References
----------
- Morison, J.R., O'Brien, M.P., Johnson, J.W., Schaaf, S.A., 1950. "The
  force exerted by surface waves on piles." Petroleum Transactions
  AIME 189, 149-154.
- Faltinsen, O.M., 1990. *Sea Loads on Ships and Offshore Structures*.
  Cambridge University Press. Chapter 4 (Morison's equation, KC number,
  drag-vs-inertia regime diagram).
- Sarpkaya, T., 2010. *Wave Forces on Offshore Structures*. Cambridge.
  Chapters 3-5 (drag/inertia coefficients, separated flow).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray

from floatsim.bodies.rigid_body import quaternion_from_euler_zyx, rotation_matrix

_AXIS_LENGTH_RTOL: Final[float] = 1.0e-9


@dataclass(frozen=True)
class MorisonElement:
    """One Morison drag (+ optional inertia) member on a body.

    Attributes
    ----------
    body_index
        Index into the global state vector's body slot. Must satisfy
        ``0 <= body_index < n_dof // 6``.
    node_a_body, node_b_body
        Length-3 body-frame coordinates of the member's two endpoints,
        in metres, relative to the body's reference point. The two
        nodes must be distinct (length > 0); the order is irrelevant
        for the drag formula but determines the sign of ``ê_axis``.
    diameter
        Hydrodynamic diameter ``D`` in metres. Must be positive.
    Cd
        Drag coefficient (dimensionless, non-negative). Typical values:
        circular cylinder in turbulent flow ``Cd ≈ 0.6–1.2``; sharp
        plate ``Cd ≈ 1.5–2.0``.
    Ca
        Added-mass coefficient (dimensionless, non-negative). Used only
        when ``include_inertia=True``. Typical values: circular
        cylinder ``Ca ≈ 1.0``; sphere ``Ca = 0.5``.
    include_inertia
        ``False`` by default — the element contributes only the drag
        term. Set ``True`` to additionally apply the Froude-Krylov +
        added-mass term ``ρ·V·(1+Ca)·a_fluid_n − ρ·V·Ca·a_body_n``.
        For a body with a non-empty BEM database the BEM ``A(ω)`` and
        ``F_exc(ω)`` already cover this term; double-counting is the
        bug :func:`startup_inertia_double_count_warnings` flags.
    """

    body_index: int
    node_a_body: NDArray[np.float64]
    node_b_body: NDArray[np.float64]
    diameter: float
    Cd: float
    Ca: float = 0.0
    include_inertia: bool = False

    def __post_init__(self) -> None:
        if self.body_index < 0:
            raise ValueError(f"body_index must be >= 0; got {self.body_index}")
        a = np.asarray(self.node_a_body, dtype=np.float64)
        b = np.asarray(self.node_b_body, dtype=np.float64)
        if a.shape != (3,) or b.shape != (3,):
            raise ValueError(
                f"node_a_body and node_b_body must have shape (3,); " f"got {a.shape}, {b.shape}"
            )
        # Re-bind to enforce the canonical dtype/shape and to freeze them.
        object.__setattr__(self, "node_a_body", a)
        object.__setattr__(self, "node_b_body", b)
        if float(np.linalg.norm(b - a)) <= _AXIS_LENGTH_RTOL:
            raise ValueError(
                f"node_a_body and node_b_body must be distinct; "
                f"got |b - a| = {float(np.linalg.norm(b - a)):.3e} m"
            )
        if self.diameter <= 0.0:
            raise ValueError(f"diameter must be positive; got {self.diameter}")
        if self.Cd < 0.0:
            raise ValueError(f"Cd must be non-negative; got {self.Cd}")
        if self.Ca < 0.0:
            raise ValueError(f"Ca must be non-negative; got {self.Ca}")

    @property
    def length_m(self) -> float:
        """Member length ``L = |node_b - node_a|`` in metres (body-frame; rigid)."""
        return float(np.linalg.norm(self.node_b_body - self.node_a_body))

    @property
    def cross_section_area_m2(self) -> float:
        """Cross-sectional area ``A_x = π·D²/4`` in m² (used by the inertia term)."""
        return float(np.pi * (self.diameter**2) / 4.0)


# ---------------------------------------------------------------------------
# Direction-dependent plate (heave-plate) drag element -- M11a PR4 (plan Q3-iii)
# ---------------------------------------------------------------------------


def _inplane_basis(n_hat: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two orthonormal in-plane vectors spanning the plane orthogonal to ``n_hat``."""
    seed = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(n_hat)))]
    e1 = seed - float(np.dot(seed, n_hat)) * n_hat
    e1 = e1 / float(np.linalg.norm(e1))
    e2 = np.cross(n_hat, e1)
    return e1, e2


def _disc_patches(
    *,
    radius: float,
    n_radial: int,
    n_azimuthal: int,
    center: NDArray[np.float64],
    e1: NDArray[np.float64],
    e2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Midpoint-rule polar quadrature of the disc face (body frame).

    Returns ``(positions, areas)``: ``positions`` are the ``P = n_radial *
    n_azimuthal`` patch centroids in body-frame coordinates (relative to the
    body reference point), ``areas`` the corresponding patch areas. Exact
    ring areas ⇒ ``areas.sum() == π·radius²`` to machine precision (the
    property GATE 3 relies on for the exact heave reduction); the ``∫∫|x|³
    dA`` moment converges to ``8·radius⁵/15`` as the counts increase
    (GATE 4).
    """
    positions: list[NDArray[np.float64]] = []
    areas: list[float] = []
    for i in range(n_radial):
        r_lo = radius * i / n_radial
        r_hi = radius * (i + 1) / n_radial
        r_mid = 0.5 * (r_lo + r_hi)
        d_area = float(np.pi * (r_hi**2 - r_lo**2) / n_azimuthal)
        for j in range(n_azimuthal):
            phi = (j + 0.5) * 2.0 * np.pi / n_azimuthal
            rho = r_mid * (float(np.cos(phi)) * e1 + float(np.sin(phi)) * e2)
            positions.append(center + rho)
            areas.append(d_area)
    return np.asarray(positions, dtype=np.float64), np.asarray(areas, dtype=np.float64)


@dataclass(frozen=True)
class PlateDragElement:
    """A direction-dependent (anisotropic) circular-plate drag element.

    A heave plate is a thin disc that resists **broadside (normal)** flow far
    more than **edge-on (tangential)** flow. A :class:`MorisonElement`
    (member-normal cylinder) is drag-*isotropic* in its normal plane -- one
    scalar ``Cd`` on the whole 2-D resultant -- so it cannot represent this
    anisotropy and mis-models a tilting plate (M11a PR4 scoping; the record
    corrected an earlier "reuse the cylinder" plan). This element decomposes
    the local relative flow into the two physically distinct regimes:

    - **Normal (broadside)** -- ``½·ρ·Cd_n·|w|·w`` per unit face area along the
      disc normal ``n̂``, where ``w = u_rel·n̂`` is the local normal speed.
      Integrated over the disc face by a body-fixed polar quadrature
      (:func:`_disc_patches`). Because ``w`` varies linearly across the disc
      when the plate tilts (``w(x) = −θ̇·x``), this captures BOTH pure heave
      (uniform ``w`` ⇒ the classic ``½ρCd_n·A·|v|v`` plate drag, GATE 3) and
      the tilting-rotational contribution (``∫∫|x|³dA = 8a⁵/15``, GATE 4).
      ``Cd_n = 5.0`` is the KNOWN heave-plate coefficient.
    - **Tangential (edge-on)** -- ``½·ρ·Cd_t·(t·2a)·|u_t|·u_t`` lumped at the
      disc centre, where ``u_t`` is the in-plane relative velocity and
      ``t·2a`` the rim frontal (silhouette) area. ``Cd_t`` is tank-pending
      (carried as a ``[1,2]`` sensitivity); this term is minor
      (``E_normal/E_tangential = 1.8–3.5`` on the OC4 buoy-pitch mode).

    **Principal stated approximation (sheared field).** ``Cd_n = 5.0`` was
    measured for *uniform* heave; applying it strip-wise to the linearly
    varying tilting field assumes local face-normal drag with no radial
    interaction. This is far better grounded than the discarded edge-on
    framing, but it is the assumption the tank rotational-decay campaign
    actually tests. See ``docs/m11-platform-plan.md`` Finding F3 and tracker
    ``INBAND-ROTATIONAL-RESONANCE`` (2f correction).

    Attributes
    ----------
    body_index
        Index into the global state vector's body slot
        (``0 <= body_index < n_dof // 6``).
    center_body
        Length-3 body-frame disc centre (metres, relative to the reference
        point).
    normal_body
        Length-3 body-frame disc normal. Normalised on construction; need
        not be unit on input. Must be non-degenerate.
    radius
        Disc radius ``a`` in metres (positive).
    thickness
        Rim thickness ``t`` in metres (non-negative). Sets the edge-on
        frontal area ``t·2a``; ``0`` disables the tangential term.
    Cd_n
        Normal (broadside) drag coefficient (non-negative). The known
        heave-plate value ``5.0`` for the OC4 buoy.
    Cd_t
        Tangential (edge-on) drag coefficient (non-negative). Tank-pending;
        literature ``≈1–2`` for a bluff thin rim.
    n_radial, n_azimuthal
        Polar quadrature counts for the disc face. Defaults ``12 × 24`` give
        the ``∫∫|x|³dA`` moment to ``−0.58 %`` of the analytical ``8a⁵/15``
        (GATE 4); the face area is exact at any counts.
    """

    body_index: int
    center_body: NDArray[np.float64]
    normal_body: NDArray[np.float64]
    radius: float
    thickness: float
    Cd_n: float
    Cd_t: float
    n_radial: int = 12
    n_azimuthal: int = 24
    # Cached quadrature (frozen; set in __post_init__).
    _n_hat_body: NDArray[np.float64] = field(init=False, repr=False)
    _patch_pos_body: NDArray[np.float64] = field(init=False, repr=False)
    _patch_area: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.body_index < 0:
            raise ValueError(f"body_index must be >= 0; got {self.body_index}")
        c = np.asarray(self.center_body, dtype=np.float64)
        n = np.asarray(self.normal_body, dtype=np.float64)
        if c.shape != (3,) or n.shape != (3,):
            raise ValueError(
                f"center_body and normal_body must have shape (3,); got {c.shape}, {n.shape}"
            )
        n_norm = float(np.linalg.norm(n))
        if n_norm <= _AXIS_LENGTH_RTOL:
            raise ValueError(f"normal_body must be non-degenerate; got |n| = {n_norm:.3e}")
        if self.radius <= 0.0:
            raise ValueError(f"radius must be positive; got {self.radius}")
        if self.thickness < 0.0:
            raise ValueError(f"thickness must be non-negative; got {self.thickness}")
        if self.Cd_n < 0.0:
            raise ValueError(f"Cd_n must be non-negative; got {self.Cd_n}")
        if self.Cd_t < 0.0:
            raise ValueError(f"Cd_t must be non-negative; got {self.Cd_t}")
        if self.n_radial < 1 or self.n_azimuthal < 1:
            raise ValueError(
                f"n_radial and n_azimuthal must be >= 1; got {self.n_radial}, {self.n_azimuthal}"
            )
        n_hat = n / n_norm
        e1, e2 = _inplane_basis(n_hat)
        pos, area = _disc_patches(
            radius=self.radius,
            n_radial=self.n_radial,
            n_azimuthal=self.n_azimuthal,
            center=c,
            e1=e1,
            e2=e2,
        )
        object.__setattr__(self, "center_body", c)
        object.__setattr__(self, "normal_body", n)
        object.__setattr__(self, "_n_hat_body", n_hat)
        object.__setattr__(self, "_patch_pos_body", pos)
        object.__setattr__(self, "_patch_area", area)

    @property
    def face_area_m2(self) -> float:
        """Disc face area ``π·a²`` in m² (equals ``_patch_area.sum()`` exactly)."""
        return float(np.pi * self.radius**2)

    @property
    def edge_area_m2(self) -> float:
        """Rim frontal (silhouette) area ``t·2a`` in m² for the edge-on term."""
        return float(self.thickness * 2.0 * self.radius)


# ---------------------------------------------------------------------------
# Force computation at one element (point evaluation, midpoint quadrature)
# ---------------------------------------------------------------------------


def _project_normal(vec: NDArray[np.float64], axis_hat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``vec − (vec·axis_hat)·axis_hat`` (member-normal projection)."""
    return vec - float(np.dot(vec, axis_hat)) * axis_hat


def morison_element_force(
    element: MorisonElement,
    *,
    midpoint_inertial: NDArray[np.float64],
    axis_hat_inertial: NDArray[np.float64],
    body_velocity_at_midpoint: NDArray[np.float64],
    body_acceleration_at_midpoint: NDArray[np.float64] | None,
    fluid_velocity: NDArray[np.float64],
    fluid_acceleration: NDArray[np.float64] | None,
    rho: float,
    reference_point_inertial: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the 6-DOF generalized force from a single Morison element.

    All vector inputs are in the inertial frame. ``midpoint_inertial`` is
    the member's midpoint in inertial coordinates, ``axis_hat_inertial``
    is the unit vector along the member axis (sign immaterial), and
    ``reference_point_inertial`` is the body reference point about which
    moments are taken (typically the body's translated reference point).

    Parameters
    ----------
    element
        The :class:`MorisonElement` to evaluate.
    midpoint_inertial, axis_hat_inertial
        Length-3 inertial-frame vectors. ``axis_hat_inertial`` is
        normalised by the caller (no-op if already unit length).
    body_velocity_at_midpoint
        Length-3 inertial-frame velocity of the body material point at
        the midpoint, ``v_body + ω_body × (r_mid − r_ref)``.
    body_acceleration_at_midpoint
        Length-3 inertial-frame acceleration of the body material point
        at the midpoint. Used only if ``element.include_inertia`` is
        ``True``; pass ``None`` otherwise. ``None`` is treated as zero
        (Phase 1 default — see module docstring).
    fluid_velocity
        Length-3 inertial-frame fluid velocity at the midpoint.
    fluid_acceleration
        Length-3 inertial-frame fluid acceleration at the midpoint.
        Required when ``element.include_inertia`` is ``True``; ignored
        otherwise.
    rho
        Water density in kg/m³ (deck-supplied).
    reference_point_inertial
        Length-3 inertial-frame coordinates of the body's reference
        point. Moments are taken about this point.

    Returns
    -------
    NDArray[np.float64]
        Length-6 generalized force ``[Fx, Fy, Fz, Mx, My, Mz]`` in the
        inertial frame, ready to deposit into the body's slot of the
        global ``state_force`` vector.

    Raises
    ------
    ValueError
        If ``rho`` is not positive, ``axis_hat_inertial`` is not
        unit-length to within ``rtol=1e-9``, or any vector input has
        the wrong shape, or ``include_inertia`` is ``True`` but
        ``fluid_acceleration`` is ``None``.
    """
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError(f"rho must be finite and positive; got {rho}")

    mid = np.asarray(midpoint_inertial, dtype=np.float64)
    ax = np.asarray(axis_hat_inertial, dtype=np.float64)
    v_body = np.asarray(body_velocity_at_midpoint, dtype=np.float64)
    u_fluid = np.asarray(fluid_velocity, dtype=np.float64)
    r_ref = np.asarray(reference_point_inertial, dtype=np.float64)
    for name, arr in (
        ("midpoint_inertial", mid),
        ("axis_hat_inertial", ax),
        ("body_velocity_at_midpoint", v_body),
        ("fluid_velocity", u_fluid),
        ("reference_point_inertial", r_ref),
    ):
        if arr.shape != (3,):
            raise ValueError(f"{name} must have shape (3,); got {arr.shape}")
    ax_norm = float(np.linalg.norm(ax))
    if abs(ax_norm - 1.0) > _AXIS_LENGTH_RTOL:
        raise ValueError(f"axis_hat_inertial must be unit-length; got |ax| = {ax_norm:.6e}")

    L = element.length_m
    D = element.diameter
    u_rel = u_fluid - v_body
    u_n = _project_normal(u_rel, ax)
    speed = float(np.linalg.norm(u_n))
    F_drag = 0.5 * rho * D * element.Cd * speed * u_n * L

    F_inertia: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
    if element.include_inertia:
        if fluid_acceleration is None:
            raise ValueError("include_inertia=True requires fluid_acceleration; got None")
        a_fluid = np.asarray(fluid_acceleration, dtype=np.float64)
        if a_fluid.shape != (3,):
            raise ValueError(f"fluid_acceleration must have shape (3,); got {a_fluid.shape}")
        if body_acceleration_at_midpoint is None:
            a_body = np.zeros(3, dtype=np.float64)
        else:
            a_body = np.asarray(body_acceleration_at_midpoint, dtype=np.float64)
            if a_body.shape != (3,):
                raise ValueError(
                    f"body_acceleration_at_midpoint must have shape (3,); " f"got {a_body.shape}"
                )
        a_fluid_n = _project_normal(a_fluid, ax)
        a_body_n = _project_normal(a_body, ax)
        Ax = element.cross_section_area_m2
        F_inertia = rho * Ax * L * ((1.0 + element.Ca) * a_fluid_n - element.Ca * a_body_n)

    F_inertial: NDArray[np.float64] = F_drag + F_inertia
    arm = mid - r_ref
    M_inertial: NDArray[np.float64] = np.cross(arm, F_inertial)

    out = np.empty(6, dtype=np.float64)
    out[0:3] = F_inertial
    out[3:6] = M_inertial
    return out


def plate_element_force(
    element: PlateDragElement,
    *,
    rotation_matrix_body: NDArray[np.float64],
    reference_velocity_inertial: NDArray[np.float64],
    angular_velocity_inertial: NDArray[np.float64],
    fluid_velocity: NDArray[np.float64],
    rho: float,
) -> NDArray[np.float64]:
    """6-DOF generalized drag force from one anisotropic plate element.

    Drag only (no inertia term -- the BEM carries added mass). The normal
    (broadside) contribution is integrated over the disc face by the cached
    polar quadrature; the tangential (edge-on) contribution is lumped at the
    disc centre. See :class:`PlateDragElement` for the model and its stated
    approximations.

    Moments are taken about the body reference point: the disc patches are
    stored as body-frame offsets from that point, so their inertial lever
    arms ``R · offset`` already reference it (the absolute reference
    position is not needed).

    Parameters
    ----------
    element
        The :class:`PlateDragElement` to evaluate.
    rotation_matrix_body
        3x3 body→inertial rotation matrix.
    reference_velocity_inertial
        Length-3 inertial-frame velocity of the reference point
        (``xi_dot[0:3]``).
    angular_velocity_inertial
        Length-3 inertial-frame body angular velocity
        (``R · xi_dot[3:6]``).
    fluid_velocity
        Length-3 inertial-frame fluid velocity, sampled once at the disc
        centre (uniform over the compact disc; zero for a calm sea).
    rho
        Water density in kg/m³ (positive).

    Returns
    -------
    NDArray[np.float64]
        Length-6 generalized force ``[Fx, Fy, Fz, Mx, My, Mz]`` (inertial,
        moments about the body reference point).

    Raises
    ------
    ValueError
        If ``rho`` is not finite and positive.
    """
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError(f"rho must be finite and positive; got {rho}")

    r_mat = np.asarray(rotation_matrix_body, dtype=np.float64)
    v_ref = np.asarray(reference_velocity_inertial, dtype=np.float64)
    omega = np.asarray(angular_velocity_inertial, dtype=np.float64)
    u_fluid = np.asarray(fluid_velocity, dtype=np.float64)

    n_hat = r_mat @ element._n_hat_body

    # --- Normal (broadside) term: integrate over the disc face patches. ---
    # Patch lever arms (inertial offset from the reference point).
    arms = element._patch_pos_body @ r_mat.T  # (P,3) = R @ pos, body->inertial (offset from ref)
    # Body material velocity at each patch: v_ref + omega x arm.
    u_body = v_ref + np.cross(omega, arms)
    u_rel = u_fluid - u_body  # (P,3); calm -> -u_body
    w = u_rel @ n_hat  # (P,) local normal speed
    # d(F_normal) = 0.5*rho*Cd_n*|w|*w * dA, directed along n_hat.
    df_n = 0.5 * rho * element.Cd_n * np.abs(w) * w * element._patch_area  # (P,)
    f_normal = n_hat * float(df_n.sum())
    # Moment: sum arm x (df_n * n_hat) = (sum df_n*arm) x n_hat.
    m_normal = np.cross(df_n @ arms, n_hat)

    # --- Tangential (edge-on) term: lumped at the disc centre. ---
    center_arm = r_mat @ element.center_body
    u_body_c = v_ref + np.cross(omega, center_arm)
    u_rel_c = u_fluid - u_body_c
    u_t = u_rel_c - float(u_rel_c @ n_hat) * n_hat  # in-plane component
    speed_t = float(np.linalg.norm(u_t))
    f_tangential = 0.5 * rho * element.Cd_t * element.edge_area_m2 * speed_t * u_t
    m_tangential = np.cross(center_arm, f_tangential)

    out = np.empty(6, dtype=np.float64)
    out[0:3] = f_normal + f_tangential
    out[3:6] = m_normal + m_tangential
    return out


# ---------------------------------------------------------------------------
# State-force closure for the integrator
# ---------------------------------------------------------------------------


FluidFieldFn = Callable[[NDArray[np.float64], float], NDArray[np.float64]]
"""Inertial-frame field sampler ``(point, t) -> vec3``.

Used for both fluid velocity and fluid acceleration. The point is in
inertial coordinates; the return value is in inertial coordinates.
"""


def _body_pose_from_xi(
    xi_body: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(translation_3, R_3x3)`` for one body's xi slice.

    ``xi_body[0:3]`` is the inertial-frame translation of the body
    reference point; ``xi_body[3:6]`` is ``(roll, pitch, yaw)`` in
    radians (ZYX-intrinsic), per ARCHITECTURE.md §3.
    """
    translation = xi_body[0:3].astype(np.float64, copy=True)
    q = quaternion_from_euler_zyx(
        roll_rad=float(xi_body[3]),
        pitch_rad=float(xi_body[4]),
        yaw_rad=float(xi_body[5]),
    )
    R = rotation_matrix(q)
    return translation, R


def _body_velocity_at(
    xi_dot_body: NDArray[np.float64],
    R_body: NDArray[np.float64],
    arm_inertial: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Inertial-frame velocity at a point on the body, given its lever arm.

    ``xi_dot_body[0:3]`` is the inertial-frame translational velocity of
    the reference point; ``xi_dot_body[3:6]`` is the body-frame angular
    velocity ``(p, q, r)``. The angular velocity is rotated to inertial
    via ``R_body`` before the cross product.
    """
    v_ref = xi_dot_body[0:3].astype(np.float64, copy=True)
    omega_body = xi_dot_body[3:6].astype(np.float64, copy=True)
    omega_inertial = R_body @ omega_body
    return v_ref + np.cross(omega_inertial, arm_inertial)


_PLATE_PARALLEL_RTOL: Final[float] = 1.0e-6
"""A cylinder counts as a spar (parallel to a plate normal) when
``|axis_hat · n_hat| >= 1 - _PLATE_PARALLEL_RTOL``."""


def _check_plate_supersession(elements: Sequence[MorisonElement | PlateDragElement]) -> None:
    """Enforce the plate/cylinder supersession rule (M11a PR4 requirement a).

    A :class:`PlateDragElement` owns its plate's normal (broadside) drag,
    integrated over the disc face. A :class:`MorisonElement` cylinder in the
    plate plane (axis perpendicular to the plate normal -- the M11a-PR1
    horizontal-cylinder heave-plate stand-in) captures that same broadside
    drag: running both double-counts the normal term. This makes the
    conflict impossible -- on any body carrying a plate element, every
    coexisting cylinder must be parallel to that plate's normal (a spar,
    whose member-normal plane is orthogonal to the plate normal and so
    captures only lateral drag). Any non-parallel cylinder raises.

    Raises
    ------
    ValueError
        If a body carries both a :class:`PlateDragElement` and a
        :class:`MorisonElement` whose axis is not parallel to the plate
        normal.
    """
    plates_by_body: dict[int, list[PlateDragElement]] = {}
    for e in elements:
        if isinstance(e, PlateDragElement):
            plates_by_body.setdefault(e.body_index, []).append(e)
    if not plates_by_body:
        return
    for e in elements:
        if not isinstance(e, MorisonElement):
            continue
        plates = plates_by_body.get(e.body_index)
        if not plates:
            continue
        axis = e.node_b_body - e.node_a_body
        axis_hat = axis / float(np.linalg.norm(axis))
        for plate in plates:
            dot = abs(float(np.dot(axis_hat, plate._n_hat_body)))
            if dot < 1.0 - _PLATE_PARALLEL_RTOL:
                raise ValueError(
                    f"body_index {e.body_index}: a MorisonElement cylinder "
                    f"(axis {axis_hat.round(4).tolist()}) coexists with a "
                    f"PlateDragElement (normal {plate._n_hat_body.round(4).tolist()}) "
                    f"but is not parallel to the plate normal "
                    f"(|axis.n| = {dot:.6f} < 1). A cylinder in the plate plane is the "
                    "M11a-PR1 horizontal-cylinder heave-plate stand-in; it double-counts "
                    "the plate's normal drag, which the PlateDragElement now owns. Remove "
                    "the stand-in cylinder (the plate element supersedes it), or make the "
                    "cylinder a spar parallel to the plate normal."
                )


def make_morison_state_force(
    elements: Sequence[MorisonElement | PlateDragElement],
    n_dof: int,
    *,
    fluid_velocity_fn: FluidFieldFn,
    fluid_acceleration_fn: FluidFieldFn | None = None,
    rho: float,
) -> Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    """Build the ``(t, xi, xi_dot) -> F`` closure consumed by ``integrate_cummins``.

    Validates body indices, the kinematics callables, and the plate/cylinder
    supersession rule up front so illegal configurations fail fast.

    Parameters
    ----------
    elements
        Sequence of :class:`MorisonElement` and/or :class:`PlateDragElement`
        instances (may be mixed). A :class:`PlateDragElement` owns the
        plate's normal (broadside) drag; see the supersession rule under
        *Raises*.
    n_dof
        Global DOF count ``6 * N`` for the system being integrated.
    fluid_velocity_fn
        Callable ``(point_inertial, t) -> u_fluid_inertial`` returning a
        length-3 array. Sampled once per element per step at the
        member's inertial-frame midpoint. For a calm sea pass a
        zero-returning lambda; for a regular Airy wave wrap
        :func:`floatsim.waves.kinematics.airy_velocity`.
    fluid_acceleration_fn
        Callable ``(point_inertial, t) -> a_fluid_inertial``. Required
        when at least one element has ``include_inertia=True``;
        ``None`` is acceptable (and saves a sampling call) when no
        element uses the inertia term.
    rho
        Water density in kg/m³. Must be positive.

    Returns
    -------
    Callable
        ``state_force(t, xi, xi_dot)`` returning a length-``n_dof``
        force vector summed across all elements.

    Raises
    ------
    ValueError
        If ``n_dof`` is not a positive multiple of 6, any element has a
        ``body_index`` outside ``[0, n_dof // 6)``, ``rho`` is
        non-positive, any :class:`MorisonElement` has ``include_inertia=True``
        but ``fluid_acceleration_fn`` is ``None``, or the plate/cylinder
        **supersession rule** is violated: a body carrying a
        :class:`PlateDragElement` may only carry :class:`MorisonElement`
        cylinders whose axis is parallel to that plate's normal (spars).
        A cylinder in the plate plane (the M11a-PR1 horizontal-cylinder
        heave-plate stand-in) captures the same broadside drag the plate
        element now owns, so allowing both would double-count the normal
        term. This is made impossible, not documented against.
    """
    if n_dof <= 0 or n_dof % 6 != 0:
        raise ValueError(f"n_dof must be a positive multiple of 6; got {n_dof}")
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError(f"rho must be finite and positive; got {rho}")
    n_bodies = n_dof // 6
    elem_list = list(elements)
    needs_acceleration = any(isinstance(e, MorisonElement) and e.include_inertia for e in elem_list)
    if needs_acceleration and fluid_acceleration_fn is None:
        raise ValueError(
            "at least one element has include_inertia=True; "
            "fluid_acceleration_fn must be provided."
        )
    for k, e in enumerate(elem_list):
        if not (0 <= e.body_index < n_bodies):
            raise ValueError(
                f"element {k}: body_index {e.body_index} outside valid range "
                f"[0, {n_bodies}) for n_dof = {n_dof}"
            )
    _check_plate_supersession(elem_list)

    def _state_force(
        t_eval: float,
        xi: NDArray[np.float64],
        xi_dot: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        F_global = np.zeros(n_dof, dtype=np.float64)

        # Per-body pose / velocity caches so we do the trig only once
        # even when a body has multiple drag elements.
        pose_cache: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}

        for e in elem_list:
            b = e.body_index
            slc = slice(6 * b, 6 * b + 6)
            if b not in pose_cache:
                pose_cache[b] = _body_pose_from_xi(xi[slc])
            r_ref, R_body = pose_cache[b]

            if isinstance(e, PlateDragElement):
                # Disc centre (inertial) -> uniform fluid sample over the
                # compact disc; then integrate the face + edge terms.
                center_inertial = r_ref + R_body @ e.center_body
                u_fluid_c = np.asarray(
                    fluid_velocity_fn(center_inertial, float(t_eval)), dtype=np.float64
                )
                omega_inertial = R_body @ xi_dot[slc][3:6]
                f6 = plate_element_force(
                    e,
                    rotation_matrix_body=R_body,
                    reference_velocity_inertial=xi_dot[slc][0:3],
                    angular_velocity_inertial=omega_inertial,
                    fluid_velocity=u_fluid_c,
                    rho=rho,
                )
                F_global[slc] += f6
                continue

            # Body-frame midpoint -> inertial via R_body, then translate.
            mid_body = 0.5 * (e.node_a_body + e.node_b_body)
            mid_inertial = r_ref + R_body @ mid_body
            axis_body = e.node_b_body - e.node_a_body
            axis_inertial = R_body @ axis_body
            axis_hat = axis_inertial / float(np.linalg.norm(axis_inertial))

            # Velocity of the body material point at the midpoint.
            arm_inertial = mid_inertial - r_ref
            v_body_at_mid = _body_velocity_at(xi_dot[slc], R_body, arm_inertial)

            u_fluid = np.asarray(fluid_velocity_fn(mid_inertial, float(t_eval)), dtype=np.float64)
            a_fluid: NDArray[np.float64] | None
            if e.include_inertia:
                assert fluid_acceleration_fn is not None  # guaranteed at build time
                a_fluid = np.asarray(
                    fluid_acceleration_fn(mid_inertial, float(t_eval)),
                    dtype=np.float64,
                )
            else:
                a_fluid = None

            f6 = morison_element_force(
                e,
                midpoint_inertial=mid_inertial,
                axis_hat_inertial=axis_hat,
                body_velocity_at_midpoint=v_body_at_mid,
                body_acceleration_at_midpoint=None,
                fluid_velocity=u_fluid,
                fluid_acceleration=a_fluid,
                rho=rho,
                reference_point_inertial=r_ref,
            )
            F_global[slc] += f6

        return F_global

    return _state_force


# ---------------------------------------------------------------------------
# Startup diagnostics (Q1: include_inertia + non-empty BEM database -> warn)
# ---------------------------------------------------------------------------


def startup_inertia_double_count_warnings(
    elements: Sequence[MorisonElement],
    bodies_with_bem: Sequence[bool],
) -> list[str]:
    """Return human-readable warnings for ``include_inertia`` double-counting.

    Per Q1 of ``docs/milestone-5-plan.md``: when a Morison element on a
    BEM-equipped body sets ``include_inertia=True``, the
    Froude-Krylov + added-mass term is already being applied via the
    BEM ``A(ω)`` and ``F_exc(ω)`` and adding the Morison inertia
    double-counts. The deck loader and CLI are expected to surface
    these messages before integration starts; the integrator itself
    is silent.

    Parameters
    ----------
    elements
        Sequence of :class:`MorisonElement` instances.
    bodies_with_bem
        ``bodies_with_bem[i]`` is ``True`` if body ``i`` has a non-empty
        BEM database (``HydroDatabase`` with finite ``A∞``,
        ``RAO``, etc.). Length must be ``>= max(e.body_index) + 1``.

    Returns
    -------
    list[str]
        One message per offending element, in element-iteration order.
        Empty if no element triggers the rule.
    """
    msgs: list[str] = []
    for k, e in enumerate(elements):
        if not e.include_inertia:
            continue
        if e.body_index >= len(bodies_with_bem):
            raise ValueError(
                f"element {k}: body_index {e.body_index} outside "
                f"bodies_with_bem (length {len(bodies_with_bem)})"
            )
        if bodies_with_bem[e.body_index]:
            msgs.append(
                f"Morison element {k} on body {e.body_index} has "
                f"include_inertia=True but the body has a non-empty BEM "
                f"database. The Froude-Krylov + added-mass term is "
                f"already in A(omega) and F_exc(omega); adding the "
                f"Morison inertia double-counts. Set include_inertia=False "
                f"or remove the BEM database from this body."
            )
    return msgs
