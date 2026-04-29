"""Gravity contribution to the linearised hydrostatic restoring matrix.

ARCHITECTURE.md §4 lists this module ("Restoring matrix, gravity/buoyancy
balance"). It was missing through M5 — the BEM readers explicitly
documented that gravity is added at assembly, but the assembly step did
not exist. The gap surfaced during the M6 OpenFAST cross-check audit
(April 2026) when an OC4 pitch-period test against a buoyancy-only
:class:`HydroDatabase` produced ``NaN`` because ``C[4, 4]`` was negative
without the ``-m·g·z_G`` term. See
``docs/post-mortems/hydrostatic-gravity-bug.md`` for the post-mortem.

What this module provides
-------------------------
:func:`gravity_restoring_contribution` returns the symmetric 6×6 matrix
``ΔC_grav`` to be **added** to the buoyancy-only BEM ``C`` by
:func:`floatsim.hydro.radiation.assemble_cummins_lhs` when the
:attr:`HydroDatabase.C_source` flag is ``"buoyancy_only"`` and the
caller supplies the body's mass, centre-of-gravity offset, and gravity.

Derivation (rotation-vector convention)
---------------------------------------
Let ``r_G = (x_G, y_G, z_G)`` be the body-frame position of the centre
of gravity, measured **from the BEM database's hydrostatic reference
origin** (not from any body-internal reference). For a small rotation
parameterised as a rotation vector ``θ = (θ_x, θ_y, θ_z)`` (so that
``R(θ) = exp(θ̂) ≈ I + θ̂ + ½θ̂²``), the inertial-frame z-coordinate of
the CoG to second order in ``θ`` is::

    z_G_inertial = z_P + z_G + (θ × r_G)_z + ½(θ × (θ × r_G))_z + O(θ³)
                 = z_P + z_G + (θ_x·y_G − θ_y·x_G)
                   + ½(θ_x·θ_z·x_G − θ_x²·z_G − θ_y²·z_G + θ_y·θ_z·y_G)

The gravitational potential energy is ``V_grav = m·g·z_G_inertial``.
Dropping the constants and the linear-in-``θ`` part (absorbed into the
static equilibrium balance), the quadratic part is::

    V_quad = ½·m·g·(θ_x·θ_z·x_G − θ_x²·z_G − θ_y²·z_G + θ_y·θ_z·y_G)

With the standard convention ``V_quad = ½·ξᵀ·C·ξ`` and ``C`` symmetric,
the gravity contribution to the restoring matrix is::

    ΔC[3, 3] = ∂²V/∂θ_x² = -m·g·z_G        (roll-roll)
    ΔC[4, 4] = ∂²V/∂θ_y² = -m·g·z_G        (pitch-pitch)
    ΔC[3, 5] = ∂²V/∂θ_x∂θ_z = ½·m·g·x_G    (roll-yaw, rotation-vector)
    ΔC[4, 5] = ∂²V/∂θ_y∂θ_z = ½·m·g·y_G    (pitch-yaw, rotation-vector)
    ΔC[5, 3] = ΔC[3, 5]                    (symmetric)
    ΔC[5, 4] = ΔC[4, 5]                    (symmetric)
    all other entries: 0

Why a factor of ½ on the cross-couplings
----------------------------------------
The textbook Faltinsen (1990, *Sea Loads*, Eq. 2.104) gives
``ΔC_46 = m·g·x_G`` (no factor of ½). Faltinsen's derivation uses
ZYX-intrinsic Euler angles for the linearisation; rotation-vector and
ZYX-Euler agree at first order in ``θ`` but differ at second order, and
the cross-coupling ``C[3, 5]`` is itself a second-order quantity — so
the parameterisation matters.

FloatSim uses **rotation-vector** parameterisation throughout the
linearised assembly: ``ξ[3:6]`` is the integrated body-frame angular
velocity (which is exactly the rotation vector at small angles), and
the integrator advances quaternions via :func:`integrate_quaternion`
rather than Euler-angle composition. The rotation-vector ½ is therefore
the convention-consistent answer here.

For the OC4 DeepCwind case and any other axisymmetric platform with
``x_G = y_G = 0``, the cross-coupling vanishes and the parameterisation
choice is moot — ``ΔC[3, 3] = ΔC[4, 4] = -m·g·z_G`` is unambiguous.
The convention only matters for non-axisymmetric mass distributions,
which Phase 1 does not target. See
``docs/post-mortems/hydrostatic-gravity-bug.md`` §"Convention notes"
for the discussion.

References
----------
- Faltinsen, O.M., 1990. *Sea Loads on Ships and Offshore Structures*,
  Cambridge University Press, Eq. 2.104 (ZYX-Euler convention).
- Newman, J.N., 1977. *Marine Hydrodynamics*, MIT Press, §6.16.
- HydroDyn theory document, NREL/TP-XXXX (currently published with
  OpenFAST), §"Hydrostatic restoring".
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

_DOF_ROLL: Final[int] = 3
_DOF_PITCH: Final[int] = 4
_DOF_YAW: Final[int] = 5


def gravity_restoring_contribution(
    *,
    mass: float,
    cog_offset_from_bem_origin: NDArray[np.floating],
    gravity: float,
) -> NDArray[np.float64]:
    """Compute the symmetric 6×6 gravity contribution to ``C``.

    Parameters
    ----------
    mass
        Rigid-body mass in kilograms. Must be positive and finite.
    cog_offset_from_bem_origin
        Length-3 body-frame vector ``(x_G, y_G, z_G)`` from the BEM
        database's hydrostatic reference origin (e.g. the WAMIT
        reference point, the Capytaine origin, or
        ``HydroDatabase.reference_point``) to the body's centre of
        gravity, in metres. **Not** the CoG offset from any
        body-internal reference; the gravity term is referenced to the
        same point as the buoyancy-only ``C`` it is being added to.
    gravity
        Gravitational acceleration in m/s². Must be positive and finite.
        Decks supply this via :class:`floatsim.io.deck.Environment`.

    Returns
    -------
    ndarray of shape (6, 6), float64
        Symmetric ``ΔC_grav``. Only the rotation block (rows/cols 3-5)
        contains non-zero entries:

        - ``ΔC[3, 3] = ΔC[4, 4] = -m·g·z_G`` (the dominant terms; the
          only ones for axisymmetric ``x_G = y_G = 0`` cases).
        - ``ΔC[3, 5] = ΔC[5, 3] = ½·m·g·x_G`` (rotation-vector
          convention; see module docstring).
        - ``ΔC[4, 5] = ΔC[5, 4] = ½·m·g·y_G`` (same).

    Raises
    ------
    ValueError
        If ``mass`` is non-positive or non-finite, ``gravity`` is
        non-positive or non-finite, or ``cog_offset_from_bem_origin``
        does not have shape ``(3,)`` or contains non-finite values.

    Notes
    -----
    This function does **not** add the buoyancy/waterplane term — that
    is the BEM database's responsibility. ``ΔC_grav`` is meant to be
    added to a buoyancy-only ``C``; the assembled total is the
    physically-meaningful linearised hydrostatic restoring.
    """
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"mass must be finite and positive; got {mass!r}")
    if not np.isfinite(gravity) or gravity <= 0.0:
        raise ValueError(f"gravity must be finite and positive; got {gravity!r}")
    r_G = np.asarray(cog_offset_from_bem_origin, dtype=np.float64)
    if r_G.shape != (3,):
        raise ValueError(f"cog_offset_from_bem_origin must have shape (3,); got {r_G.shape}")
    if not np.all(np.isfinite(r_G)):
        raise ValueError(f"cog_offset_from_bem_origin must be all-finite; got {r_G!r}")

    x_G, y_G, z_G = float(r_G[0]), float(r_G[1]), float(r_G[2])
    m_g = mass * gravity

    dC = np.zeros((6, 6), dtype=np.float64)
    # Diagonal terms — dominant for any non-zero CoG depth.
    dC[_DOF_ROLL, _DOF_ROLL] = -m_g * z_G
    dC[_DOF_PITCH, _DOF_PITCH] = -m_g * z_G
    # Cross-couplings (rotation-vector convention; zero for axisymmetric mass).
    dC[_DOF_ROLL, _DOF_YAW] = 0.5 * m_g * x_G
    dC[_DOF_YAW, _DOF_ROLL] = 0.5 * m_g * x_G
    dC[_DOF_PITCH, _DOF_YAW] = 0.5 * m_g * y_G
    dC[_DOF_YAW, _DOF_PITCH] = 0.5 * m_g * y_G
    return dC
