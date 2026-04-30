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

Derivation (Newton-Euler / Faltinsen 1990 Eq. 2.104)
----------------------------------------------------
Let ``r_G = (x_G, y_G, z_G)`` be the body-frame position of the centre
of gravity, measured **from the BEM database's hydrostatic reference
origin** (not from any body-internal reference). For a small rotation
parameterised by ``ξ = (θ_x, θ_y, θ_z)`` (the linearised rotation
vector / integrated body-frame angular velocity), the linearised
moment of gravity about the body reference point is computed from
first principles as ``δM = (R(θ)·r_G - r_G) × F_grav`` with
``F_grav = (0, 0, -m·g)`` and ``R(θ) ≈ I + θ̂``::

    δr_G_inertial = θ × r_G
                  = (θ_y·z_G - θ_z·y_G, θ_z·x_G - θ_x·z_G,
                     θ_x·y_G - θ_y·x_G)
    δM = δr_G_inertial × F_grav
       = (-m·g·(θ_z·x_G - θ_x·z_G), -m·g·(θ_y·z_G - θ_z·y_G), 0)
       = (m·g·z_G·θ_x - m·g·x_G·θ_z,
          m·g·z_G·θ_y - m·g·y_G·θ_z,
          0)

By the Cummins convention ``C·ξ = -δM_total``, the gravity
contribution to the restoring matrix is::

    ΔC[3, 3] = -m·g·z_G       (roll-roll diagonal)
    ΔC[4, 4] = -m·g·z_G       (pitch-pitch diagonal)
    ΔC[3, 5] = +m·g·x_G       (roll-yaw cross-coupling)
    ΔC[4, 5] = +m·g·y_G       (pitch-yaw cross-coupling)
    ΔC[5, 3] = +m·g·x_G       (symmetric companion of [3, 5])
    ΔC[5, 4] = +m·g·y_G       (symmetric companion of [4, 5])
    all other entries: 0

Convention settled — ½ vs 1 on the cross-couplings
--------------------------------------------------
The cross-couplings carry a factor of 1, **not** ½. The factor was
explicitly probed in
:mod:`tests.validation.test_gravity_restoring_asymmetric_cog`
against a Newton-Euler-from-first-principles reference: applying a
small yaw perturbation ``θ_z`` and computing ``δM_x = -m·g·x_G·θ_z``
gives ``C[3, 5] = m·g·x_G``. A rotation-vector V-Hessian alternative
would put a ½ here, but that ½ is parameterisation-artifact noise and
does not match the linearised moment that the Cummins integrator
actually consumes.

The symmetric companions ``C[5, 3] = C[3, 5]`` and
``C[5, 4] = C[4, 5]`` are populated by the ``C = C^T`` requirement
of any conservative restoring (so the energy
``V = ½·ξ^T·C·ξ`` is well-defined) — they do not arise from a
single Newton-Euler perturbation but from the conservation
structure. This matches the symmetric form used by HydroDyn, AQWA,
WAMIT, and Faltinsen 1990 Eq. 2.104.

For the OC4 DeepCwind case and any axisymmetric platform with
``x_G = y_G = 0`` the cross-couplings vanish and the convention
choice is moot — only ``ΔC[3, 3] = ΔC[4, 4] = -m·g·z_G`` is
exercised. The asymmetric-CoG regression test is the only place
where the factor matters; see
``docs/post-mortems/hydrostatic-gravity-bug.md``
§"Asymmetric CoG verification" for the resolution log.

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
        - ``ΔC[3, 5] = ΔC[5, 3] = m·g·x_G`` (Faltinsen 1990
          Eq. 2.104; settled by the asymmetric-CoG test, see module
          docstring).
        - ``ΔC[4, 5] = ΔC[5, 4] = m·g·y_G`` (same).

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
    # Cross-couplings (Faltinsen / Newton-Euler convention; zero for
    # axisymmetric mass). The factor-of-1 (no ½) is settled by
    # tests/validation/test_gravity_restoring_asymmetric_cog.py against
    # the Newton-Euler-from-first-principles reference; the rotation-
    # vector V-Hessian alternative would put a ½ here but does not match
    # the linearised moment that the Cummins integrator actually uses.
    # See docs/post-mortems/hydrostatic-gravity-bug.md
    # §"Asymmetric CoG verification" for the resolution.
    dC[_DOF_ROLL, _DOF_YAW] = m_g * x_G
    dC[_DOF_YAW, _DOF_ROLL] = m_g * x_G
    dC[_DOF_PITCH, _DOF_YAW] = m_g * y_G
    dC[_DOF_YAW, _DOF_PITCH] = m_g * y_G
    return dC
