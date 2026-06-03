"""M7-Foundation PR3 Step A -- hand prediction of catenary 6-vector force.

Per docs/m7-foundation-plan.md Q6 PR3 row: predict the 6-vector
generalised force on body 0 at the M6 PR5 OC4 moored equilibrium
via DIRECT solve_catenary calls (the PR5 hand-wired path), so the
PR3 identity test has explicit numerical targets BEFORE running
the make_catenary_state_force composer.

Two prediction points:
  1. xi_eq -- the PR5 equilibrium (3-fold symmetric, surge=0).
     Discriminates the dominant vertical tension term.
  2. xi_offset -- xi_eq + 5 m surge.
     Asymmetric load on the 3 lines; full 6-vector non-trivial
     (translational forces in all 3 directions, non-zero moments).
     This is the discriminator. A composer that drops the moment
     transfer at the fairlead arm would fail this point silently
     and pass point 1.

For each point we print the per-line 6-vector AND the summed
total at rtol=1e-12 precision (15 sig figs) so the identity test
can hard-code the numbers.

Geometry source: tests/validation/test_m6_openfast_moored_eq.py
(M6 PR5 fixture, locked at commit 36d08c2).
"""

from __future__ import annotations

import numpy as np

from floatsim.mooring.catenary_analytic import CatenaryLine, solve_catenary

# ---------------------------------------------------------------------------
# OC4 mooring geometry (verbatim from test_m6_openfast_moored_eq.py)
# ---------------------------------------------------------------------------

_RHO_KG_M3: float = 1025.0
_G_M_S2: float = 9.80665

_LINE_DIAM_M: float = 0.0766
_LINE_MASS_AIR_KG_PER_M: float = 113.35
_LINE_A_CROSS_M2: float = float(np.pi * _LINE_DIAM_M**2 / 4.0)
_LINE_W_SUB_N_PER_M: float = (
    _LINE_MASS_AIR_KG_PER_M - _RHO_KG_M3 * _LINE_A_CROSS_M2
) * _G_M_S2

_LINE_PROPS = CatenaryLine(
    length=835.35,
    weight_per_length=_LINE_W_SUB_N_PER_M,
    EA=7.536e8,
)
_SEABED_DEPTH_M: float = 200.0

# Inertial-frame anchor positions (MoorDyn POINTS 1-3).
_ANCHORS_3D = np.array(
    [
        [+418.80, +725.38, -200.0],
        [-837.60, 0.00, -200.0],
        [+418.80, -725.38, -200.0],
    ],
    dtype=np.float64,
)

# Body-frame fairlead positions (offsets from body reference, IDs 4-6).
_FAIRLEADS_BODY = np.array(
    [
        [+20.43, +35.39, -14.0],
        [-40.87, 0.00, -14.0],
        [+20.43, -35.39, -14.0],
    ],
    dtype=np.float64,
)

# Platform properties (for the equilibrium close at point 1).
_M_TOTAL_KG: float = 1.4074e7
_C33_HEAVE_N_PER_M: float = 3.836e6
_PTFM_VOL0_M3: float = 13917.0


# ---------------------------------------------------------------------------
# Per-line solver -- replicates PR5's _solve_line_at_body_offset exactly
# ---------------------------------------------------------------------------


def _per_line_6vector_force(line_idx: int, xi: np.ndarray) -> np.ndarray:
    """Return the 6-DOF generalised force on body 0's reference from line `line_idx`.

    Small-angle linear: the fairlead's inertial position at body
    pose xi is::

        r_fairlead_inertial = (xi[0], xi[1], xi[2]) + fairlead_body
                              + small-angle Euler contribution (omitted at
                              theta = 0 for this Step A prediction; the
                              PR5 fixture has xi[3:6] = 0 throughout)

    Catenary forces ar planar (vertical plane containing anchor and
    fairlead). Solve in 2D, map back to 3D inertial::

      H pulls the fairlead horizontally TOWARD the anchor
      V_fairlead pulls the fairlead DOWN (positive V_fairlead = down,
      matching solve_catenary's sign convention; see PR5's
      `F_mooring_z -= sol.V_fairlead`).
    """
    assert xi.shape == (6,)
    anchor = _ANCHORS_3D[line_idx]
    fairlead_body = _FAIRLEADS_BODY[line_idx]

    # Fairlead position in inertial frame -- small-angle linear is just
    # body translation + fairlead_body (theta = 0 here so no rotation).
    r_fairlead = np.array([xi[0], xi[1], xi[2]]) + fairlead_body

    # 3D vector from fairlead to anchor; horizontal-plane projection.
    dxy = anchor[:2] - r_fairlead[:2]
    horizontal_span = float(np.hypot(dxy[0], dxy[1]))
    if horizontal_span < 1.0e-9:
        raise ValueError(f"line {line_idx}: degenerate horizontal span")

    azimuth_rad = float(np.arctan2(dxy[1], dxy[0]))

    # 2D catenary solve in the local plane (anchor at origin, fairlead at
    # +x direction with z-coordinate from the inertial frame).
    anchor_2d = np.array([0.0, float(anchor[2])])
    fairlead_2d = np.array([horizontal_span, float(r_fairlead[2])])
    sol = solve_catenary(
        line=_LINE_PROPS,
        anchor_pos=anchor_2d,
        fairlead_pos=fairlead_2d,
        seabed_depth=_SEABED_DEPTH_M,
    )

    # Map (H, V_fairlead) back to a 3D force at the fairlead, INERTIAL frame:
    #   * Horizontal: H along (anchor - fairlead) / horizontal_span unit
    #     vector (toward anchor; pulls body toward the anchor).
    #   * Vertical: -V_fairlead (V_fairlead positive = down = -z).
    cos_az = np.cos(azimuth_rad)
    sin_az = np.sin(azimuth_rad)
    F_fairlead_inertial = np.array(
        [sol.H * cos_az, sol.H * sin_az, -sol.V_fairlead], dtype=np.float64
    )

    # Map to 6-DOF generalised force on body reference at xi[0:3]:
    #   * Translation: F_fairlead unchanged.
    #   * Moment: (r_fairlead_inertial - r_body_ref) x F_fairlead_inertial
    #     = (fairlead_body + small-angle xi displacement above ref) x F.
    #     At theta=0 this is just fairlead_body x F + (xi[0:3] - xi[0:3]) x F
    #     = fairlead_body x F (the arm from body ref to fairlead).
    r_arm_inertial = r_fairlead - np.array([xi[0], xi[1], xi[2]])  # = fairlead_body
    moment = np.cross(r_arm_inertial, F_fairlead_inertial)

    F_6 = np.zeros(6, dtype=np.float64)
    F_6[:3] = F_fairlead_inertial
    F_6[3:] = moment
    return F_6


def _total_6vector_force(xi: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Sum the 3 per-line 6-vector forces on body 0 at pose xi."""
    per_line = [_per_line_6vector_force(i, xi) for i in range(3)]
    total = sum(per_line, np.zeros(6, dtype=np.float64))
    return total, per_line


# ---------------------------------------------------------------------------
# PR5's equilibrium close (replicated verbatim)
# ---------------------------------------------------------------------------


def _net_z_force(heave_m: float, surge_m: float = 0.0) -> tuple[float, float]:
    """Newton residual on heave: F_buoy + F_weight + Σ -V_fairlead = 0."""
    xi = np.array([surge_m, 0.0, heave_m, 0.0, 0.0, 0.0])
    F_total, _ = _total_6vector_force(xi)
    F_buoy = _RHO_KG_M3 * _PTFM_VOL0_M3 * _G_M_S2 - _C33_HEAVE_N_PER_M * heave_m
    F_weight = -_M_TOTAL_KG * _G_M_S2
    return F_buoy + F_weight + F_total[2], F_total[2]


def _solve_heave_eq(tol_n: float = 1.0e2, max_iter: int = 50) -> tuple[float, bool]:
    heave = 0.0
    for _ in range(max_iter):
        F_net, _ = _net_z_force(heave)
        if abs(F_net) < tol_n:
            return heave, True
        dz = F_net / _C33_HEAVE_N_PER_M
        heave += dz
    return heave, False


# ---------------------------------------------------------------------------
# Main: write the two prediction points
# ---------------------------------------------------------------------------


def main() -> None:
    print("M7-Foundation PR3 Step A -- catenary 6-vector force prediction")
    print("=" * 70)

    heave_eq, converged = _solve_heave_eq()
    print(f"\nPR5 heave equilibrium: {heave_eq:+.10e} m  (converged={converged})")

    # Point 1: PR5 equilibrium (surge=0, theta=0, heave=heave_eq).
    xi_eq = np.array([0.0, 0.0, heave_eq, 0.0, 0.0, 0.0])
    F_total_eq, F_per_line_eq = _total_6vector_force(xi_eq)
    print(f"\n--- Point 1: xi_eq = {xi_eq.tolist()} ---")
    for i, F_i in enumerate(F_per_line_eq):
        print(f"  line {i} 6-vector force on body 0 (N, N*m):")
        print(f"    Fx = {F_i[0]:+.12e}   Mx = {F_i[3]:+.12e}")
        print(f"    Fy = {F_i[1]:+.12e}   My = {F_i[4]:+.12e}")
        print(f"    Fz = {F_i[2]:+.12e}   Mz = {F_i[5]:+.12e}")
    print(f"  TOTAL 6-vector on body 0:")
    print(f"    Fx = {F_total_eq[0]:+.12e}   Mx = {F_total_eq[3]:+.12e}")
    print(f"    Fy = {F_total_eq[1]:+.12e}   My = {F_total_eq[4]:+.12e}")
    print(f"    Fz = {F_total_eq[2]:+.12e}   Mz = {F_total_eq[5]:+.12e}")

    # Point 2: discriminator -- xi = (5m surge, 0, heave_eq, 0, 0, 0).
    # Asymmetric load on the 3 lines; full 6-vector non-trivial.
    xi_offset = np.array([5.0, 0.0, heave_eq, 0.0, 0.0, 0.0])
    F_total_off, F_per_line_off = _total_6vector_force(xi_offset)
    print(f"\n--- Point 2: xi_offset = {xi_offset.tolist()} (discriminator) ---")
    for i, F_i in enumerate(F_per_line_off):
        print(f"  line {i} 6-vector force on body 0 (N, N*m):")
        print(f"    Fx = {F_i[0]:+.12e}   Mx = {F_i[3]:+.12e}")
        print(f"    Fy = {F_i[1]:+.12e}   My = {F_i[4]:+.12e}")
        print(f"    Fz = {F_i[2]:+.12e}   Mz = {F_i[5]:+.12e}")
    print(f"  TOTAL 6-vector on body 0:")
    print(f"    Fx = {F_total_off[0]:+.12e}   Mx = {F_total_off[3]:+.12e}")
    print(f"    Fy = {F_total_off[1]:+.12e}   My = {F_total_off[4]:+.12e}")
    print(f"    Fz = {F_total_off[2]:+.12e}   Mz = {F_total_off[5]:+.12e}")

    # Numpy repr at full precision so the identity test can copy-paste.
    print("\n--- Numerical targets for tests/unit/test_catenary_state_force.py ---")
    np.set_printoptions(precision=15, suppress=False, linewidth=200)
    print(f"heave_eq = {heave_eq!r}")
    print(f"F_total_at_xi_eq = np.{repr(F_total_eq)}")
    print(f"F_total_at_xi_offset = np.{repr(F_total_off)}")
    print(f"F_per_line_at_xi_eq:")
    for i, F_i in enumerate(F_per_line_eq):
        print(f"  line {i}: np.{repr(F_i)}")
    print(f"F_per_line_at_xi_offset:")
    for i, F_i in enumerate(F_per_line_off):
        print(f"  line {i}: np.{repr(F_i)}")


if __name__ == "__main__":
    main()
