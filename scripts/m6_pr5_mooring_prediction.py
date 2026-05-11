"""M6 PR5 Step B — Predict OC4 moored static equilibrium + per-line
fairlead/anchor tensions from FloatSim's analytic catenary, then
compare to OpenFAST's S4 last-30 s means (Step C).

Per the locked R1b plan, this script runs AFTER the S4 fixture has
been re-extracted with TMax = 1200 s (so surge has time to settle).

Method
------
For each of the OC4 mooring lines (3 catenary lines at 120° spacing):

  1. Place the body at trial heave + zero surge/sway (3-fold symmetry
     forces surge ≈ 0; assert that empirically too).
  2. Rotate the line's anchor/fairlead 3D positions into the line's
     2D vertical plane (defined by anchor azimuth from origin).
  3. Call ``solve_catenary(line, anchor_2d, fairlead_2d,
     seabed_depth=200.0)`` to get H, V_fairlead, T_fairlead, regime.
  4. Sum vertical components ΣV_F across the 3 lines (= net downward
     force on body from mooring).

Iterate on trial heave until the vertical force balance closes:

  buoyancy = m_total * g + ΣV_F

For OC4: the platform mass is calibrated such that buoyancy = weight
at z = 0 INCLUDING the static mooring loads (i.e., the platform's
specified mass implicitly assumes mooring pretension). Heave
equilibrium with mooring should therefore be near zero.

Compare predicted (T_F_i, T_A_i, heave_eq) to OpenFAST's
last-30 s means.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from floatsim.mooring.catenary_analytic import (  # noqa: E402
    CatenaryLine,
    CatenarySolution,
    solve_catenary,
)
from tests.support.openfast_csv import load_openfast_history  # noqa: E402

# ---------------------------------------------------------------------------
# OC4 mooring geometry (from the s4_moored_eq MoorDyn deck)
# ---------------------------------------------------------------------------

# Line properties (all 3 lines identical "main" type from MoorDyn deck).
# Submerged weight per unit length:
#   w_sub = (mass_air_per_length - rho_water * A_cross) * g
#   A_cross = π * D^2 / 4 = π * 0.0766^2 / 4 = 4.61e-3 m^2
#   w_sub = (113.35 - 1025 * 4.61e-3) * 9.80665
#         = (113.35 - 4.72) * 9.80665
#         = 1065.4 N/m
# The 4.2% reduction from the air-weight value matches the empirical
# tension discrepancy with OpenFAST. Documented as the per-line buoyancy
# correction (see PR5 retrospective).
_LINE_DIAM_M = 0.0766
_LINE_MASS_AIR_KG_PER_M = 113.35
_LINE_A_CROSS_M2 = np.pi * _LINE_DIAM_M**2 / 4.0
_LINE_W_SUB_N_PER_M = (_LINE_MASS_AIR_KG_PER_M - 1025.0 * _LINE_A_CROSS_M2) * 9.80665

LINE_PROPS = CatenaryLine(
    length=835.35,
    weight_per_length=_LINE_W_SUB_N_PER_M,
    EA=7.536e8,
)
SEABED_DEPTH_M = 200.0  # z = -200 m (anchor depth)
RHO_KG_M3 = 1025.0  # not used by catenary solver (line weight is in N/m)
G_M_S2 = 9.80665

# Anchor positions (Fixed; from MoorDyn deck "POINTS" section, IDs 1-3)
ANCHORS_3D = np.array(
    [
        [+418.80, +725.38, -200.0],  # line 1: azimuth +60°
        [-837.60, 0.00, -200.0],  # line 2: azimuth 180°
        [+418.80, -725.38, -200.0],  # line 3: azimuth -60°
    ]
)

# Fairlead positions on the vessel body (body-frame; offsets from the
# platform reference point). IDs 4-6 in the MoorDyn deck.
FAIRLEADS_BODY = np.array(
    [
        [+20.43, +35.39, -14.0],  # line 1
        [-40.87, 0.00, -14.0],  # line 2
        [+20.43, -35.39, -14.0],  # line 3
    ]
)

# Platform mass: Robertson "platform-with-ballast" + tower + RNA per
# Setup B (the M6 PR3/PR4 combined-deck mass, calibrated against the
# S2/S3/S5 OpenFAST decks). Heave restoring stiffness from Robertson.
M_TOTAL_KG = 1.4074e7  # Setup B combined-deck mass (matches fragility script)
C33_HEAVE_N_PER_M = 3.836e6  # Robertson C_33 (PR2/PR3 reference)

# Platform displaced volume at undisplaced reference position (PtfmVol0 in
# HydroDyn deck for OC4). Sets the free-floating equilibrium offset.
PTFM_VOL0_M3 = 13917.0


def _net_z_force(heave_m: float) -> tuple[float, list[CatenarySolution], list[float]]:
    """Net vertical force on body at trial heave (mooring + buoyancy - weight).

    Force balance components:
      F_buoyancy(z) = ρ · V₀ · g  + C_33 · (-z)         (linear in z)
      F_weight     = -m · g                              (constant)
      F_mooring    = -Σ V_F(z)                          (mostly downward)

    Returns the SIGNED net F_z; root-find on this for equilibrium.
    """
    F_mooring_z, sols, azimuths = _vertical_force_balance(heave_m)
    F_buoyancy = RHO_KG_M3 * PTFM_VOL0_M3 * G_M_S2 - C33_HEAVE_N_PER_M * heave_m
    F_weight = -M_TOTAL_KG * G_M_S2
    F_net = F_buoyancy + F_weight + F_mooring_z
    return F_net, sols, azimuths


# ---------------------------------------------------------------------------
# 3D <-> 2D mapping
# ---------------------------------------------------------------------------


def _solve_line(
    anchor_3d: NDArray[np.float64],
    fairlead_3d: NDArray[np.float64],
) -> tuple[CatenarySolution, float]:
    """Solve catenary for one line in its own vertical plane.

    Returns (CatenarySolution, azimuth_deg) where azimuth_deg is the
    horizontal-plane angle (degrees, +X = 0°, CCW positive) from origin
    to the anchor projection. Used by the caller to project the
    horizontal force back to inertial frame.
    """
    # Horizontal vector from fairlead to anchor.
    dxy = anchor_3d[:2] - fairlead_3d[:2]
    horizontal_span = float(np.hypot(dxy[0], dxy[1]))
    azimuth_rad = float(np.arctan2(dxy[1], dxy[0]))
    # 2D positions in the line's plane: anchor at (horizontal_span, z_anchor),
    # fairlead at (0, z_fairlead). solve_catenary requires fairlead RIGHT
    # of anchor, so flip: anchor at (0, z_a), fairlead at (h, z_f).
    anchor_2d = np.array([0.0, float(anchor_3d[2])])
    fairlead_2d = np.array([horizontal_span, float(fairlead_3d[2])])
    sol = solve_catenary(
        line=LINE_PROPS,
        anchor_pos=anchor_2d,
        fairlead_pos=fairlead_2d,
        seabed_depth=SEABED_DEPTH_M,
    )
    return sol, np.degrees(azimuth_rad)


def _line_force_on_body(sol: CatenarySolution, azimuth_deg: float) -> NDArray[np.float64]:
    """3D force vector applied by one line on the body at the fairlead.

    Line pulls body in the direction of the line's outward tangent at
    the fairlead — radially outward (toward anchor) in horizontal plane,
    and DOWNWARD vertically (toward seabed anchor).
    """
    az = np.radians(azimuth_deg)
    Fx = sol.H * np.cos(az)
    Fy = sol.H * np.sin(az)
    Fz = -sol.V_fairlead
    return np.array([Fx, Fy, Fz])


# ---------------------------------------------------------------------------
# Equilibrium solve (1-DOF heave iteration, surge = 0 by symmetry)
# ---------------------------------------------------------------------------


def _vertical_force_balance(
    heave_m: float, surge_m: float = 0.0
) -> tuple[float, list[CatenarySolution], list[float]]:
    """Compute net vertical force on body for trial (surge, heave).

    Returns (sum_F_z_mooring_only, per_line_solutions, per_line_azimuths_deg).

    Caller iterates ``heave_m`` until ``F_z_total = -m·g + buoyancy + ΣF_z_lines``
    closes; here we return just the mooring contribution (no buoyancy
    explicitly modelled — the iteration uses the heave restoring
    stiffness ``C_33`` instead, which is the local-linear approximation
    of buoyancy near the reference floating equilibrium).
    """
    body_offset = np.array([surge_m, 0.0, heave_m])
    F_sum = np.zeros(3, dtype=np.float64)
    sols: list[CatenarySolution] = []
    azimuths: list[float] = []
    for i in range(3):
        anchor = ANCHORS_3D[i]
        fairlead = FAIRLEADS_BODY[i] + body_offset
        sol, az = _solve_line(anchor, fairlead)
        sols.append(sol)
        azimuths.append(az)
        F_sum += _line_force_on_body(sol, az)
    return float(F_sum[2]), sols, azimuths


def find_heave_equilibrium(
    tol_n: float = 1.0e2,
) -> tuple[float, list[CatenarySolution], list[float]]:
    """Iterate heave to close the vertical force balance.

    Newton-step iteration:
        z_{k+1} = z_k + F_net(z_k) / C_33
    using the linearised buoyancy ``dF_buoyancy/dz = -C_33`` so the
    Jacobian of F_net w.r.t. heave is dominated by the buoyancy slope.
    The mooring contribution ``dF_mooring/dz`` is small (catenary tension
    barely changes for small heave) so we ignore it in the Jacobian.
    """
    heave = 0.0
    for _ in range(50):
        F_net, sols, azimuths = _net_z_force(heave)
        # Newton step: F_net(z + dz) ≈ F_net(z) - C_33 · dz = 0 → dz = F_net / C_33
        dz = F_net / C33_HEAVE_N_PER_M
        heave_new = heave + dz
        if abs(F_net) < tol_n:
            return heave_new, sols, azimuths
        heave = heave_new
    raise RuntimeError(f"heave equilibrium iteration did not converge; last heave = {heave}")


# ---------------------------------------------------------------------------
# Comparison against OpenFAST
# ---------------------------------------------------------------------------


def _of_means(csv_path: Path, window_s: float = 200.0) -> dict[str, float]:
    """Compute time-averaged means + stds over the last ``window_s`` of the
    OpenFAST CSV.

    Default window is 200 s (NOT the PR2/PR3 30-s precedent) -- OC4 moored
    surge has a ~ 100-s natural period with very light damping, so a 30-s
    window samples one half-cycle and is biased by the oscillation phase.
    The 200-s window covers 2 full periods and gives a clean mean.

    Heave + tensions are well-settled over both windows; the longer
    average is harmless for them.
    """
    h = load_openfast_history(csv_path)
    t = h.t
    mask = t >= t[-1] - window_s
    out: dict[str, float] = {
        "surge": float(np.mean(h.xi[mask, 0])),
        "sway": float(np.mean(h.xi[mask, 1])),
        "heave": float(np.mean(h.xi[mask, 2])),
        "roll_deg": float(np.degrees(np.mean(h.xi[mask, 3]))),
        "pitch_deg": float(np.degrees(np.mean(h.xi[mask, 4]))),
        "yaw_deg": float(np.degrees(np.mean(h.xi[mask, 5]))),
    }
    for ch in (
        "fair_ten_line1_n",
        "fair_ten_line2_n",
        "fair_ten_line3_n",
        "anch_ten_line1_n",
        "anch_ten_line2_n",
        "anch_ten_line3_n",
    ):
        if ch in h.extra_columns:
            out[ch + "_mean"] = float(np.mean(h.extra_columns[ch][mask]))
            out[ch + "_std"] = float(np.std(h.extra_columns[ch][mask]))
    out["surge_std"] = float(np.std(h.xi[mask, 0]))
    out["heave_std"] = float(np.std(h.xi[mask, 2]))
    out["t_max"] = float(t[-1])
    return out


def main() -> None:
    print("M6 PR5 Step B + C -- moored static equilibrium prediction vs OpenFAST")
    print("=" * 78)
    print()
    print("Geometry:")
    print(f"  Line: L = {LINE_PROPS.length} m")
    print(f"        w = {LINE_PROPS.weight_per_length:.2f} N/m, EA = {LINE_PROPS.EA:.3e}")
    print(f"  Seabed at z = {-SEABED_DEPTH_M} m")
    print(f"  M_total = {M_TOTAL_KG:.4e} kg, C_33 = {C33_HEAVE_N_PER_M:.3e} N/m")
    print()

    heave_eq, sols, azimuths = find_heave_equilibrium()
    print(f"FloatSim predicted heave equilibrium: {heave_eq:.4f} m  (surge = 0 by symmetry)")
    print()

    # Per-line predictions.
    print(
        f"{'line':>5}  {'azimuth':>8}  {'regime':>10}  {'H [N]':>11}  "
        f"{'V_F [N]':>11}  {'T_F [N]':>11}  {'T_A [N]':>11}"
    )
    print("-" * 80)
    for i, (sol, az) in enumerate(zip(sols, azimuths, strict=True), start=1):
        T_F = sol.T_fairlead
        T_A = sol.H if sol.regime == "touchdown" else float(np.hypot(sol.H, sol.V_anchor))
        print(
            f"{i:>5}  {az:>+8.2f}°  {sol.regime:>10}  {sol.H:>11.4e}  "
            f"{sol.V_fairlead:>11.4e}  {T_F:>11.4e}  {T_A:>11.4e}"
        )
    print()

    # Compare to OpenFAST S4 reference (re-extracted at TMax = 1200 s).
    csv = REPO_ROOT / "tests/fixtures/openfast/oc4_deepcwind/inputs/s4_moored_eq/s4_moored_eq.csv"
    if not csv.is_file():
        print(f"OpenFAST S4 CSV not yet present at {csv}; run/extraction pending.")
        return
    of = _of_means(csv)
    print(f"OpenFAST last-30s means (TMax = {of['t_max']:.0f} s):")
    print(f"  surge = {of['surge']:>+8.4f} m   (std = {of['surge_std']:.4f} m)")
    print(f"  sway  = {of['sway']:>+8.4f} m")
    print(f"  heave = {of['heave']:>+8.4f} m   (std = {of['heave_std']:.4f} m)")
    print(f"  roll  = {of['roll_deg']:>+8.4f} deg")
    print(f"  pitch = {of['pitch_deg']:>+8.4f} deg")
    print(f"  yaw   = {of['yaw_deg']:>+8.4f} deg")
    for i in (1, 2, 3):
        f_k = f"fair_ten_line{i}_n_mean"
        a_k = f"anch_ten_line{i}_n_mean"
        f_std = f"fair_ten_line{i}_n_std"
        if f_k in of:
            std_pct = of[f_std] / of[f_k] * 100
            print(
                f"  line {i}: FairTen mean = {of[f_k]:.4e} N "
                f"(std/mean = {std_pct:.2f}%), AnchTen mean = {of[a_k]:.4e} N"
            )

    print()
    print("Comparison (FS prediction vs OF last-30s mean):")
    delta_cm = abs(heave_eq - of["heave"]) * 100
    print(f"  heave: FS {heave_eq:+.4f}, OF {of['heave']:+.4f}, |delta| = {delta_cm:.2f} cm")
    print()
    for i, sol in enumerate(sols, start=1):
        f_k = f"fair_ten_line{i}_n_mean"
        a_k = f"anch_ten_line{i}_n_mean"
        if f_k not in of:
            continue
        T_F_pred = sol.T_fairlead
        T_A_pred = sol.H if sol.regime == "touchdown" else float(np.hypot(sol.H, sol.V_anchor))
        f_rel = (T_F_pred - of[f_k]) / of[f_k]
        a_rel = (T_A_pred - of[a_k]) / of[a_k]
        print(f"  line {i} FairTen: FS {T_F_pred:.4e}, OF {of[f_k]:.4e}, rel-err {f_rel:+.2%}")
        print(f"  line {i} AnchTen: FS {T_A_pred:.4e}, OF {of[a_k]:.4e}, rel-err {a_rel:+.2%}")


if __name__ == "__main__":
    main()
