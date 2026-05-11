"""M6 PR6 Step A — derive (Cd*D*L)_eq aggregate from the s5 HydroDyn deck.

Per the locked PR6 plan (P2 with strengthening): build the heave-equivalent
quadratic-drag coefficient by summing per-member cylindrical drag and per-joint
axial drag, projected onto the heave (z) direction.

Step B: predict the hyperbolic decay constant delta from the aggregate +
m_eff = M + A_inf_33 (first principles, derived in this script's main body).

Step C: compare delta_predicted to OpenFAST's measured value (0.309 1/m
from the S5 reference; see pre-flight diagnostic at end of PR6 setup).

Hyperbolic decay derivation (locked from first principles in this script):

  m xi_ddot + R xi_dot |xi_dot| + C xi = 0,    R [kg/m]: total quadratic drag

  Using ξ(t) = A cos(ω_n t), energy lost per cycle:
      ΔE = R ∫₀^T |ẋ|³ dt = R · A³ · ω_n² · (8/3)
  Total energy at amplitude A: E = (1/2) C A²
  dE/dn = C A dA/dn ⇒ dA/dn = -(8/3) R A² / m (since ω_n² = C/m)
  Solving: 1/A(n) = 1/A(0) + (8/3) (R/m) n
  ⇒ K = δ = (8/3) R / m  [units 1/m]

Cylindrical Morison contribution per member:
  f_z = -0.5 ρ Cd D L · v_z |v_z| · sin³(θ_from_vertical)
  R_cyl_member = 0.5 ρ Cd D L sin³(θ_from_vertical)

Axial drag contribution per joint (**HydroDyn convention, NOT standard Morison**):
  HydroDyn's Morison.f90 (line 3085 + 4742) precomputes
      DragConst_End = JAxCd · rho / (4 · |An_drag|²)
  where An_drag = sum over connected members of (sgn · k · pi · R²) — the
  outward-facing area-vector. The runtime force is
      F_D_End_i = An_End_i · DragConst_End · |vmag| · vmag,  vmag = vrel · An_End
  Algebraic reduction at a joint with ONE attached vertical member (one axial
  drag node) yields
      F_z = -(1/4) · rho · A_x · JAxCd · v_z · |v_z|         A_x = pi*D^2/4
  i.e., **half the standard Morison factor of 1/2** for axial flow. The
  effective JAxCd is implicitly a "two-face combined disc" coefficient;
  the per-face Morison equivalent is JAxCd / 2. See conventions doc Item 30.

  R_axial_joint = 0.25 · rho · A_x · JAxCd · cos³(θ_from_vertical)

Total R = Σ_cyl_members (0.5 ρ Cd D L sin³θ) + Σ_axial_joints (0.25 ρ A_x AxCd cos³θ).
"""

from __future__ import annotations

import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HYDRODYN_DAT = (
    REPO_ROOT
    / "tests/fixtures/openfast/oc4_deepcwind/inputs/s5_drag_decay/s5_drag_decay_HydroDyn.dat"
)

RHO = 1025.0  # seawater density [kg/m^3]


def _parse_data_block(lines: list[str], start_marker_substr: str, n_rows: int) -> list[list[str]]:
    """Find a section header that contains start_marker_substr, then read n_rows
    of data rows skipping comment / unit lines.
    """
    rows: list[list[str]] = []
    found = False
    for line in lines:
        if not found and start_marker_substr in line:
            found = True
            continue
        if not found:
            continue
        toks = line.split()
        if not toks:
            continue
        # Skip column-header rows (start with non-numeric like "JointID", "MemberID", etc.)
        # and unit rows (start with "(-)", "(m)", "(deg)", etc., or "!")
        if toks[0].startswith("(") or toks[0].startswith("!"):
            continue
        # Numeric-leading row?
        try:
            int(toks[0])
        except ValueError:
            try:
                float(toks[0])
            except ValueError:
                continue
        rows.append(toks)
        if len(rows) == n_rows:
            break
    if len(rows) != n_rows:
        raise ValueError(f"expected {n_rows} rows after {start_marker_substr!r}; got {len(rows)}")
    return rows


def main() -> None:
    print("M6 PR6 Step A -- (Cd*D*L)_eq aggregate from S5 HydroDyn deck")
    print("=" * 78)
    print()
    text = HYDRODYN_DAT.read_text()
    lines = text.splitlines()

    # Axial coefficient sets (NAxCoef=2).
    ax_coefs: dict[int, dict[str, float]] = {}
    for row in _parse_data_block(lines, "NAxCoef", n_rows=2):
        ax_coefs[int(row[0])] = {
            "AxCd": float(row[1]),
            "AxCa": float(row[2]),
            "AxCp": float(row[3]),
        }
    print("Axial coefficient sets:")
    for k, v in ax_coefs.items():
        print(f"  AxCoefID {k}: AxCd={v['AxCd']:.3f}, AxCa={v['AxCa']:.3f}")
    print()

    # Joints (NJoints=44).
    joints: dict[int, dict[str, float]] = {}
    for row in _parse_data_block(lines, "NJoints", n_rows=44):
        joints[int(row[0])] = {
            "x": float(row[1]),
            "y": float(row[2]),
            "z": float(row[3]),
            "ax_id": int(row[4]),
        }

    # Property sets (NPropSetsCyl=4).
    prop_sets: dict[int, float] = {}
    for row in _parse_data_block(lines, "NPropSetsCyl", n_rows=4):
        prop_sets[int(row[0])] = float(row[1])  # diameter D
    print("Cylindrical property sets (diameter D):")
    for k, v in prop_sets.items():
        print(f"  PropSetID {k}: D = {v:.2f} m")
    print()

    # Member-based Cd (NCoefMembersCyl=25).
    member_cds: dict[int, dict[str, float]] = {}
    for row in _parse_data_block(lines, "NCoefMembersCyl", n_rows=25):
        member_cds[int(row[0])] = {
            "Cd1": float(row[1]),
            "Cd2": float(row[2]),
        }

    # Members (NMembers=25).
    members: list[dict[str, object]] = []
    for row in _parse_data_block(lines, "NMembers ", n_rows=25):
        members.append(
            {
                "id": int(row[0]),
                "j1": int(row[1]),
                "j2": int(row[2]),
                "prop_id": int(row[3]),  # MPropSetID1
            }
        )

    # ---------------------------------------------------------------------
    # Cylindrical Morison contribution: R_cyl = 0.5 rho Cd D L sin^3(theta)
    # where theta = angle from vertical (z-axis).
    # ---------------------------------------------------------------------
    print("Cylindrical Morison contribution to heave drag")
    print(
        "(only members with at least one underwater node contribute -- "
        "skip those entirely above SWL)"
    )
    print()
    print(
        f"{'memb':>4}  {'j1':>3}  {'j2':>3}  {'L':>7}  {'D':>5}  {'Cd':>5}  "
        f"{'theta_deg':>9}  {'sin^3 th':>9}  {'R_cyl [kg/m]':>14}  {'note':<18}"
    )
    print("-" * 95)
    R_cyl_total = 0.0
    for m in members:
        mid = m["id"]
        j1 = joints[m["j1"]]
        j2 = joints[m["j2"]]
        D = prop_sets[m["prop_id"]]
        Cd_avg = 0.5 * (member_cds[mid]["Cd1"] + member_cds[mid]["Cd2"])
        dx = j2["x"] - j1["x"]
        dy = j2["y"] - j1["y"]
        dz = j2["z"] - j1["z"]
        L = math.sqrt(dx * dx + dy * dy + dz * dz)
        # Angle of axis from vertical: cos(theta) = |dz| / L
        cos_theta = abs(dz) / L if L > 0 else 0.0
        cos_theta = min(1.0, max(0.0, cos_theta))
        theta = math.acos(cos_theta)
        sin_theta_cubed = math.sin(theta) ** 3
        # Skip members entirely above SWL (z > 0).
        z_min = min(j1["z"], j2["z"])
        z_max = max(j1["z"], j2["z"])
        note = ""
        if z_min >= 0.0:
            note = "above SWL"
            R_member = 0.0
        else:
            # Use submerged length only.
            if z_max > 0.0:
                # Member crosses SWL; use submerged fraction.
                L_submerged = L * (-z_min / (z_max - z_min))
                note = f"partial L_sub={L_submerged:.2f}"
            else:
                L_submerged = L
            R_member = 0.5 * RHO * Cd_avg * D * L_submerged * sin_theta_cubed
        R_cyl_total += R_member
        print(
            f"{mid:>4}  {m['j1']:>3}  {m['j2']:>3}  {L:>7.2f}  {D:>5.1f}  {Cd_avg:>5.2f}  "
            f"{math.degrees(theta):>9.2f}  {sin_theta_cubed:>9.5f}  {R_member:>14.3e}  {note:<18}"
        )
    print()
    print(f"  Total cylindrical Morison R: {R_cyl_total:.4e} kg/m")
    print()

    # ---------------------------------------------------------------------
    # Axial drag contribution per joint: R_axial_joint = 0.5 rho A_x AxCd cos^3(theta)
    # The axial direction at a joint is the axis of the (any) attached
    # member; for OC4 the heave plates (ax_id=2) are at z=-20 attached to
    # vertical columns, so theta=0 from vertical and cos^3=1.
    # ---------------------------------------------------------------------
    print("Axial drag contribution at joints with non-zero AxCd")
    print(
        f"{'joint':>5}  {'(x,y,z)':>22}  {'D_attached':>10}  {'A_x':>9}  {'AxCd':>5}  "
        f"{'theta_deg':>9}  {'cos^3':>6}  {'R_ax [kg/m]':>13}"
    )
    print("-" * 95)
    R_ax_total = 0.0
    for jid, j in joints.items():
        if j["ax_id"] not in ax_coefs:
            continue
        AxCd = ax_coefs[j["ax_id"]]["AxCd"]
        if AxCd == 0.0:
            continue
        # Find any member that attaches to this joint and use its diameter.
        # OC4 has each special joint attached to a vertical column.
        attached = [m for m in members if m["j1"] == jid or m["j2"] == jid]
        if not attached:
            continue
        # Use the first attachment's prop_id.
        prop_id = attached[0]["prop_id"]
        D = prop_sets[prop_id]
        A_x = math.pi * D * D / 4.0
        # Angle from vertical for the axis.
        m0 = attached[0]
        j_other = joints[m0["j2"] if m0["j1"] == jid else m0["j1"]]
        dx = j_other["x"] - j["x"]
        dy = j_other["y"] - j["y"]
        dz = j_other["z"] - j["z"]
        L = math.sqrt(dx * dx + dy * dy + dz * dz)
        cos_theta = abs(dz) / L if L > 0 else 1.0
        cos_theta = min(1.0, max(0.0, cos_theta))
        theta = math.acos(cos_theta)
        cos_theta_cubed = cos_theta**3
        # HydroDyn convention (Item 30): 1/4 factor, NOT standard Morison's 1/2.
        R_joint = 0.25 * RHO * A_x * AxCd * cos_theta_cubed
        R_ax_total += R_joint
        coords = f"({j['x']:>+6.2f},{j['y']:>+6.2f},{j['z']:>+6.2f})"
        print(
            f"{jid:>5}  {coords:>22}  {D:>10.2f}  {A_x:>9.2f}  {AxCd:>5.2f}  "
            f"{math.degrees(theta):>9.2f}  {cos_theta_cubed:>6.3f}  {R_joint:>13.4e}"
        )
    print()
    print(f"  Total axial R: {R_ax_total:.4e} kg/m")
    print()

    R_total = R_cyl_total + R_ax_total
    print(f"GRAND TOTAL R = {R_total:.4e} kg/m")
    print(f"  cylindrical: {R_cyl_total:.4e} ({R_cyl_total/R_total*100:.1f}%)")
    print(f"  axial:       {R_ax_total:.4e} ({R_ax_total/R_total*100:.1f}%)")
    print()

    # ---------------------------------------------------------------------
    # Step B: predict delta from first principles.
    # delta = (8/3) R / m_eff
    # m_eff = M + A_inf_33
    # ---------------------------------------------------------------------
    print("Step B -- delta prediction from first principles")
    print()
    # Use the same M_total + A_inf as PR3/PR4 fragility script (Setup B mass).
    # Robertson platform-with-ballast = 1.347e7 kg; combined-deck adds
    # ~5e5 tower + RNA ~ 1.4074e7 (from fragility output).
    M_combined = 1.4074e7
    A_inf_33 = 1.4960e7  # from marin_semi.1, post-WAMIT-dim (fragility script)
    m_eff = M_combined + A_inf_33
    print(f"  M_combined (platform+tower+RNA) = {M_combined:.4e} kg")
    print(f"  A_inf_33 (marin_semi)            = {A_inf_33:.4e} kg")
    print(f"  m_eff = M + A_inf                = {m_eff:.4e} kg")
    print()

    delta_predicted = (8.0 / 3.0) * R_total / m_eff
    print(f"  delta_predicted = (8/3) * R / m_eff = {delta_predicted:.4f} 1/m")
    print()

    # ---------------------------------------------------------------------
    # Step C: validate against OpenFAST.
    # ---------------------------------------------------------------------
    delta_OF = 0.309
    rel_err = abs(delta_predicted - delta_OF) / delta_OF
    print("Step C -- validation against OpenFAST measurement")
    print(f"  delta_OF (measured from S5 reference, peaks 0-1) = {delta_OF:.4f} 1/m")
    print(f"  delta_predicted                                  = {delta_predicted:.4f} 1/m")
    print(f"  rel-err |pred - OF| / OF                         = {rel_err*100:.2f}%")
    if rel_err < 5.0e-2:
        print("  VERDICT: aggregation validated (rel-err < 5%). Proceed to Step D.")
    elif rel_err < 0.5:
        print("  VERDICT: marginal (rel-err 5-50%). Investigate before Step D.")
    else:
        print("  VERDICT: large disagreement. Aggregation has an error or the")
        print("  drag formulation differs from what HydroDyn applies. Investigate.")


if __name__ == "__main__":
    main()
