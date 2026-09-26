"""Analytical check of the free single buoy's pitch natural period (DESIGN-BASIS Phase I).

The buoy floats free in surge (no restoring), so surge and pitch form one mode. With every
coefficient about the CoG (rigid-body M15 = 0), the 2-DOF eigenproblem
    det([[-w² M11, -w² M15], [-w² M15, C55 - w² M55]]) = 0,  M = rigid + added,
gives, with surge condensed out,
    w² = C55 / (I55 + A55 - A15² / (M + A11))                                        (1)
    C55 = rho g (I_wp + V (z_B - z_G))   (hydrostatics about the CoG)                  (2)
The uncoupled estimate w² = C55 / (I55 + A55) ignores the surge coupling. The mode's rotation
centre is A15 / (M + A11) above the CoG.

Three evaluations of (1):
- HAND: (2) from the BEM mesh's geometry (pipe to the waterline + the placeholder disc) and
  slender-body strip theory for the pipe's added mass (Ca = 1, a = rho pi R² per metre); the
  disc adds only its rotation about a diameter, 16/45 rho a⁵ (edge-on otherwise). No BEM,
  no FloatSim.
- BEM: the same formula with the database's C55 and A(w), iterated at the mode, for the flume
  database (12-sided waterline) and the converged 96-sided one; compared with FloatSim's modal
  periods recorded in bem_waterline.json.
- AS-BUILT: (2) with the underwater volume the CAD shows outside the pipe
  (../osu-test-buoy/underwater_volume.json) plus the lead (not in the CAD) at the inertia
  model's position, at the same 967 mm waterline; and the sensitivity to the two unmeasured
  inputs, the CoG height and the pitch inertia.

Run: python pitch_analytic_check.py  ->  pitch_analytic_check.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import floatsim_decks as fd  # puts platform-12buoy/flume-wall-effect on sys.path
import numpy as np

from floatsim.hydro.readers.capytaine import read_capytaine

# isort: split
import coupled_bem_osu as cb  # needs the sys.path entry floatsim_decks adds

HERE = Path(__file__).resolve().parent
RHO, G = cb.RHO, cb.G
R, Z_BOT, RP, ZP, ZG = cb.SPAR_R, cb.Z_BOT, cb.PLATE_R, cb.PLATE_Z, cb.COG_Z
TP = 0.02  # placeholder disc thickness (coupled_bem_osu.buoy_mesh)
M, I55 = fd.aw.M_BUOY, fd.aw.IYY
M_LEAD, Z_LEAD, RHO_LEAD = 13.36, -1.383, 11350.0  # osu-test-buoy/inertia_from_step.py; lead
DATABASES = {
    "flume BEM, 12-sided": "single_osu_open_psd.nc",
    "converged, 96-sided": "single_osu_open_nt96_psd.nc",
}


def period(c55: float, i55: float, a11: float, a15: float, a55: float, m: float = M) -> float:
    """Pitch period from (1), coupled with surge."""
    return float(2 * np.pi * np.sqrt((i55 + a55 - a15**2 / (m + a11)) / c55))


def c55_of(vols: list[tuple[float, float]], zg: float = ZG) -> float:
    """C55 from (2) for a list of (volume m³, centroid z) plus the pipe's waterplane."""
    v = sum(x for x, _ in vols)
    zb = sum(x * z for x, z in vols) / v
    return float(RHO * G * (np.pi * R**4 / 4 + v * (zb - zg)))


def strip() -> tuple[float, float, float]:
    """Pipe added mass by strip theory (Ca = 1) about the CoG, + the disc's rotation term."""
    a = RHO * np.pi * R**2
    z0, z1 = Z_BOT - ZG, -ZG
    return (a * (z1 - z0), a * (z1**2 - z0**2) / 2, a * (z1**3 - z0**3) / 3 + 16 / 45 * RHO * RP**5)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    out: dict = {}
    pipe = (np.pi * R**2 * -Z_BOT, Z_BOT / 2)
    disc = (np.pi * RP**2 * TP, ZP)
    a11, a15, a55 = strip()
    c55 = c55_of([pipe, disc])
    out["hand"] = {
        "V_m3": pipe[0] + disc[0],
        "C55": c55,
        "A11": a11,
        "A15": a15,
        "A55": a55,
        "T_coupled_s": period(c55, I55, a11, a15, a55),
        "T_uncoupled_s": float(2 * np.pi * np.sqrt((I55 + a55) / c55)),
        "rotation_centre_above_CoG_m": a15 / (M + a11),
    }

    recorded = json.loads((HERE / "bem_waterline.json").read_text())["articles"]["buoy_free"]
    fs = {
        "flume BEM, 12-sided": recorded["12"]["tilt"],
        "converged, 96-sided": recorded["12"]["tilt"] - recorded["12"]["shift_vs_nt96_s"]["tilt"],
    }
    out["bem"] = {}
    for tag, f in DATABASES.items():
        h = read_capytaine(HERE / f)
        w = 2 * np.pi / 2.7
        for _ in range(100):  # A(w) iterated at the mode
            a = [float(np.interp(w, h.omega, h.A[i, j, :])) for i, j in ((0, 0), (0, 4), (4, 4))]
            w_new = 2 * np.pi / period(float(h.C[4, 4]), I55, *a)
            if abs(w_new - w) < 1e-12:
                break
            w = w_new
        out["bem"][tag] = {
            "C55": float(h.C[4, 4]),
            "omega": w,
            "A11": a[0],
            "A15": a[1],
            "A55": a[2],
            "T_formula_s": 2 * np.pi / w,
            "T_FloatSim_modal_s": fs[tag],
        }

    uw = json.loads((HERE.parent / "osu-test-buoy" / "underwater_volume.json").read_text())
    ext = (uw["external_underwater_volume_L"] / 1e3, uw["external_centroid_z_wl_m"])
    lead = (M_LEAD / RHO_LEAD, Z_LEAD)
    vols = [pipe, ext, lead]
    v_asb = sum(x for x, _ in vols)
    m_asb = RHO * v_asb  # floating at the same 967 mm waterline
    c_asb = c55_of(vols)
    t_asb = period(c_asb, I55, a11, a15, a55, m_asb)
    out["as_built"] = {
        "V_mesh_m3": pipe[0] + disc[0],
        "V_spreadsheet_m3": M / RHO,
        "V_as_built_m3": v_asb,
        "external_CAD_m3": ext[0],
        "external_CAD_z": ext[1],
        "lead_m3": lead[0],
        "mass_at_967mm_kg": m_asb,
        "C55": c_asb,
        "T_coupled_s": t_asb,
        "T_CoG_minus_2cm_s": period(c55_of(vols, ZG - 0.02), I55, a11, a15, a55, m_asb),
        "T_CoG_plus_2cm_s": period(c55_of(vols, ZG + 0.02), I55, a11, a15, a55, m_asb),
        "T_Iyy_minus_10pct_s": period(c_asb, 0.9 * I55, a11, a15, a55, m_asb),
        "T_Iyy_plus_10pct_s": period(c_asb, 1.1 * I55, a11, a15, a55, m_asb),
    }
    (HERE / "pitch_analytic_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    hd = out["hand"]
    print(
        f"HAND: C55 {hd['C55']:.2f} N·m/rad, A11 {hd['A11']:.2f} kg, A15 {hd['A15']:.3f} kg·m, "
        f"A55 {hd['A55']:.3f} kg·m² -> T {hd['T_coupled_s']:.3f} s "
        f"(uncoupled {hd['T_uncoupled_s']:.3f} s; rotation centre "
        f"{hd['rotation_centre_above_CoG_m']:.2f} m above the CoG)"
    )
    for tag, b in out["bem"].items():
        print(
            f"BEM {tag}: C55 {b['C55']:.2f}, A11 {b['A11']:.2f}, A15 {b['A15']:.3f}, "
            f"A55 {b['A55']:.3f} -> T {b['T_formula_s']:.4f} s; FloatSim modal "
            f"{b['T_FloatSim_modal_s']:.4f} s"
        )
    ab = out["as_built"]
    print(
        f"AS-BUILT: V mesh {ab['V_mesh_m3'] * 1e3:.2f} L, spreadsheet "
        f"{ab['V_spreadsheet_m3'] * 1e3:.2f} L, as-built {ab['V_as_built_m3'] * 1e3:.2f} L "
        f"(mass at 967 mm {ab['mass_at_967mm_kg']:.2f} kg); C55 {ab['C55']:.2f} -> "
        f"T {ab['T_coupled_s']:.3f} s; CoG ∓2 cm {ab['T_CoG_minus_2cm_s']:.3f} / "
        f"{ab['T_CoG_plus_2cm_s']:.3f} s; Iyy ∓10 % {ab['T_Iyy_minus_10pct_s']:.3f} / "
        f"{ab['T_Iyy_plus_10pct_s']:.3f} s"
    )


if __name__ == "__main__":
    main()
