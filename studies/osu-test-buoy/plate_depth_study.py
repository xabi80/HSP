"""Parametric study: heave-plate (heave drag device) depth vs PITCH performance.

Holds the BALLAST fixed — mass 21.52 kg, CoG −0.907 m, inertia 10.2 kg·m² — so the
stability set-point (pitch restoring C55) is unchanged, and moves ONLY the heave drag
device (the Capytaine BEM disc + its Morison drag element) in z. For each depth we rebuild
the BEM, assemble the Cummins system, and run heave + pitch free-decay, reporting the
natural period and the first-swing damping ratio.

The plate is the PLACEHOLDER solid equal-area disc (added mass is a potential-flow upper
bound; the real perforated frame adds less), so read the TRENDS, not the absolute values.
Free-decay only → excitation is written as zeros (diffraction skipped for speed).

Writes plate_depth_pitch_study.png next to this script. Requires capytaine. SLOW.
Usage: python plate_depth_study.py [quick]
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

warnings.simplefilter("ignore")
import capytaine as cpt  # noqa: E402

cpt.set_logging("ERROR")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))
from floatsim.bodies.mass_properties import rigid_body_mass_matrix  # noqa: E402
from floatsim.hydro.morison import MorisonElement, PlateDragElement, make_morison_state_force  # noqa: E402
from floatsim.hydro.radiation import assemble_cummins_lhs  # noqa: E402
from floatsim.hydro.readers.capytaine import read_capytaine  # noqa: E402
from floatsim.hydro.retardation import compute_retardation_kernel  # noqa: E402
from floatsim.solver.equilibrium import solve_static_equilibrium  # noqa: E402
from floatsim.solver.newmark import integrate_cummins  # noqa: E402

# --- fixed ballast / body (the stability set-point does NOT move) ---
RHO, G = 998.0, 9.806
M_BODY, CoG_Z = 21.52, -0.907
I_XX = I_YY = 10.2
I_ZZ = 0.063
DT, KERNEL_TMAX = 0.01, 30.0
# --- geometry ---
R, Z_BOT, Z_TOP = 0.07965, -0.967, 0.717            # spar (fixed)
A_PLATE = float(np.sqrt(0.328 * 0.198 / np.pi))     # equal-area disc radius (placeholder)
DOFS = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]
# --- spar drag (fixed) + plate drag (moves with the device) ---
_SPAR_D, _SPAR_CD, _WL = 0.1593, 1.2, 0.0
_PLATE_R, _PLATE_CD_N, _PLATE_CD_T, _PLATE_T = 0.1437, 5.0, 1.5, 0.0039
_OVR = "OSU plate-depth study: small body, B not fully asymptotic at omega_max"
_TMP = _HERE.parent  # scratch .nc alongside (overwritten per depth)


def _omega_grid():
    ref = xr.open_dataset(_HERE.parent.parent / "studies/spar-fin-decay/capytaine_bem.nc")
    om = np.sort(np.asarray(ref["omega"].values))
    om = np.ascontiguousarray(om[np.isfinite(om)])
    ref.close()
    return om


def build_bem(z_plate: float, out_path: Path, coarse: bool = True) -> None:
    """Rebuild the 6-DOF BEM with the disc at z_plate (rotation center fixed at CoG)."""
    sres = (2, 32, 44) if coarse else (4, 48, 70)
    pres = (8, 32, 2) if coarse else (12, 48, 2)
    spar = cpt.mesh_vertical_cylinder(length=Z_TOP - Z_BOT, radius=R,
                                      center=(0, 0, (Z_BOT + Z_TOP) / 2), resolution=sres)
    plate = cpt.mesh_vertical_cylinder(length=0.02, radius=A_PLATE, center=(0, 0, z_plate),
                                       resolution=pres)
    b = cpt.FloatingBody(mesh=spar.join_meshes(plate), mass=M_BODY, center_of_mass=(0, 0, CoG_Z))
    b.rotation_center = np.array([0.0, 0.0, CoG_Z])
    b.add_all_rigid_body_dofs()
    b = b.immersed_part()
    b.rotation_center = np.array([0.0, 0.0, CoG_Z])
    hs = b.compute_hydrostatics(rho=RHO, g=G)
    C = np.array([[float(hs["hydrostatic_stiffness"].sel(radiating_dof=a, influenced_dof=c))
                   for c in DOFS] for a in DOFS])
    om = _omega_grid()
    om_all = np.append(om, np.inf)
    solver = cpt.BEMSolver()
    nA = len(om_all)
    A = np.zeros((nA, 6, 6)); B = np.zeros((nA, 6, 6))
    for i, w in enumerate(om_all):
        for j, dof in enumerate(DOFS):
            res = solver.solve(cpt.RadiationProblem(body=b, omega=w, radiating_dof=dof,
                                                    rho=RHO, g=G, water_depth=np.inf), keep_details=False)
            A[i, j, :] = [res.added_masses[d] for d in DOFS]
            if np.isfinite(w):
                B[i, j, :] = [res.radiation_dampings[d] for d in DOFS]
    Fexc = np.zeros((nA, 1, 6), complex)  # free-decay only → excitation unused
    ds = xr.Dataset(
        data_vars=dict(
            added_mass=(("omega", "radiating_dof", "influenced_dof"), A),
            radiation_damping=(("omega", "radiating_dof", "influenced_dof"), B),
            hydrostatic_stiffness=(("radiating_dof", "influenced_dof"), C),
            excitation_force=(("complex", "omega", "wave_direction", "influenced_dof"),
                              np.stack([Fexc.real, Fexc.imag], 0)),
        ),
        coords=dict(omega=("omega", om_all), wave_direction=("wave_direction", [0.0]),
                    radiating_dof=("radiating_dof", DOFS), influenced_dof=("influenced_dof", DOFS),
                    complex=("complex", ["re", "im"])),
        attrs=dict(rho=RHO, g=G, water_depth="inf", body_name=f"osu_plate_z{z_plate:+.3f}"),
    )
    ds.to_netcdf(out_path)


def make_drag(z_plate: float, n_seg: int = 10):  # type: ignore[no-untyped-def]
    elems: list = []
    edges = np.linspace(Z_BOT, _WL, n_seg + 1)
    for i in range(n_seg):
        elems.append(MorisonElement(body_index=0, node_a_body=np.array([0.0, 0.0, edges[i]]),
                                     node_b_body=np.array([0.0, 0.0, edges[i + 1]]),
                                     diameter=_SPAR_D, Cd=_SPAR_CD))
    elems.append(PlateDragElement(body_index=0, center_body=np.array([0.0, 0.0, z_plate]),
                                  normal_body=np.array([0.0, 0.0, 1.0]), radius=_PLATE_R,
                                  thickness=_PLATE_T, Cd_n=_PLATE_CD_N, Cd_t=_PLATE_CD_T))
    return make_morison_state_force(elems, n_dof=6, fluid_velocity_fn=lambda p, t: np.zeros(3), rho=RHO)


def decay(hdb, drag, z_plate: float, duration: float):  # type: ignore[no-untyped-def]
    r = np.array([0.0, 0.0, CoG_Z])
    i_ref = np.diag([I_XX, I_YY, I_ZZ]) + M_BODY * ((r @ r) * np.eye(3) - np.outer(r, r))
    M = rigid_body_mass_matrix(mass=M_BODY, inertia_at_reference=i_ref, cog_offset_body=r)
    lhs = assemble_cummins_lhs(rigid_body_mass=M, hdb=hdb, mass=M_BODY,
                               cog_offset_from_bem_origin=r, gravity=G)
    kernel = compute_retardation_kernel(hdb, t_max=KERNEL_TMAX, dt=DT,
                                        asymptote_check_override=_OVR, kernel_decay_floor_override=_OVR)
    eq = solve_static_equilibrium(lhs=lhs, state_force=drag)
    out = {"C55": float(lhs.C[4, 4]), "A55": float(lhs.M_plus_Ainf[4, 4] - (I_YY + M_BODY * r @ r)),
           "A33": float(lhs.M_plus_Ainf[2, 2] - M_BODY)}
    for dof, key in [(2, "heave"), (4, "pitch")]:
        xi0 = eq.xi_eq.copy(); xi0[dof] += 0.10
        res = integrate_cummins(lhs=lhs, kernel=kernel, xi0=xi0, xi_dot0=np.zeros(6),
                                duration=duration, dt=DT, state_force=drag)
        x = res.xi[:, dof] - eq.xi_eq[dof]
        pk, _ = find_peaks(x, height=1e-4)
        if len(pk) < 3:
            out[f"T_{key}"], out[f"z_{key}"] = np.nan, np.nan
            continue
        out[f"T_{key}"] = float(np.mean(np.diff(res.t[pk][:6])))
        d = np.log(x[pk][:-1] / x[pk][1:]); d = d[np.isfinite(d) & (d > 0)]
        out[f"z_{key}"] = float(np.mean(d[:2]) / (2 * np.pi) * 100)
    return out


_JSON = _HERE / "plate_depth_results.json"


def run_sweep(depths: list[float], duration: float) -> list[dict]:
    tmp = _TMP / "_plate_sweep_tmp.nc"
    rows = []
    for z in depths:
        L = abs(z - CoG_Z)
        try:
            build_bem(z, tmp, coarse=True)
            hdb = read_capytaine(tmp)
            d = decay(hdb, make_drag(z), z, duration)
        except Exception as exc:  # noqa: BLE001  keep the sweep alive; log the bad depth
            print(f"z_plate={z:+.3f} (L={L:.3f} m)  SKIPPED: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0]}", flush=True)
            d = {k: float("nan") for k in ("T_pitch", "z_pitch", "T_heave", "z_heave", "A55", "C55", "A33")}
        d["z"] = z
        rows.append(d)
        print(f"z_plate={z:+.3f} (L={L:.3f} m)  A55={d['A55']:5.2f}  C55={d['C55']:6.1f}  "
              f"PITCH T={d['T_pitch']:.2f}s ζ={d['z_pitch']:.1f}%  |  "
              f"HEAVE T={d['T_heave']:.2f}s ζ={d['z_heave']:.1f}%", flush=True)
    if tmp.exists():
        tmp.unlink()
    return rows


def make_plot(rows: list[dict]) -> None:
    z = np.array([r["z"] for r in rows])
    Tp = np.array([r["T_pitch"] for r in rows]); zp = np.array([r["z_pitch"] for r in rows])
    Th = np.array([r["T_heave"] for r in rows]); zh = np.array([r["z_heave"] for r in rows])
    cur = int(np.argmin(np.abs(z + 1.383)))
    d_Tp = np.nanmax(Tp) - np.nanmin(Tp); d_zp = np.nanmax(zp) - np.nanmin(zp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    CYP, CYH = "#0c8b96", "#c98a2b"
    # x runs deeper (left, more negative z) -> shallower (right); no axis inversion.
    for ax, yp, yh, ylab, title, note in [
        (ax1, Tp, Th, "natural period (s)", "PERIOD vs plate depth",
         f"pitch spans only {d_Tp:.2f} s (~{100 * d_Tp / np.nanmin(Tp):.0f}%) across the range"),
        (ax2, zp, zh, "first-swing damping ζ (%)", "DAMPING vs plate depth",
         f"pitch spans only {d_zp:.1f} pt across the range"),
    ]:
        ax.plot(z, yp, "-o", color=CYP, lw=2.2, ms=7, label="pitch")
        ax.plot(z, yh, "-s", color=CYH, lw=2.0, ms=6, label="heave")
        ax.plot(z[cur], yp[cur], "o", ms=14, mfc="none", mec="#d1543a", mew=2.2)
        ax.axvline(-1.383, ls=":", color="#d1543a", lw=1.2)
        ax.set_xlabel("heave-plate depth  z (m)     ← deeper            shallower →")
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=11); ax.grid(alpha=0.3); ax.legend(loc="center right")
        ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=8.5, color="#54636d")
    ax2.annotate("current −1.383 m", (z[cur], zp[cur]), textcoords="offset points",
                 xytext=(10, 10), fontsize=8, color="#d1543a")
    fig.suptitle("OSU buoy — heave-plate depth barely moves pitch (ballast fixed; C55 ≈ const; placeholder disc → trend)",
                 fontsize=11.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = _HERE / "plate_depth_pitch_study.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if "replot" in sys.argv:
        rows = json.loads(_JSON.read_text())
    else:
        quick = "quick" in sys.argv
        # depths keep a clean gap below the spar bottom (−0.967); a disc kissing the cap
        # corrupts the high-ω heave radiation (a placeholder-disc meshing artifact, not physics).
        depths = [-1.383] if quick else [-1.10, -1.25, -1.383, -1.55, -1.70]
        rows = run_sweep(depths, 20.0 if quick else 50.0)
        if quick:
            print("quick OK"); return
        _JSON.write_text(json.dumps(rows, indent=2))
    make_plot(rows)


if __name__ == "__main__":
    main()
