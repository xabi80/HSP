"""OSU buoy hydrostatics against sinkage, for the payload-stability study (Capytaine).

A mass on the cluster's hub sinks every buoy. FloatSim's restoring is a constant matrix fixed at
the BEM draft (Level 2 is not wired: docs/phase2-followups.md LEVEL2-INTEGRATOR-UNWIRED), so each
load level needs the buoy's hydrostatics at its own draft. This script re-meshes the flume BEM hull
(coupled_bem_osu: spar, heave plate, NT panels round) sunk by s below its design draft and computes,
about the buoy's CoG (rotation centre = CoG, as single_buoy_c), with Capytaine:
- the displaced volume V and the extra buoyancy dW = rho g (V(s) - V(0)) that a pin load must
  supply to hold the buoy at that sinkage;
- C33 and C44 = C55 about the CoG: rho g A_wp and rho g (I_wp + V (z_B - z_G)).
The unloaded buoy floats at the design draft (s = 0): the hull's design weight is taken as
balanced there, so only V(s) - V(0) matters (the BEM mesh's 1 L volume deficit, OSU geometry doc,
is a deep feature that does not change with s).

Writes buoy_hydrostatics_vs_draft.json (NT 12 and 36; NT 96 at s = 0 as the convergence check).
Run: python draft_hydrostatics.py
"""

# ruff: noqa: E402  -- sys.path bootstrap precedes the study imports
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FWE = HERE.parent / "platform-12buoy" / "flume-wall-effect"
sys.path.insert(0, str(FWE))
warnings.simplefilter("ignore")

import capytaine as cpt
import coupled_bem_osu as cb

# m below the design draft: buoys rising on the light side (down to -0.95: the spar bottom is
# 0.967 m below the design waterline) to just short of the spar top (+0.717 m) going under
SINK = np.round(np.concatenate([np.arange(-0.95, 0.701, 0.025), [0.71, 0.715]]), 4)
DOF6 = cb.DOF6


def buoy_mesh_sunk(s: float, nt: int):  # type: ignore[no-untyped-def]
    """coupled_bem_osu.buoy_mesh at the origin with nt panels round, lowered by s, wetted part."""
    spar = cpt.mesh_vertical_cylinder(length=cb.Z_TOP - cb.Z_BOT, radius=cb.SPAR_R,
                                      center=(0.0, 0.0, (cb.Z_BOT + cb.Z_TOP) / 2 - s),
                                      resolution=(2, nt, 24))
    plate = cpt.mesh_vertical_cylinder(length=0.02, radius=cb.PLATE_R,
                                       center=(0.0, 0.0, cb.PLATE_Z - s), resolution=(4, nt, 2))
    return cpt.FloatingBody(mesh=spar.join_meshes(plate)).immersed_part().mesh


def hydrostatics(s: float, nt: int) -> dict:
    cog = np.array([0.0, 0.0, cb.COG_Z - s])
    b = cpt.FloatingBody(mesh=buoy_mesh_sunk(s, nt), center_of_mass=tuple(cog))
    b.rotation_center = cog
    b.add_all_rigid_body_dofs()
    hs = b.compute_hydrostatics(rho=cb.RHO, g=cb.G)
    K = np.array([[float(hs["hydrostatic_stiffness"].sel(radiating_dof=a, influenced_dof=c))
                   for c in DOF6] for a in DOF6])
    zb = float(np.asarray(hs["center_of_buoyancy"])[2])
    return {"sink_m": float(s), "nt": nt, "V_m3": float(hs["disp_volume"]),
            "Awp_m2": float(hs["waterplane_area"]), "zB_rel_cog_m": zb - float(cog[2]),
            "C33": K[2, 2], "C44": K[3, 3], "C55": K[4, 4], "C6": K.tolist()}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    out: dict = {"rho": cb.RHO, "g": cb.G, "z_cog_design_m": cb.COG_Z, "z_pin_design_m": cb.Z_TOP,
                 "pin_above_cog_m": cb.Z_TOP - cb.COG_Z, "rows": {}}
    for nt in (12, 36):
        rows = [hydrostatics(float(s), nt) for s in SINK]
        v0 = next(r["V_m3"] for r in rows if abs(r["sink_m"]) < 1e-9)
        for r in rows:
            r["dW_N"] = cb.RHO * cb.G * (r["V_m3"] - v0)       # pin load holding this sinkage
        out["rows"][str(nt)] = rows
        print(f"NT {nt}: sink  V(L)   C33    C55   zB-zG   dW(N)")
        for r in rows[::6]:
            print(f"       {r['sink_m']:+.3f} {1e3 * r['V_m3']:6.2f} {r['C33']:6.1f} "
                  f"{r['C55']:6.2f} {r['zB_rel_cog_m']:+.3f} {r['dW_N']:7.1f}")
    out["nt96_design"] = hydrostatics(0.0, 96)
    d96 = out["nt96_design"]
    print(f"NT 96 at s = 0: C33 {d96['C33']:.2f}, C55 {d96['C55']:.2f}")
    (HERE / "buoy_hydrostatics_vs_draft.json").write_text(json.dumps(out, indent=1),
                                                          encoding="utf-8")
    print("wrote buoy_hydrostatics_vs_draft.json")


if __name__ == "__main__":
    main()
