"""As-built underwater volume of the OSU Test Buoy outside the pipe, from the STEP file (gmsh).

The FloatSim/BEM buoy (flume-mooring `single_osu_open*`, coupled_bem_osu.buoy_mesh) meshes the
sealed pipe plus a placeholder solid disc (diameter 0.2874 m, 0.02 m thick, 1.297 L at
z = -1.383 m) for the ballast/heave-plate frame. Its mass, 21.52 kg, is rho times the
spreadsheet's PARAMETRIC displaced volume, not an as-built one: 21.57 L = the pipe to the
967 mm waterline (19.27 L) + an assumed "low hemisphere" (1.06 L) + an assumed ballast volume
(1.25 L); `OSU Spar Buoy Platform Metric.xlsx` B45 + B47 + B48.

This script measures what the real buoy displaces outside the pipe:

- every CAD solid minus the pipe envelope (r = 79.65 mm, the pipe's full length) and minus
  everything above the waterline (967 mm above the pipe bottom);
- per part: the remaining volume and centroid, in the waterline frame (z up, still water 0).

The lead ballast is not in the CAD (structure = all parts minus the ballast box, 8.16 kg);
its volume, 13.36 kg / 11350 kg/m³, is added by the consumer (flume-mooring
`pitch_analytic_check.py`) at the inertia model's `Z_LEAD`.

Requires: gmsh (contributor tool, not a FloatSim dependency) and the STEP file.
Run: python underwater_volume.py  ->  underwater_volume.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gmsh

HERE = Path(__file__).resolve().parent
STEP = r"C:/Users/xlama/OneDrive/Documents/buoy/OSU Test Buoy.stp"
WL_MM = 967.0  # waterline above the pipe bottom (osu_buoy_common / spreadsheet B3·L)
R_MM = 159.3 / 2  # pipe outer radius
L_MM = 1683.801  # pipe length


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(STEP)
    gmsh.model.occ.synchronize()
    solids = [t for _, t in gmsh.model.getEntities(3)]
    names = {t: gmsh.model.getEntityName(3, t).split("/")[-1] or f"solid {t}" for t in solids}
    boxes = {t: gmsh.model.occ.getBoundingBox(3, t) for t in solids}
    big = 5000.0
    env = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L_MM, R_MM)
    above = gmsh.model.occ.addBox(-big, -big, WL_MM, 2 * big, 2 * big, big)
    parts = []
    for t in solids:
        tool = gmsh.model.occ.copy([(3, env), (3, above)])
        out, _ = gmsh.model.occ.cut([(3, t)], tool, removeObject=False, removeTool=True)
        gmsh.model.occ.synchronize()
        vol = sum(gmsh.model.occ.getMass(d, k) for d, k in out)  # mm³
        if vol > 1.0:
            z = (
                sum(
                    gmsh.model.occ.getMass(d, k) * gmsh.model.occ.getCenterOfMass(d, k)[2]
                    for d, k in out
                )
                / vol
            )
            b = boxes[t]
            parts.append(
                {
                    "tag": t,
                    "name": names[t],
                    "volume_L": round(vol / 1e6, 4),
                    "z_wl_m": round(z / 1000 - WL_MM / 1000, 4),
                    "y_range_mm": [round(b[1]), round(b[4])],
                }
            )
        gmsh.model.occ.remove(out, recursive=True)
    gmsh.finalize()
    parts.sort(key=lambda p: p["z_wl_m"])
    groups = {
        "ballast frame (base plate + webs)": [p for p in parts if p["z_wl_m"] < -1.05],
        "lower cap (outside the pipe)": [p for p in parts if p["name"] == "Lower cap 6 inch"],
        "side parts on +y (outside the pipe)": [
            p for p in parts if -1.0 < p["z_wl_m"] < -0.3 and p["name"] != "Lower cap 6 inch"
        ],
        "other": [p for p in parts if p["z_wl_m"] >= -0.3],
    }
    total = sum(p["volume_L"] for p in parts)
    res = {
        "waterline_mm_above_pipe_bottom": WL_MM,
        "external_underwater_volume_L": round(total, 4),
        "external_centroid_z_wl_m": round(
            sum(p["volume_L"] * p["z_wl_m"] for p in parts) / total, 4
        ),
        "groups": {
            k: {
                "volume_L": round(sum(p["volume_L"] for p in v), 4),
                "z_wl_m": (
                    round(
                        sum(p["volume_L"] * p["z_wl_m"] for p in v) / sum(p["volume_L"] for p in v),
                        4,
                    )
                    if v
                    else None
                ),
            }
            for k, v in groups.items()
        },
        "parts": parts,
    }
    (HERE / "underwater_volume.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, g in res["groups"].items():
        print(f"{k:38} {g['volume_L']:6.3f} L at z {g['z_wl_m']}")
    print(
        f"external underwater volume {total:.3f} L, centroid z {res['external_centroid_z_wl_m']} m"
    )


if __name__ == "__main__":
    main()
