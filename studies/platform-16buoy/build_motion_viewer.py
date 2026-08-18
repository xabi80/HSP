"""Build a 16-buoy motion viewer by (1) synthesizing all-body 6-DOF motion from the
VALIDATED equivalent-linearized-drag FD (drag_fd.py -- reproduces the measured buoy
RAO to <0.1%) for the no-fin and 0.215-fin resonances, and (2) splicing that DATA into
a copy of the proven 12-buoy viewer HTML (studies/platform-12buoy/fin_study/
platform_motion.html), which is fully data-driven (bodies from DATA.bodies, platform
by type, buoy by name) -- so ZERO rendering-code change. Output: platform16_motion.html.

Steady-state motion of every body: xi_body(t) = Re{xhat[6b:6b+6] e^{iωt}}, xhat the
126-DOF constrained FD response. Same JSON schema as platform_motion_export.py.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

_R = Path("studies")
sys.path.insert(0, str(_R / "platform-12buoy"))
sys.path.insert(0, str(_R / "platform-16buoy"))
sys.path.insert(0, "floatsim")
warnings.simplefilter("ignore")

import drag_fd  # noqa: E402  (solve, harmonic_drag)
import platform16_common as pc16  # noqa: E402
import platform16_fin_fan as pff  # noqa: E402
import platform16_rao as prp  # noqa: E402

_L_TOP = pc16.Z_HUB_REF - pc16.Z_BUOY_REF
_NF = 120
_VIEWER = _R / "platform-12buoy/fin_study/platform_motion.html"
_OUT_HTML = _R / "platform-16buoy/fin_study/platform16_motion.html"

# (fin tag, plate_r, cd, T, H, label). SAME 5 operating points the 12-buoy motion
# viewer used (platform_motion_export.py) so the two viewers compare case-for-case.
# NOTE: the viewer does label.split("--")[1] for the subtitle, so labels MUST contain
# a " -- " separator (a "—" em-dash breaks draw() with a TypeError). 0215 first for build reuse.
_CASES = [
    ("0215", 0.215, 5.0, 3.141, 0.04, "0.215 m fin · T=3.14 s -- H=0.04 m"),
    ("0215", 0.215, 5.0, 3.141, 0.12, "0.215 m fin · T=3.14 s -- H=0.12 m"),
    ("0215", 0.215, 5.0, 2.500, 0.08, "0.215 m fin · T=2.5 s (off-res) -- H=0.08 m"),
    ("none", pff._R_SPAR, 5.0, 2.500, 0.08, "no fin · T=2.5 s -- H=0.08 m"),
    ("none", pff._R_SPAR, 5.0, 2.500, 0.04, "no fin · T=2.5 s -- H=0.04 m"),
]
_BUILD_CACHE: dict = {}  # reuse the (BEM db, Cummins system) per fin across its cases


def _bodies():  # type: ignore[no-untyped-def]
    deck = prp._deck_with_drag()
    out = []
    for i, b in enumerate(deck.bodies):
        rp = [float(v) for v in b.reference_point]
        if b.name == "platform":
            typ, parent = "platform", 20
            parent = -1
        elif b.name.startswith("hub"):
            typ, parent = "hub", 20
        else:
            typ, parent = "buoy", 5 * (i // 5) + 4
        out.append({"name": b.name, "type": typ, "parent": parent,
                    "x0": round(rp[0], 5), "y0": round(rp[1], 5), "z0": round(rp[2], 5)})
    return out


def _series(cvec, omega):  # type: ignore[no-untyped-def]
    """(disp, vel, acc) frame lists for a complex 3-vector point response."""
    disp, vel, acc = [], [], []
    for f in range(_NF):
        ph = np.exp(1j * 2 * np.pi * f / _NF)
        disp.append([round(float(np.real(c * ph)), 6) for c in cvec])
        vel.append([round(float(np.real(1j * omega * c * ph)), 6) for c in cvec])
        acc.append([round(float(np.real(-omega**2 * c * ph)), 5) for c in cvec])
    return {"disp": disp, "vel": vel, "acc": acc}


def _make_case(fin, plate_r, T, H, label, bodies):  # type: ignore[no-untyped-def]
    if (fin, plate_r) not in _BUILD_CACHE:
        hdb = pff._hdb(fin)
        _BUILD_CACHE[(fin, plate_r)] = (hdb, pff._build(plate_r, 5.0, hdb))
    hdb, setup = _BUILD_CACHE[(fin, plate_r)]
    hydro = np.asarray(prp._hydro_dof(prp._deck_with_drag()))
    bhv = [6 * prp._buoy_body_index(k) + 2 for k in range(16)]
    phv = 6 * prp._buoy_body_index_platform() + 2
    omega = 2 * np.pi / T
    amp = 0.5 * H
    xhat, _ = drag_fd.solve(hdb, setup, hydro, omega, amp, bhv, phv, drag=True)
    nb = xhat.shape[0] // 6
    frames = [[[round(float(np.real(xhat[6 * b + d] * np.exp(1j * 2 * np.pi * f / _NF))), 6)
                for d in range(6)] for b in range(nb)] for f in range(_NF)]
    buoys = [i for i, b in enumerate(bodies) if b["type"] == "buoy"]
    up = min(buoys, key=lambda i: bodies[i]["x0"])
    dn = max(buoys, key=lambda i: bodies[i]["x0"])
    plat = next(i for i, b in enumerate(bodies) if b["type"] == "platform")

    def pt(bi, lever):  # type: ignore[no-untyped-def]
        x = xhat[6 * bi:6 * bi + 6]
        return _series(np.array([x[0] + x[4] * lever, x[1] - x[3] * lever, x[2]]), omega)

    sig = {"plat": pt(plat, 0.0), "up": pt(up, _L_TOP), "dn": pt(dn, _L_TOP)}
    rao_b = float(abs(xhat[bhv[6]]) / amp)
    rao_p = float(abs(xhat[phv]) / amp)
    print(f"  {fin}: buoy7 RAO {rao_b:.3f}  platform RAO {rao_p:.3f}  ratio {rao_b / rao_p:.2f}", flush=True)
    case = {"fin": fin, "label": label, "H": H, "T": T, "omega": round(float(omega), 5),
            "amp_m": round(float(amp), 5), "rao_platform": round(rao_p, 4),
            "rao_buoy7": round(rao_b, 4), "settled": True, "n_frames": _NF,
            "dt_frame_s": round(T / _NF, 5), "frames": frames, "sig": sig}
    return case, up, dn


def _splice(data_obj):  # type: ignore[no-untyped-def]
    h = _VIEWER.read_text(encoding="utf-8")
    d = h.find("const DATA")
    i = h.find("{", d)
    depth, instr, esc, end = 0, False, False, None
    for k in range(i, len(h)):
        ch = h[k]
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
        elif ch == '"':
            instr = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = k
                break
    semi = h.find(";", end)
    newjson = json.dumps(data_obj, separators=(",", ":"))
    newh = h[:d] + "const DATA=" + newjson + h[semi:]
    newh = newh.replace("Platform Motion Viewer — FloatSim",
                        "16-buoy Platform Motion — FloatSim (overshoot vs fin)", 1)
    # Layout: the 16-buoy deck sits high enough to touch the top-left case caption.
    # Shift the whole scene down ~10% of canvas height so the deck clears the caption
    # at every phase (caption is also trimmed to 2 short lines via _CASES labels).
    cam_old, cam_new = "cy:h*0.5+gsy*s", "cy:h*0.6+gsy*s"
    if cam_old not in newh:
        raise SystemExit(f"camera string {cam_old!r} not found -- viewer changed, fix the splice")
    newh = newh.replace(cam_old, cam_new)
    _OUT_HTML.write_text(newh, encoding="utf-8")
    return len(newh)


def main() -> None:
    bodies = _bodies()
    print(f"{len(bodies)} bodies (expect 21); "
          f"buoys={sum(b['type'] == 'buoy' for b in bodies)} hubs={sum(b['type'] == 'hub' for b in bodies)}")
    cases, up, dn = [], None, None
    for fin, pr, cd, T, H, lab in _CASES:
        c, up, dn = _make_case(fin, pr, T, H, lab, bodies)
        cases.append(c)
    data = {
        "meta": {"scale": "1:50 (model)", "n_bodies": len(bodies),
                 "dof_order": ["surge", "sway", "heave", "roll", "pitch", "yaw"],
                 "units": "m / rad", "heading_deg": 0.0,
                 "note": "16-buoy; steady-state motion synthesized from the validated "
                         "equivalent-linearized-drag FD (drag_fd.py); displacements from equilibrium"},
        "bodies": bodies, "upwave_buoy": up, "downwave_buoy": dn, "cases": cases}
    n = _splice(data)
    print(f"wrote {_OUT_HTML} ({n / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
