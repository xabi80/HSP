"""Export a compact JSON of the 12-buoy platform's time-domain motion for the
interactive artifact. Re-runs a few representative cases (fin 0.215 at/off
resonance + the no-fin resonance), pulls ALL 17 bodies' 6-DOF from the settled
window of ``run_case``, keeps an integer number of wave periods (clean loop),
decimates to ~180 frames, and writes ``platform_motion.json``:

  bodies:  equilibrium (x0,y0,z0), type, parent index (for drawing the arms)
  cases:   per case, frames[frame][body] = [surge,sway,heave,roll,pitch,yaw]
           (displacements from equilibrium, m / rad)

One-shot (not the chunked fan): 5 cases + 2 builds in a single process peaks
~16 GB, well under the box -- the ~2 GB/case integrator retention only matters
for the 220-case sweep.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # fin_study (platform_timeseries)
sys.path.insert(0, str(_HERE.parent))  # platform-12buoy
sys.path.insert(0, str(_HERE.parent.parent / "cluster-3buoy-rigid"))

import platform_fin_fan as pff  # noqa: E402
import platform_rao_pilot as prp  # noqa: E402
import platform_timeseries as pts  # noqa: E402  (point kinematics for the scope signals)

_OUT = _HERE / "platform_motion.json"
_N_FRAMES = 180
_LOOP_PERIODS = 4

# (fin tag, plate_r, cd, period_s, height_m, label) -- all 0215 first (build reuse).
_CASES = [
    ("0215", 0.215, 5.0, 3.141, 0.04, "fin 0.215 m -- resonance (T=3.14 s), H=0.04 m"),
    ("0215", 0.215, 5.0, 3.141, 0.12, "fin 0.215 m -- resonance (T=3.14 s), H=0.12 m"),
    ("0215", 0.215, 5.0, 2.500, 0.08, "fin 0.215 m -- off-resonance (T=2.5 s), H=0.08 m"),
    ("none", pff._R_SPAR, 5.0, 2.500, 0.08, "no fin -- resonance (T=2.5 s), H=0.08 m"),
    ("none", pff._R_SPAR, 5.0, 2.500, 0.04, "no fin -- resonance (T=2.5 s), H=0.04 m"),
]


def _bodies(deck):  # type: ignore[no-untyped-def]
    """17 bodies in deck order: [3 buoys, hub] x 4 clusters, then platform."""
    out = []
    for i, b in enumerate(deck.bodies):
        rp = [float(v) for v in b.reference_point]
        if i == 16:
            typ, parent = "platform", -1
        elif i % 4 == 3:
            typ, parent = "hub", 16
        else:
            typ, parent = "buoy", 4 * (i // 4) + 3
        out.append({"name": b.name, "type": typ, "parent": parent,
                    "x0": round(rp[0], 5), "y0": round(rp[1], 5), "z0": round(rp[2], 5)})
    return out


def _signals(c, t, m, idx, sig_points):  # type: ignore[no-untyped-def]
    """Per-point displacement/velocity/acceleration time histories (true amplitude,
    m / m·s⁻¹ / m·s⁻²) on the SAME window+decimation as the animation frames, via
    the rigid-body point kinematics (``platform_timeseries._point_kinematics``:
    reference accel from the integrator's xi_ddot + the buoy-top lever terms)."""
    xi = np.asarray(c["xi"], dtype=float)
    acc = np.asarray(c["acc"], dtype=float)
    out = {}
    for key, bi, lever in sig_points:
        sl = slice(6 * bi, 6 * bi + 6)
        dsp, vel, ac = pts._point_kinematics(xi[:, sl], acc[:, sl], t, lever)
        dw, vw, aw = dsp[m][idx], vel[m][idx], ac[m][idx]
        out[key] = {
            "disp": [[round(float(dw[f, j]), 6) for j in range(3)] for f in range(dw.shape[0])],
            "vel": [[round(float(vw[f, j]), 6) for j in range(3)] for f in range(vw.shape[0])],
            "acc": [[round(float(aw[f, j]), 5) for j in range(3)] for f in range(aw.shape[0])],
        }
    return out


def _frames(c, sig_points):  # type: ignore[no-untyped-def]
    """Keep an integer number of periods (clean loop), decimate to _N_FRAMES.
    Also returns the scope signals on the identical window+decimation."""
    t = np.asarray(c["t"], dtype=float)
    xi = np.asarray(c["xi"], dtype=float)  # (nw, 102)
    T = 2.0 * np.pi / float(c["omega"])
    dur = float(t[-1] - t[0])
    k = max(1, min(_LOOP_PERIODS, int(dur // T)))
    t0 = t[-1] - k * T
    m = t >= t0 - 1e-9
    xw = xi[m]
    idx = np.clip(np.round(np.linspace(0, xw.shape[0], _N_FRAMES, endpoint=False)).astype(int),
                  0, xw.shape[0] - 1)
    xf = xw[idx]  # (N, 102)
    nb = xf.shape[1] // 6
    frames = [[[round(float(xf[f, 6 * b + d]), 6) for d in range(6)] for b in range(nb)]
              for f in range(xf.shape[0])]
    dt_frame = k * T / _N_FRAMES
    signals = _signals(c, t, m, idx, sig_points)
    return frames, dt_frame, signals


def main() -> None:
    deck = prp._deck_with_drag()
    bodies = _bodies(deck)
    hydro_dof = prp._hydro_dof(deck)
    buoys = [i for i, b in enumerate(bodies) if b["type"] == "buoy"]
    upwave = min(buoys, key=lambda i: bodies[i]["x0"])  # wave (heading 0) arrives at -x first
    downwave = max(buoys, key=lambda i: bodies[i]["x0"])
    plat_idx = next(i for i, b in enumerate(bodies) if b["type"] == "platform")
    # Scope signal points: platform centre (lever 0) + upwave/downwave buoy TOPS
    # (spar-top attach point, body-frame lever Z_HUB-Z_BUOY). Same names the viewer
    # oscilloscope reads (plat / up / dn).
    sig_points = [("plat", plat_idx, 0.0), ("up", upwave, pts._L_TOP), ("dn", downwave, pts._L_TOP)]
    print(f"{len(bodies)} bodies; upwave={bodies[upwave]['name']} "
          f"(x={bodies[upwave]['x0']}), downwave={bodies[downwave]['name']}; "
          f"buoy-top lever {pts._L_TOP:.3f} m", flush=True)

    hdb_cache: dict = {}
    built: dict = {}
    cases_out = []
    for tag, plate_r, cd, T, H, label in _CASES:
        if tag not in hdb_cache:
            hdb_cache[tag] = pff._hdb(tag)
        hdb = hdb_cache[tag]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if tag not in built:
                built[tag] = pff._build(plate_r, cd, hdb)
            setup = built[tag]
            c = prp.run_case(setup, hdb, hydro_dof, height_m=H, period_s=T,
                             ramp_s=20.0, cap_settle_s=350.0, window_periods=6.0, dt=0.01)
        frames, dt_frame, signals = _frames(c, sig_points)
        cases_out.append({
            "fin": tag, "label": label, "H": H, "T": T,
            "omega": round(float(c["omega"]), 5), "amp_m": round(float(c["amp_m"]), 5),
            "rao_platform": round(float(c["rao"]["platform_heave"]), 4),
            "rao_buoy7": round(float(c["rao"]["buoy7_heave"]), 4),
            "settled": bool(c["settled"]),
            "n_frames": len(frames), "dt_frame_s": round(dt_frame, 5), "frames": frames,
            "sig": signals})
        print(f"  {label}: {len(frames)} frames, dt={dt_frame:.4f}s, settled={c['settled']}",
              flush=True)

    out = {
        "meta": {"scale": "1:50 (model)", "n_bodies": len(bodies),
                 "dof_order": ["surge", "sway", "heave", "roll", "pitch", "yaw"],
                 "units": "m / rad", "heading_deg": 0.0,
                 "sig": {"points": ["plat (platform centre)", "up (upwave buoy top)",
                                    "dn (downwave buoy top)"],
                         "buoy_top_lever_m": round(float(pts._L_TOP), 5),
                         "quantities": {"disp": "m", "vel": "m/s", "acc": "m/s^2"},
                         "comp_order": ["x (surge)", "y (sway)", "z (heave)"],
                         "note": "true-amplitude point kinematics on the same window+decimation "
                                 "as frames; scope reads case.sig[point][quantity][frame][comp]"},
                 "note": "displacements from equilibrium; heading 0 => wave travels +x, "
                         "arrives at -x (upwave) first"},
        "bodies": bodies, "upwave_buoy": upwave, "downwave_buoy": downwave, "cases": cases_out}
    with _OUT.open("w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"wrote {_OUT.name} ({_OUT.stat().st_size / 1e6:.2f} MB, {len(cases_out)} cases)")


if __name__ == "__main__":
    main()
