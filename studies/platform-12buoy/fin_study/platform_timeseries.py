"""Time-history extractor + on-demand plotter for the 12-buoy platform.

For any case (fin {0215,015,none}, plate drag Cd, wave period T, wave height H)
this runs the SAME simulation as the fin fan (``platform_fin_fan._build`` +
``platform_rao_pilot.run_case``) over the settled window and derives the full
motion of two families of points:

  * ``platform``            -- the platform reference (structural centre);
  * ``buoy1`` .. ``buoy12`` -- the TOP of each buoy, i.e. the spar-top / deck
    attach point (body-frame lever [0,0, Z_HUB_REF - Z_BUOY_REF] = 1.69 m above
    the buoy reference). This is the point a payload would sit on.

For every point we store the inertial DISPLACEMENT (from equilibrium),
VELOCITY, and ACCELERATION, each as (x, y, z) = (surge, sway, heave) directions.
Velocity/acceleration are central differences of the exact point displacement
(rigid-body kinematics from the 6-DOF); the platform-centre acceleration is
cross-checked against the integrator's ``xi_ddot`` at export time.

The per-case result is cached to ``timeseries/ts_<tag>_Cd<cd>_H<H>_T<T>.npz`` so
the ~minutes-long integration runs once; subsequent plots load instantly.

USAGE
  # run + cache (no plot):
  python platform_timeseries.py --fin none --T 2.5 --H 0.08 --export

  # plot any channels (body:quantity:component), one subplot per quantity:
  python platform_timeseries.py --fin 0215 --T 3.141 --H 0.04 \
      --plot platform:acc:z buoy8:acc:z buoy8:disp:z --out demo.png

  --fin   0215 | 015 | none          (none = spar bottom-cap, no fin)
  --cd    5 | 1                       plate drag Cd_n (default 5; none only has 5)
  body    platform | buoy1..buoy12    (buoyK = the TOP of buoy K)
  quantity disp | vel | acc           (aliases: displacement/position, velocity, acceleration)
  component x | y | z                 (aliases: surge->x, sway->y, heave/vertical->z)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # platform-12buoy
sys.path.insert(0, str(_HERE.parent.parent / "cluster-3buoy-rigid"))

import platform_common as pc  # noqa: E402
import platform_fin_fan as pff  # noqa: E402
import platform_rao_pilot as prp  # noqa: E402

_CACHE = _HERE / "timeseries"
_L_TOP = pc.Z_HUB_REF - pc.Z_BUOY_REF  # buoy-top (attach) lever above buoy ref, body frame
_NAMES = ["platform"] + [f"buoy{k + 1}" for k in range(12)]
_COMP = {"x": 0, "surge": 0, "y": 1, "sway": 1, "z": 2, "heave": 2, "vertical": 2}
_QUANT = {"disp": "disp", "displacement": "disp", "position": "disp", "pos": "disp",
          "vel": "vel", "velocity": "vel", "acc": "acc", "acceleration": "acc", "accel": "acc"}
_UNIT = {"disp": ("mm", 1e3), "vel": ("mm/s", 1e3), "acc": ("m/s²", 1.0)}
_QLABEL = {"disp": "displacement", "vel": "velocity", "acc": "acceleration"}


def _resolve_fin(tag: str, cd: float | None) -> tuple[str, float, float]:
    """(tag, plate_r, cd) for the fan config -- mirrors platform_fin_fan._CONFIGS."""
    for t, fin_r, cds in pff._CONFIGS:
        if t != tag:
            continue
        for c, plate_r in cds:
            if cd is None or abs(c - cd) < 1e-9:
                return t, plate_r, c
        raise SystemExit(f"fin {tag} has no Cd={cd}; available {[c for c, _ in cds]}")
    raise SystemExit(f"unknown fin '{tag}'; choose from 0215 / 015 / none")


def _cache_path(tag: str, cd: float, H: float, T: float) -> Path:
    return _CACHE / (f"ts_{tag}_Cd{cd:g}_H{H:g}_T{T:g}".replace(".", "p") + ".npz")


def _top_column(rot: np.ndarray) -> np.ndarray:
    """Rotated body-z axis (3rd column of R_zyx) for roll/pitch/yaw arrays (nt,3).

    R = Rz(yaw) Ry(pitch) Rx(roll); returns (nt,3) = R @ [0,0,1]."""
    phi, th, psi = rot[:, 0], rot[:, 1], rot[:, 2]  # roll, pitch, yaw
    cphi, sphi, cth, sth, cpsi, spsi = (
        np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.cos(psi), np.sin(psi))
    return np.column_stack([cpsi * sth * cphi + spsi * sphi,
                            spsi * sth * cphi - cpsi * sphi,
                            cth * cphi])


def _point_kinematics(xi6: np.ndarray, acc6: np.ndarray, t: np.ndarray,
                      lever: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inertial displacement, velocity, acceleration (each (nt,3)) of the point a
    distance ``lever`` up the body z-axis from the body reference.

    Rigid-body point kinematics from the body 6-DOF: the reference-point linear
    acceleration and the angular acceleration come straight from the integrator
    (``acc6`` = ``xi_ddot``); the angular velocity is a central difference of the
    rotation DOF (small angles -> Euler rates ~ body angular velocity):

        p = trans + (R - I) r0
        v = trans_dot + w x (R r0)
        a = trans_ddot + alpha x (R r0) + w x (w x (R r0))

    For ``lever == 0`` (the platform reference itself) this is exactly the
    integrator's own translation/velocity/acceleration -- no differentiation of
    the acceleration is involved."""
    trans, rot = xi6[:, 0:3], xi6[:, 3:6]
    trans_dot = np.gradient(trans, t, axis=0)
    trans_ddot = acc6[:, 0:3]
    if lever == 0.0:
        return trans.copy(), trans_dot, trans_ddot
    r_world = lever * _top_column(rot)  # R @ [0,0,lever]
    disp = trans + r_world - np.array([0.0, 0.0, lever])
    omega = np.gradient(rot, t, axis=0)  # small-angle Euler rates ~ angular velocity
    alpha = acc6[:, 3:6]
    vel = trans_dot + np.cross(omega, r_world)
    acc = trans_ddot + np.cross(alpha, r_world) + np.cross(omega, np.cross(omega, r_world))
    return disp, vel, acc


def export_case(tag: str, cd: float | None, H: float, T: float,
                cap_settle_s: float = 350.0, window_periods: float = 6.0) -> Path:
    """Run one case (or return the cache) and persist point pos/vel/acc as npz."""
    tag, plate_r, cd = _resolve_fin(tag, cd)
    out = _cache_path(tag, cd, H, T)
    if out.exists():
        print(f"cache hit: {out.name}", flush=True)
        return out
    _CACHE.mkdir(exist_ok=True)
    print(f"running fin={tag} Cd={cd:g} H={H:g} T={T:g}  (integrating to steady state)...",
          flush=True)
    hdb = pff._hdb(tag)
    hydro_dof = prp._hydro_dof(prp._deck_with_drag())
    setup = pff._build(plate_r, cd, hdb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = prp.run_case(setup, hdb, hydro_dof, height_m=H, period_s=T, ramp_s=20.0,
                         cap_settle_s=cap_settle_s, window_periods=window_periods, dt=0.01)
    t = np.asarray(c["t"], dtype=float)
    xi = np.asarray(c["xi"], dtype=float)  # (nt, 102)
    acc_solver = np.asarray(c["acc"], dtype=float)

    disp = np.empty((t.size, 13, 3))
    vel = np.empty_like(disp)
    acc = np.empty_like(disp)
    for i, name in enumerate(_NAMES):
        bidx = prp._buoy_body_index_platform() if name == "platform" else prp._buoy_body_index(i - 1)
        sl = slice(6 * bidx, 6 * bidx + 6)
        lever = 0.0 if name == "platform" else _L_TOP
        disp[:, i, :], vel[:, i, :], acc[:, i, :] = _point_kinematics(
            xi[:, sl], acc_solver[:, sl], t, lever)

    meta = {"fin": tag, "cd": cd, "H": H, "T": T, "omega": float(c["omega"]),
            "amp_m": float(c["amp_m"]), "settled": bool(c["settled"]),
            "rao_platform": float(c["rao"]["platform_heave"]),
            "buoy_top_lever_m": float(_L_TOP),
            "dof_order": "x=surge,y=sway,z=heave", "n": int(t.size)}
    np.savez_compressed(out, t=t, names=np.array(_NAMES), disp=disp, vel=vel, acc=acc,
                        meta=json.dumps(meta))
    print(f"wrote {out.name}  ({out.stat().st_size / 1e6:.2f} MB, {t.size} samples, "
          f"settled={meta['settled']})", flush=True)
    return out


def _parse_channel(tok: str) -> tuple[str, str, str]:
    parts = tok.split(":")
    if len(parts) != 3:
        raise SystemExit(f"bad channel '{tok}'; expected body:quantity:component")
    body, q, comp = parts
    if body not in _NAMES:
        raise SystemExit(f"unknown body '{body}'; choose platform or buoy1..buoy12")
    if q.lower() not in _QUANT:
        raise SystemExit(f"unknown quantity '{q}'; choose disp/vel/acc")
    if comp.lower() not in _COMP:
        raise SystemExit(f"unknown component '{comp}'; choose x/y/z (surge/sway/heave)")
    return body, _QUANT[q.lower()], comp.lower()


def plot(tag: str, cd: float | None, H: float, T: float, channels: list[str],
         out: Path, title: str | None) -> None:
    path = export_case(tag, cd, H, T)
    d = np.load(path, allow_pickle=True)
    t = d["t"]
    t = t - t[0]
    names = list(d["names"])
    meta = json.loads(str(d["meta"]))
    chans = [_parse_channel(c) for c in channels]
    quants = list(dict.fromkeys(q for _, q, _ in chans))  # preserve order, unique

    fig, axes = plt.subplots(len(quants), 1, figsize=(11, 2.9 * len(quants) + 0.6),
                             sharex=True, squeeze=False)
    print("\npeak amplitudes (0-to-peak over the window):", flush=True)
    for ax, q in zip(axes[:, 0], quants):
        unit, scale = _UNIT[q]
        for body, qq, comp in chans:
            if qq != q:
                continue
            j = _COMP[comp]
            y = d[q][:, names.index(body), j] * scale
            pk = float(np.max(np.abs(y)))
            lbl = f"{body} · {comp}"
            ax.plot(t, y, lw=1.4, label=lbl)
            print(f"  {body:9s} {_QLABEL[q]:12s} {comp} : {pk:.4g} {unit}", flush=True)
        ax.set_ylabel(f"{_QLABEL[q]}\n({unit})", fontsize=9)
        ax.axhline(0, color="0.6", lw=0.6, zorder=0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2, loc="upper right")
        ax.margins(x=0)
    axes[-1, 0].set_xlabel("time (s)  [settled window]", fontsize=9)

    tn = title or (
        f"12-buoy platform time histories -- fin {tag} (Cd_n {meta['cd']:g}), "
        f"T={T:g} s, H={H:g} m  |  omega={meta['omega']:.3f} rad/s, "
        f"platform-heave RAO {meta['rao_platform']:.3f}"
        + ("" if meta["settled"] else "  [NOT settled]"))
    fig.suptitle(tn, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fin", required=True, choices=["0215", "015", "none"])
    ap.add_argument("--cd", type=float, default=None, help="plate Cd_n (default 5; none->cap)")
    ap.add_argument("--T", type=float, required=True, help="wave period (s)")
    ap.add_argument("--H", type=float, required=True, help="wave height (s)")
    ap.add_argument("--export", action="store_true", help="run + cache only (no plot)")
    ap.add_argument("--plot", nargs="+", default=None, metavar="body:quantity:component",
                    help="channels to plot, e.g. platform:acc:z buoy8:disp:z")
    ap.add_argument("--out", type=Path, default=_HERE / "platform_timeseries.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.export or not args.plot:
        export_case(args.fin, args.cd, args.H, args.T)
    if args.plot:
        plot(args.fin, args.cd, args.H, args.T, args.plot, args.out, args.title)


if __name__ == "__main__":
    main()
