"""Single-buoy mooring redesign (flume-mooring Phase C, item 2), from FloatSim.

The single buoy's four pin-level lines (spar top, +0.717 m; anchors at the walls in the pin
plane; dry weight 0.3 N/m, in air) carry the lowest pretension of the three articles
(T0 = 2.23 N), so they sag the most. Two things are redesigned together:

(a) PRETENSION for line-to-water clearance and slack. The critical line is the SLACK-side
    (downstream) one at the record's design state (``mooring_sizing.xspread``): the mean offset
    under the recorded drift bound, plus the wave amplitude A_w = H/2. The mean offset is a
    FloatSim static solve with the bound applied as a steady force at the spar waterline (the
    Phase D in-run method). At that state the line's lowest point must clear the crest (0.6 H)
    by >= 0.10 m (criterion 5), and its tension must stay >= 0.15 T0 (criterion 3).
(b) YAW RESTRAINT: the lines attach to a collar of radius r instead of the spar axis.
      radial   -- each line at the collar point facing its anchor. Linearised yaw stiffness
                  K = sum T r (1 + r/l): the pretension's moment arm dominates.
      pinwheel -- each line leaves its collar point tangentially, two turning each way (the
                  pretension moments cancel). K = sum k_eff r^2 at the calm position; the
                  pretension term returns once an offset turns the lines off tangency.
    Surge, sway and yaw periods must all be >= 4 x 3.5 s = 14 s (criterion 2).

Everything is FloatSim: the buoy deck and BEM (floatsim_decks), the lines' forces from
``make_catenary_state_force`` (each line's lowest point from the same Irvine closed form at the
H, V_A that FloatSim solved), static solves with ``solve_static_equilibrium``, and periods as
Rayleigh quotients on the rigid surge / sway / yaw patterns with FloatSim's M + A(omega)
(BEM, iterated at the mode) and K = C + the linearised FloatSim catenary force.

Writes single_buoy_redesign.json.  Run: python single_buoy_redesign.py
"""
# ruff: noqa: E402  -- sys.path bootstrap first
from __future__ import annotations

import dataclasses
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import design_basis as db
import floatsim_decks as fd
import mooring_sizing as ms

import floatsim.driver as fsd
from floatsim.mooring.catenary_analytic import make_catenary_state_force

PIN_B, Z_PIN = db.PIN_B, db.Z_PIN
W_AIR = 0.3                        # N/m, line dry weight (assumed, pending hardware)
H_CASES = (0.12, 0.2, 0.35, 0.5)   # top of the operational band + the extreme band
CREST = 0.6                        # crest height / H (steep regular waves)
CLEAR_MIN, SLACK_MIN = 0.10, 0.15  # criteria 5 and 3
T_SEP = 4 * 3.5                    # criterion 2: >= 4 x T_wave,max
OUT = HERE / "single_buoy_redesign.json"


def drift_envelope(H: float) -> tuple[float, float]:
    """Max recorded drift bound per spar over T = 1.4-3.5 s with H/lambda <= 0.08."""
    Ts = np.linspace(1.4, 3.5, 211)
    f = [sum(ms.drift_per_spar(H, T, steep=ms.STEEP_MATRIX)[:2]) for T in Ts]
    i = int(np.argmax(f))
    return float(f[i]), float(Ts[i])


def setup(T0: float | None, r: float, collar: str = "radial"):  # type: ignore[no-untyped-def]
    dk, lines = fd.moored(fd.deck("buoy", drag=False), "buoy", fairlead=PIN_B, anchor_z=Z_PIN,
                          w_line=W_AIR, T0=T0, collar_r=r, collar=collar)
    hdb = fd.hdbs("buoy")["bem_databases"]["buoy"]
    return fd.build_single(dk, hdb, True), dk, lines, hdb


def _drift_force(F: float, n: int = 6):  # type: ignore[no-untyped-def]
    """The recorded bound as a steady +x force at the spar's calm waterline (body frame)."""
    arm = np.asarray(fd.WL_B, dtype=float)

    def f(_t: float, xi: np.ndarray, _xd: np.ndarray) -> np.ndarray:
        out = np.zeros(n)
        r_arm = arm + np.cross(xi[3:6], arm)
        out[0] = F
        out[3:6] = np.cross(r_arm, np.array([F, 0.0, 0.0]))
        return out

    return f


K_YAW_REG = 1.0e-4   # N m/rad: numerical yaw spring for the static solve only (see mean_offset)


def mean_offset(s, F: float) -> np.ndarray:  # type: ignore[no-untyped-def]
    """FloatSim static solve under the drift bound, by load continuation (F/4 ... F, each from
    the previous solution): hybr from the calm state can stall on the stiffening lines.

    With the lines on the spar axis (r = 0) the buoy has NO yaw stiffness, so the 6-DOF static
    Jacobian is singular; a numerical yaw spring K_YAW_REG (a 157 s yaw period) regularises it
    and replaces the solver's default Tikhonov term (regularization=0): with that term
    (1e-8 max C = 1.9e-6) hybr stalls on this problem, whose near-empty yaw row couples to roll
    through the small-angle arm. Yaw does not couple to the other DOFs in this symmetric
    head-sea state, so the spring changes no reported number."""
    xi = np.asarray(s.xi0, dtype=float)
    z = np.zeros(6)
    f_prev = 0.0
    for frac in (0.25, 0.5, 0.75, 1.0):
        fdrift = _drift_force(frac * F)

        def sf(t: float, x: np.ndarray, xd: np.ndarray, _f=fdrift) -> np.ndarray:  # type: ignore[no-untyped-def]
            f = s.state_force(t, x, xd) + _f(t, x, xd)
            f[5] -= K_YAW_REG * x[5]
            return f

        # Linear predictor for the load increment (FloatSim's linearised stiffness): hybr started
        # ON the previous equilibrium can stop at once and report success (a false xtol
        # convergence), so the residual is also checked explicitly below.
        K = db._stiffness(dataclasses.replace(s, xi0=xi))
        K[5, 5] += K_YAW_REG
        dF = _drift_force(frac * F - f_prev)(0.0, xi, z)
        x_start = xi + np.linalg.lstsq(K, dF, rcond=None)[0]
        xi = np.asarray(fsd.solve_static_equilibrium(lhs=s.lhs, state_force=sf, xi0=x_start,
                                                     tol=1e-6, regularization=0.0,
                                                     allow_failure=True).xi_eq)
        res = float(np.abs(s.lhs.C @ xi - sf(0.0, xi, z)).max())
        if res > 1e-4:                              # 0.1 mN: 15 um of offset
            raise RuntimeError(f"static solve under drift {frac * F:.4f} N: residual {res:.2e} N")
        f_prev = frac * F
    return xi


def line_states(dk, xi: np.ndarray) -> list[dict]:  # type: ignore[no-untyped-def]
    """Per line: FloatSim tension and lowest point at state xi."""
    ref = np.array([b.reference_point for b in dk.bodies], dtype=float)
    names = fsd._validate_body_names(dk)
    out = []
    for c in dk.connections:
        att = fsd._materialise_catenary(c, names)
        F = make_catenary_state_force([att], n_dof=6, body_reference_points=ref)(
            0.0, xi, np.zeros(6))[:3]
        L, w, EA = att.line.length, att.line.weight_per_length, att.line.EA
        H, VF = float(np.hypot(F[0], F[1])), float(-F[2])
        VA = VF - w * L                               # Irvine S3
        z_low = float(att.anchor_global[2])
        if VA < 0.0 < VF:                             # the line dips below its anchor end
            s_low = -VA / w
            z_low += (H - np.hypot(H, VA)) / w + (VA * s_low + 0.5 * w * s_low**2) / EA
        fair = ref[0] + xi[:3] + att.fairlead_body + np.cross(xi[3:6], att.fairlead_body)
        z_low = min(z_low, float(fair[2]))
        out.append({"anchor_x": float(att.anchor_global[0]), "T_fair": float(np.hypot(H, VF)),
                    "H": H, "z_low": z_low, "L0": L, "k": EA / L})
    return out


def periods(s, dk, hdb) -> dict:  # type: ignore[no-untyped-def]
    K = db._stiffness(s)
    Ma = np.asarray(s.lhs.M_plus_Ainf)
    A_w, A_inf, om = np.asarray(hdb.A), np.asarray(hdb.A_inf), np.asarray(hdb.omega)
    out = {}
    for name, pat in db._patterns(dk).items():
        w_r = 0.4
        for _ in range(8):
            A = np.array([[np.interp(w_r, om, A_w[i, j]) for j in range(6)] for i in range(6)])
            M = Ma + A - A_inf
            kq, mq = float(pat @ K @ pat), float(pat @ M @ pat)
            w_r = float(np.sqrt(kq / mq)) if kq > 0 else 0.0
            if w_r == 0.0:
                break
        out[name] = {"T_s": (2 * np.pi / w_r) if w_r > 0 else float("inf"), "K": kq, "M": mq}
    return out


def design_state(s, dk, H: float, xi_c: np.ndarray) -> dict:  # type: ignore[no-untyped-def]
    """Design state at height H; ``xi_c`` is the calm equilibrium (``mean_offset(s, 0)``)."""
    F, T_at = drift_envelope(H)
    xi_m = mean_offset(s, F)
    xi_d = xi_m.copy()
    xi_d[0] += 0.5 * H                                # + the design wave amplitude A_w = H/2
    ls = line_states(dk, xi_d)
    ls0 = line_states(dk, xi_c)
    T0 = float(np.mean([ln["T_fair"] for ln in ls0]))
    slack = min(ls, key=lambda ln: ln["T_fair"])
    return {"H": H, "drift_N": F, "drift_T_s": T_at, "mean_offset_m": float(xi_m[0] - xi_c[0]),
            "mean_tilt_deg": float(np.degrees(xi_m[4] - xi_c[4])),
            "T0_actual_N": T0, "T_min_N": slack["T_fair"], "T_min_ratio": slack["T_fair"] / T0,
            "T_max_N": max(ln["T_fair"] for ln in ls),
            "stretch_max_m": max(ln["T_fair"] / ln["k"] for ln in ls),
            "z_low_m": min(ln["z_low"] for ln in ls),
            "clearance_m": min(ln["z_low"] for ln in ls) - CREST * H}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    res: dict = {"T0_sweep": [], "collar": [], "notes": {}}
    des = fd.line_design("buoy", 1)
    print(f"record design: k {des['k_line']:.3f} N/m, T0 {des['T0']:.3f} N", flush=True)
    for H in H_CASES:
        F, T = drift_envelope(H)
        print(f"drift bound H {H}: {F:.4f} N/spar at T {T:.2f} s", flush=True)
    # (a) pretension: clearance and slack at the design states, r = 0
    for T0 in (des["T0"], 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0):
        s, dk, _lines, hdb = setup(T0, 0.0)
        xi_c = mean_offset(s, 0.0)
        s = dataclasses.replace(s, xi0=xi_c)          # stiffness about the precise calm state
        row = {"T0": T0, "periods": periods(s, dk, hdb),
               "calm_sag_m": PIN_B[2] + fd.aw.ZB - min(ln["z_low"] for ln in line_states(dk, xi_c)),
               "states": [design_state(s, dk, H, xi_c) for H in H_CASES]}
        res["T0_sweep"].append(row)
        p = row["periods"]
        print(f"T0 {T0:5.2f}: surge {p['surge']['T_s']:5.1f} sway {p['sway']['T_s']:5.1f} "
              f"heave {p['heave']['T_s']:.3f} | " + "  ".join(
                  f"H{st['H']}: off {st['mean_offset_m']:.3f} clr {st['clearance_m']:+.3f} "
                  f"Tmin/T0 {st['T_min_ratio']:.2f} Tmax {st['T_max_N']:.2f}"
                  for st in row["states"]), flush=True)
    # (b) collar radius at the chosen pretension (the smallest T0 meeting criteria 3 and 5)
    ok = [r for r in res["T0_sweep"]
          if all(st["clearance_m"] >= CLEAR_MIN and st["T_min_ratio"] >= SLACK_MIN
                 for st in r["states"])]
    T0_sel = min(r["T0"] for r in ok) if ok else None
    res["notes"]["T0_selected"] = T0_sel
    print(f"smallest T0 meeting clearance and slack at every H: {T0_sel}", flush=True)
    T0_c = T0_sel if T0_sel is not None else des["T0"]
    for collar in ("radial", "pinwheel"):
        for r in (0.0, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.3):
            s, dk, _lines, hdb = setup(T0_c, r, collar)
            s = dataclasses.replace(s, xi0=mean_offset(s, 0.0))
            p = periods(s, dk, hdb)
            row = {"collar": collar, "r": r, "T0": T0_c, "periods": p}
            if r > 0:                                 # does the H = 0.5 m mean offset keep K_yaw?
                try:
                    s_off = dataclasses.replace(s, xi0=mean_offset(s, drift_envelope(0.5)[0]))
                    row["yaw_at_H0.5_offset"] = periods(s_off, dk, hdb)["yaw"]
                except RuntimeError as exc:           # no static equilibrium: yaw runs away
                    row["yaw_at_H0.5_offset"] = {"T_s": None, "K": None,
                                                 "failed": str(exc)[:160]}
            res["collar"].append(row)
            yo = row.get("yaw_at_H0.5_offset")
            extra = ("" if yo is None else " | H0.5 offset: NO static equilibrium"
                     if yo["K"] is None
                     else f" | H0.5 offset: yaw {yo['T_s']:.2f} s (K {yo['K']:+.5f})")
            print(f"{collar:8s} r {r:6.4f}: surge {p['surge']['T_s']:5.1f} sway "
                  f"{p['sway']['T_s']:5.1f} yaw {p['yaw']['T_s']:7.2f} s "
                  f"(K_yaw {p['yaw']['K']:.5f}, I {p['yaw']['M']:.4f}){extra}", flush=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
