"""Flume mooring design basis (Phase A(d) and the Phase B caveats): numbers derived from FloatSim's
assembled matrices and the record. No time-domain runs.

1. Mooring natural periods (surge, sway, yaw) of the three articles with PIN-LEVEL attachment and
   0.3 N/m lines (in air), plus the heave mode from the same analysis:
     M(w) = FloatSim M + A_inf with A_inf replaced by the BEM A(w) at the mode frequency (iterated);
     K    = FloatSim hydrostatic C + the linearised FloatSim catenary force (central differences of
            the setup's state force at the settled equilibrium; drag is zero at zero velocity);
     constrained generalised eigenproblem on null(G) (the joint Jacobian), as the driver's
     restoring-PSD gate does. Modes are picked by mass-weighted MAC against rigid surge / sway /
     yaw / heave patterns.
   Equilibria: buoy -- FloatSim static solve (one body, balanced lines); cluster -- the cached
   settle cluster:pin_level_w0.3; platform -- the cached settles at 0.02 and 1.0 N/m, which
   bracket 0.3 N/m (a 0.3 N/m platform settle would be a new run).
2. ka of the flume spars and plates across the band, and the recorded drift bound's
   potential-flow share (mooring_sizing.drift_per_spar).
3. FloatSim's drag drift on the FIXED article (wave-relative drag through build_system's wiring,
   body held at the reference state, averaged over one period).
4. Wave slope kA over the test matrix (H/L <= 1/15 cap), for the small-angle (LEVEL2) screen.

Writes design_basis.json.  Run: python design_basis.py
"""
# ruff: noqa: E402  -- sys.path bootstrap first
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, null_space

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import floatsim_decks as fd
import mooring_sizing as ms

import floatsim.driver as fsd
from floatsim.solver.ramp import HalfCosineRamp
from floatsim.waves.regular import RegularWave

PIN_B = np.array([0.0, 0.0, 1.624])
Z_PIN = -0.907 + 1.624
OUT = HERE / "design_basis.json"


def _pin_opts(w: float) -> dict:
    return {"fairlead": PIN_B, "anchor_z": Z_PIN, "w_line": w}


def _setup(article: str, w: float, tag: str | None):  # type: ignore[no-untyped-def]
    dk0 = fd.deck(article, drag=False)
    dkm, lines = fd.moored(dk0, article, **_pin_opts(w))
    hd = fd.hdbs(article)
    if article == "buoy":
        s = fd.build_single(dkm, hd["bem_databases"]["buoy"], True)
        return s, dkm, hd, lines
    xi = fd.moored_equilibrium(article, tag=tag, **_pin_opts(w))
    s = fsd.build_system(fd.with_positions(dkm, xi), dt=fd.DT, t_max_kernel=fd.T_KERNEL,
                         solve_equilibrium=False, **hd)
    return s, dkm, hd, lines


def _stiffness(s, h: float = 1.0e-4) -> np.ndarray:  # type: ignore[no-untyped-def]
    n = s.lhs.n_dof
    z = np.zeros(n)
    K = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        K[:, j] = -(s.state_force(0.0, s.xi0 + e, z) - s.state_force(0.0, s.xi0 - e, z)) / (2 * h)
    return np.asarray(s.lhs.C) + 0.5 * (K + K.T)


def _patterns(dk) -> dict[str, np.ndarray]:  # type: ignore[no-untyped-def]
    n = 6 * len(dk.bodies)
    p = {k: np.zeros(n) for k in ("surge", "sway", "heave", "yaw")}
    for b, body in enumerate(dk.bodies):
        x, y = body.reference_point[0], body.reference_point[1]
        p["surge"][6 * b] = 1.0
        p["sway"][6 * b + 1] = 1.0
        p["heave"][6 * b + 2] = 1.0
        p["yaw"][6 * b:6 * b + 2] = (-y, x)
        p["yaw"][6 * b + 5] = 1.0
    return p


def modes(article: str, w: float, tag: str | None) -> dict:
    s, dk, hd, lines = _setup(article, w, tag)
    hdb = hd.get("shared_hydro_database") or hd["bem_databases"]["buoy"]
    idx = fd.hydro_dof(dk)
    K = _stiffness(s)
    Ma = np.asarray(s.lhs.M_plus_Ainf).copy()
    A_w, A_inf, om = np.asarray(hdb.A), np.asarray(hdb.A_inf), np.asarray(hdb.omega)
    if s.constraints is not None:
        N = null_space(np.asarray(s.constraints.jacobian(s.xi0)))
    else:
        N = np.eye(K.shape[0])
    out = {}
    for name, pat in _patterns(dk).items():
        w_m = 0.4
        for _ in range(6):                      # iterate A(w) at the mode frequency
            A = np.stack([np.interp(w_m, om, A_w[i, j]) for i in range(A_w.shape[0])
                          for j in range(A_w.shape[1])]).reshape(A_w.shape[:2])
            M = Ma.copy()
            M[np.ix_(idx, idx)] += A - A_inf
            lam, V = eigh(N.T @ K @ N, N.T @ M @ N)
            Phi = N @ V
            mac = [(ph @ M @ pat) ** 2 / ((ph @ M @ ph) * (pat @ M @ pat)) for ph in Phi.T]
            j = int(np.argmax(mac))
            w_new = float(np.sqrt(max(lam[j], 0.0)))
            if w_new == 0.0 or abs(w_new - w_m) < 1e-6:
                w_m = w_new
                break
            w_m = w_new
        # Rayleigh quotient on the RIGID pattern (A(w) iterated): the consistent measure for the
        # rigid mooring modes -- the pins do no work in a rigid motion, so the joint-reaction
        # geometric stiffness (absent from this linearisation) does not enter.
        w_r = 0.4
        for _ in range(8):
            A = np.array([[np.interp(w_r, om, A_w[i, jj]) for jj in range(A_w.shape[1])]
                          for i in range(A_w.shape[0])])
            M = Ma.copy()
            M[np.ix_(idx, idx)] += A - A_inf
            kq, mq = float(pat @ K @ pat), float(pat @ M @ pat)
            w_r = float(np.sqrt(kq / mq)) if kq > 0 else 0.0
        out[name] = {"T_eig_s": (2 * np.pi / w_m) if w_m > 0 else float("inf"),
                     "lam_eig": float(lam[j]), "mac": float(mac[j]),
                     "T_rayleigh_s": (2 * np.pi / w_r) if w_r > 0 else float("inf"),
                     "K_rigid": kq, "M_rigid": mq}
    out["lowest_eigs"] = [float(v) for v in lam[:6]]
    out["lines"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ln.items()
                     if k in ("name", "anchor", "k", "T0", "L0", "chord", "w")} for ln in lines]
    return out


def ka_table() -> dict:
    a_spar, a_plate, z_plate = ms.SPAR_D / 2, ms.PLATE_R, -1.383
    rows = []
    for T in (1.4, 1.6, 2.0, 2.2, 2.5, 2.9, 3.5, 4.0):
        k = ms.k_fin(T)
        fd_, fp, hu = ms.drift_per_spar(0.5, T)
        rows.append({"T": T, "k": k, "ka_spar": k * a_spar, "ka_plate": k * a_plate,
                     "plate_decay_e2kd": float(np.exp(-2 * k * abs(z_plate))),
                     "H_used_at_0.5": hu, "drift_drag_N": fd_, "drift_pot_N": fp,
                     "pot_share": fp / (fd_ + fp)})
    return {"rows": rows}


def fixed_body_floatsim_drift(article: str, H: float, T: float) -> dict:
    """Mean of FloatSim's wave-relative drag on the article held fixed (after the ramp)."""
    dk = fd.deck(article)
    n = 6 * len(dk.bodies)
    wave = RegularWave(amplitude=0.5 * H, omega=2 * np.pi / T, heading_deg=0.0)
    f = fsd._build_drag_state_force(dk, n, rho=dk.environment.water_density, wave=wave,
                                    ramp=HalfCosineRamp(duration=1.0))
    ts = 10.0 + np.arange(400) * T / 400
    F = np.mean([f(t, np.zeros(n), np.zeros(n)) for t in ts], axis=0)
    Fmax = np.max([np.abs(f(t, np.zeros(n), np.zeros(n))[0::6]).sum() for t in ts[::20]])
    nb = sum(1 for b in dk.bodies if b.hydro_body_label or b.hydro_database)
    return {"article": article, "H": H, "T": T, "mean_Fx_total_N": float(F[0::6].sum()),
            "mean_Fx_per_spar_N": float(F[0::6].sum() / nb), "peak_abs_Fx_sum_N": float(Fmax),
            "bound_per_spar_N": float(sum(ms.drift_per_spar(H, T)[:2]))}


def slope_table() -> list:
    rows = []
    for H in ms.H_LIST:
        for T in (1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5, 4.0):
            k = ms.k_fin(T)
            hu = min(H, ms.STEEP * 2 * np.pi / k)
            rows.append({"H": H, "T": T, "H_used": hu, "kA": k * hu / 2})
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    res = {"modes": {}, "ka": ka_table(), "slope": slope_table(), "fixed_drift": []}
    for art, w, tag in (("buoy", 0.3, None), ("cluster", 0.3, "cluster:pin_level_w0.3"),
                        ("platform", 0.02, "platform:pin_level"),
                        ("platform", 1.0, "platform:pin_level_w1")):
        m = modes(art, w, tag)
        res["modes"][f"{art}@{w:g}"] = m
        print(f"{art:8s} w {w:4.2f}: " + "  ".join(
            f"{k} {m[k]['T_rayleigh_s']:.2f} s (eig {m[k]['T_eig_s']:.2f}, MAC {m[k]['mac']:.2f})"
            for k in ("surge", "sway", "yaw", "heave")), flush=True)
        print(f"          K_rigid surge/sway/yaw {m['surge']['K_rigid']:.2f} / "
              f"{m['sway']['K_rigid']:.2f} / {m['yaw']['K_rigid']:.3f}; lowest eigs "
              + " ".join(f"{v:+.3e}" for v in m["lowest_eigs"]), flush=True)
    for art in ("cluster", "platform"):
        for H, T in ((0.5, 2.25), (0.3, 1.70)):
            r = fixed_body_floatsim_drift(art, H, T)
            res["fixed_drift"].append(r)
            print(f"fixed {art} H {H} T {T}: FloatSim mean drag Fx "
                  f"{r['mean_Fx_total_N']:+.2e} N "
                  f"({r['mean_Fx_per_spar_N']:+.2e}/spar) vs bound "
                  f"{r['bound_per_spar_N']:.3f}/spar; peak |Fx| {r['peak_abs_Fx_sum_N']:.2f} N",
                  flush=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
