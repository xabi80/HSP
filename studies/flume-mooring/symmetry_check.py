"""Mirror symmetry about the wave axis (flume-mooring, decision 2 condition (a)).

Criterion 2 (mooring periods >= 4 x T_wave) is applied only to modes the waves drive. At heading
0, yaw, sway and roll of an article that is mirror-symmetric about the wave axis (y -> -y) have
no first-order or mean wave forcing. This script checks that symmetry on FloatSim's OWN
assembled system, for the moored design configuration of each article (pin level, 0.3 N/m; the
single buoy with its r = 0.2 m radial collar; the cluster and the platform at 45 deg):

  P = the mirror operator: body k -> the body at its mirrored reference point, DOFs scaled by
      (1, -1, 1, -1, 1, -1) (x, y, z; roll, pitch, yaw as a pseudo-vector);
  checks  P M P^T = M,  P C P^T = C,  P K(t) P^T = K(t)  (radiation kernel),
          f_exc(t) = P f_exc(t)  (heading-0 excitation),
          F(P xi, P xi_dot) = P F(xi, xi_dot)  (catenary + wave-relative drag at heading 0),
          P Pi(xi) P^T = Pi(P xi)  (Pi = projector on the joints' feasible velocities null(G)).

Each is reported as a relative residual. Then a short FloatSim wave run (H = 0.04 m at the tilt
resonance) reports the largest yaw, sway and roll: zero up to round-off unless something breaks
the symmetry or amplifies round-off.

Writes symmetry_check.json.  Run: python symmetry_check.py [buoy|cluster|platform ...]
"""
# ruff: noqa: E402  -- sys.path bootstrap first
from __future__ import annotations

import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import null_space

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import floatsim_decks as fd
import resonance_bandwidth as rb

OUT = HERE / "symmetry_check.json"
SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
T_RES = {"buoy": 2.55, "cluster": 2.60, "platform": 2.65}
RUN_SETTLE_S = 120.0                           # after the 15 s ramp, like a Phase D case


def mirror_operator(dk) -> tuple[np.ndarray, list[int]]:  # type: ignore[no-untyped-def]
    refs = np.array([b.reference_point for b in dk.bodies], dtype=float)
    n = 6 * len(refs)
    P = np.zeros((n, n))
    perm = []
    for k, r in enumerate(refs):
        m = r * np.array([1.0, -1.0, 1.0])
        j = int(np.argmin(np.linalg.norm(refs - m, axis=1)))
        if np.linalg.norm(refs[j] - m) > 1e-9:
            raise ValueError(f"body {dk.bodies[k].name}: no body at the mirrored point {m}")
        a, b = dk.bodies[k], dk.bodies[j]
        same = (a.mass == b.mass and a.inertia.Ixx == b.inertia.Ixx
                and a.inertia.Iyy == b.inertia.Iyy and a.inertia.Izz == b.inertia.Izz
                and a.inertia.Ixy == -b.inertia.Ixy and a.inertia.Iyz == -b.inertia.Iyz
                and a.inertia.Ixz == b.inertia.Ixz and len(a.drag_elements) == len(b.drag_elements))
        if not same:
            raise ValueError(f"{a.name} and its mirror {b.name} differ in mass/inertia/drag")
        perm.append(j)
        P[6 * j:6 * j + 6, 6 * k:6 * k + 6] = np.diag(SIGN)
    return P, perm


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def check(article: str) -> dict:
    t0 = time.perf_counter()
    opts = rb.mooring_opts(article)
    dk, _ = fd.moored(fd.deck(article), article, **opts)
    xi_eq = None if article == "buoy" else fd.moored_equilibrium(
        article, tag=f"{article}:pin_level_w{rb.W_AIR:g}", **rb.mooring_opts(article))
    T = T_RES[article]
    s, hd, wave, ramp = fd.wave_setup(dk, article, T, 0.04, xi_eq=xi_eq)
    P, perm = mirror_operator(dk)
    n = P.shape[0]
    out: dict = {"article": article, "bodies": len(dk.bodies),
                 "self_mirrored_bodies": int(sum(1 for k, j in enumerate(perm) if k == j))}
    M, C = np.asarray(s.lhs.M_plus_Ainf), np.asarray(s.lhs.C)
    out["M_plus_Ainf"] = _rel(P @ M @ P.T, M)
    out["C"] = _rel(P @ C @ P.T, C)
    K = np.asarray(s.kernel.K)
    out["kernel"] = max(_rel(P @ K[:, :, i] @ P.T, K[:, :, i]) if np.abs(K[:, :, i]).max() else 0.0
                        for i in range(0, K.shape[2], max(1, K.shape[2] // 50)))
    hdb = hd.get("shared_hydro_database") or hd["bem_databases"]["buoy"]
    from floatsim.hydro.excitation import make_regular_wave_force
    f_wave = make_regular_wave_force(hdb=hdb, wave=wave, body_position=(0.0, 0.0, 0.0), ramp=ramp)
    idx = fd.hydro_dof(dk)
    exc = []
    for t in np.linspace(20.0, 20.0 + T, 9):
        f = np.zeros(n)
        f[idx] = f_wave(t)
        exc.append(_rel(P @ f, f))
    out["excitation"] = max(exc)
    rng = np.random.default_rng(4)
    xi0 = np.asarray(s.xi0)
    sf = []
    for t in (0.0, 20.0, 21.3):
        xi = xi0 + rng.normal(0.0, 0.01, n)
        xd = rng.normal(0.0, 0.05, n)
        a = s.state_force(t, P @ xi, P @ xd)
        b = P @ s.state_force(t, xi, xd)
        sf.append(_rel(a, b))
    out["state_force"] = max(sf)
    if s.constraints is not None:
        pj = []
        for _ in range(3):
            xi = xi0 + rng.normal(0.0, 0.01, n)
            N1 = null_space(np.asarray(s.constraints.jacobian(xi)))
            N2 = null_space(np.asarray(s.constraints.jacobian(P @ xi)))
            pj.append(_rel(P @ (N1 @ N1.T) @ P.T, N2 @ N2.T))
        out["joint_projector"] = max(pj)
    # dynamic: heading-0 run at the tilt resonance; antisymmetric DOFs must stay at round-off
    r = fd.run_case(s, hd, dk, wave, ramp, RUN_SETTLE_S, 2)
    d = r.xi - xi0
    anti = {name: float(np.abs(d[:, j::6]).max()) for name, j in (("sway", 1), ("roll", 3),
                                                                     ("yaw", 5))}
    # the trend: the largest antisymmetric excursion per 30 s window (steady = forced by the
    # kernel's ~1e-5 asymmetry; growing = round-off amplified by an instability)
    edges = np.arange(fd.RAMP_S, r.t[-1] + 1e-9, 30.0)
    windows = {name: [float(np.abs(d[(r.t >= a) & (r.t < b), j::6]).max())
                      for a, b in itertools.pairwise(edges)]
               for name, j in (("sway", 1), ("roll", 3), ("yaw", 5))}
    sym = {name: float(np.abs(d[:, j::6]).max()) for name, j in (("surge", 0), ("heave", 2),
                                                                    ("pitch", 4))}
    out["run"] = {"H": 0.04, "T": T, "duration_s": float(r.t[-1]), "antisymmetric_max": anti,
                  "symmetric_max": sym, "antisymmetric_per_30s_window": windows}
    out["wall_min"] = (time.perf_counter() - t0) / 60
    print(f"{article:8s}: M {out['M_plus_Ainf']:.1e}  C {out['C']:.1e}  K {out['kernel']:.1e}  "
          f"f_exc {out['excitation']:.1e}  F_state {out['state_force']:.1e}  joints "
          f"{out.get('joint_projector', 0.0):.1e} | run: max sway {anti['sway']:.1e} m, roll "
          f"{np.degrees(anti['roll']):.1e} deg, yaw {np.degrees(anti['yaw']):.1e} deg "
          f"(pitch {np.degrees(sym['pitch']):.2f} deg) [{out['wall_min']:.1f} min]",
          flush=True)
    roll_w = [f"{np.degrees(v):.2e}" for v in windows["roll"]]
    print(f"          roll per 30 s window (deg): {roll_w}", flush=True)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    arts = sys.argv[1:] or ["buoy", "cluster", "platform"]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for a in arts:
        res[a] = check(a)
        OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
