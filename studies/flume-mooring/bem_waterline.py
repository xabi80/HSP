"""Flume BEM waterline resolution: how much the 12-sided mesh moves the heave and pitch/tilt
resonance periods of each article (flume-mooring, decision 5).

The flume databases mesh each spar and plate with NT = 12 panels round (coupled_bem_osu.NT): the
12-sided waterline and hull hold 0.9549 of the circle's area, so C33 and C44/C55 are 4.5 % low.
Single-buoy databases at NT = 24 / 36 / 48 / 96 come from the SAME pipeline
(``BEM_NT=<n> python bem_cluster.py single``). Their FloatSim periods are compared with NT = 12:

  * single buoy (free, and moored in the design configuration: pin level, 0.3 N/m, T0 = 4 N,
    r = 0.2 m collar): FloatSim setups with each database;
  * cluster and platform: the coupled NT = 12 databases PATCHED with the single-buoy change --
    each buoy's hydrostatic block replaced by the fine single-buoy block (the coupled C is exactly
    block-diagonal in the single-buoy C), and the fine-minus-coarse single-buoy A(omega) and A_inf
    added to each buoy's self block (the interaction blocks, a smaller effect, and B stay
    NT = 12).

Periods are FloatSim's modal ones: K = C + the linearised FloatSim catenary force at the
equilibrium, M = M + A(omega) with A(omega) iterated at the mode, the generalised eigenproblem on
the joints' null space; heave by mass-weighted MAC against rigid heave, tilt as the wave-band
mode with the largest in-phase buoy-pitch participation (only modes shorter than 10 s). The
criterion: a resonance that moves by more than half the fine period step (0.025 s) means the
databases must be regenerated before Phase D.

Writes bem_waterline.json.  Run: python bem_waterline.py
"""
# ruff: noqa: E402  -- sys.path bootstrap first
from __future__ import annotations

import dataclasses
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, null_space

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import design_basis as db
import floatsim_decks as fd
import resonance_bandwidth as rb

from floatsim.driver import build_system
from floatsim.hydro.readers.capytaine import read_capytaine

OUT = HERE / "bem_waterline.json"
NTS = (12, 24, 36, 48, 96)
HALF_STEP = 0.025


def single_db(nt: int):  # type: ignore[no-untyped-def]
    return read_capytaine(HERE / f"single_osu_open{'' if nt == 12 else f'_nt{nt}'}_psd.nc")


def _patterns(dk) -> dict[str, np.ndarray]:  # type: ignore[no-untyped-def]
    n = 6 * len(dk.bodies)
    heave, tilt = np.zeros(n), np.zeros(n)
    for k, b in enumerate(dk.bodies):
        heave[6 * k + 2] = 1.0
        if b.hydro_body_label or b.hydro_database:
            tilt[6 * k + 4] = 1.0                  # buoy pitch (selection by participation)
    return {"heave": heave, "tilt": tilt}


def periods(s, dk, hdb) -> dict[str, float]:  # type: ignore[no-untyped-def]
    K = db._stiffness(s)
    Ma = np.asarray(s.lhs.M_plus_Ainf)
    idx = fd.hydro_dof(dk)
    A_w, A_inf, om = np.asarray(hdb.A), np.asarray(hdb.A_inf), np.asarray(hdb.omega)
    fin = np.isfinite(om)
    N = (null_space(np.asarray(s.constraints.jacobian(s.xi0))) if s.constraints is not None
         else np.eye(K.shape[0]))
    out = {}
    for name, pat in _patterns(dk).items():
        w = 2 * np.pi / 2.6
        for _ in range(8):
            A = np.empty(A_inf.shape)
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    A[i, j] = np.interp(w, om[fin], A_w[i, j, fin])
            M = Ma.copy()
            M[np.ix_(idx, idx)] += A - A_inf
            lam, V = eigh(N.T @ K @ N, N.T @ M @ N)
            Phi = N @ V
            if name == "heave":
                mac = [(ph @ M @ pat) ** 2 / ((ph @ M @ ph) * (pat @ M @ pat)) for ph in Phi.T]
            else:           # in-phase buoy pitch participation, per unit modal mass
                mac = [(ph @ pat) ** 2 / (ph @ M @ ph) for ph in Phi.T]
            # only wave-band modes (T < 10 s): drops the free buoy's zero-stiffness surge/sway/yaw
            # and the moorings' slow modes, which the tilt pattern's surge part would also match
            band = (2 * np.pi / 10.0) ** 2
            mac = [m if lm > band else -1.0 for m, lm in zip(mac, lam, strict=True)]
            j = int(np.argmax(mac))
            w_new = float(np.sqrt(max(lam[j], 0.0)))
            if abs(w_new - w) < 1e-7:
                break
            w = w_new
        out[name] = 2 * np.pi / w
        out[f"{name}_selector"] = float(mac[j])
    return out


def patched(hdb_c, s12, sfine):  # type: ignore[no-untyped-def]
    """The coupled database with each buoy's self block corrected by the single-buoy change: C
    replaced, A(omega) and A_inf corrected. B is left at NT = 12 -- it sets damping, not the
    periods, and adding the single-buoy B difference (which carries the irregular-frequency
    differences) breaks FloatSim's multi-body PSD gate."""
    assert np.allclose(np.asarray(hdb_c.omega), np.asarray(s12.omega), equal_nan=True)
    assert np.allclose(np.asarray(sfine.omega), np.asarray(s12.omega), equal_nan=True)
    A = np.array(hdb_c.A, copy=True)
    A_inf, C = np.array(hdb_c.A_inf, copy=True), np.array(hdb_c.C, copy=True)
    dA = np.asarray(sfine.A) - np.asarray(s12.A)
    dAi = np.asarray(sfine.A_inf) - np.asarray(s12.A_inf)
    for b in range(C.shape[0] // 6):
        sl = slice(6 * b, 6 * b + 6)
        assert np.allclose(C[sl, sl], np.asarray(s12.C)), "coupled C is not the single-buoy C"
        C[sl, sl] = np.asarray(sfine.C)
        A[sl, sl] += dA
        A_inf[sl, sl] += dAi
    return dataclasses.replace(hdb_c, A=A, A_inf=A_inf, C=C)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    res: dict = {"single": {}, "articles": {}}
    dbs = {nt: single_db(nt) for nt in NTS}
    for nt, h in dbs.items():
        res["single"][nt] = {"C33": float(h.C[2, 2]), "C55": float(h.C[4, 4]),
                             "A33_inf": float(h.A_inf[2, 2]), "A55_inf": float(h.A_inf[4, 4])}
    # single buoy: free and moored (design)
    dk_free = fd.deck("buoy", drag=False)
    dk_m, _ = fd.moored(fd.deck("buoy", drag=False), "buoy", **rb.mooring_opts("buoy"))
    for label, dk in (("buoy_free", dk_free), ("buoy", dk_m)):
        res["articles"][label] = {}
        for nt in NTS:
            s = fd.build_single(dk, dbs[nt], True)
            res["articles"][label][nt] = periods(s, dk, dbs[nt])
    # cluster and platform: patched coupled databases
    for art in ("cluster", "platform"):
        opts = rb.mooring_opts(art)
        dk, _ = fd.moored(fd.deck(art, drag=False), art, **opts)
        xi = fd.moored_equilibrium(art, tag=f"{art}:pin_level_w{rb.W_AIR:g}", **opts)
        hd = fd.hdbs(art)
        res["articles"][art] = {}
        for nt in NTS:
            h = hd["shared_hydro_database"] if nt == 12 else patched(
                hd["shared_hydro_database"], dbs[12], dbs[nt])
            s = build_system(fd.with_positions(dk, xi), dt=fd.DT, t_max_kernel=fd.T_KERNEL,
                             solve_equilibrium=False, **{**hd, "shared_hydro_database": h})
            res["articles"][art][nt] = periods(s, dk, h)
            print(f"{art} NT {nt}: {res['articles'][art][nt]}", flush=True)
    for label, rows in res["articles"].items():
        ref = rows[96]
        for nt in NTS:
            rows[nt]["shift_vs_nt96_s"] = {m: rows[nt][m] - ref[m] for m in ("heave", "tilt")}
        print(f"{label:9s}: " + " | ".join(
            f"NT {nt}: heave {rows[nt]['heave']:.3f} tilt {rows[nt]['tilt']:.3f} "
            f"(d {rows[nt]['shift_vs_nt96_s']['heave']:+.3f}/"
            f"{rows[nt]['shift_vs_nt96_s']['tilt']:+.3f})"
            for nt in NTS), flush=True)
    res["verdict"] = {label: {nt: bool(max(abs(v) for v in rows[nt]["shift_vs_nt96_s"].values())
                                      > HALF_STEP) for nt in NTS}
                      for label, rows in res["articles"].items()}
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
