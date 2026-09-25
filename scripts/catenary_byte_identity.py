"""Byte-identity gate for the catenary path (flume-mooring Phase C1 / C2).

Hashes (sha256 of raw float64 bytes):
  * ``solve_catenary`` (H, V_fairlead, V_anchor) over 300 random geometries on which the cold
    start converges -- suspended and touchdown -- spanning offshore (OC4-like) and flume lines;
  * the catenary state force of the OC4 S4 attachments (tests/validation/test_m6_openfast_moored_eq)
    and of the M7 PR4 two-line deck (scripts/m7_pr4_driver_prediction) at 40 random body states;
  * the driver-built state force of the flume moored decks (buoy / cluster / platform, SWL and
    pin-level attachment) at 40 random states near their reference;
  * a short moored single-buoy integration.

Run before and after a change; the JSON outputs must be identical.
    python scripts/catenary_byte_identity.py <out.json>
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
warnings.simplefilter("ignore")


def _h(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:24]


def _geometries() -> str:
    from floatsim.mooring.catenary_analytic import CatenaryLine, solve_catenary

    rng = np.random.default_rng(5)
    out = []
    kept = 0
    while kept < 300:
        if rng.random() < 0.5:        # offshore line, anchor on the seabed
            line = CatenaryLine(length=float(rng.uniform(400, 900)), weight_per_length=float(
                rng.uniform(300, 1200)), EA=float(rng.uniform(1e8, 1e9)))
            anchor = np.array([0.0, -200.0])
            fair = np.array([float(rng.uniform(300, 800)), float(rng.uniform(-20, 0))])
            depth = 200.0
        else:                          # flume spring line, suspended
            L = float(rng.uniform(2.5, 5.0))
            line = CatenaryLine(length=L, weight_per_length=float(rng.uniform(0.01, 1.0)),
                                EA=float(rng.uniform(5.0, 60.0)))
            anchor = np.array([0.0, 0.0])
            fair = np.array([float(rng.uniform(L, 1.5 * L)), float(rng.uniform(-0.1, 0.8))])
            depth = 200.0
        try:
            s = solve_catenary(line=line, anchor_pos=anchor, fairlead_pos=fair, seabed_depth=depth)
        except (RuntimeError, ValueError):
            continue
        out.append([s.H, s.V_fairlead, s.V_anchor])
        kept += 1
    return _h(np.asarray(out))


def main() -> None:
    sys.path.insert(0, str(ROOT))
    for sub in ("studies/flume-mooring",):
        sys.path.insert(0, str(ROOT / sub))
    res: dict[str, str] = {"solve_catenary:300 geometries": _geometries()}
    from floatsim.mooring.catenary_analytic import make_catenary_state_force
    from tests.validation import test_m6_openfast_moored_eq as m6

    rng = np.random.default_rng(9)
    f = make_catenary_state_force(m6._build_oc4_attachments(), n_dof=6)
    res["oc4 s4 state force"] = _h(np.stack([f(0.0, rng.normal(0, [5, 5, 1, .02, .02, .05]),
                                               np.zeros(6)) for _ in range(40)]))
    from scripts import m7_pr4_driver_prediction as m7

    hw = m7.build_m4_pr6_hand_wired()
    sf = hw["state_force"]
    n = int(np.asarray(hw["lhs_global"].C).shape[0])
    res["m7 pr4 hand-wired equilibrium"] = _h(np.asarray(hw["xi0_post_equilibrium"]))
    res["m7 pr4 hand-wired state force"] = _h(np.stack([sf(0.0, rng.normal(0, 2.0, n),
                                                            np.zeros(n)) for _ in range(40)]))
    import floatsim_decks as fd

    import floatsim.driver as fsd
    pin = {"fairlead": np.array([0.0, 0.0, 1.624]), "anchor_z": 0.717}
    for art in ("buoy", "cluster", "platform"):
        for tag, opts in (("swl", {}), ("pin", pin)):
            dk, _ = fd.moored(fd.deck(art, drag=False), art, **opts)
            nd = 6 * len(dk.bodies)
            names = fsd._validate_body_names(dk)
            cats = [fsd._materialise_catenary(c, names) for c in dk.connections]
            kwargs = {}
            if "body_reference_points" in make_catenary_state_force.__code__.co_varnames:
                kwargs["body_reference_points"] = np.array([b.reference_point for b in dk.bodies])
            g = make_catenary_state_force(cats, n_dof=nd, **kwargs)
            res[f"flume {art} {tag} state force"] = _h(np.stack(
                [g(0.0, rng.normal(0, 0.02, nd), np.zeros(nd)) for _ in range(40)]))
    from floatsim.solver.newmark import integrate_cummins

    dkm, _ = fd.moored(fd.deck("buoy"), "buoy")
    s, _ = fd.build(dkm, "buoy")
    xi0 = s.xi0.copy()
    xi0[0] += 0.3
    r = integrate_cummins(lhs=s.lhs, kernel=s.kernel, xi0=xi0, xi_dot0=s.xi_dot0, duration=10.0,
                          dt=fd.DT, rho_inf=0.8, state_force=s.state_force)
    res["integrate: moored buoy surge 10 s"] = _h(r.xi)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{v}  {k}")


if __name__ == "__main__":
    main()
