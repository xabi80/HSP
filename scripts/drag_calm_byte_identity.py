"""Byte-identity gate for the drag path in calm water (STEP 5 PR1, relative-velocity drag).

For every committed deck that carries Morison drag elements, hash (sha256 of the raw float64
bytes):
  * the driver's drag state force ``floatsim.driver._build_drag_state_force(deck, ...)`` (no wave
    = calm water, the default path) at 40 fixed random states;
  * two short calm-water integrations through ``build_system`` + ``integrate_cummins`` (the
    12-buoy platform, whose bodies sit away from the origin, and the flume cluster).

Run on the commit before a change and after it; the JSON outputs must be identical.

    python scripts/drag_calm_byte_identity.py <out.json>
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
warnings.simplefilter("ignore")


def _decks() -> dict:  # type: ignore[type-arg]
    from floatsim.io.deck import load_deck

    out: dict = {}  # type: ignore[type-arg]
    for rel in ("examples/two_body_semisub_barge.yml",
                "studies/cluster-3buoy-rigid/deck_bem_morison.yaml"):
        try:
            out[rel] = load_deck(ROOT / rel)
        except Exception as exc:  # a stale YAML is recorded, not fatal
            out[rel] = f"UNLOADABLE: {type(exc).__name__}"
    for sub in ("studies/cluster-3buoy-rigid", "studies/platform-12buoy",
                "studies/platform-12buoy/flume-wall-effect", "studies/platform-16buoy",
                "studies/flume-mooring", "studies/osu-test-buoy", "studies/spar-fin-decay"):
        sys.path.insert(0, str(ROOT / sub))
    os.environ.setdefault("PLAT_ROT_DEG", "45")  # the 45-deg platform, as floatsim_decks sets
    import articulated_wall as aw
    import cluster_rao
    import floatsim_decks as fd
    import osu_buoy_common as oc
    import platform16_rao
    import platform_rao_pilot as prp
    import sparfin_rao

    out["cluster_rao(Cd5)"] = cluster_rao._deck(5.0)
    out["platform_rao_pilot"] = prp._deck_with_drag()
    out["platform16_rao"] = platform16_rao._deck_with_drag()
    out["articulated_wall"] = aw.deck()
    for art in ("buoy", "cluster", "platform"):
        out[f"floatsim_decks.{art}+moored"] = fd.moored(fd.deck(art), art)[0]
    out["osu_buoy_common"] = oc.deck(drag=oc.drag_elements())
    out["sparfin_rao(Cd5)"] = sparfin_rao._drag_deck(5.0)
    return out


def _h(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:24]


def main() -> None:
    import floatsim.driver as fsd

    res: dict[str, str] = {}
    for name, dk in _decks().items():
        if isinstance(dk, str):
            res[f"drag:{name}"] = dk
            continue
        n = 6 * len(dk.bodies)
        f = fsd._build_drag_state_force(dk, n, rho=dk.environment.water_density)
        rng = np.random.default_rng(7)
        vals = [f(float(rng.uniform(0, 30)), rng.normal(0, 0.05, n), rng.normal(0, 0.3, n))
                for _ in range(40)]
        res[f"drag:{name}"] = _h(np.stack(vals))
    # full calm-water integrations through build_system
    sys.path.insert(0, str(ROOT / "studies/flume-mooring"))
    import floatsim_decks as fd
    import platform_rao_pilot as prp

    from floatsim.hydro.readers.capytaine import read_capytaine
    from floatsim.solver.newmark import integrate_cummins

    dk = prp._deck_with_drag()
    s = fsd.build_system(dk, bem_databases={}, dt=0.01, t_max_kernel=30.0, solve_equilibrium=False,
                         shared_hydro_database=read_capytaine(prp._PLAT_NC),
                         asymptote_check_override=prp._ASYMPTOTE_OVR,
                         kernel_decay_floor_override=prp._KERNEL_EXEMPT)
    xi0 = s.xi0.copy()
    xi0[2::6] += 0.02
    r = integrate_cummins(lhs=s.lhs, kernel=s.kernel, xi0=xi0, xi_dot0=s.xi_dot0, duration=4.0,
                          dt=0.01, rho_inf=0.8, constraints=s.constraints,
                          state_force=s.state_force, projection_interval=1)
    res["integrate:platform_rao_pilot"] = _h(r.xi)
    dkc = fd.deck("cluster")
    s, _ = fd.build(dkc, "cluster")
    xi0 = s.xi0.copy()
    xi0[4::6] += 0.02
    r = integrate_cummins(lhs=s.lhs, kernel=s.kernel, xi0=xi0, xi_dot0=s.xi_dot0, duration=4.0,
                          dt=0.01, rho_inf=0.8, constraints=s.constraints,
                          state_force=s.state_force, projection_interval=1)
    res["integrate:flume cluster"] = _h(r.xi)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{v}  {k}")


if __name__ == "__main__":
    main()
