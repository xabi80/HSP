"""Rev C extreme cord set (flume-mooring, 2026-09-25; Xabier's decisions on rev B).

Two cord sets on the SAME anchors and attachment points at the SAME nominal pretension T0: the
operational set (rev B, attachment_sweep.py's choice) for the response tests (H <= 0.12 m), and
a stiffer extreme set for the H = 0.2-0.5 m load / survival tests. The extreme set's cord
stiffness is a multiple m of the operational k; this picks the smallest m that keeps the article
within the tracking window at H = 0.5 m (T = 2.35 and 2.65 s, the conservative drift sum applied
in-run; line_hardware.extreme_run).

Pretension. "The same T0" must mean the same AT-REST tension at the calm geometry: the code's
nominal T0 is the tension at the design (untilted) chord, and the calm 1 deg tilt moves each
attachment delta towards its anchor, so the at-rest tension is T0 - k delta -- lower for a stiffer
cord (the platform's legs: 1.42 N with the operational cord, ~0.9 N at k x5 with the same nominal
T0). With the at-rest tension matched (nominal T0_ext = T_rest + m (T0 - T_rest)) the line forces
at the calm geometry are the operational set's, so the operational settle IS the extreme set's
calm equilibrium (the same 1 deg tilt; FloatSim's joint residual checks it). The k scan (``runs``)
used the nominal T0 (offsets differ by ~1 cm); ``final`` re-runs the chosen k at-rest-matched, and
those runs are the spec's.

Selection: the confirmed Phase B criterion 4 is mean offset + dynamic amplitude <= 1.0 m, i.e.
the run's maximum surge <= 1.0 m; Xabier's instruction states the mean offset <= 1.0 m. Both m
are reported; the spec uses criterion 4 (the record).

Steps (FloatSim throughout):
  predict  -- a quasi-static estimate: the rev B H = 0.5 m runs' mean offset gives the mean force
              (the operational set's rigid-pull curve at that offset); for each m the rigid-pull
              offset under that force, plus rev B's dynamic amplitude. Brackets the runs.
  runs M.. -- FloatSim extreme runs at the multiples M (from the operational settle, accepted
              within FloatSim's equilibrium tolerance with the stiffer lines: line_hardware
              ._setup(reuse_eq=True)).
  choose   -- the smallest m run that meets criterion 4 at both periods (and the mean-only m).
  final    -- the chosen m with the at-rest-matched pretension, both periods (the spec's runs).
  settle   -- FloatSim's settle of the chosen extreme set: the calm tilt and at-rest tensions.
  declare  -- the chosen extreme set's tilt / heave periods vs the unmoored article (modal, as
              attachment_sweep.py) -- DECLARED, not a criterion (load / survival tests).
  t0       -- optional (cheap, cluster only): T0 raised to the confirmed 3 deg mean-tilt
              allowance on the chosen extreme set -- does it cut the peak re-tension?

Writes extreme_set.json.  Run: python extreme_set.py predict|runs M1 M2 ..|choose|final|declare|t0
"""
# ruff: noqa: E402  -- sys.path bootstrap first
from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import attachment_sweep as asw
import floatsim_decks as fd
import line_hardware as lh
import tank_predictions as tp

OUT = HERE / "extreme_set.json"
ARTS = ("cluster", "platform")
H_EXT, T_EXT = 0.5, (2.35, 2.65)
FOV = 1.0
T0_TILT_CAP_DEG = 3.0                      # the confirmed mean-tilt allowance (criterion 1)


def rest_ratio(art: str, m: float) -> float:
    """Nominal-T0 factor that gives k x m the operational set's at-rest tension at the calm
    geometry: (T_rest + m (T0 - T_rest)) / T0, per line (spec_statics.json, line 1)."""
    ln = json.loads((HERE / "spec_statics.json").read_text())[art]["lines"][0]
    T0, Tr = ln["T0_nominal_N"], ln["T_at_rest_N"]
    return (Tr + m * (T0 - Tr)) / T0


def opts(art: str, m: float, t0_scale: float = 1.0, rest: bool = False) -> dict:
    """The operational (rev B) options with k x m and T0 x t0_scale (x rest_ratio if ``rest``)."""
    o = dict(asw.chosen(art)["opts"])
    o["k_scale"] = o["k_scale"] * m
    o["T0"] = o["T0"] * t0_scale * (rest_ratio(art, m) if rest else 1.0)
    return o


def _pull(art: str, o: dict):  # type: ignore[no-untyped-def]
    """Rigid-surge pull force of these lines from the operational settle, F(x) (N)."""
    dk, _ = fd.moored(fd.deck(art), art, **o)
    xi = np.asarray(json.loads(fd.EQ_CACHE.read_text())[asw.chosen(art)["tag"]]["xi"])
    lf = tp._line_forces(dk)
    z = np.zeros(xi.size)

    def res(x: np.ndarray) -> np.ndarray:
        return tp._resultant(dk, x, np.sum([f(0.0, x, z) for f in lf], axis=0))

    F0 = res(xi)[0]
    return lambda x: float(F0 - res(tp.rigid(dk, xi, "surge", x))[0])


def predict() -> dict:
    ad = json.loads((HERE / "attachment_design.json").read_text())
    out = {}
    for art in ARTS:
        base = _pull(art, opts(art, 1.0))
        rows = {r["T"]: r for r in ad["extremes"] if r["article"] == art and r["H"] == H_EXT}
        pred = []
        for m in (1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
            pm = _pull(art, opts(art, m))
            per = {}
            for T, r in rows.items():
                F = base(r["mean_offset_m"])
                lo, hi = 0.0, 5.0
                for _ in range(40):                       # pm is monotonic in x
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if pm(mid) < F else (lo, mid)
                amp = r["surge_max_m"] - r["mean_offset_m"]
                per[T] = {"F_mean_N": F, "mean_m": hi, "max_m": hi + amp}
            pred.append({"m": m, **{f"T{T:g}": v for T, v in per.items()}})
            print(f"{art:8s} k x{m:<4g}: " + ", ".join(
                f"T {T}: mean {v['mean_m']:.2f} max {v['max_m']:.2f} m (F {v['F_mean_N']:.1f} N)"
                for T, v in per.items()), flush=True)
        out[art] = pred
    return out


def run_one(args: tuple) -> dict:
    art, m, T, t0s = args[:4]
    rest = len(args) > 4 and args[4]
    des = asw.chosen(art)
    o = opts(art, m, t0s, rest)
    d = {"opts": o, "tag": des["tag"] if t0s == 1.0 else f"{art}:ext_k{m:g}_t0x{t0s:g}",
         "K_surge": _k_surge(art, o), "reuse_eq": t0s == 1.0}
    r = lh.extreme_run((art, H_EXT, T, True, 1.0, d))
    return {**r, "m": m, "t0x": t0s, "pretension": "at-rest matched" if rest else "nominal"}


def _k_surge(art: str, o: dict) -> float:
    p = _pull(art, o)
    return (p(1e-3) - p(-1e-3)) / 2e-3


def slack_retension(r: dict) -> float:
    """Peak tension of lines that went slack (T_min < 0.15 T0) in the steady window; 0 if none."""
    v = [tx for tx, tn, t0 in zip(r["T_max_N"], r["T_min_N"], r["T0_N"], strict=True)
         if tn < 0.15 * t0]
    return max(v) if v else 0.0


def choose(runs: list[dict]) -> dict:
    out = {}
    for art in ARTS:
        scan = [r for r in runs if r["t0x"] == 1.0 and r.get("pretension", "nominal") == "nominal"]
        ms = sorted({r["m"] for r in scan if r["article"] == art})

        def ok(m: float, key: str, art: str = art, scan: list = scan) -> bool:
            rr = [r for r in scan if r["article"] == art and r["m"] == m]
            return len(rr) == len(T_EXT) and all(r[key] <= FOV for r in rr)

        crit4 = next((m for m in ms if ok(m, "surge_max_m")), None)
        mean = next((m for m in ms if ok(m, "mean_offset_m")), None)
        out[art] = {"m_criterion4": crit4, "m_mean_only": mean, "m_run": ms}
        print(f"{art:8s}: smallest k multiple run with max surge <= {FOV} m: {crit4}; "
              f"with mean <= {FOV} m: {mean} (runs {ms})", flush=True)
    return out


def settle(art: str) -> dict:
    res = json.loads(OUT.read_text())
    m = res["choice"][art]["m_criterion4"]
    tag = f"{art}:ext_k{m:g}"
    fd.moored_equilibrium(art, tag=tag, **opts(art, m))
    rec = {k: v for k, v in json.loads(fd.EQ_CACHE.read_text())[tag].items() if k != "xi"}
    rec["tag"] = tag
    print(f"{art:8s} extreme set (k x{m:g}) calm settle: tilt {rec['max_buoy_tilt_deg']:.3f} deg "
          f"(operational set 1.00 deg)", flush=True)
    return rec


def declare(art: str) -> dict:
    m = json.loads(OUT.read_text())["choice"][art]["m_criterion4"]
    d = asw.chosen(art)["row"]
    sweep = json.loads(asw.OUT.read_text())
    r = asw._shifts(asw.evaluate(art, asw._free(art), asw._kin(art), d["height"],
                                 d["t0_scale"] * rest_ratio(art, m), d["k_scale"] * m),
                    sweep[art][0]["free"])
    out = {k: r[k] for k in ("tilt_T_s", "heave_T_s", "tilt_shift_pct", "heave_shift_pct",
                             "periods_s", "K", "k_line", "T0_line_N")}
    print(f"{art:8s} extreme set (k x{m:g}): tilt {r['tilt_shift_pct']:+.2f} %, heave "
          f"{r['heave_shift_pct']:+.2f} %, surge {r['periods_s']['surge']:.1f} s (declared)",
          flush=True)
    return {**out, "m": m}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    step = sys.argv[1]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if step == "predict":
        res["predict"] = predict()
    elif step == "runs":
        ms = [float(x) for x in sys.argv[2:]]
        todo = [(a, m, T, 1.0) for a in ARTS for m in ms for T in T_EXT]
        with ProcessPoolExecutor(max_workers=len(todo)) as ex:
            new = list(ex.map(run_one, todo))
        res = json.loads(OUT.read_text()) if OUT.exists() else {}    # others may have written
        keep = [r for r in res.get("runs", []) if (r["article"], r["m"], r["T"], r["t0x"])
                not in {(n["article"], n["m"], n["T"], n["t0x"]) for n in new}]
        res["runs"] = keep + new
    elif step == "choose":
        res["choice"] = choose(res["runs"])
    elif step == "final":
        todo = [(a, res["choice"][a]["m_criterion4"], T, 1.0, True) for a in ARTS for T in T_EXT]
        with ProcessPoolExecutor(max_workers=len(todo)) as ex:
            new = list(ex.map(run_one, todo))
        res = json.loads(OUT.read_text())
        res["final"] = new
        for a in ARTS:
            rr = [r for r in new if r["article"] == a]
            ok = all(r["surge_max_m"] <= FOV for r in rr)
            res["choice"][a]["final_passes"] = ok
            print(f"{a:8s} final (k x{rr[0]['m']:g}, at-rest matched): max surge "
                  f"{max(r['surge_max_m'] for r in rr):.3f} m -> {'PASS' if ok else 'FAIL'}",
                  flush=True)
    elif step == "settle":
        with ProcessPoolExecutor(max_workers=len(ARTS)) as ex:
            res["settle"] = dict(zip(ARTS, ex.map(settle, ARTS), strict=True))
    elif step == "declare":
        res["declare"] = {a: declare(a) for a in ARTS}
    elif step == "t0":
        art = "cluster"
        m = res["choice"][art]["m_criterion4"]
        t0s = T0_TILT_CAP_DEG / res["settle"][art]["max_buoy_tilt_deg"]     # linear in T0
        tag = f"{art}:ext_k{m:g}_t0x{t0s:g}"
        fd.moored_equilibrium(art, tag=tag, **opts(art, m, t0s))
        with ProcessPoolExecutor(max_workers=len(T_EXT)) as ex:
            new = list(ex.map(run_one, [(art, m, T, t0s) for T in T_EXT]))
        res["t0_raise"] = {"article": art, "m": m, "t0x": t0s, "tag": tag,
                           "calm_tilt_deg": json.loads(fd.EQ_CACHE.read_text())[tag]
                           ["max_buoy_tilt_deg"], "runs": new}
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
