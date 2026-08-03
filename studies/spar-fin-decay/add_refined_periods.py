"""Incrementally add the refined periods {2.2, 2.3, 2.4} s to each fin-study
config and append to the existing rao_summary_*.csv (the original 7-point grid
straddled the no-fin 2.31 s resonance). Runs only the NEW periods; a fresh
`sparfin_fin_fan.py` (now on the 10-point grid) reproduces the full set."""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import sparfin_fin_fan as fff  # noqa: E402
import sparfin_rao as srp  # noqa: E402
import study_common as sc  # noqa: E402

from floatsim.hydro.readers.capytaine import read_capytaine  # noqa: E402
from floatsim.hydro.retardation import compute_retardation_kernel  # noqa: E402
from floatsim.solver.equilibrium import solve_static_equilibrium  # noqa: E402

_NEW = [2.2, 2.3, 2.4]
# (bem_tag, cd_label, drag_plate_radius, cd_n)
_CONFIGS = [
    ("0215", "Cd5", 0.215, 5.0),
    ("0215", "Cd1", 0.215, 1.0),
    ("015", "Cd5", 0.15, 5.0),
    ("015", "Cd1", 0.15, 1.0),
    ("none", "cap", 0.0841, 5.0),  # no-fin BEM + spar bottom-cap drag
]


def main() -> None:
    for bem_tag, cdl, drag_r, cd in _CONFIGS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hdb = read_capytaine(fff._STUDY / f"capytaine_fin{bem_tag}.nc")
            lhs = sc.build_lhs(hdb)
            kernel = compute_retardation_kernel(
                hdb, t_max=60.0, dt=sc.DT, asymptote_check_override=sc._OVERRIDE)
            eq = solve_static_equilibrium(lhs=lhs, state_force=None)
        drag = fff._drag(drag_r, cd)
        print(f"=== {bem_tag} {cdl} (plate r={drag_r}) ===", flush=True)
        new_rows = []
        for p in _NEW:
            for h in fff._HEIGHTS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    c = srp.run_case(lhs, kernel, hdb, drag, eq.xi_eq, height_m=h,
                                     period_s=p, cap_settle_s=fff._CAP)
                fname = f"case_fin{bem_tag}_{cdl}_H{h:g}_T{p:g}".replace(".", "p") + ".csv"
                srp._write_case_csv(fff._STUDY / fname, c)
                new_rows.append(dict(
                    height_m=h, period_s=p, omega=c["omega"], amp_m=c["amp_m"],
                    rao_center=c["rao_heave"], acc_center_amp=c["acc_heave_amp"],
                    acc_center_peak=c["acc_heave_peak"], rao_buoy=c["rao_heave"],
                    acc_buoy_amp=c["acc_heave_amp"], acc_buoy_peak=c["acc_heave_peak"],
                    settled=c["settled"], converged_early=c["converged_early"],
                    duration_used_s=c["duration_used_s"]))
                print(f"  T={p:.2f} H={h:.2f}: RAO={c['rao_heave']:.3f} "
                      f"acc={c['acc_heave_amp']:.3f} settled={c['settled']}", flush=True)
        summ = fff._STUDY / f"rao_summary_fin{bem_tag}_{cdl}.csv"
        existing = list(csv.DictReader(summ.open()))
        allrows = existing + [{k: str(v) for k, v in r.items()} for r in new_rows]
        allrows.sort(key=lambda r: (float(r["period_s"]), float(r["height_m"])))
        with summ.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(existing[0].keys()))
            w.writeheader()
            w.writerows(allrows)
        print(f"  -> {summ.name}: {len(existing)} + {len(new_rows)} = {len(allrows)} rows")
    print("\nDone.")


if __name__ == "__main__":
    main()
