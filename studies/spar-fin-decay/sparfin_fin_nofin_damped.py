"""Re-run the no-fin case with a small, physical heave damper so it converges:
the spar's own flat-bottom form drag, modelled as a heave plate at the spar
radius (0.0841 m, ~15% of the 0.215 fin area). The pure no-fin idealization has
literally zero heave damping (B33~0, vertical-spar Morison ~0 in heave) and
diverges at resonance; the bare spar's bottom cap is the realistic minimum.
Uses the no-fin BEM (A33=1.3 kg) + this bottom-cap drag. Writes
rao_summary_finnone_cap.csv (reusing sparfin_fin_fan._fan)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import sparfin_fin_fan as fff  # noqa: E402
import study_common as sc  # noqa: E402

from floatsim.hydro.readers.capytaine import read_capytaine  # noqa: E402
from floatsim.hydro.retardation import compute_retardation_kernel  # noqa: E402
from floatsim.solver.equilibrium import solve_static_equilibrium  # noqa: E402

_R_CAP = 0.0841  # spar-bottom cap radius = spar radius (the realistic minimum drag)


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hdb = read_capytaine(fff._STUDY / "capytaine_finnone.nc")
        lhs = sc.build_lhs(hdb)
        kernel = compute_retardation_kernel(
            hdb, t_max=60.0, dt=sc.DT, asymptote_check_override=sc._OVERRIDE)
        eq = solve_static_equilibrium(lhs=lhs, state_force=None)
    tn = 2 * np.pi * np.sqrt(lhs.M_plus_Ainf[2, 2] / lhs.C[2, 2])
    print(f"no-fin + bottom-cap drag (r={_R_CAP} m, Cd_n=5): T_n={tn:.3f} s "
          f"(A33={lhs.M_plus_Ainf[2, 2] - 28.67:.2f})", flush=True)
    # _fan(tag, fin_r_for_drag, cd_n, lhs, kernel, hdb, eq, label): the BEM is no-fin,
    # the drag plate radius is the bottom cap.
    nun = fff._fan("none", _R_CAP, 5.0, lhs, kernel, hdb, eq, "cap")
    print(f"unsettled: {nun}/35")


if __name__ == "__main__":
    main()
