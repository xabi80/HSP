"""Per-buoy collective-heave radiation damping across the 4 models (no-fin), at
each model's own heave resonance, vs the measured peak buoy RAO -- to show the
array wave-interaction 'park effect': damping minimum (RAO maximum) at 12 buoys."""
import csv
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "floatsim")
warnings.simplefilter("ignore")
from floatsim.hydro.readers.capytaine import read_capytaine

_R = Path("studies")
MODELS = [
    ("single", 1, _R / "spar-fin-decay/fin_study/capytaine_finnone.nc",
     _R / "spar-fin-decay/fin_study/rao_summary_finnone_cap.csv"),
    ("3-cluster", 3, _R / "cluster-3buoy-rigid/fin_study/capytaine_cluster_finnone.nc",
     _R / "cluster-3buoy-rigid/fin_study/rao_summary_cluster_finnone_cap.csv"),
    ("12-buoy", 12, _R / "platform-12buoy/fin_study/capytaine_platform_finnone.nc",
     _R / "platform-12buoy/fin_study/rao_summary_platform_finnone_cap.csv"),
    ("16-buoy", 16, _R / "platform-16buoy/fin_study/capytaine_platform16_finnone.nc",
     _R / "platform-16buoy/fin_study/rao_summary_platform16_finnone_cap.csv"),
]


def peak_rao_and_T(csvp):  # type: ignore[no-untyped-def]
    rows = list(csv.DictReader(open(csvp)))
    key = "rao_buoy" if "rao_buoy" in rows[0] else ("rao_center" if "rao_center" in rows[0]
                                                    else "rao_heave")
    pk = max(rows, key=lambda r: float(r[key]))
    return float(pk[key]), float(pk["period_s"])


labels, Ns, Bpb, Xpb, RAOp, RAOproxy = [], [], [], [], [], []
print(f"{'model':10} {'N':>3} {'T_res':>6} {'RAO':>6} {'B/buoy':>8} {'X/buoy':>8} "
      f"{'B_block':>8} {'|X_mod|':>9} {'proxy':>7}")
for lab, N, nc, csvp in MODELS:
    rao_pk, Tres = peak_rao_and_T(csvp)
    h = read_capytaine(nc)
    w = np.asarray(h.omega)
    wn = 2 * np.pi / Tres
    i = int(np.argmin(np.abs(w - wn)))
    hv = [6 * b + 2 for b in range(N)]
    B = np.asarray(h.B)[:, :, i]
    Bblk = float(B[np.ix_(hv, hv)].sum())
    X = np.asarray(h.RAO)[hv, i, 0]
    Xmod = float(abs(X.sum()))
    proxy = Xmod / (wn * Bblk)
    labels.append(lab); Ns.append(N); Bpb.append(Bblk / N); Xpb.append(abs(X).mean())
    RAOp.append(rao_pk); RAOproxy.append(proxy)
    print(f"{lab:10} {N:3d} {Tres:6.2f} {rao_pk:6.2f} {Bblk / N:8.3f} {abs(X).mean():8.2f} "
          f"{Bblk:8.2f} {Xmod:9.1f} {proxy:7.2f}")

x = np.arange(len(labels))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.plot(x, Bpb, "o-", color="#0c8b96", lw=2.2, ms=8)
a1.set_ylabel("per-buoy collective-heave\nradiation damping  B_block / N", color="#0c8b96")
a1b = a1.twinx()
a1b.plot(x, RAOp, "s--", color="#d1543a", lw=2, ms=7)
a1b.set_ylabel("measured peak buoy RAO", color="#d1543a")
a1.set_title("Per-buoy radiation damping is MONOTONIC, peak RAO is NOT\n-> radiation damping "
             "does NOT set the buoy RAO", fontsize=10)
a1.set_xticks(x); a1.set_xticklabels(labels)
a1.grid(True, alpha=0.3)

a2.plot(x, RAOproxy, "o-", color="#0c8b96", lw=2.2, ms=8, label="proxy |X_modal|/(w*B_block)")
a2.plot(x, RAOp, "s--", color="#d1543a", lw=2, ms=7, label="measured peak RAO")
a2.set_title("Resonance RAO: radiation-damping proxy vs measured\n(excitation per buoy is ~flat)",
             fontsize=10)
a2.set_xticks(x); a2.set_xticklabels(labels)
a2.grid(True, alpha=0.3); a2.legend(fontsize=9)
fig.suptitle("No-fin heave, single / 3-cluster / 12 / 16 buoys: per-buoy radiation damping rises "
             "monotonically and is ~20x below the bottom-cap drag,\nso it does NOT explain the "
             "non-monotonic peak buoy RAO -- that is a coupled mode-shape effect (needs eigen-analysis)",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = _R / "platform-16buoy/fin_study/array_radiation_damping_1_3_12_16.png"
fig.savefig(out, dpi=140)
print(f"\nwrote {out}")
