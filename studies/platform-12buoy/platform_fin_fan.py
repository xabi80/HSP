"""Fin-sensitivity RAO + Nz-accel fan for the 12-buoy platform. Uses the parametric
platform BEMs (platform_fin_bem.py) per fin {0.215, 0.15, none}; the plate drag
radius is matched to the fin (no-fin: spar bottom-cap). Hydrostatic from
reference_single (C33 is draft-independent, so the single-buoy-tiled C is correct
for all fins). Refined grid. Reuses platform_rao_pilot.run_case (platform ref
centre + buoy7). Uniform cross-model schema.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "cluster-3buoy-rigid"))

import platform_rao_pilot as prp  # noqa: E402

from floatsim.driver import build_system  # noqa: E402
from floatsim.hydro.database import HydroDatabase  # noqa: E402
from floatsim.hydro.readers.capytaine import read_capytaine  # noqa: E402

_STUDY = _HERE / "fin_study"
_REF = _HERE.parent / "cluster-3buoy-rigid" / "reference_single_bem.nc"
_HEIGHTS = [0.04, 0.06, 0.08, 0.10, 0.12]
_PERIODS = [2.0, 2.3, 2.4, 2.5, 2.6, 2.65, 2.8, 3.0, 3.141, 3.257, 3.3]
_CAP = 350.0
_CENTER = 6 * prp._buoy_body_index_platform() + 2  # platform ref heave DOF = 98
_BUOY = 6 * prp._buoy_body_index(6) + 2  # buoy7 (cluster C) heave DOF = 50
_R_SPAR = 0.0841
_CONTAM: dict[str, tuple[float, ...]] = {"0215": (2.977,), "015": (2.977,), "none": (2.977,)}
_CONFIGS = [
    ("0215", 0.215, [(5.0, 0.215), (1.0, 0.215)]),
    ("015", 0.15, [(5.0, 0.15), (1.0, 0.15)]),
    ("none", None, [(5.0, _R_SPAR)]),
]


def _hdb(tag: str) -> HydroDatabase:
    h = read_capytaine(_STUDY / f"capytaine_platform_fin{tag}.nc")
    drop = _CONTAM.get(tag, ())
    if not drop:
        return h
    w = np.asarray(h.omega)
    di = {int(np.argmin(np.abs(w - c))) for c in drop}
    keep = np.array([k for k in range(w.size) if k not in di])
    return HydroDatabase(
        omega=h.omega[keep],
        heading_deg=h.heading_deg,
        A=h.A[:, :, keep],
        B=h.B[:, :, keep],
        A_inf=h.A_inf,
        C=h.C,
        RAO=h.RAO[:, keep, :],
        reference_point=h.reference_point,
        C_source=h.C_source,
        metadata=dict(h.metadata),
        body_labels=h.body_labels,
    )


def _acc_amp(acc: np.ndarray, dof: int) -> tuple[float, float]:
    v = acc[:, dof]
    return 0.5 * (v.max() - v.min()), float(np.max(np.abs(v)))


def _build(plate_r: float, cd_n: float, hdb: HydroDatabase):  # type: ignore[no-untyped-def]
    prp._PLATE_R = plate_r
    prp._PLATE_CD_N = cd_n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_system(
            prp._deck_with_drag(),
            bem_databases={},
            dt=0.01,
            t_max_kernel=60.0,
            solve_equilibrium=False,
            shared_hydro_database=hdb,
            hydrostatic_database=read_capytaine(_REF),
            asymptote_check_override=prp._ASYMPTOTE_OVR,
            kernel_decay_floor_override="platform parametric spar-fin small-body",
        )


_FIELDS = [
    "height_m",
    "period_s",
    "omega",
    "amp_m",
    "rao_center",
    "acc_center_amp",
    "acc_center_peak",
    "rao_buoy",
    "acc_buoy_amp",
    "acc_buoy_peak",
    "settled",
    "converged_early",
    "duration_used_s",
]
_MAX_NEW = int(os.environ.get("FIN_MAX_NEW", "12"))  # new cases per process (memory bound)


def _grid() -> list[tuple[float, float]]:
    return [(h, p) for p in _PERIODS for h in _HEIGHTS]


def _row_path(tag: str, label: str, h: float, p: float) -> Path:
    return _STUDY / (f"row_platform_fin{tag}_{label}_H{h:g}_T{p:g}".replace(".", "p") + ".json")


def _missing(tag: str, label: str) -> list[tuple[float, float]]:
    return [(h, p) for (h, p) in _grid() if not _row_path(tag, label, h, p).exists()]


def _write_summary(tag: str, label: str) -> None:
    """Assemble the per-config summary CSV from the per-case row JSONs (all present)."""
    rows = [json.load(_row_path(tag, label, h, p).open()) for (h, p) in _grid()]
    with (_STUDY / f"rao_summary_platform_fin{tag}_{label}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)
    nun = sum(1 for r in rows if not r["settled"])
    print(
        f"  [{tag} {label}] summary written ({len(rows)} cases; {nun} did not settle)", flush=True
    )


def _run_one(tag, label, setup, hdb, hydro_dof, h, p):  # type: ignore[no-untyped-def]
    """Run one case; persist per-case CSV + a per-case row JSON (the resume unit)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = prp.run_case(
            setup,
            hdb,
            hydro_dof,
            height_m=h,
            period_s=p,
            ramp_s=20.0,
            cap_settle_s=_CAP,
            window_periods=6.0,
            dt=0.01,
        )
    ca, cp = _acc_amp(c["acc"], _CENTER)
    ba, bp = _acc_amp(c["acc"], _BUOY)
    fname = f"case_platform_fin{tag}_{label}_H{h:g}_T{p:g}".replace(".", "p") + ".csv"
    with (_STUDY / fname).open("w", newline="") as fh:
        wc = csv.writer(fh)
        wc.writerow(
            [
                "t_s",
                "center_heave_m",
                "center_heave_acc_mps2",
                "buoy_heave_m",
                "buoy_heave_acc_mps2",
            ]
        )
        wc.writerows(
            np.column_stack(
                [
                    c["t"],
                    c["xi"][:, _CENTER],
                    c["acc"][:, _CENTER],
                    c["xi"][:, _BUOY],
                    c["acc"][:, _BUOY],
                ]
            ).tolist()
        )
    row = dict(
        height_m=h,
        period_s=p,
        omega=c["omega"],
        amp_m=c["amp_m"],
        rao_center=c["rao"]["platform_heave"],
        acc_center_amp=ca,
        acc_center_peak=cp,
        rao_buoy=c["rao"]["buoy7_heave"],
        acc_buoy_amp=ba,
        acc_buoy_peak=bp,
        settled=c["settled"],
        converged_early=c["converged_early"],
        duration_used_s=c["duration_s"],
    )
    with _row_path(tag, label, h, p).open("w") as fh:
        json.dump(row, fh)
    print(
        f"  platform fin={tag} {label} H={h:.2f} T={p:.3f}: "
        f"RAO_ctr={row['rao_center']:.3f} RAO_b7={row['rao_buoy']:.3f} "
        f"settled={row['settled']}",
        flush=True,
    )


def _selected(only_tag: str | None, only_cd: str | None) -> list[tuple[str, str, float, float]]:
    """Expand _CONFIGS into (tag, label, plate_r, cd) for the argv selection."""
    out = []
    for tag, fin_r, cds in _CONFIGS:
        if only_tag and tag != only_tag:
            continue
        for cd, plate_r in cds:
            label = "cap" if fin_r is None else f"Cd{cd:g}"
            target = None if only_cd is None else ("cap" if only_cd == "cap" else f"Cd{only_cd}")
            if target is not None and label != target:
                continue
            out.append((tag, label, plate_r, cd))
    return out


def main() -> None:
    """One CHUNK of the fin fan per process. Runs up to FIN_MAX_NEW (env, default
    12) NEW cases for the FIRST incomplete config, persisting each case as a row
    JSON (the resume unit); a config's summary CSV is assembled once all 55 rows
    exist. Prints ``REMAINING: n`` so ``run_platform_fin_fan.py`` can loop fresh
    processes until 0. Chunking (not per-config) is REQUIRED: the constrained
    integrator's per-step KKT allocations retain ~2 GB/case via native-heap
    fragmentation (gc-immune; the convolution buffer was exonerated in isolation),
    so even one 55-case config OOMs a single process -- see tracker
    CONSTRAINED-INTEGRATOR-SWEEP-MEMORY. Optional argv: ``<tag> [<cd>]`` to scope
    to one fin / drag config.

    """
    only_tag = sys.argv[1] if len(sys.argv) > 1 else None
    only_cd = sys.argv[2] if len(sys.argv) > 2 else None
    selected = _selected(only_tag, only_cd)

    ran = False
    for tag, label, plate_r, cd in selected:
        # A finished config is one with a summary CSV (0215_Cd5 completed under the
        # pre-chunk fan and has no per-case row JSONs -- do NOT re-run it).
        if (_STUDY / f"rao_summary_platform_fin{tag}_{label}.csv").exists():
            continue
        miss = _missing(tag, label)
        if not miss:
            _write_summary(tag, label)
            continue
        if ran:
            continue  # one build per process; leave the rest for later invocations
        hdb = _hdb(tag)
        hydro_dof = prp._hydro_dof(prp._deck_with_drag())
        setup = _build(plate_r, cd, hdb)
        print(
            f"\n=== platform fin {tag} {label} (plate r={plate_r}); "
            f"{len(miss)} missing, running up to {_MAX_NEW} ===",
            flush=True,
        )
        for h, p in miss[:_MAX_NEW]:
            _run_one(tag, label, setup, hdb, hydro_dof, h, p)
        ran = True
        if not _missing(tag, label):
            _write_summary(tag, label)

    with (_STUDY / "platform_fin_manifest.json").open("w") as fh:
        json.dump(
            {
                "grid_heights_m": _HEIGHTS,
                "grid_periods_s": _PERIODS,
                "note": "parametric platform BEM per fin; reference_single hydrostatic; "
                "chunked per-case resume (integrator KKT path retains ~2 GB/case)",
            },
            fh,
            indent=2,
        )
    total_remaining = sum(
        len(_missing(t, lbl))
        for t, lbl, _, _ in selected
        if not (_STUDY / f"rao_summary_platform_fin{t}_{lbl}.csv").exists()
    )
    (_STUDY / "_fin_remaining.txt").write_text(str(total_remaining))
    print(f"\nREMAINING: {total_remaining}", flush=True)


if __name__ == "__main__":
    main()
