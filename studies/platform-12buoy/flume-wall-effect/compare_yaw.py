"""Yaw sensitivity of the sidewall effect for the 2.4 m square platform (PLAT_SQUARE_M=2.4).

The platform is tested flat-on: sides parallel to the walls, PLAT_ROT_DEG = 45 in the study's
convention (0 = corner-on). Here yaw psi is measured from flat-on, so PLAT_ROT_DEG = 45 - psi.
psi and -psi are mirror images across the flume centreline, and the square repeats every 90 deg,
so 0 <= psi <= 45 covers every heading. The plates reach the walls at psi = 38.5 deg, so
corner-on (psi = 45) does not fit.

Flat-on and corner-on are mirror-symmetric about the flume centreline. In head seas they excite
only surge / heave / pitch, and only the symmetric cross-flume modes (n = 2: 1.53 s). An
intermediate yaw breaks that symmetry: head seas then also excite sway / roll / yaw, and the
antisymmetric n = 1 cross-flume mode (cut-on 2.19 s) can couple. This script measures what that
does to the wall effect, with the study's own metrics, per yaw:
- the coupled BEM's rigid-platform excitation, walled vs open, all six modes
  (coupled_osu_{open,walled}<suf>.nc);
- the platform-heave radiation (added mass, damping) near the decay period (*_psd.nc);
- the articulated FloatSim free-decay period shift (articulated_decay<suf>.json);
- the heave RAO (articulated_rao<suf>.npz, or the identical deck-centre heave acceleration of
  accel_multidof<suf>.npz: it is omega^2 x the same response);
- the accelerations at the deck centre and four hubs (accel_multidof<suf>.npz).
<suf> = _rot<45-psi>_sq2p4. Writes yaw_compare_sq2p4.png and .json, and prints the table.
Run: python compare_yaw.py
"""

# ruff: noqa: E702, RUF001  -- compact figure-assembly lines, as compare_sizes.py; en dashes in
# the figure labels (display typography).
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
SQ = 2.4
S_ARM = SQ * np.sqrt(2.0) / 3.0                 # coupled_bem_osu.S_ARM for PLAT_SQUARE_M=2.4
PLATE_R, FLUME_W = 0.1437, 3.66                 # coupled_bem_osu.PLATE_R, W
M_PLATFORM = 16 * 21.52 + 4 * 2.0 + 6.0         # kg, as flume-mooring mooring_sizing.ARTICLES
YAWS = [0, 10, 20, 30]                          # deg from flat-on
BAND = (1.4, 4.0)                               # s, the test matrix's wave periods
COLORS = ["#0c8b96", "#6a8a2b", "#c9772b", "#b2432c"]


def suf(psi: float) -> str:
    return f"_rot{int(45 - psi)}_sq2p4"


def centres(psi: float) -> np.ndarray:
    """Buoy centres (x, y) in coupled_bem_osu.centers() order (buoy1..buoy16)."""
    rot = np.deg2rad(45.0 - psi)
    a = np.deg2rad([0.0, 90.0, 180.0, 270.0])
    return np.array([(S_ARM * np.cos(pc + rot) + 0.5 * S_ARM * np.cos(tb + rot),
                      S_ARM * np.sin(pc + rot) + 0.5 * S_ARM * np.sin(tb + rot))
                     for pc in a for tb in a])


def clearance(psi: float) -> float:
    return float(FLUME_W / 2 - (np.abs(centres(psi)[:, 1]).max() + PLATE_R))


def rigid_excitation(ds: xr.Dataset, xy: np.ndarray) -> dict[str, np.ndarray]:
    """Generalised excitation of the rigid platform modes about (0, 0, z_CoG): per-buoy forces and
    moments (each about its own CoG, all at the same z) transferred to the platform centre."""
    f = ds.excitation_force.values
    F = (f[0] + 1j * f[1])[:, 0, :]                         # (omega, dof)
    dofs = [str(d) for d in ds.influenced_dof.values]

    def col(i: int, name: str) -> np.ndarray:
        return F[:, dofs.index(f"buoy{i + 1}__{name}")]
    out = {k: 0.0 for k in ("surge", "sway", "heave", "roll", "pitch", "yaw")}
    for i, (x, y) in enumerate(xy):
        fx, fy, fz = col(i, "Surge"), col(i, "Sway"), col(i, "Heave")
        out["surge"] = out["surge"] + fx
        out["sway"] = out["sway"] + fy
        out["heave"] = out["heave"] + fz
        out["roll"] = out["roll"] + col(i, "Roll") + y * fz
        out["pitch"] = out["pitch"] + col(i, "Pitch") - x * fz
        out["yaw"] = out["yaw"] + col(i, "Yaw") + x * fy - y * fx
    return {k: np.abs(v) for k, v in out.items()}


def heave_radiation(psi: float, T: float) -> dict[str, float]:
    """Platform-heave added mass and radiation damping (summed heave block) at period T, open and
    walled, from the PSD-projected databases FloatSim uses."""
    out = {}
    for mode in ("open", "walled"):
        ds = xr.load_dataset(HERE / f"coupled_osu_{mode}{suf(psi)}_psd.nc")
        hv = [d for d in ds.radiating_dof.values if str(d).endswith("Heave")]
        om = ds.omega.values; ok = np.isfinite(om)
        A = ds.added_mass.sel(radiating_dof=hv, influenced_dof=hv).sum(
            ("radiating_dof", "influenced_dof")).values[ok]
        B = ds.radiation_damping.sel(radiating_dof=hv, influenced_dof=hv).sum(
            ("radiating_dof", "influenced_dof")).values[ok]
        w = 2 * np.pi / T
        out[mode] = {"A33": float(np.interp(w, om[ok], A)), "B33": float(np.interp(w, om[ok], B))}
    return out


def one(psi: float) -> dict:
    s = suf(psi)
    o = xr.load_dataset(HERE / f"coupled_osu_open{s}.nc")
    w = xr.load_dataset(HERE / f"coupled_osu_walled{s}.nc")
    om = o.omega.values
    band = np.isfinite(om) & (om > 0)
    per = np.full(om.shape, np.nan); per[band] = 2 * np.pi / om[band]
    band &= (per >= BAND[0]) & (per <= BAND[1])
    eo, ew = rigid_excitation(o, centres(psi)), rigid_excitation(w, centres(psi))
    idx = np.argsort(per[band])
    Tb = per[band][idx]
    r: dict = {"yaw_deg": psi, "plat_rot_deg": 45 - psi, "clearance_m": clearance(psi),
               "T_band_s": Tb.tolist()}
    for k in ("surge", "heave", "pitch"):                   # symmetric modes: % change
        pct = 100 * (ew[k][band] / eo[k][band] - 1)
        r[f"exc_{k}_pct"] = pct[idx].tolist()
        r[f"exc_{k}_max_abs_pct"] = float(np.abs(pct).max())
    # antisymmetric modes (zero at flat-on): band peak relative to the band peak of the symmetric
    # counterpart in open water, and the wall change where the open-water force is not negligible
    # (>= 5 % of that peak; a ratio of near-zero forces is meaningless)
    for k, ref in (("sway", "surge"), ("roll", "pitch"), ("yaw", "pitch")):
        peak = float(np.max(eo[ref][band]))
        r[f"exc_{k}_open_pct_of_{ref}"] = float(100 * np.max(eo[k][band]) / peak)
        r[f"exc_{k}_walled_pct_of_{ref}"] = float(100 * np.max(ew[k][band]) / peak)
        ok = eo[k][band] >= 0.05 * peak
        pct = np.where(ok, 100 * (ew[k][band] / np.where(ok, eo[k][band], 1.0) - 1), np.nan)
        r[f"exc_{k}_pct"] = pct[idx].tolist()
        r[f"exc_{k}_max_abs_pct"] = float(np.nanmax(np.abs(pct))) if ok.any() else None
    dec = HERE / f"articulated_decay{s}.json"
    if dec.exists():
        d = json.loads(dec.read_text())
        r["decay_T_open_s"], r["decay_T_walled_s"] = d["open"]["heave_T"], d["walled"]["heave_T"]
        r["decay_shift_pct"] = 100 * (d["walled"]["heave_T"] / d["open"]["heave_T"] - 1)
        r["decay_zeta_open"] = d["open"]["heave_zeta"]
        r["decay_zeta_walled"] = d["walled"]["heave_zeta"]
        rad = heave_radiation(psi, d["open"]["heave_T"])
        r["A33_shift_pct"] = 100 * (rad["walled"]["A33"] / rad["open"]["A33"] - 1)
        r["B33_shift_pct"] = 100 * (rad["walled"]["B33"] / rad["open"]["B33"] - 1)
        r["zeta_rad_open_pct"] = 100 * rad["open"]["B33"] / (
            2 * (2 * np.pi / d["open"]["heave_T"]) * (M_PLATFORM + rad["open"]["A33"]))
    rao = HERE / f"articulated_rao{s}.npz"
    if rao.exists():
        z = np.load(rao)
        r["rao_P_s"] = z["P"].tolist()
        r["rao_heave_pct"] = (100 * (z["Rw"] / z["Ro"] - 1)).tolist()
    acc = HERE / f"accel_multidof{s}.npz"
    if acc.exists():
        z = np.load(acc)
        pts = sorted({k[2:].rsplit("_", 1)[0] for k in z.files if k.startswith("o_")})
        if "rao_heave_pct" not in r:   # the deck-centre heave acceleration IS w^2 x the heave RAO
            r["rao_P_s"] = z["P"].tolist()
            r["rao_heave_pct"] = (100 * (z["w_platform_centre_heave"]
                                         / z["o_platform_centre_heave"] - 1)).tolist()
        for d in ("surge", "heave", "pitch"):
            r[f"acc_{d}_max_abs_pct"] = float(max(
                np.max(np.abs(100 * (z[f"w_{p}_{d}"] / z[f"o_{p}_{d}"] - 1))) for p in pts))
        ref = max(np.max(np.abs(z[f"o_{p}_heave"])) for p in pts)
        for d in ("sway", "roll", "yaw"):
            r[f"acc_{d}_open_pct_of_heave"] = float(100 * max(np.max(np.abs(z[f"o_{p}_{d}"]))
                                                              for p in pts) / ref)
            r[f"acc_{d}_walled_pct_of_heave"] = float(100 * max(np.max(np.abs(z[f"w_{p}_{d}"]))
                                                                for p in pts) / ref)
            r[f"acc_{d}_change_pct_of_heave"] = float(100 * max(
                np.max(np.abs(z[f"w_{p}_{d}"] - z[f"o_{p}_{d}"])) for p in pts) / ref)
    return r


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    rows = [one(p) for p in YAWS if (HERE / f"coupled_osu_walled{suf(p)}.nc").exists()]
    (HERE / "yaw_compare_sq2p4.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(" yaw | clear. | heave exc | surge exc | pitch exc | sway/surge o→w | yaw/pitch o→w | "
          "decay shift | RAO heave | acc s/h/p | acc sway,roll,yaw Δ (% heave)")
    for r in rows:
        acc = (" / ".join(f"{r[f'acc_{d}_max_abs_pct']:.1f}" for d in ("surge", "heave", "pitch"))
               if "acc_heave_max_abs_pct" in r else "—")
        accz = (", ".join(f"{r[f'acc_{d}_change_pct_of_heave']:.2f}"
                          for d in ("sway", "roll", "yaw"))
                if "acc_sway_change_pct_of_heave" in r else "—")
        rao = (" / ".join(f"{v:+.1f}" for v in r["rao_heave_pct"]) if "rao_heave_pct" in r else "—")
        dec = f"{r['decay_shift_pct']:+.2f} %" if "decay_shift_pct" in r else "—"
        print(f" {r['yaw_deg']:3.0f}° | {r['clearance_m']:.3f} m | "
              f"{r['exc_heave_max_abs_pct']:5.1f} % | {r['exc_surge_max_abs_pct']:5.1f} % | "
              f"{r['exc_pitch_max_abs_pct']:5.1f} % | "
              f"{r['exc_sway_open_pct_of_surge']:.1f}→{r['exc_sway_walled_pct_of_surge']:.1f} % | "
              f"{r['exc_yaw_open_pct_of_pitch']:.1f}→{r['exc_yaw_walled_pct_of_pitch']:.1f} % | "
              f"{dec} | {rao} % | {acc} % | {accz}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for r, col in zip(rows, COLORS, strict=False):
        a1.plot(r["T_band_s"], r["exc_heave_pct"], "o-", color=col, lw=2,
                label=f"yaw {r['yaw_deg']:.0f}° ({r['clearance_m']:.2f} m/side)")
    a1.axhspan(-5, 5, color="0.92", zorder=0); a1.axhline(0, color="0.4", lw=0.8)
    for tc, lab in ((2.19, "n=1 cut-on"), (1.53, "n=2")):
        a1.axvline(tc, color="0.5", ls=":", lw=1)
        a1.text(tc + 0.01, a1.get_ylim()[1] * 0.9, lab, fontsize=8, color="0.4")
    a1.set_xlabel("wave period T (s)"); a1.set_ylabel("sidewall effect on heave excitation (%)")
    a1.set_title("Heave excitation, walls in vs out (coupled BEM)", fontsize=10.5)
    a1.legend(fontsize=8.5); a1.grid(alpha=0.3)
    yaw = [r["yaw_deg"] for r in rows]
    for key, lab, mk in (("exc_heave_max_abs_pct", "heave excitation (max |%| 1.4–4 s)", "o-"),
                         ("exc_surge_max_abs_pct", "surge excitation", "s--"),
                         ("exc_pitch_max_abs_pct", "pitch excitation", "^--"),
                         ("exc_yaw_max_abs_pct", "yaw-moment excitation (where ≥ 5 % of pitch)",
                          "v-."),
                         ("acc_heave_max_abs_pct", "heave acceleration (FloatSim)", "D-")):
        ys = [np.nan if r.get(key) is None else r[key] for r in rows]
        a2.plot(yaw, ys, mk, lw=1.8, label=lab)
    a2.set_xlabel("yaw from flat-on (deg)"); a2.set_ylabel("largest sidewall effect (%)")
    a2.set_title("Largest wall effect vs yaw (plates touch the walls at 38.5°)", fontsize=10.5)
    a2.axvspan(38.5, 45, color="#f3d9d3", zorder=0); a2.set_xlim(-2, 45)
    a2.grid(alpha=0.3); a2.legend(fontsize=8.5, loc="upper left")
    ax = a2.twinx(); ax.plot(yaw, [r["clearance_m"] for r in rows], "k:", lw=1)
    ax.set_ylabel("clearance per side (m, dotted)", fontsize=9)
    fig.suptitle("2.4 m square platform in the OSU LWF — sidewall effect vs platform yaw",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(HERE / "yaw_compare_sq2p4.png", dpi=130, bbox_inches="tight")
    print("\nwrote yaw_compare_sq2p4.png and yaw_compare_sq2p4.json")


if __name__ == "__main__":
    main()
