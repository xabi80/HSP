"""Constrained frequency-domain sweep: articulated vs whole-chain-rigid 16-buoy.

For each configuration and wave period, solve the saddle system
``[Z Gᵀ; G 0][ξ;λ] = [F;0]`` with ``Z(ω) = -ω²(M+A(ω)) + iωB(ω) + C`` (RADIATION
damping only), ``G`` the joint Jacobian at the reference (rigid or yaw_locked), and
``F`` the coupled wave-excitation RAO (heading 0°). Extract the platform's
heave / roll / pitch response (deck stillness) AND the joint reactions ``λ`` (the
connection loads — the moment a rigid weld carries that a pin cannot).

CAVEAT (from mode_shape_fd.py): radiation-only linear FD gives resonance PLACEMENT
and mode STRUCTURE robustly, but magnitudes near resonance are upper bounds — the
real response is drag-limited (Morison drag ≫ radiation). Magnitudes are confirmed
drag-limited in pvr_td.py. Read this as the map, not the amplitude.

Writes pvr_fd_summary.csv + pvr_fd_compare.png next to this script.
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import platform16_common as pc16  # noqa: E402
import pvr_common as pvr  # noqa: E402

warnings.simplefilter("ignore")

_PLAT = pc16.platform_body_index()
_HEAVE, _ROLL, _PITCH = 6 * _PLAT + 2, 6 * _PLAT + 3, 6 * _PLAT + 4


def _hydro_dof():  # type: ignore[no-untyped-def]
    import platform16_rao as prp16
    return prp16._hydro_dof(pvr.build_deck(False))


def _joint_row_spans(constraints):  # type: ignore[no-untyped-def]
    """(joint, force_slice, moment_slice, is_buoy_to_hub) per joint in λ order."""
    spans = []
    off = 0
    plat = _PLAT
    for j in constraints.joints:
        nr = j.n_rows
        spans.append((j, slice(off, off + 3), slice(off + 3, off + nr), j.body_b != plat))
        off += nr
    return spans


def fd_response(hdb, setup, hydro_dof, periods):  # type: ignore[no-untyped-def]
    w = np.asarray(hdb.omega)
    ndof = setup.lhs.M_plus_Ainf.shape[0]
    Ainf = np.zeros((ndof, ndof))
    Ainf[np.ix_(hydro_dof, hydro_dof)] = np.asarray(hdb.A_inf)
    M = setup.lhs.M_plus_Ainf - Ainf
    C = setup.lhs.C
    G = setup.constraints.jacobian(setup.xi0)
    m = G.shape[0]
    A = np.asarray(hdb.A)
    B = np.asarray(hdb.B)
    RAO = np.asarray(hdb.RAO)
    spans = _joint_row_spans(setup.constraints)
    rows = []
    for T in periods:
        wn = 2 * np.pi / T
        i = int(np.argmin(np.abs(w - wn)))
        Ag = np.zeros((ndof, ndof))
        Bg = np.zeros((ndof, ndof))
        Ag[np.ix_(hydro_dof, hydro_dof)] = A[:, :, i]
        Bg[np.ix_(hydro_dof, hydro_dof)] = B[:, :, i]
        Z = -wn**2 * (M + Ag) + 1j * wn * Bg + C
        F = np.zeros(ndof, complex)
        F[hydro_dof] = RAO[:, i, 0]
        K = np.zeros((ndof + m, ndof + m), complex)
        K[:ndof, :ndof] = Z
        K[:ndof, ndof:] = G.T
        K[ndof:, :ndof] = G
        sol = np.linalg.solve(K, np.concatenate([F, np.zeros(m)]))
        xi, lam = sol[:ndof], sol[ndof:]
        # joint loads: max |force| and |moment| over buoy->hub and hub->platform sets
        f_bh = m_bh = f_hp = m_hp = 0.0
        for _j, fsl, msl, is_bh in spans:
            fmag = float(np.linalg.norm(np.abs(lam[fsl])))
            mmag = float(np.linalg.norm(np.abs(lam[msl]))) if msl.stop > msl.start else 0.0
            if is_bh:
                f_bh, m_bh = max(f_bh, fmag), max(m_bh, mmag)
            else:
                f_hp, m_hp = max(f_hp, fmag), max(m_hp, mmag)
        rows.append(dict(
            T=float(T), omega=wn,
            heave=float(np.abs(xi[_HEAVE])),
            roll=float(np.abs(xi[_ROLL])),
            pitch=float(np.abs(xi[_PITCH])),
            jf_buoyhub=f_bh, jm_buoyhub=m_bh, jf_hubplat=f_hp, jm_hubplat=m_hp,
        ))
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    hdb = pvr.load_hdb("0215")
    w = np.asarray(hdb.omega)
    print(f"BEM omega in [{w.min():.3f}, {w.max():.3f}] rad/s -> T in "
          f"[{2 * np.pi / w.max():.2f}, {2 * np.pi / w.min():.2f}] s")
    periods = np.round(np.arange(1.8, 6.01, 0.1), 3)
    hydro_dof = _hydro_dof()

    results = {}
    for rigid, key in [(False, "artic"), (True, "rigid")]:
        setup = pvr.build_setup(rigid, hdb)
        results[key] = fd_response(hdb, setup, hydro_dof, periods)
        pk_h = max(results[key], key=lambda r: r["heave"])
        pk_p = max(results[key], key=lambda r: r["pitch"])
        print(f"[{key}] heave FD peak T={pk_h['T']:.2f}s (|{pk_h['heave']:.2f}|); "
              f"pitch FD peak T={pk_p['T']:.2f}s (|{pk_p['pitch']*1e3:.2f} mrad/m|)")

    # CSV
    with (_HERE / "pvr_fd_summary.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["config", "T_s", "heave_RAO", "roll_RAO_radpm", "pitch_RAO_radpm",
                     "Fjoint_buoyhub_N", "Mjoint_buoyhub_Nm", "Fjoint_hubplat_N",
                     "Mjoint_hubplat_Nm"])
        for key in ("artic", "rigid"):
            for r in results[key]:
                wr.writerow([key, r["T"], r["heave"], r["roll"], r["pitch"],
                             r["jf_buoyhub"], r["jm_buoyhub"], r["jf_hubplat"], r["jm_hubplat"]])

    # figure
    T = np.array([r["T"] for r in results["artic"]])
    CA, CR = "#0c8b96", "#c0392b"
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    for key, col, lab in [("artic", CA, "articulated (pin)"), ("rigid", CR, "rigid (weld)")]:
        rr = results[key]
        ax[0, 0].plot(T, [r["heave"] for r in rr], "-o", ms=3, color=col, label=lab)
        ax[0, 1].plot(T, [r["pitch"] * 1e3 for r in rr], "-o", ms=3, color=col, label=lab)
        ax[1, 0].plot(T, [r["jm_buoyhub"] for r in rr], "-o", ms=3, color=col, label=lab)
        ax[1, 1].plot(T, [r["jf_buoyhub"] for r in rr], "-o", ms=3, color=col, label=lab)
    ax[0, 0].set_title("Platform HEAVE RAO (deck vertical motion)")
    ax[0, 0].set_ylabel("|heave| (m per m wave-amp)")
    ax[0, 1].set_title("Platform PITCH RAO (deck TILT — the rocket/deck metric)")
    ax[0, 1].set_ylabel("|pitch| (mrad per m wave-amp)")
    ax[1, 0].set_title("Buoy→hub connection MOMENT (the weld's structural cost)")
    ax[1, 0].set_ylabel("max |joint moment| (N·m per m)")
    ax[1, 1].set_title("Buoy→hub connection FORCE")
    ax[1, 1].set_ylabel("max |joint force| (N per m)")
    for a in ax.flat:
        a.set_xlabel("wave period T (s)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("Articulated vs whole-chain-rigid 16-buoy — constrained FD (radiation-only: "
                 "resonance PLACEMENT + loads; magnitudes are upper bounds, see pvr_td.py)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(_HERE / "pvr_fd_compare.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {_HERE / 'pvr_fd_summary.csv'} and pvr_fd_compare.png")


if __name__ == "__main__":
    main()
