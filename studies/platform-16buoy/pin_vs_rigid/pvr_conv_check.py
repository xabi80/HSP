"""Does the RadiationConvolution rectangular-vs-trapezoid defect change the
pin-vs-rigid TD sweep? Re-runs the drag-limited platform cases with the CURRENT
(rectangular) evaluate and with a TRAPEZOID-fixed evaluate, at an operational
period (2.5 s, off-resonance) and the pitch resonance (3.2 s), for both configs.
Extracts platform pitch RAO so we can see if the deck-tilt verdict moves.

Resumable per-case JSON. Run from the local mirror. See PIN-VS-RIGID.md / the
convolution-defect note.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import platform16_common as pc16  # noqa: E402
import platform16_rao as prp16  # noqa: E402
import platform_rao_pilot as prp  # noqa: E402
import pvr_common as pvr  # noqa: E402

from floatsim.hydro.retardation import RadiationConvolution  # noqa: E402

warnings.simplefilter("ignore")

_PLAT = pc16.platform_body_index()
_HEAVE, _PITCH = 6 * _PLAT + 2, 6 * _PLAT + 4
_ROWDIR = _HERE / "pvr_conv_rows"
_PERIODS = [2.5, 3.2]

_orig = RadiationConvolution.evaluate


def _trap(self):
    mu = self._dt * np.einsum("ijk,kj->i", self._K, self._buffer)
    mu -= 0.5 * self._dt * (self._K[:, :, 0] @ self._buffer[0]
                            + self._K[:, :, -1] @ self._buffer[-1])
    return mu


def _run(setup, hdb, hydro_dof, T):  # type: ignore[no-untyped-def]
    c = prp16.run_case(setup, hdb, hydro_dof, height_m=0.10, period_s=T,
                       ramp_s=20.0, cap_settle_s=150.0, window_periods=6.0, dt=0.01)
    amp, omega, t, xi = c["amp_m"], c["omega"], c["t"], c["xi"]
    return dict(
        heave_RAO=float(prp._fit_amplitude(t, xi[:, _HEAVE], omega) / amp),
        pitch_RAO=float(prp._fit_amplitude(t, xi[:, _PITCH], omega) / amp),
        settled=bool(c["settled"]),
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    _ROWDIR.mkdir(exist_ok=True)
    hdb = pvr.load_hdb("0215")
    hydro_dof = prp16._hydro_dof(pvr.build_deck(False))

    for rigid, cfg in [(False, "artic"), (True, "rigid")]:
        setup = None
        for rule, patch in [("rect", None), ("trap", _trap)]:
            for T in _PERIODS:
                rp = _ROWDIR / f"conv_{cfg}_{rule}_T{T:g}".replace(".", "p")
                rp = rp.with_suffix(".json")
                if rp.exists():
                    continue
                if setup is None:
                    setup = pvr.build_setup(rigid, hdb)
                RadiationConvolution.evaluate = patch if patch else _orig
                try:
                    row = _run(setup, hdb, hydro_dof, T)
                finally:
                    RadiationConvolution.evaluate = _orig
                row.update(config=cfg, rule=rule, T=T)
                rp.write_text(json.dumps(row))
                print(f"[{cfg} {rule}] T={T}: pitch={row['pitch_RAO']*1e3:.1f} mrad/m  "
                      f"heave={row['heave_RAO']:.3f}  settled={row['settled']}", flush=True)

    # summary: pitch RAO rect vs trap, and the rigid/artic ratio each way
    def get(cfg, rule, T):  # type: ignore[no-untyped-def]
        p = (_ROWDIR / f"conv_{cfg}_{rule}_T{T:g}".replace(".", "p")).with_suffix(".json")
        return json.loads(p.read_text()) if p.exists() else None

    print("\n=== platform pitch RAO (mrad/m): rect (buggy) vs trap (fixed) ===")
    for T in _PERIODS:
        line = f"T={T}s: "
        for cfg in ("artic", "rigid"):
            r, t = get(cfg, "rect", T), get(cfg, "trap", T)
            if r and t:
                pr, pt = r["pitch_RAO"] * 1e3, t["pitch_RAO"] * 1e3
                line += f"{cfg} {pr:.1f}->{pt:.1f} ({pt/pr:.2f}x)  "
        ra_r, ri_r = get("artic", "rect", T), get("rigid", "rect", T)
        ra_t, ri_t = get("artic", "trap", T), get("rigid", "trap", T)
        if all([ra_r, ri_r, ra_t, ri_t]):
            rat_r = ri_r["pitch_RAO"] / ra_r["pitch_RAO"]
            rat_t = ri_t["pitch_RAO"] / ra_t["pitch_RAO"]
            line += f"| rigid/artic {rat_r:.2f}x(rect) -> {rat_t:.2f}x(trap)"
        print(line)


if __name__ == "__main__":
    main()
