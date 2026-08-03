# Fin-size sensitivity (single buoy) — RAO + max acceleration

Rigorous per-fin BEM study to inform the fin-size decision. Fin radius
{0.215 (baseline), 0.15, none}, corrected heights 0.04–0.12 m, plate Cd_n {5, 1}
(no-fin has no plate → spar-only). Each fin uses its **own** parametric Capytaine
BEM (correct added mass AND excitation); the plate drag radius is matched.

**Method.** A parametric axisymmetric spar+annular-fin mesh (`sparfin_fin_bem.py`),
validated to reproduce the GDF-mesh baseline to ~2% at R=0.215 (A₃₃ 21.7 vs 21.1,
C₃₃/T_n <0.5%). The fin is a ~4 mm plate with negligible buoyancy, so **draft,
C₃₃ (=221 N/m), and equilibrium are fin-independent** — only A₃₃, B₃₃(≈0),
F_exc, and drag change.

## Hydrodynamics per fin

| fin R | A₃₃ | T_n | plate drag area |
|-------|-----|-----|-----------------|
| 0.215 | 21.7 kg | 2.99 s | πr² = 0.145 m² |
| 0.15 | 6.0 kg (−72%) | 2.48 s | 0.071 m² (−51%) |
| none | 1.3 kg (spar only) | 2.31 s | 0 |

## Peak RAO / peak Nz-acceleration (over the H,T grid)

| fin R | Cd5 RAO | Cd5 accel | Cd1 RAO | Cd1 accel |
|-------|---------|-----------|---------|-----------|
| 0.215 | 1.62 | 0.25 m/s² | 3.45 | 0.53 m/s² |
| 0.15 | **1.97** | **0.43 m/s²** | **4.40** | **0.95 m/s²** |
| none | diverges @ res. | — | diverges @ res. | — |
| none (off-res) | ≤1.56, amplitude-independent | | | |

## Findings

1. **A smaller fin is worse on BOTH RAO and acceleration.** 0.215 → 0.15:
   peak RAO **+22% (Cd5) / +28% (Cd1)**; peak Nz-accel **+74% (Cd5) / +79%
   (Cd1)**. Two compounding causes: (i) less plate area → less drag → less
   damping → higher RAO; (ii) less added mass → shorter T_n → higher ω² → higher
   acceleration for the same displacement (a ∝ ω²·z). Shrinking the fin buys
   nothing and costs ~¾ more peak acceleration.

2. **No fin → the heave resonance is undamped and diverges.** With no plate the
   only heave damping is radiation, which is ~0 (B₃₃≈0, and a vertical spar's
   Morison drag is ~zero in heave). At/near its 2.31 s resonance the response
   runs away (does not settle; NaN). Off-resonance the response is finite but
   **amplitude-independent (linear)** — removing the plate also removes the
   quadratic-drag amplitude-gating. **The fin is the buoy's only meaningful
   heave damper.**

3. **Bigger is better, within the tested range.** Motion decreases monotonically
   with fin size (0.215 < 0.15 < none in RAO and accel). If material/cost allows,
   a fin ≥ 0.215 m is favourable; **do not shrink to 0.15 m** (~75% more peak
   acceleration), and a fin is **essential** (none = undamped).

## Caveat / next step

Single buoy in isolation. In the cluster/platform each buoy carries more
structural mass (longer T_n, more inertia), shifting the resonance and absolute
levels, but the fin's role — the dominant added mass **and** the only real heave
damper — is per-buoy identical. Extending this fin sweep to the 3-buoy cluster
and 12-buoy platform (each needs new coupled BEMs per fin) is the pending step.

## Files

- `sparfin_fin_bem.py` — parametric mesh + BEM per fin (`capytaine_fin{0215,015,none}.nc`).
- `sparfin_fin_fan.py` — the RAO+accel fan (matched plate drag; t_max=60 kernel).
- `fin_plots.py` → `fin_sensitivity.png`.
- `rao_summary_fin*.csv` + per-case CSVs + `manifest.json`.
