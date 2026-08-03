# Cross-model heave RAO + Nz-acceleration wave sweep (1 vs 3 vs 12 buoys)

Heave RAO and vertical (Nz) acceleration vs (wave height, wave period) for the
single spar-fin buoy, the 3-buoy articulated cluster, and the 12-buoy platform,
at the corrected model-scale wave heights, both plate drag settings.

## Scale (1:50 model)

The simulation runs at **model scale**. Froude scaling to full scale (λ=50):

| quantity | model | full-scale |
|----------|-------|------------|
| wave height H | 0.04–0.12 m | **2–6 m** |
| wave period T | 2.0–3.3 s | 14.1–23.3 s |
| heave natural period | 2.97 / 3.11 / 3.14 s | 21.0 / 22.0 / 22.2 s |
| **acceleration** | — | **Froude-invariant (same m/s²)** |

The previous sweep (0.05–1.0 m) reached 50 m full-scale waves; 0.04–0.12 m
(= 2–6 m full-scale) is the realistic operational band.

## Grid & drag

- Heights {0.04, 0.06, 0.08, 0.10, 0.12} m × periods {2.0, 2.5, 2.8, 3.0,
  3.141, 3.257, 3.3} s × plate `Cd_n` ∈ {5.0 operational, 1.0 light}.
- **Identical** per-buoy drag on all three models — distributed spar cylinder
  (D=0.1682 m, Cd=1.2, 10 seg) + heave plate (r=0.215 m, Cd_n swept, Cd_t=1.5)
  — so differences isolate coupling/draft, not the drag model.
- Adaptive settle; all 3×70 cases converged (0 unsettled).

## Findings

**Peak buoy heave RAO / peak buoy Nz-accel (m/s²):**

| model | T_n | Cd=5 RAO | Cd=5 accel | Cd=1 RAO | Cd=1 accel |
|-------|-----|----------|-----------|----------|-----------|
| 1 buoy | 2.97 s | 1.66 | 0.25 | 3.53 | 0.54 |
| 3-buoy cluster | 3.11 s | 1.75 | 0.24 | 3.86 | 0.54 |
| 12-buoy platform | 3.14 s | 1.92 | 0.26 | 4.20 | 0.57 |

1. **The resonance walks right with model size** — the buoy's heave natural
   period lengthens 2.97 → 3.11 → 3.14 s as it is embedded in larger
   structures, and the RAO peak follows. Decomposing
   `T_n = 2π√((M_eff + A₃₃)/C₃₃)` per buoy shows this is driven **mostly by the
   dry structural mass each buoy carries, not by added mass**:

   | model | M_eff (kg) | A₃₃ (kg) | M+A | C₃₃ (N/m) | T_n (s) |
   |-------|-----------|----------|-----|-----------|---------|
   | 1 buoy | 28.67 | 21.11 | 49.78 | 221.1 | 2.98 |
   | 3-buoy cluster | 32.67 | 21.34 | 54.01 | 221.1 | 3.11 |
   | 12-buoy platform | 33.50 | 21.74 | 55.24 | 221.1 | 3.14 |

   Of the single→platform change in the `M+A` numerator (+5.46 kg):
   **effective mass +4.83 kg (89%)** — each buoy carries a growing share of the
   dry arms/platform (which is also what sinks it deeper); **added mass +0.63 kg
   (11%, +3.0%)** — real but secondary, part deeper-submergence and part genuine
   inter-buoy hydrodynamic coupling (~2%, STEP A); **stiffness +0.00** —
   `C₃₃ = ρg·πR²_spar` is the spar cylinder's waterplane, which is
   **draft-independent**, so despite the drafts differing it contributes nothing
   to the spread. (`scratchpad tn_decomp.py`; A₃₃ at each model's own ω_n.)
2. **Peak buoy RAO *rises* with model size** (1.66→1.75→1.92 at Cd5;
   3.53→3.86→4.20 at Cd1) — the coupled added-mass/radiation environment
   amplifies the resonance relative to the isolated buoy.
3. **Peak buoy Nz-accel is ~flat across models** (~0.25 m/s² Cd5, ~0.54–0.57
   Cd1) — because acceleration ∝ ω²·displacement and the natural period
   lengthens with size, the higher RAO is offset by the lower ω². The absolute
   acceleration a buoy sees is nearly model-independent; the RAO is not.
4. **Amplitude-gated, sharply, at these heights** — RAO roughly doubles from
   Cd=5 to Cd=1, and at the smallest height (0.04 m) the Cd=1 platform buoy
   overshoots to RAO 4.2. All peaks sit at H=0.04 m (least quadratic damping).

## Files

- `cross_model_rao.png`, `cross_model_nz_accel.png` — 1-vs-3-vs-12 overlays vs
  period at H=0.08 m, Cd=5 and Cd=1.
- `<model>_rao.png`, `<model>_nz_accel.png` — per-model surfaces (centre & buoy
  × Cd=5/1); the single buoy has one location (centre = buoy).
- Data: `../spar-fin-decay/sparfin_rao_out/`,
  `../cluster-3buoy-rigid/cluster_rao_out/`,
  `../platform-12buoy/platform_rao_lowh_out/` (per-case CSVs + `rao_summary_Cd{5,1}.csv`
  + `manifest.json`, uniform schema).
- Regenerate: run the three `*_rao*.py` / `*_fan_lowh.py` scripts, then
  `python studies/rao_cross_model_plots.py`.

## Related

- `../spar-fin-decay/ADDED-MASS-VALIDATION.md` — sanity-check of the buoy heave
  added mass (A₃₃≈21 kg, 78% of the ideal disc, ~94% from the fin) and the key
  finding that **radiation damping B₃₃≈0**, so all heave damping is Morison drag
  (Cd) — the root of the Cd-sensitivity and amplitude-gating seen here.
