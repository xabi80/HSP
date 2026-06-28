# Spar+fin buoy decay study

**Scope.** Bounded warm-up exercise on real geometry before Tier 3
scoping. Single-buoy heave decay, two runs (BEM-only and BEM+Morison),
compared against analytical first-principles predictions.

**This is NOT a milestone.** This is a scratch-branch study under
`studies/spar-fin-decay/`. No FloatSim source modifications. The
results inform whether the multi-buoy Tier 3 path (per
[`docs/m7-foundation-closure.md`](../../docs/m7-foundation-closure.md)
§6) is the right next step.

## Layout

```
studies/spar-fin-decay/
  README.md                    -- this file
  RUNBOOK.md                   -- final explanation deliverable (Step G+)
  mesh/test2_spar_fin.gdf      -- WAMIT GDF, 1488 panels
  capytaine_run.py             -- Step A: Capytaine BEM
  capytaine_bem.nc             -- Step A output (xarray Dataset)
  deck_bem_only.yaml           -- Step C: FloatSim deck, no Morison
  deck_bem_morison.yaml        -- Step C: FloatSim deck, with heave-plate
  equilibrium_check.py         -- Step D: static eq solve + finding
  decay_run.py                 -- Step E: both decay integrations
  analytical_predictions.py    -- Step F: hand-derived comparison targets
  compare_plots.py             -- Step G: plots
  results/
    equilibrium.json           -- Step D finding disposition (scenario A/B/C/D)
    decay_bem_only.csv         -- Step E: heave + all 6 DOF traces
    decay_bem_morison.csv      -- Step E: heave + all 6 DOF traces
    summary.md                 -- Step F: written-up findings
    figures/
      heave_decay.png          -- both runs on same axes
      decay_envelope_log.png   -- log-scale envelope vs analytical
      cross_dof_silence.png    -- silent-DOF check
```

## Locked inputs (do not modify without Xabier consultation)

| quantity | value | notes |
|---|---|---|
| Water density `rho` | 1025.0 kg/m^3 | salt water |
| Gravity `g` | 9.81 m/s^2 | matches GDF header `1.0 9.810000` |
| Buoy mass `M` | 28.67 kg | from OrcaFlex inertia screenshot |
| CoG z-coordinate | -0.8317 m | body frame; below mesh origin = below waterline |
| Inertia at CoM (kg*m^2) | I_xx = I_yy = 24.0; I_zz = 0.114 | OrcaFlex tonne*m^2 x 1000 already applied |
| Initial heave displacement | 0.10 m | for decay run |
| Morison C_D (heave plate) | 5.0 | literature (Tao & Cai 2004) |
| Morison projected area | 0.1452 m^2 | pi * (0.215)^2 |
| Mesh panels | 1488 | sanity-check value |

## Locked analytical targets (pre-FloatSim-run predictions)

- **C33 hydrostatic restoring**: 223.43 N/m (rho*g*pi*r_spar^2;
  r_spar ~ 0.084 m).
- **Heave natural period** `T_n`: 3.2-3.8 s predicted band; pinned
  exactly once Capytaine A_inf_heave is known via
  `T_n = 2*pi*sqrt((M + A_inf_heave) / C33)`.
- **Radiation damping ratio** `zeta_rad` (BEM-only): 5-15 % of
  critical; pinned by `zeta = B(omega_n) / (2*sqrt((M+A_inf)*C33))`.
- **Cross-DOF silence**: `|xi_k(t)| < 1e-11 m` (or rad) for any
  k != heave (deliberately not excited).
- **KC number at IC = 0.10 m, T_n ~ 3.5 s**: ~1.5 (supports C_D = 5.0
  per Tao & Cai's Re-KC regime).

## Stop conditions

The study stops and reports to Xabier on:

- Capytaine run failure or sanity-check failure (Step A/B).
- Equilibrium solver non-convergence (Step D, scenario D).
- Morison heave-plate topology unsupported in
  `floatsim.hydro.morison` (Step C; do not modify floatsim/, use
  closest approximation and document the gap).
- Period rel-err > 5 % or damping rel-err > 20 % (Step F): complete
  the study so the gap is visible, then report.

## Reproducibility

After Step G or a STOP, [`RUNBOOK.md`](RUNBOOK.md) walks through
what was done at each step, what commands to run, what the
analytical predictions were, what the actual results were, and
what physical interpretation follows.

## Scope discipline

Per the study spec: **NO new code in `floatsim/`** — all work goes
under `studies/spar-fin-decay/`. After the study completes (or
STOPs), await Xabier's review before scoping anything further.
**Do NOT** proceed to a 3-buoy cluster study or any successor work
without explicit direction.
