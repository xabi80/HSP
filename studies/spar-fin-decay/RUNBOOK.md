# Spar-fin decay study — RUNBOOK

Reproduction walkthrough for the single-buoy heave free-decay
study, as completed at the M7.5 resumption (2026-07-04,
free-floating configuration). All commands run from
`studies/spar-fin-decay/`. No `floatsim/` modifications; the
study hand-assembles the Cummins system because the small-body
BEM cannot pass the deck driver's (non-overridable) Item-25
kernel gate.

## Pipeline

| step | script | output |
|---|---|---|
| mesh prep (workflow alpha) | `prepare_mesh.py` | `mesh/test2_spar_fin_fullfix.gdf`, `..._eqdraft.gdf` |
| waterline balance | `waterline_balance.py` | dz = 0.1846 m (stdout) |
| BEM (eqdraft) | `capytaine_run.py` | `capytaine_bem.nc` |
| pre-flight re-verify | `verify_preflight.py` | PF1/PF2 pass (stdout) |
| equilibrium + decay (D, E) | `decay_run.py` | `results/decay_*.csv`, `results/equilibrium.json` |
| analysis + plots (F, G) | `analyze_and_plot.py` | `results/summary.md`, `results/figures/*.png` |

## Commands

```bash
python prepare_mesh.py         # ORIGINAL -> fullfix (216 flips) -> eqdraft (dz=0.1846 m)
python waterline_balance.py    # confirms 24.47 kg @ z=0 -> sinks 0.185 m
python capytaine_run.py        # regenerate BEM on eqdraft mesh (~2 min)
python verify_preflight.py     # PF1 reader + PF2 kernel-override smoke check
python decay_run.py            # Step D equilibrium gate + Step E both decays
python analyze_and_plot.py     # Step F analysis + Step G figures + summary.md
```

## Locked inputs

See `README.md`. Resumption updates: eqdraft mesh (buoy sunk
dz = 0.1846 m to the true free-floating waterline); CoG z =
-1.0163 m; A_inf(heave) = 21.12 kg (measured; the earlier
22.74 kg expectation was tied to the withdrawn 0.54 m draft).

## Key results (eqdraft, IC heave = 0.10 m)

- **T_n = 2.982 s** — analytical vs FloatSim agree to 0.008%.
- **zeta_rad = 1.13e-4** (0.011% critical) — agree to 1.28%;
  the spar+fin is a very-low-radiation heave geometry.
- **BEM + Morison plate**: effective zeta ~ 2.5% critical
  (amplitude-dependent; heave-plate drag dominates radiation by
  ~2 orders and kills the decay in a few periods).
- **Cross-DOF** coupling at numerical-noise level (max 9.4e-13).

## Interpretation

The buoy is a lightly-radiation-damped heave oscillator; without
the heave plate it would ring for tens of periods. The plate's
quadratic drag (Cd=5, A=0.1452 m²) is the dominant damping
mechanism, which is the physical point of the spar+fin design.
The BEM-only vs BEM+Morison contrast quantifies that. Absolute
damping numbers for the plate await tank data; the study's role
was to validate the FloatSim heave dynamics against
first-principles (period + radiation damping), which it does.

## Provenance / corrections

- Buoyancy interpretation corrected: tier-2 `check_hydrostatic_volume`
  is fully-submerged reserve buoyancy (~40.9 kg), not waterline
  displacement (24.47 kg). See STEP-A-FINDING.md addendum §3 and
  main commit `907a2b2`.
- Mesh normals fixed via `floatsim.hydro.mesh_hygiene` (216
  panels), superseding the deprecated `fix_mesh_normals.py`
  (192-panel z-band heuristic).
- Capytaine A/B symmetrization now handled by the FloatSim reader
  (M7.5 PR2); the study's local workaround was retired.
