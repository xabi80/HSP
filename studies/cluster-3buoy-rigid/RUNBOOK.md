# 3-buoy rigid-cluster heave decay — RUNBOOK

Reproduction walkthrough. Composite single rigid body: three spar-fin
fullfix hulls on a 0.5 m circle (120° apart, one on +x) + a 12 kg dry
arm structure. Built on the M7.5-hardened stack and the spar-fin
study's eqdraft hull mesh. All commands from
`studies/cluster-3buoy-rigid/`. No `floatsim/` modifications; the
system is hand-assembled (`cluster_study_common.py`) because the
small-body composite BEM cannot pass the deck driver's non-overridable
Item-25 kernel gate — same reason as the single-buoy study.

## Pipeline

| step | script | output |
|---|---|---|
| 1 composite mesh | `build_cluster_mesh.py` | `mesh/cluster3_fullfix.gdf` (4464 panels; 0 inward, 288 open edges) |
| 2 waterline + mass | `cluster_balance.py` | dz2 = 0.17937 m; `results/mass_properties.json` |
| 3a reference BEM | `cluster_bem.py reference` | `reference_single_bem.nc` |
| 3b probe | `cluster_bem.py probe` | runtime estimate (~80 s) |
| 3b composite BEM | `cluster_bem.py composite` | `composite_bem.nc` (179 s) |
| 3 interaction | `cluster_interaction.py` | `results/interaction.json` (R, PF1/PF2) |
| 4+5 run | `cluster_decay_run.py` | `results/decay_*.csv`, `results/equilibrium.json` |
| 5 analysis | `cluster_analyze.py` | `results/summary.md`, `results/figures/*.png` |

## Geometry / mass (measured, Step 2)

- Cluster radius 0.5 m; buoy centres at 0°, 120°, 240° (one on +x).
- Additional sink dz2 = 0.17937 m below the single-buoy waterline to
  carry the 12 kg arm mass; composite displaces 98.007 kg at z=0.
- Composite mass 98.01 kg; CoG (0, 0, −0.9889 m); inertia about CoG
  Ixx=Iyy=113.29, Izz=22.84 kg·m².

## Headline result: hydrodynamic interaction

- **R(∞) = A33_composite / (3·A33_single) = 1.0113** (+1.1% heave
  added mass). R(ω_n) = 1.0106. Radiation damping slightly
  destructive: B33 ratio at ω_n = 2.894 (−3.5% vs 3×).
- Interaction lengthens the heave period by **+6.85 ms (+0.22%)**:
  T_n 3.10609 s (with) vs 3.09924 s (3× single, no interaction).
- Weak coupling is expected: hull spacing ~0.87 m is ~4× the plate
  radius.

## Validation

- T_n analytical 3.10609 s vs FloatSim 3.10533 s — rel-err **0.024%**
  (gate 1e-2).
- zeta_rad predicted 2.884e-4 vs FloatSim 2.886e-4 — rel-err
  **0.039%** (gate 5e-2).
- BEM+Morison effective zeta ~2.5% (amplitude-dependent, no gate;
  plate drag dominates radiation by ~2 orders).
- Cross-DOF coupling max 2.6e-10 (numerical-noise level; larger than
  the single-buoy 9.4e-13 from three-hull panel asymmetries but
  negligible).

## Commands

```bash
python build_cluster_mesh.py       # composite mesh + mesh_hygiene
python cluster_balance.py          # refine dz2 -> 98.01 kg + mass props
python cluster_bem.py reference    # single hull at cluster draft
python cluster_bem.py probe        # confirm composite runtime < 3 h
python cluster_bem.py composite    # composite BEM (~3 min)
python cluster_interaction.py      # R + PF1/PF2 smoke check
python cluster_decay_run.py        # equilibrium gate + both decays
python cluster_analyze.py          # analysis + 4 figures + summary.md
```

## Provenance

Builds on the spar-fin single-buoy study (`studies/spar-fin-decay/`,
merged to main): its eqdraft fullfix hull is the cluster's building
block, and the same hand-assembly / Item-25-override pattern applies.
Free-floating, no moorings; heave-only (pitch/roll not exercised).
