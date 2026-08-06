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

Period grid `{2.0, 2.2, 2.3, 2.4, 2.5, 2.8, 3.0, 3.141, 3.257, 3.3}` s —
refined near 2.3 s so every fin's resonance is sampled at its natural period
(the no-fin's 2.31 s fell between the original grid's 2.0 and 2.5 points and was
under-reported).

| fin R | Cd5 RAO | Cd5 accel | Cd1 RAO | Cd1 accel | peak @ T |
|-------|---------|-----------|---------|-----------|----------|
| 0.215 | 1.62 | 0.25 m/s² | 3.45 | 0.53 m/s² | 3.0–3.14 s |
| 0.15 | **1.97** | **0.44 m/s²** | **4.40** | **0.95 m/s²** | 2.4–2.5 s |
| none (+bottom-cap drag) | **3.30** | **0.84 m/s²** | — | — | 2.3 s |

The pure no-fin idealization has literally zero heave damping (B₃₃≈0; a vertical
spar's Morison drag is ~0 in heave) and **diverges** at its 2.31 s resonance. To
get a comparable number, the no-fin case above includes the spar's own
flat-bottom form drag — a heave plate at the spar radius (0.084 m, ~15% of the
0.215 fin area), the realistic minimum for a finless spar. Even with that, its
peak RAO (3.30, at its 2.30 s natural period) and acceleration (0.84 m/s²)
exceed both fins by a wide margin.

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

3. **Bigger is better — monotonic.** Peak acceleration (Cd5) climbs
   0.25 → 0.44 → 0.84 m/s² as the fin goes 0.215 → 0.15 → none: shrinking to
   0.15 costs **+74%**, removing it costs another **+90%** (×3.4 vs the 0.215
   fin). Peak RAO likewise 1.62 → 1.97 → 3.30. If material/cost allows a fin
   ≥ 0.215 m is favourable; **do not shrink to 0.15 m**, and a fin is
   **essential** (none = undamped/diverges without even its bottom-cap drag).

## 3-buoy cluster — confirms the single-buoy conclusion

The fin sweep was repeated on the 3-buoy articulated cluster with rigorous
**coupled** BEMs per fin (parametric mesh replicated at the cluster positions,
validated: heave-block A₃₃ 65.8 kg vs the existing cluster's 64.1, 2.8%). Cluster
heave modes 3.12 / 2.63 / 2.47 s (0.215/0.15/none) — ~0.15 s longer than the
single buoy (arm mass). Peak **buoy** heave (Cd5), single **/** cluster:

| fin | buoy RAO (single/cluster) | buoy Nz-accel (single/cluster) |
|-----|---------------------------|--------------------------------|
| 0.215 | 1.62 / 1.73 | 0.25 / 0.24 m/s² |
| 0.15 | 1.97 / 2.23 | 0.44 / 0.44 m/s² |
| none (+cap) | 3.30 / 3.81 | 0.84 / 0.86 m/s² |

The buoy tracks tightly across the two models — accel within a few percent, RAO
modestly higher in the cluster (see
`../../cluster-3buoy-rigid/fin_study/fin_single_vs_cluster.png` and
`cluster_fin_sensitivity.png`; the full 3-model picture is in the platform
section below). The monotonic "smaller fin = more motion" conclusion is
model-independent, as expected — the fin's role (dominant added mass **and** the
only real heave damper) is per-buoy identical.

## 12-buoy platform — confirms it at full scale

The sweep was repeated on the full 12-buoy articulated platform (102 global
DOF, 16 yaw-locked joints) with a rigorous **72-DOF coupled** BEM per fin
(parametric mesh replicated at the 12 platform positions; heave-block A₃₃
266 / 74 / 18 kg for 0.215 / 0.15 / none, i.e. ~22 / 6.2 / 1.5 kg per buoy,
matching the per-buoy single/cluster values). Platform heave natural periods
**3.15 / 2.66 / 2.50 s** (0.215 / 0.15 / none) — a touch longer than the cluster
(more coupled added mass).

All cross-model comparisons below are for **the buoy** — the payload body, the
same physical object in every model. (The per-config "center" channel is the
buoy for the single model but the hub / platform structural reference node for
the cluster / platform; those central nodes move *less* than the buoys and are
not comparable across models — comparing them is what makes an apples-to-oranges
"cluster > single" artifact.) Peak buoy heave (Cd_n=5), all three models:

| fin | buoy RAO (single / cluster / platform) | buoy Nz-accel (single / cluster / platform) |
|-----|----------------------------------------|----------------------------------------------|
| 0.215 | 1.62 / 1.73 / **1.84** | 0.25 / 0.24 / **0.25** m/s² |
| 0.15 | 1.97 / 2.23 / **2.37** | 0.44 / 0.44 / **0.47** m/s² |
| none (+cap) | 3.30 / 3.81 / **4.27** | 0.84 / 0.86 / **0.92** m/s² |

Two clean, consistent trends. **(i)** Buoy heave *RAO* rises modestly with model
size (single → platform, ~+14 % at 0.215) — a buoy embedded in a larger
articulated structure resonates a little higher. **(ii)** Peak *acceleration* is
essentially **model-independent** (0.25 / 0.24 / 0.25 m/s² at 0.215; tight at
every fin): the extra coupled added mass that raises the RAO also lengthens T_n,
and the lower ω² exactly cancels it, since `a = ω²·(RAO·H/2)`. Both trends are
monotonic in fin size in **every** model — the "smaller fin = more motion"
conclusion is fully model-independent, as expected, because the fin's role
(dominant heave added mass **and** the only real heave damper) is per-buoy
identical. Under Cd_n=1 the buoy peaks roughly double, landing on each fin's
resonance (3.14 / 2.65 s). See
`../../platform-12buoy/fin_study/platform_fin_sensitivity.png` and
`../../platform-12buoy/fin_study/fin_single_vs_cluster_vs_platform.png`.

**Run infrastructure note.** The platform fin fan surfaced a memory leak in the
constrained (KKT) integrator — ~2 GB/case retained via native-heap fragmentation
over the ~40 k steps/case (the retardation convolution buffer was suspected from
the OOM traceback but exonerated in isolation; see tracker
`CONSTRAINED-INTEGRATOR-SWEEP-MEMORY`). The 220-case sweep is therefore run as a
sequence of bounded fresh subprocesses (`run_platform_fin_fan.py`, 12 cases each,
per-case row-JSON resume), which caps peak RAM ~24 GB and reclaims it on each
process exit. RAO values are byte-identical to a single-process run.

## Decision summary (all three models)

**Keep a full-size fin (≥ 0.215 m).** Shrinking to 0.15 m costs ~+20-30 % peak
buoy RAO and ~+80-90 % peak buoy vertical acceleration (0.25 → 0.44-0.47 m/s²);
removing the fin roughly doubles peak RAO again (to ~4 at platform scale) and
leaves the heave resonance essentially undamped (finite only because of the
modelled spar bottom-cap drag; a bare finless spar diverges). The fin is the
buoy's dominant heave added mass **and** its only meaningful heave damper, so
this holds identically across the single buoy, the 3-cluster, and the 12-buoy
platform.

## Files

- `sparfin_fin_bem.py` — parametric mesh + BEM per fin (`capytaine_fin{0215,015,none}.nc`).
- `sparfin_fin_fan.py` — the RAO+accel fan (matched plate drag; t_max=60 kernel).
- `fin_plots.py` → `fin_sensitivity.png`.
- `rao_summary_fin*.csv` + per-case CSVs + `manifest.json`.
- 3-buoy cluster: `../../cluster-3buoy-rigid/fin_study/` (BEM, fan, plots).
- 12-buoy platform: `../../platform-12buoy/fin_study/` + `platform_fin_bem.py`,
  `platform_fin_fan.py`, `run_platform_fin_fan.py` (chunked runner),
  `platform_fin_plots.py`.

- `sparfin_fin_bem.py` — parametric mesh + BEM per fin (`capytaine_fin{0215,015,none}.nc`).
- `sparfin_fin_fan.py` — the RAO+accel fan (matched plate drag; t_max=60 kernel).
- `fin_plots.py` → `fin_sensitivity.png`.
- `rao_summary_fin*.csv` + per-case CSVs + `manifest.json`.
