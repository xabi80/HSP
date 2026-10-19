# Payload stability: the 4-buoy cluster (2026-10-14) and the 4x4 platform (2026-10-15)

**Question (Xabier):** add mass on top of the cluster until its stability is compromised, move
that mass around parallel to the waterline, and give an envelope of mass and CG. Then the same
exercise for the platform: see [The platform](#the-platform-2026-10-15) below.

## Decisions

| Item | Choice |
|---|---|
| Mass position | A point mass on the hub, in the pin plane (+0.717 m), moved horizontally |
| Condition | Free-floating, calm water |
| Stability criterion | Loss of restoring, **and the pins must stay dry** |
| Hub tilt | Free on the gimbals: no mechanical stop |
| Method | Capytaine hydrostatics for each draft, plus large-angle statics outside FloatSim (an approved exception). FloatSim cross-checks the small-angle range. |

### Why not FloatSim alone

FloatSim cannot reach the stability limit on its own:
- **Its restoring is a constant matrix at the BEM draft.** Level 2 is not wired:
  `docs/phase2-followups.md` LEVEL2-INTEGRATOR-UNWIRED. Each kilogram on the hub sinks the cluster
  12.6 mm, and the buoy's tilt stiffness triples before the spar tops go under.
- **Its validity ends at 0.1 rad.** An off-centre mass tilts the hub past that long before any
  limit.
- **The hub's weight never loads the joints.** FloatSim's linear model has no weight term.

## Method

1. **`draft_hydrostatics.py`.** Capytaine hydrostatics of the flume BEM hull, re-meshed sunk by
   s = −0.95 … +0.715 m. It uses 36 panels round; the 12-panel mesh reads C55 4.5 % low. At each
   s it gives:
   - the displaced volume;
   - the pin load holding that sinkage, dW = ρg(V(s) − V(0));
   - C33 and C55 about the buoy's CoG.

   The 96-panel mesh at the design draft gives C55 = 74.08 N·m/rad, the record value
   (flume-mooring DESIGN-BASIS §C8). Output: `buoy_hydrostatics_vs_draft.json`.
2. **`cluster_statics.py`: the hinged cluster, exact at any hub tilt.**
   - Every load is vertical, so each buoy hangs vertically from its gimbal. Its pin load is
     dW(s_i), where s_i is the depth of its pin below the design pin plane.
   - The hub (2 kg plus the payload at offset e) is in equilibrium where the potential energy is
     stationary, U(z, roll, pitch) = Σ∫dW + g m_hub z + g m z_payload.
   - The hub is stable while the Hessian of U is positive definite.
   - Each buoy is stable on its gimbal while C55(s_i) − P_i·1.624 m > 0, where 1.624 m is the
     height from the buoy's CoG to the pin.
   - The payload mass is raised from 0 along the equilibrium branch until a pin goes awash, the
     restoring is lost, or the hub rolls onto its side (beyond 85°, the model's range).
   - Grid: offsets 0–0.5 m in 0.025 m steps, in 4 directions (0° = between two buoys, 45° =
     towards a buoy). The square's symmetry covers every direction.
   - Output: `cluster_envelope.json`.
3. **`floatsim_crosscheck.py`.** FloatSim's own cluster model (`mooring_verify.article_cluster`,
   12-panel BEM, gimbal joints), solved statically with its linear constrained system, against the
   statics on the same 12-panel table. At small tilt they agree exactly:
   - sinkage 52.65 / 157.94 mm for 2 / 10 kg;
   - tilt 0.869° and 3.475° vs 3.477° for 1 kg at 0.1 m and 2 kg at 0.2 m, in every direction.
4. **`envelope_plot.py`** draws `cluster_envelope.png`.

## Results

![envelope](cluster_envelope.png)

- **A centred mass never makes the cluster unstable; it only sinks it.**
  - The limit is **54.7 kg on the hub**, when all four spar tops (the pins) reach the waterline.
  - The hub's own 2 kg already sinks the cluster 25 mm.
  - The buoys' tilt stiffness on their gimbals rises with draft almost as fast as the pin load
    erodes it. Its margin never falls below 24 N·m/rad, so **no buoy ever flops over** on its
    gimbal.
- **An off-centre mass tilts the hub like a seesaw.** The buoys stay upright; only their
  differential sinkage resists the tilt, 67 N·m/rad. The tilt depends almost only on mass × offset,
  in any direction:

  | Hub tilt | 2° | 5° | 10° | 20° | 30° |
  |---|---|---|---|---|---|
  | mass × offset | 0.24 kg·m | 0.60 kg·m | 1.19 kg·m | 2.35 kg·m | 3.44 kg·m |
  | e.g. at 0.1 m | 2.4 kg | 6.0 kg | 11.9 kg | 23.5 kg | 34.4 kg |
  | e.g. at 0.3 m | 0.8 kg | 2.0 kg | 4.0 kg | 7.8 kg | 11.5 kg |

- **Limits** (payload mass on the hub; tilt and whole-cluster CG offset at the limit):

  | Offset | Between buoys (0°) | Towards a buoy (45°, worst) | Buoys may touch (45°) |
  |---|---|---|---|
  | 0 | 54.7 kg, pins awash | 54.7 kg | — |
  | 0.1 m | 40.9 kg (36° tilt, CG 32 mm) | 36.9 kg (33°, CG 30 mm) | 40.8 kg |
  | 0.2 m | 32.6 kg (71°, CG 54 mm) | 27.8 kg (54°, CG 48 mm) | 28.8 kg |
  | 0.3 m | 22.8 kg, hub on its side | 22.3 kg, pin awash (79°, CG 61 mm) | 21.5 kg |
  | 0.4 m | 17.1 kg, hub on its side | 17.1 kg, hub on its side | 16.1 kg |
  | 0.5 m | 13.7 kg, hub on its side | 13.7 kg, hub on its side | 12.9 kg |

  - Up to about 0.3 m of offset, the first pin going awash sets the limit.
  - Beyond about 0.3 m, the hub rolls onto its side (≥ 85°) before any pin goes under. That
    happens at mass × offset ≈ 6.8 kg·m in every direction.
  - Past about 0.2 m, the buoys may touch each other before either limit.
- **The limits all sit at very large hub tilts** (30–85°). With a free gimbal, the hub tilt, not
  buoyancy, is what an operator will notice first. A practical allowable is better taken from the
  tilt lines, e.g. ≤ 10° hub tilt ↔ mass × offset ≤ 1.2 kg·m.

## Presentation

`Cluster_payload_stability.pptx` (15 slides, `make_payload_ppt.py`) explains the physics with
pictures:
- the cluster and its loads, drawn to scale (`sketch_cluster.png`). Every picture is drawn from
  one fixed direction, 22.5° off the buoy diagonals: seen square-on or along a diagonal, buoys
  hide behind each other, but from 22.5° all four show. A top-view inset gives the payload's
  position;
- three loadings as still strips and animations, from the solved equilibria
  (`payload_figures.py`): the centred mass sinking (`seq_centred.png`, `anim_centred.gif`), the
  seesaw (`seq_offset.png`, `anim_offset.gif`) and the hub rolling onto its side
  (`seq_rollover.png`, `anim_rollover.gif`). The GIFs play in PowerPoint's slide show;
- native, editable charts of a buoy's pin load and gimbal margin against sinkage, the
  tilt-against-moment collapse (`tilt_paths.json`) and both envelopes;
- the cross-checks, guidance and limits.

A copy is in `studies/` (PRESENTATIONS.md). The slide helpers are shared with the platform deck in
`deck_kit.py`.

## Limits of this analysis

- **Static, calm water.** Waves, dynamics and the mooring (its 1° pretension lean) are not
  included and would reduce the margins.
- **The payload sits in the pin plane.** A raised payload adds m·g·h of destabilising moment to
  the hub tilt and needs a new run.
- **The hub tilts freely.**
  - Gimbal travel, payload retention and hub buoyancy are not modelled.
  - Buoy contact is only flagged: axes closer than a plate diameter, 0.287 m.
- **Hull geometry.** The BEM hull lacks the real buoy's stem volume (about 1 L, below the
  waterline). It doesn't change with draft, so dW(s) is unaffected.

## Reproduce

```
python draft_hydrostatics.py   # Capytaine, ~10 min
python cluster_statics.py      # ~10 min
python floatsim_crosscheck.py  # needs flume-mooring/cluster_osu_open_rot45_psd.nc
python envelope_plot.py
python payload_figures.py     # pictures and animations, ~10 min
python make_payload_ppt.py    # the deck
```

## The platform (2026-10-15)

**Question (Xabier):** "the same exercise as for the cluster" for the 4x4 platform.

### Model

- **The platform is the record's** (`articulated_wall.deck` with PLAT_SQUARE_M 2.4 and
  PLAT_ROT_DEG 45, the model `mooring_verify.article_platform` runs):
  - 16 buoys on a 0.8 m grid (outer buoy-centre span 2.4 m), in four 2 × 2 clusters;
  - each buoy hangs from its hub (2 kg) by a gimbal at its pin (+0.717 m);
  - each hub hangs from the deck frame (6 kg, +0.90 m) by a gimbal at the hub centre, in the
    pin plane.
- **Decisions carried over from the cluster:** free-floating, calm water, every gimbal free,
  pins dry; the limit is the loss of restoring or the first pin awash.
- **Payload:** a point mass on the deck frame (+0.90 m, 0.183 m above the pins), moved out to the
  outer buoy-centre line (1.2 m), in 4 directions (0° = between hubs, along the grid; 45° =
  towards a hub, on the diagonal). The deck sits above the pins, so the payload's CG height
  matters. It is run as a sensitivity: full envelopes with the CG 0.5 and 1.0 m above the deck,
  and the centred limit for CG heights 0–2 m.
- **Each hub hangs level, exactly.** The deck's load reaches a hub at its gimbal: the hub centre,
  in the plane of its four pins. That load has no lever arm to tilt the hub, so its four buoys
  sink together and each cluster acts as one vertical spring under the deck (4 × 194 N/m,
  1.131 m from the centre). Only the deck tilts. Equilibrium is solved on the deck's heave, roll
  and pitch. The stability check uses all 11 coordinates, the hub tilts included, and the hub
  moment residual stays ≤ 2e-14 N·m.

Scripts:
- `platform_statics.py`: the platform, exact at any deck tilt, on the cluster's draft table and
  building blocks (`cluster_statics.Buoy`, `rot`, `awash`). It writes:
  - `platform_envelope_h0p0.json`, `_h0p5.json` and `_h1p0.json` (one per payload CG height);
  - `platform_vs_height.json`.
- `platform_crosscheck.py`: FloatSim's own platform model, solved statically.
- `platform_envelope_plot.py`: draws `platform_envelope.png`.

### Results

![platform envelope](platform_envelope.png)

- **A centred mass on the deck only sinks the platform.**
  - The limit is **212.7 kg** on the deck, when all 16 pins reach the waterline.
  - That is about four times the cluster's limit, less the deck: 4 × 54.7 − 6.
  - Each kilogram sinks the platform 3.16 mm. The deck and hubs alone sink it 44 mm.
- **Off-centre, the whole deck tilts, and much more stiffly than the cluster's hub.**
  - The hubs stay level; the clusters' differential sinkage resists the tilt at **1987 N·m/rad**,
    29 times the cluster's hub (16 buoys at about 0.8 m, against 4 at 0.3 m).
  - The tilt again depends mostly on mass × offset, the same in every direction. The values below
    are the smallest over the offsets: heavy masses at small offsets tilt a little more, because
    they sit above the pins.

    | Deck tilt | 1° | 2° | 5° | 10° | 20° |
    |---|---|---|---|---|---|
    | mass × offset | 3.3 kg·m | 6.2 kg·m | 15.1 kg·m | 30.1 kg·m | 60.8 kg·m |

- **Limits** (payload mass on the deck; deck tilt and whole-platform CG offset at the limit):

  | Offset | Between hubs (0°), CG on the deck | Towards a hub (45°, worst), CG on the deck | 45°, CG 1.0 m above the deck |
  |---|---|---|---|
  | 0 | 212.7 kg, pins awash | 212.7 kg | 170.4 kg, the level deck tips over |
  | 0.2 m | 164.3 kg (11°, CG 63 mm) | 150.8 kg (10°, CG 59 mm) | 105.6 kg (17°) |
  | 0.4 m | 134.9 kg (18°, CG 109 mm) | 118.3 kg (15°, CG 99 mm) | 84.5 kg (21°) |
  | 0.6 m | 114.9 kg (23°, CG 146 mm) | 97.4 kg (19°, CG 128 mm) | 71.7 kg (23°) |
  | 0.8 m | 100.5 kg (26°, CG 175 mm) | 83.6 kg (21°, CG 151 mm) | 62.7 kg (24°) |
  | 1.0 m | 89.2 kg (29°, CG 199 mm) | 72.9 kg (23°, CG 169 mm) | 56.0 kg (25°) |
  | 1.2 m | 80.3 kg (31°, CG 220 mm) | 64.9 kg (24°, CG 184 mm) | 50.7 kg (26°) |

  - With the payload on the deck, every limit is one cluster's four pins going awash together.
    The deck never tips over.
  - Towards a hub is the worst direction, by up to 19 %.
  - The limits sit at 10–31° of deck tilt (up to 37° with a raised CG). As for the cluster, a
    practical allowable is better taken from the tilt lines.
- **A raised payload can tip the deck over: the platform's new limit.**
  - As the deck tilts, a mass at a height H above the pins moves sideways by H × the angle. That
    gives a tipping moment m g H per radian against the clusters' 1987 N·m/rad.
  - The level deck loses its restoring at m (h + 0.183 m) = 1987/g − 6 × 0.183 = **201.6 kg·m**,
    where h is the CG height above the deck. The full calculation follows this rule exactly
    (panel c).
  - Centred, the pins limit the payload while its CG is below about 0.77 m above the deck. Above
    that, the deck tips first: 170 kg with the CG 1 m up; at most 1.83 / 1.16 / 0.82 m for
    100 / 150 / 200 kg.
  - Off-centre, a raised CG lowers the limits (purple lines) and amplifies the tilt by
    1 / (1 − m H / 201.6 kg·m).
- **The buoys behave as in the cluster.** They never flop on their gimbals: the margin is
  ≥ 23.9 N·m/rad at every limit. They never come near each other either: 0.47 m between axes at
  worst, against a 0.287 m heave plate.
- **FloatSim cross-check** (`platform_crosscheck.py`; FloatSim's 12-panel coupled BEM, C33
  186.3 N/m per buoy).
  - **Like for like** (every weight in the pin plane, all FloatSim's linear model can represent),
    FloatSim and the statics agree exactly:
    - sinkage 79.0 / 210.6 mm for 10 / 50 kg;
    - tilt 0.736° for 5 kg at 0.5 m, and 2.946° vs 2.947° for 10 kg at 1.0 m, in every direction.
  - **With the real deck and payload heights,** the statics tilt up to 1.6 % more: the m g h
    effect FloatSim omits.

### Presentation

`Platform_payload_stability.pptx` (15 slides, `make_platform_ppt.py`), built like the cluster
deck, with the same shared helpers in `deck_kit.py`:
- **Pictures** (`platform_figures.py`), with one fixed view for every drawing. The view is
  atan(1/4) ≈ 14° off the grid, where the 16 spars project evenly, 0.19 m apart, so none hides
  another. A top-view inset gives the payload's position. The pictures are:
  - the labelled sketch, `sketch_platform.png`;
  - three loadings as strips and animations: centred (`pseq_centred.png`, `panim_centred.gif`),
    1.2 m towards a corner (`pseq_offset`, `panim_offset`), and the CG 1 m up, 0.2 m off centre
    (`pseq_raised`, `panim_raised`).
- **Native, editable charts:** tilt against mass × offset (`platform_tilt_paths.json`), the
  centred limit against CG height, and both envelopes.
- **The comparison with the cluster,** the cross-check, guidance and limits.

A copy is in `studies/` (PRESENTATIONS.md).

### Limits of this analysis (platform)

As for the cluster:
- **Static, in calm water, free-floating.** Waves and the mooring would reduce the margins, most
  for raised payloads.
- **The payload is a point mass,** within the outer buoy-centre line (1.2 m).
- **The deck frame is rigid.** Gimbal travel, payload retention and the deck's own flexibility
  are not modelled.

### Reproduce (platform)

```
python platform_statics.py envelope 0     # ~2 min each
python platform_statics.py envelope 0.5
python platform_statics.py envelope 1.0
python platform_statics.py heights
python platform_crosscheck.py             # needs platform-12buoy/flume-wall-effect/coupled_osu_open_rot45_sq2p4_psd.nc
python platform_envelope_plot.py
python platform_figures.py                # pictures and animations, ~5 min
python make_platform_ppt.py               # the deck
```

## Dynamics in waves with a mass-shift tilt (2026-10-16)

**Question (Xabier):** a few wave periods for 5° and 10° of tilt caused by a mass shift, to see
the effect on the dynamics.

### Decisions (Xabier)

| Item | Choice |
|---|---|
| Method | FloatSim, with the BEM re-run at each tilted equilibrium |
| Articles | Both |
| Mass shift | 50 kg on the platform deck and 12.5 kg (50 / 4) on the cluster hub, shifted along the waves to 5° and 10°; compared with the same mass centred (level) and the unloaded article |
| Waves | Regular, H = 0.1 m, along the shift, T = 1.5 / 2.0 / 2.35 / 2.65 / 3.5 / 5.0 s |

### Method

1. **`tilted_cases.py`** finds each case's static equilibrium with the large-angle statics, on
   the 12-panel table the coupled BEM uses. It gives each body's position, each buoy's draft and
   pin load, and the payload offset for exactly 5° and 10°:
   - platform: 0.32 m and 0.64 m;
   - cluster: 46 mm and 92 mm.
2. **`tilted_bem.py`** runs the coupled Capytaine BEM at each case's geometry. Every buoy has its
   own position and draft and its own hydrostatic matrix; the records tile one design-draft buoy.
   Otherwise it follows the records: 12 panels round, deep water, 48 frequencies, PSD
   projection. Run time is ~30 min per platform case and ~0.5 min per cluster case.
3. **`tilted_dynamics.py`** runs FloatSim's own articulated model, rebuilt about each equilibrium.
   - Buoys sit at their drafts and still hang vertically; the hubs are level; the deck or hub is
     tilted.
   - The payload is folded into the deck or hub body: mass, CG and full inertia tensor.
   - The joints attach at the equilibrium pins and hub centres.
   - Each spar's drag runs up to its own waterline. Drag acts relative to the wave's orbital
     velocity.
   - FloatSim's static solve has no weight term, so it keeps this geometry as the equilibrium.
     The linear dynamics about it hold for small wave motions, whatever the static tilt.
   - A very soft horizontal hold on the deck or hub stops the free article drifting (~0.5 m per
     run otherwise), which would put FloatSim's drag, which follows the body, out of phase with
     the BEM excitation, which stays at the mean position. The hold's surge period is ~45 s, and
     it changes the wave response by ~1 % at 5 s and less below.
   - Each run is 90 s with a 15 s ramp; amplitudes are fitted over the last 30 s.
   - The script also gives natural periods (`mooring_verify` mode tracking), with and without the
     pin-load tipping. The tipping is an estimate added for the periods only, not to the time
     domain: each constant vertical load F at an arm a from a body's reference adds
     (a·F) I − ½(a Fᵀ + F aᵀ) to the body's rotations.
   - Run time is ~25 min per cluster case and ~60 min per platform case (4 in parallel).
4. **`tilted_dynamics_plot.py`** draws `tilted_dynamics.png`.

### Results

![dynamics](tilted_dynamics.png)

| Case | Offset | Heave T | Deck / hub pitch T | Buoys' rocking T: FloatSim / + pin tipping | At 2.65 s: heave, pitch | Smallest pin freeboard: still water → in waves |
|---|---|---|---|---|---|---|
| **Cluster**, unloaded | 0 | 2.59 s | 2.60 s | 2.79 / 2.95 s | 73 mm, 2.49° | 691 → 588 mm |
| 12.5 kg centred (level) | 0 | 2.71 s | 2.60 s | 2.86 / 4.22 s | 67 mm, 2.16° | 526 → 436 mm |
| 12.5 kg, 5° | 46 mm | 2.71 s | 2.59 s | 2.86 / 4.22 s | 67 mm, 2.89° | 500 → 410 mm |
| 12.5 kg, 10° | 92 mm | 2.72 s | 2.59 s | 2.85 / 4.21 s | 66 mm, 3.69° | 475 → 383 mm |
| **Platform**, unloaded | 0 | 2.60 s | 2.61 s | 2.81 / 3.10 s | 65 mm, 2.22° | 671 → 573 mm |
| 50 kg centred (level) | 0 | 2.72 s | 2.60 s | 2.86 / 4.38 s | 61 mm, 1.26° | 506 → 414 mm |
| 50 kg, 5° | 0.32 m | 2.74 s | 2.60 s | 2.85 / 4.36 s | 58 mm, 1.89° | 437 → 354 mm |
| 50 kg, 10° | 0.64 m | 2.78 s | 2.60 s | 2.85 / 4.30 s | 53 mm, 2.45° | 367 → 285 mm |

The smallest freeboard in waves comes at T = 2.65 s in every case.

- **The payload's mass, not the tilt, moves the heave period:** +0.12 s for both articles.
  Below resonance, heave is a little smaller (e.g. 46 → 36 mm for the cluster at 2.35 s).
- **The tilt couples heave into pitch near resonance.**
  - At 2.65 s, the hub pitch grows from 2.2° (level) to 2.9° (5°) and 3.7° (10°) on the
    cluster. The deck pitch grows from 1.3° to 1.9° and 2.5° on the platform, whose heave drops
    from 61 to 53 mm.
  - Away from resonance (1.5 s and 5 s) the tilt changes almost nothing.
  - The deck / hub pitch mode is the clusters (buoys) heaving against each other, so its period
    stays at ~2.6 s.
- **The tilt's main effect on the pins is the still-water freeboard it takes away.**
  - Waves take ~80–100 mm more at the worst period (2.65 s): the pins move up to ~80 mm relative
    to the water, 1.6× the wave amplitude.
  - At 10° the platform's lowest pin keeps 285 mm and the cluster's 383 mm in H = 0.1 m waves,
    far from awash.
  - Linearly, a pin would reach the water at 10° only with H ≈ 0.45 m (platform) or 0.57 m
    (cluster) near 2.65 s. This is indicative: drag is not linear.
- **The buoys' rocking on their gimbals is NOT reliable from FloatSim without its weight
  preload.** Superseded by the run with it: see
  [With FloatSim's weight preload](#with-floatsims-weight-preload-2026-10-17).
  - FloatSim has no weight term, so it leaves out the pin load tipping each buoy (−P L).
  - At 10° that load reaches 73 / 117 N·m/rad of the loaded cluster buoy's C55, and 106 / 141
    on the platform.
  - With it, the buoys' in-phase rocking mode moves from ~2.85 s to **~4.2–4.4 s** in every
    loaded case, against 2.95–3.10 s unloaded.
  - So FloatSim's gimbal angles (7–8° near 2.65 s) put the resonance in the wrong place: the
    real rocking peaks in the 3.5–5 s waves.
  - Heave and deck / hub pitch feel this term less. Their periods move by up to 0.08 s in every
    case but the platform at 10°, where the coupled heave–pitch pair moves from 2.78 / 2.60 s to
    2.64 / 2.86 s. So the platform's tilted resonance region is also only approximate.
  - The term is present unloaded too (the hubs' and deck's weight): it lengthens the platform's
    rocking mode from 2.81 to 3.10 s, which applies to the records' flume runs as well.
- **The six periods are coarse around resonance** (2.35 / 2.65 / 3.5 s). The peaks between them
  are not resolved. A seventh, T = 4.2 s, was added on 2026-10-17 for the preload comparison.

### With FloatSim's weight preload (2026-10-17)

Option 2 of the follow-ups, "teach FloatSim the weight term", is done in FloatSim itself (main
`05128b7`, `dd889f2`; `docs/weight-preload-plan.md`). It is opt-in, with
`simulation.weight_preload: true` and a per-body `displaced_volume`.
- **The static loads:** every body's weight, and each hydro body's buoyancy at its BEM draft.
- **The statics:** a joint-constrained static solve gives the joints their static preload.
- **The dynamics:** the preload acts through the joint Jacobian's configuration dependence.
  `SimulationSetup.K_geo` gives its stiffness to matrix tools.
- **Validated:** against an analytic pinned float (periods to 1e-4, and the finite-amplitude
  correction) and against this study's statics on the flume cluster.

`tilted_dynamics.py --preload` re-runs every case with it.
- Each buoy's `displaced_volume` is its weight plus its pin load from the statics.
- **Check 1:** FloatSim's constrained static solve leaves the statics geometry where it is, to
  ≤ 4e-15 m in all 8 cases.
- **Check 2:** its own `C + K_geo` natural periods equal this study's earlier estimate exactly.

![preload](tilted_dynamics_preload.png)

Both columns are at H = 0.1 m, without / with the preload.

| Case | Buoys' rocking period | Largest gimbal angle | At 2.65 s: deck / hub pitch | At 2.65 s: heave | Smallest pin freeboard |
|---|---|---|---|---|---|
| **Cluster**, unloaded | 2.79 / 2.95 s | 11.7° / 8.2° (2.65 s) | 2.49 / 2.54° | 73 / 75 mm | 588 / 604 mm |
| 12.5 kg centred | 2.86 / 4.22 s | 6.9° (2.65 s) / **12.3° (4.2 s)** | 2.16 / 2.15° | 67 / 68 mm | 436 / 444 mm |
| 12.5 kg, 5° | 2.86 / 4.22 s | 7.2° (2.65 s) / **12.3° (4.2 s)** | 2.89 / 2.91° | 67 / 67 mm | 410 / 417 mm |
| 12.5 kg, 10° | 2.85 / 4.21 s | 7.5° (2.65 s) / **12.6° (4.2 s)** | 3.69 / 3.80° | 66 / 66 mm | 383 / 389 mm |
| **Platform**, unloaded | 2.81 / 3.10 s | 10.7° / 6.6° (2.65 s) | 2.22 / 2.29° | 65 / 66 mm | 573 / 588 mm |
| 50 kg centred | 2.86 / 4.38 s | 7.2° (2.65 s) / **11.1° (4.2 s)** | 1.26 / **1.82°** | 61 / 60 mm | 414 / 424 mm |
| 50 kg, 5° | 2.85 / 4.36 s | 7.5° (2.65 s) / **11.2° (4.2 s)** | 1.89 / **2.48°** | 58 / 56 mm | 354 / 354 mm |
| 50 kg, 10° | 2.85 / 4.30 s | 7.9° (2.65 s) / **11.2° (4.2 s)** | 2.45 / **3.07°** | 53 / 50 mm | 285 / 282 mm |

- **The payload moves the buoys' rocking resonance into the longer waves.**
  - The pin load tipping each buoy takes away 55–75 % of its rocking stiffness (P L against
    C55, `tilted_cases.json`). The
    in-phase rocking mode moves from ~2.85 s (FloatSim without the preload) to 4.2–4.4 s.
  - There, H = 0.1 m waves rock the loaded buoys **11–12.6° on their gimbals**, in every loaded
    case, level or tilted. Without the preload FloatSim gave 0.2–0.7° at 4.2 s.
  - The rocking peak it showed at 2.65 s (7–8°) was an artefact of the missing term; with the
    preload it falls to 3–4.3°.
  - 11–12.6° is past FloatSim's small-angle range (5.7°), so the size is indicative. The
    resonance and its period are not.
- **The tilt hardly changes the rocking, though each buoy is loaded differently.**
  - **Each buoy alone would differ.** At 10° the cluster's near pair is at 44 N·m/rad
    (alone: ~4.45 s) and its far pair at 54 N·m/rad (~4.0 s); level, all four are at 49.
  - **The hub makes them rock together.** It carries all the pins at once, so in the mode the
    waves drive every buoy leans the same way. The period follows the sum of their stiffnesses:
    2×44 + 2×54 = 196 against 4×49 = 195 level, so 4.21 s against 4.22 s.
  - **Why the sum stays put.** A buoy's net stiffness falls almost linearly as it sinks: C55
    gains ~205 and P L ~300 N·m/rad per metre. Shifting the payload sinks one side and lifts
    the other by equal amounts, and the mean sinkage is fixed by the total weight.
  - **The platform's wider spread** (72–350 mm of sinkage) shows C55's curvature with draft:
    4.30 s at 10° against 4.38 s level.
  - **Evidence.** In the mode analysis the cluster's amplitudes are 0.96 : 1.00. At 4.2 s all
    16 platform buoys rock within 10.1–11.2° of each other.
  - **Other modes.** The buoys can rock against each other only by sliding sideways relative to
    one another (the hub won't let the pins separate). That brings in their whole mass, so those
    modes sit near 9 s. A cross-wise twisting pair (3.7 / 4.4 s at 10°) can't be driven by waves
    along the shift (by symmetry); waves from the side or at an angle could.
- **Unloaded, the hubs' and deck's own weight already matters.** The platform's rocking mode
  is 3.10 s, not 2.81 s, and its 2.65 s rocking falls from 10.7° to 6.6°. This also applies to
  the records' flume runs, which have no preload.
- **The platform's deck pitch near resonance rises by about 25–45 %** with the preload
  (1.26 → 1.82° level, 2.45 → 3.07° at 10°, at 2.65 s). The deck's and payload's weight above
  the pins (m g h) softens the deck's tilt and shifts the coupled heave–pitch modes. The
  cluster's payload sits in the pin plane, so its hub pitch barely changes.
- **Heave and pin freeboard change little:** heave by ≤ 6 %, the smallest freeboard by ≤ 3 %.
  The tightest freeboard is still at
  2.65 s: 282 mm (platform, 10°) and 389 mm (cluster, 10°). At the new 4.2 s rocking resonance
  it is 332 mm and 435 mm.
- **What it means for operation.** With a payload, keep clear of waves near 4.2 s. That holds
  however level the payload is: the buoys rock hard on their gimbals there, whatever the
  heave says.

### Accelerations at the payload (2026-10-18)

`tilted_dynamics.py` also records the accelerations (from `xi_ddot`, a + α × d) at the payload
and at the hub / deck centre, for every run with the preload. "Lateral" here is horizontal along
the waves, which is also the direction of the mass shift. Across the waves the acceleration is
zero in these runs, by symmetry: the article, the shift and the waves all share one vertical
plane. `tilted_accelerations_plot.py`
draws `tilted_accelerations.png`.

![accelerations](tilted_accelerations.png)

Accelerations at the payload, H = 0.1 m, amplitude in m/s² (g = 9.81 m/s²):

| | Vertical, largest | Lateral as felt, largest dynamic | Static tilt, g sin θ | Total lateral, largest |
|---|---|---|---|---|
| Cluster, unloaded | 0.42 (2.65 s) | 1.16 (2.65 s) | 0 | 1.16 |
| Cluster, 12.5 kg level / 5° / 10° | 0.38 / 0.38 / 0.37 | 0.63 / 0.69 / 0.79 | 0 / 0.85 / 1.70 | 0.63 / 1.55 / **2.49 (0.25 g)** |
| Platform, unloaded | 0.37 (2.65 s) | 0.79 (2.35 s) | 0 | 0.79 |
| Platform, 50 kg level / 5° / 10° | 0.34 / 0.32 / 0.30 | 0.52 / 0.53 / 0.60 | 0 / 0.85 / 1.70 | 0.52 / 1.39 / **2.30 (0.23 g)** |

- **Vertical:** small (≤ 0.04 g) and about the same on both articles, peaking near the heave
  resonance (2.65 s). The tilt changes it little.
- **Lateral (horizontal, along the waves and the shift), dynamic:**
  - **The platform is calmer in short waves.** Its 2.4 m span averages them out: 0.1 m/s² at
    1.5 s, against 0.6 on the cluster.
  - **Near 2.35–2.65 s the tilt adds lateral load,** because the coupled pitch swings gravity
    along the deck: cluster 0.61 → 0.79, platform 0.49 → 0.60 m/s², level → 10°.
  - **At 4.2 s, with a payload, both feel 0.4–0.6 m/s².** That is the buoys' rocking resonance:
    the buoys pivot about their CoGs on the gimbals, and the hub or deck slides back and forth along the waves on top
    (~0.25 m).
- **Lateral, total:** the static tilt dominates. A payload tilted 10° feels a constant 1.7 m/s²
  (0.17 g) laterally from gravity, before any wave. The dynamic part adds at most 0.8 m/s².
- **Inertial against felt:** the felt lateral acceleration includes `−g θ` from the deck's dynamic pitch.
  At 3.5 s it partly cancels the inertial part; near 2.35–2.65 s it adds to it.

### Presentation (dynamics)

`Payload_tilt_in_waves.pptx` (12 slides, `make_tilt_waves_ppt.py`) summarises both articles.
- **Contents:** the cases (pictures from `tilted_figures.py`), the method, why the weight
  preload changes the answer, and heave, pitch, pin freeboard and gimbal rocking against
  wave period (native charts).
- **Then:** the single rocking period of all the buoys, the guidance, and an appendix on
  FloatSim's validation.
- A copy is in `studies/` (PRESENTATIONS.md).

### Limits of this analysis (dynamics)

- **FloatSim's model:** linear, about each equilibrium. The weight term is in only with
  `--preload` (above). It uses the 12-panel BEM; 36 panels would lengthen the periods by
  ~0.05 s (flume-mooring DESIGN-BASIS).
- **Large rocking.** 11–12.6° of gimbal rocking is past the small-angle range; the gimbals'
  real travel and the plate drag at those angles are not modelled.
- **Regular waves along the shift only.** A shift across the waves would couple into roll and
  sway instead.
- **Free-floating with a soft hold, not the flume mooring.**

### Reproduce (dynamics)

```
python tilted_cases.py
python tilted_bem.py cluster                   # ~2 min
python tilted_bem.py platform <case>           # ~30 min each: unloaded, centred, tilt5, tilt10
python tilted_dynamics.py <article> <case> [--preload]   # ~25 min cluster / ~60 min platform
python tilted_dynamics.py add-period <article> <case> 4.2 [--preload]
python tilted_dynamics.py summary [--preload]
python tilted_dynamics_plot.py              # tilted_dynamics.png
python tilted_dynamics_plot.py --compare    # tilted_dynamics_preload.png
python tilted_accelerations_plot.py         # tilted_accelerations.png (needs the --preload runs)
```

Run several cases in parallel with `OMP_NUM_THREADS=1` (and `OPENBLAS_` / `MKL_NUM_THREADS=1`).
Without it, each process starts numpy threads on every core. Four platform runs then thrashed
the machine: a 206 × 206 solve took 0.5 s, and the runs went ~36× slower.

The BEM databases (`tilted_bem/`, ~60 MB) and the per-case outputs (`tilted_dynamics/`,
`tilted_dynamics_preload/`) are regenerated by these runs and not committed.
