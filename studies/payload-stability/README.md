# Payload stability: the 4-buoy cluster (2026-10-14)

**Question (Xabier):** add mass on top of the cluster until its stability is compromised, move
that mass around parallel to the waterline, and give an envelope of mass and CG. The platform
follows later.

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
- the cluster and its loads, drawn to scale (`sketch_cluster.png`);
- three loadings as still strips and animations, from the solved equilibria
  (`payload_figures.py`): the centred mass sinking (`seq_centred.png`, `anim_centred.gif`), the
  seesaw (`seq_offset.png`, `anim_offset.gif`) and the hub rolling onto its side
  (`seq_rollover.png`, `anim_rollover.gif`). The GIFs play in PowerPoint's slide show;
- native, editable charts of a buoy's pin load and gimbal margin against sinkage, the
  tilt-against-moment collapse (`tilt_paths.json`) and both envelopes;
- the cross-checks, guidance and limits.

A copy is in `studies/` (PRESENTATIONS.md).

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

## Next: the platform

The same exercise for the 2.4 m square platform (four hubs and a deck at +0.90 m) needs the
deck's height above the pins in the statics. A payload on the deck sits above the pin plane, so
the hub-tilt destabilisation that is zero for the cluster appears there.
