# Flume mooring specification — OSU Large Wave Flume (HWRL), 1:50 test articles

**REV B, 2026-09-25: the attachment-height design.** It supersedes rev A (`a91e1d5`: pin-plane
lines, cluster/platform pretension +60 %). Rev A is **withdrawn**: its lines shifted the buoys'
tilt resonance ~9 %, several times what the resonance tolerates (see *Design rule*).

Rev B is still issued ahead of full verification: Phase D follows before the test (see
*Follow-ups*). It is built by `build_spec.py` from FloatSim results (`spec_statics.json`,
`attachment_sweep.json`, `attachment_design.json`, `buoy_stability_check.json`). Basis and
derivations: `DESIGN-BASIS.md`, Phase F.

## Page 1 — Facility assumptions to CONFIRM WITH HWRL BEFORE BUYING HARDWARE

None of these is a confirmed facility fact. The design rests on all of them.

| # | Assumption | Value this specification uses | If it proves false |
|---|---|---|---|
| F1 | Anchors can be fixed on the flume walls **under water, at each article's attachment depth** | anchors at x = ±5.0 m, y = ±1.83 m, **z = -0.50 m (buoy), -0.15 m (cluster), -0.15 m (platform)** | A higher or lower anchor slopes the line. Its vertical pull at the attachment adds calm tilt and tilt coupling, so the sweep must be re-run for the real anchor depth. (Rev A's question, anchors 0.72 m *above* still water, no longer arises.) |
| F2 | Wall anchor rating | per-anchor working-load limits (×3 on the FloatSim peak): **buoy 10.6 N, cluster 45.7 N, platform 174.8 N** | Lower than these is unlikely to bind; confirm. |
| F3 | Wave height per period, steepness | H/λ ≤ 0.08 at 2.7 m depth: H 0.5 m only for T ≥ 2.01 s (and from 2.35 s by the 3° tilt rule), H 0.35 m for T ≥ 1.675 s | Cases outside the wavemaker envelope drop out. |
| F4 | Water depth | 2.7 m | The drift bound falls for deeper water; periods unchanged. |
| F5 | Test-section position vs the wavemaker and beach | a clean regular-wave window ≥ 2–3 min at the article, which sits on the flume centreline | Shorter windows do not settle the lightly damped resonances (ζ ≈ 3–4 %). |
| F6 | Tracking field of view | surge −0.3 … **+1.0 m** (down-flume), sway ±0.1 m, plus the article footprint (platform 2.06 m wide at 45°) | Rev B's lines are softer: in the extremes the articles pass +1.0 m (see §5). Those cases must stop earlier (see *Test procedure*). |
| F7 | **Standard soft-mooring hardware, usable submerged** | elastic shock cord: stiffness and elongation per the tables below (secant stiffness within −13 % / +30 % of target over the working stretch), submerged weight ≈ 0.02 N/m | Without it, the lines need farther anchors (longer lines, lower strain) or another soft element. Redesign. |

## Headline

| | single buoy | cluster | 4×4 platform |
|---|---|---|---|
| attachment | radial collar r = 0.2 m on the spar at **z = -0.50 m** (submerged) | each spar at **z = -0.15 m** | the 8 up-/down-stream row spars at **z = -0.15 m** (bridles) |
| anchors (walls, submerged) | x ±5.0, y ±1.83, z -0.50 m | x ±5.0, y ±1.83, z -0.15 m | x ±5.0, y ±1.83, z -0.15 m |
| lines (legs) | 4 | 4 | 8 |
| cord stiffness k (band) | 2.01 N/m (1.75–2.61) | 4.15 N/m (3.61–5.40) | 8.32 N/m (7.24–10.81) |
| pretension T0 per line/leg | **2.40 N** | **1.49 N** | **1.54 N** |
| unstretched length L0 | 3.930 m | 4.589 m | 4.038 / 4.208 m |
| at-rest stretch | 1.19 m | 0.35 m | 0.17 m |
| peak stretch (worst predicted case) | 1.76 m | 3.67 m | 3.54 m |
| elongation capacity ×1.25 | 2.21 m (56 % of L0) | 4.59 m (100 % of L0) | 4.42 m (110 % of L0) |
| max line tension / working load ×3 | 3.54 / 10.6 N | 15.23 / 45.7 N | 29.43 / 88.3 N |
| anchor working-load limit ×3 | **10.6 N** | **45.7 N** | **174.8 N** |
| calm static tilt (FloatSim; ≤ 1°, proposed) | 0.00° | 1.00° | 1.00° |
| tilt period vs unmoored (limit ζ/3) | -1.37 % (2.76 → 2.72 s; ≤ 1.42 %) | -0.73 % (2.86 → 2.84 s; ≤ 1.19 %) | -0.72 % (2.92 → 2.90 s; ≤ 1.05 %) |
| heave period vs unmoored (≤ 1 %) | -0.50 % | -0.08 % | -0.03 % |
| operational band T_min/T0 (≥ 0.15) | 0.94 (kinematic) / 0.93 (moored FloatSim) | 0.56 (kinematic) / 0.56 (moored FloatSim) | 0.18 (kinematic) / 0.19 (moored FloatSim) |
| surge / sway / yaw period | 16.6 / 26.4 / 1.12 s | 23.1 / 51.6 / 24.78 s | 23.0 / 59.0 / 33.40 s |
| pull stiffness surge / sway | 7.31 / 2.60 N/m | 15.05 / 2.64 N/m | 60.63 / 8.11 N/m |
| pull stiffness yaw | 2.00 N·m/rad | 3.06 N·m/rad | 28.96 N·m/rad |

Frame for all coordinates (the TRUE flume frame): the origin is at the article's calm centre, on
the flume centreline, at still water; x runs down-flume (the wave direction), y across, z up. Each
anchor is at its attachment's depth, so every line runs level at rest.

What is conservative in these numbers:
- the **drift sum**: the full recorded fixed-body drift bound, applied in-run at each spar's
  waterline, plus FloatSim's own mean wave force on the moving body;
- **working loads ×3** and **elongation capacity ×1.25** on the worst predicted FloatSim values.

## Design rule and selection (why rev B)

**Why the attachment moved.**
- Rev A attached at the pin plane (+0.717 m) to remove calm-water tilt.
- It accepted the ~9 % tilt stiffening of those lines without comparing it to the resonance
  bandwidth. The comparison, now recorded: at the pin plane with the design pretension and
  stiffness, the tilt period shifts **buoy -9.2 % against ±1.42 % (6.5 ×), cluster -8.8 % against ±1.19 % (7.4 ×), platform -9.0 % against ±1.05 % (8.6 ×)** (the multiple of the tolerance).
- The two effects act about different points:
  - The **static** pretension moment acts about the PIN. It is zero at the pin plane and grows
    with depth below it.
  - The **dynamic** coupling (≈ k·h², where h is the attachment height above the tilt mode's
    rotation centre) acts about the rotation centre, near the CoG (M11a F1). It is largest at the
    pin plane and vanishes at the CoG.
  - No single height satisfies both unless the pretension is low. Rev B lowers the pretension
    and moves the attachment below still water.

**The rule** (Xabier, 2026-09-25; select, do not ask).

HARD criteria, per article:
1. **Tilt period shift vs the unmoored article ≤ ζ/3.** ζ is the tilt damping at H = 0.04 m
   (buoy 4.25 %, cluster 3.56 %, platform 3.15 %), so the limits are buoy 1.42 %, cluster 1.19 %, platform 1.05 %. A detuning δ scales the resonant response by
   1/√(1 + (δ/ζ)²); δ ≤ ζ/3 keeps it within 5 %.
2. **Heave period shift ≤ 1 %.**
3. **Calm static tilt ≤ 1°.** This criterion is **PROPOSED; Xabier to confirm**.
4. **Surge period ≥ 14 s.**
5. **No slack in the operational band** (H ≤ 0.12 m) with the drift sum: T_min ≥ 0.15 T0.

ALLOWED: slack in the extreme cases. The peak re-tension load is reported.

Selection:
- Among passing designs, the MOST pretension.
- Tie-break: the attachment closest to the waterline.
- A third tie-break was added: the stiffest surge, i.e. the least excursion (the confirmed
  Phase B criterion 4). It decides only the buoy: at the same T0 and depth it picks k ×1 over
  ×0.5. That halves the offsets and the cord strain, with a tilt shift still inside ζ/3.

**How the sweep was run** (`attachment_sweep.py`, FloatSim modal analysis and statics):
- **Grid:** heights pin, SWL, −0.15, −0.30, −0.50 m and the CoG (−0.907 m). Pretension ×1, 0.75,
  0.6, 0.5, 0.35 and 0.25 of the design, plus the operational minimum and, for the articulated
  articles, the 1° static cap. Stiffness k ×1 and ×0.5.
- **Tilt and heave periods:** K = C + the linearised FloatSim catenary force; M + A(ω) iterated.
- **Static tilt:** estimated from the per-buoy tilt stiffness. The chosen articulated designs'
  pretension was then trimmed to FloatSim's own settle: the estimate had put the tilt 1.4 %
  (cluster) and 6.4 % (platform) too low.
- **Operational slack:**
  - Estimated from the attachment paths of free FloatSim runs at H = 0.12 m (the tilt resonance
    and 1.4 s) under the numerical restraint, plus the drift sum's mean offset.
  - Then checked with the moored design in FloatSim (§5).

The sweep. Each row is the most pretension that passes at that height and k. Where none passes,
it is the most that keeps the static tilt and the operational band, and the row shows what fails.
The static tilt is the per-buoy-stiffness estimate, except in the chosen rows (FloatSim's
settle):

| article | attachment z | k | T0 per line/leg | tilt shift (limit) | heave shift | calm static tilt | surge | op. T_min/T0 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| buoy | pin +0.717 m | ×1 | 4.00 N | -9.19 % (1.42) | -0.83 % | 0.00° | 16.4 s | 0.80 | fails: tilt -9.19 % |
| buoy | pin +0.717 m | ×0.5 | 4.00 N | -6.23 % (1.42) | -0.83 % | 0.00° | 22.6 s | 0.90 | fails: tilt -6.23 % |
| buoy | +0.00 m | ×1 | 4.00 N | -4.32 % (1.42) | -0.83 % | 0.00° | 16.4 s | 0.90 | fails: tilt -4.32 % |
| buoy | +0.00 m | ×0.5 | 4.00 N | -3.23 % (1.42) | -0.83 % | 0.00° | 22.6 s | 0.95 | fails: tilt -3.23 % |
| buoy | -0.15 m | ×1 | 4.00 N | -3.43 % (1.42) | -0.83 % | 0.00° | 16.4 s | 0.92 | fails: tilt -3.43 % |
| buoy | -0.15 m | ×0.5 | 1.40 N | -1.40 % (1.42) | -0.29 % | 0.00° | 23.4 s | 0.88 | pass |
| buoy | -0.30 m | ×1 | 1.00 N | -1.23 % (1.42) | -0.21 % | 0.00° | 16.8 s | 0.75 | pass |
| buoy | -0.30 m | ×0.5 | 2.00 N | -1.37 % (1.42) | -0.42 % | 0.00° | 23.2 s | 0.94 | pass |
| buoy | -0.50 m | ×1 | 2.40 N | -1.37 % (1.42) | -0.50 % | 0.00° | 16.6 s | 0.94 | **CHOSEN** |
| buoy | -0.50 m | ×0.5 | 2.40 N | -1.29 % (1.42) | -0.50 % | 0.00° | 23.1 s | 0.96 | pass |
| buoy | -0.91 (CoG) m | ×1 | 2.00 N | -1.24 % (1.42) | -0.42 % | 0.00° | 16.6 s | 0.94 | pass |
| buoy | -0.91 (CoG) m | ×0.5 | 2.40 N | -1.33 % (1.42) | -0.50 % | 0.00° | 23.1 s | 0.96 | pass |
| cluster | pin +0.717 m | ×1 | 9.01 N | -8.83 % (1.19) | -0.49 % | 0.00° | 16.2 s | 0.63 | fails: tilt -8.83 % |
| cluster | pin +0.717 m | ×0.5 | 9.01 N | -4.66 % (1.19) | -0.49 % | 0.00° | 22.7 s | 0.82 | fails: tilt -4.66 % |
| cluster | +0.00 m | ×1 | 1.92 N | -2.42 % (1.19) | -0.10 % | 1.05° | 16.4 s | 0.15 | fails: tilt -2.42 %, static 1.05°, op slack 0.1498 (even the least T0 that keeps the operational band taut) |
| cluster | +0.00 m | ×0.5 | 1.83 N | -1.21 % (1.19) | -0.10 % | 1.00° | 23.1 s | 0.55 | fails: tilt -1.21 % |
| cluster | -0.15 m | ×1 | 1.52 N | -1.48 % (1.19) | -0.08 % | 1.00° | 16.4 s | 0.15 | fails: tilt -1.48 % |
| cluster | -0.15 m | ×0.5 | 1.49 N | -0.73 % (1.19) | -0.08 % | 1.00° | 23.1 s | 0.56 | **CHOSEN** |
| cluster | -0.30 m | ×1 | 1.29 N | -0.76 % (1.19) | -0.07 % | 1.00° | 16.5 s | 0.26 | pass |
| cluster | -0.30 m | ×0.5 | 1.29 N | -0.37 % (1.19) | -0.07 % | 1.00° | 23.2 s | 0.63 | pass |
| cluster | -0.50 m | ×1 | 1.08 N | -0.16 % (1.19) | -0.06 % | 1.00° | 16.5 s | 0.52 | pass |
| cluster | -0.50 m | ×0.5 | 1.08 N | -0.07 % (1.19) | -0.06 % | 1.00° | 23.2 s | 0.66 | pass |
| cluster | -0.91 (CoG) m | ×1 | 0.81 N | -0.26 % (1.19) | -0.05 % | 1.00° | 16.8 s | 0.24 | pass |
| cluster | -0.91 (CoG) m | ×0.5 | 0.81 N | -0.13 % (1.19) | -0.04 % | 1.00° | 23.4 s | 0.58 | pass |
| platform | pin +0.717 m | ×1 | 18.03 N | -9.04 % (1.05) | -0.10 % | 0.00° | 16.1 s | 0.63 | fails: tilt -9.04 % |
| platform | pin +0.717 m | ×0.5 | 18.03 N | -4.89 % (1.05) | -0.10 % | 0.00° | 22.5 s | 0.82 | fails: tilt -4.89 % |
| platform | +0.00 m | ×1 | 3.79 N | -2.42 % (1.05) | -0.05 % | 1.91° | 16.2 s | 0.15 | fails: tilt -2.42 %, static 1.91° (even the least T0 that keeps the operational band taut) |
| platform | +0.00 m | ×0.5 | 1.98 N | -1.19 % (1.05) | -0.03 % | 1.00° | 23.0 s | 0.18 | fails: tilt -1.19 % |
| platform | -0.15 m | ×1 | 3.02 N | -1.48 % (1.05) | -0.04 % | 1.84° | 16.2 s | 0.15 | fails: tilt -1.48 %, static 1.84° (even the least T0 that keeps the operational band taut) |
| platform | -0.15 m | ×0.5 | 1.54 N | -0.72 % (1.05) | -0.03 % | 1.00° | 23.0 s | 0.18 | **CHOSEN** |
| platform | -0.30 m | ×1 | 2.26 N | -0.76 % (1.05) | -0.03 % | 1.61° | 16.3 s | 0.15 | fails: static 1.61° (even the least T0 that keeps the operational band taut) |
| platform | -0.30 m | ×0.5 | 1.40 N | -0.36 % (1.05) | -0.02 % | 1.00° | 23.0 s | 0.34 | pass |
| platform | -0.50 m | ×1 | 1.26 N | -0.16 % (1.05) | -0.02 % | 1.08° | 16.4 s | 0.15 | fails: static 1.08° (even the least T0 that keeps the operational band taut) |
| platform | -0.50 m | ×0.5 | 1.17 N | -0.07 % (1.05) | -0.02 % | 1.00° | 23.1 s | 0.50 | pass |
| platform | -0.91 (CoG) m | ×1 | 1.59 N | -0.33 % (1.05) | -0.03 % | 1.82° | 16.3 s | 0.15 | fails: static 1.82°, op slack 0.1485 (even the least T0 that keeps the operational band taut) |
| platform | -0.91 (CoG) m | ×0.5 | 0.88 N | -0.15 % (1.05) | -0.02 % | 1.00° | 23.3 s | 0.18 | pass |

**Residual interference.**
- The chosen designs still shift the tilt resonance (headline). The shift is within ζ/3 and
  modelled by FloatSim.
- That residual is acceptable for validating FloatSim, because the model includes the lines.
- It **must be declared** in any claim about the free (unmoored) response.

⚠ **The buoy's conflict.** The collar's pretension stiffens pitch in proportion to T0·r, whatever
the collar's depth. The same product sets its yaw stiffness.
- So every buoy design that passes ζ/3 has yaw stiffness ≤ ~2.0 N·m/rad: a yaw period ≥ 1.12 s.
  That is the zone where round 2 found the r = 0.12 m collar parametrically unstable at
  H = 0.5 m (DESIGN-BASIS §C5).
- The rule's choice (most pretension) is the stiffest yaw among the passing designs. FloatSim
  confirms the operational band is stable.
- In part of the extreme band, yaw grows or diverges (§5).
- A pitch-neutral collar (e.g. a cross-flume bar) could decouple the two. It is NOT designed
  here; see *Follow-ups*.

⚠ **The excursion consequence (cluster, platform).**
- The rule's first criterion (most pretension) picks k ×0.5 for both. The lines are then soft
  (surge ~23 s) and the pretension is low, so the down-flume lines go slack in the extremes and
  the articles drift far past the +1.0 m field of view (§5).
- The cluster has a passing alternative with about half the offset: z = −0.30 m, k ×1,
  T0 ≈ 1.3 N (surge stiffness 29.7 against 15.1 N/m, surge 16.5 s). It has less pretension, so
  the rule does not select it.
- The platform has no passing k ×1 design.
- The rule does not weigh excursion. Xabier to decide whether it should.

## 1. Layout and line make-up (all articles)

- An X-spread of four soft lines to the flume walls, 20.1° off the flume axis. The platform's four
  lines are 2-leg bridles (8 legs) to the two row spars of each up- and down-stream half-row.
- Each line or leg is an **elastic shock cord**, with short stiff leads or shackles if needed, of
  unstretched length L0. It is pre-stretched to the pretension T0 when attached: the tension at
  rest ≈ k × (chord − L0).
- **Attachments are below still water, and each anchor is at the same depth**, so every line runs
  level:
  - **single buoy:** a stiff radial collar of radius **r = 0.2 m** on the spar at z = -0.50 m, one
    line at the point facing each anchor. The collar is **submerged**, so it is a hydrodynamic
    appendage that the model does not include:
    - make it slender (e.g. a cross of two 0.4 m bars ≈ 25 mm in diameter): added mass
      ≈ 0.4 kg, drag area ≈ 0.02 m²;
    - a solid disk of r = 0.2 m would add ≈ (8/3)ρr³ ≈ 21 kg of heave added mass (the heave
      plate's is ≈ 7.9 kg): **do not use a disk**;
  - **cluster:** each spar at z = -0.15 m (a clamp or eye on the spar);
  - **platform:** the 8 up- and down-stream row spars at z = -0.15 m.
- Pretension is set by L0. Check it at installation against the at-rest tension and stretch in the
  tables.

## 2. Per-article line tables

The **up-flume lines (anchors at x = −5 m) carry the drift** and set every peak value. The
down-flume lines relax towards T0 and below. **Buy and rig every line to its article's up-flume
values** (the headline) so the lines are interchangeable.

Peak values are over every *predicted* FloatSim run of the design:
- cluster and platform: H = 0.5 m at 2.35 and 2.65 s, and the operational checks;
- buoy: every run that does not diverge (§5).

### 2a. Single buoy

| line | anchor (x, y, z) m | attachment (x, y, z) m | L0 unstretched | k target (band −13 / +30 %) | pretension T0 | at-rest stretch | peak stretch | elongation capacity ×1.25 | max tension | working load ×3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (buoy) | (-5.000, +1.830, -0.500) | (-0.188, +0.069, -0.501) | 3.930 m | 2.01 N/m (1.75–2.61) | 2.40 N | 1.19 m | 1.76 m | 2.21 m (56 % of L0) | 3.54 N | 10.6 N |
| 2 (buoy) | (-5.000, -1.830, -0.500) | (-0.188, -0.069, -0.501) | 3.930 m | 2.01 N/m (1.75–2.61) | 2.40 N | 1.19 m | 1.76 m | 2.21 m (56 % of L0) | 3.54 N | 10.6 N |
| 3 (buoy) | (+5.000, +1.830, -0.500) | (+0.188, +0.069, -0.501) | 3.930 m | 2.01 N/m (1.75–2.61) | 2.40 N | 1.19 m | 1.27 m | 1.58 m (40 % of L0) | 2.54 N | 7.6 N |
| 4 (buoy) | (+5.000, -1.830, -0.500) | (+0.188, -0.069, -0.501) | 3.930 m | 2.01 N/m (1.75–2.61) | 2.40 N | 1.19 m | 1.27 m | 1.58 m (40 % of L0) | 2.54 N | 7.6 N |

| anchor (x, y, z) m | lines | anchor design load (sum of line maxima) | anchor working-load limit ×3 |
|---|---|---|---|
| (-5.000, +1.830, -0.500) | 1 | 3.54 N | **10.6 N** |
| (-5.000, -1.830, -0.500) | 2 | 3.54 N | **10.6 N** |
| (+5.000, +1.830, -0.500) | 3 | 2.54 N | **7.6 N** |
| (+5.000, -1.830, -0.500) | 4 | 2.54 N | **7.6 N** |

### 2b. Cluster (4 buoys at 45°)

| line | anchor (x, y, z) m | attachment (x, y, z) m | L0 unstretched | k target (band −13 / +30 %) | pretension T0 | at-rest stretch | peak stretch | elongation capacity ×1.25 | max tension | working load ×3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (buoy2) | (-5.000, +1.830, -0.150) | (-0.322, +0.303, -0.150) | 4.589 m | 4.15 N/m (3.61–5.40) | 1.49 N | 0.35 m | 3.67 m | 4.59 m (100 % of L0) | 15.23 N | 45.7 N |
| 2 (buoy3) | (-5.000, -1.830, -0.150) | (-0.322, -0.303, -0.150) | 4.589 m | 4.15 N/m (3.61–5.40) | 1.49 N | 0.35 m | 3.67 m | 4.59 m (100 % of L0) | 15.23 N | 45.7 N |
| 3 (buoy1) | (+5.000, +1.830, -0.150) | (+0.322, +0.303, -0.150) | 4.589 m | 4.15 N/m (3.61–5.40) | 1.49 N | 0.35 m | 0.50 m | 0.62 m (14 % of L0) | 2.06 N | 6.2 N |
| 4 (buoy4) | (+5.000, -1.830, -0.150) | (+0.322, -0.303, -0.150) | 4.589 m | 4.15 N/m (3.61–5.40) | 1.49 N | 0.35 m | 0.50 m | 0.62 m (14 % of L0) | 2.06 N | 6.2 N |

| anchor (x, y, z) m | lines | anchor design load (sum of line maxima) | anchor working-load limit ×3 |
|---|---|---|---|
| (-5.000, +1.830, -0.150) | 1 | 15.23 N | **45.7 N** |
| (-5.000, -1.830, -0.150) | 2 | 15.23 N | **45.7 N** |
| (+5.000, +1.830, -0.150) | 3 | 2.06 N | **6.2 N** |
| (+5.000, -1.830, -0.150) | 4 | 2.06 N | **6.2 N** |

### 2c. 4×4 platform (16 buoys at 45°)

| line | anchor (x, y, z) m | attachment (x, y, z) m | L0 unstretched | k target (band −13 / +30 %) | pretension T0 | at-rest stretch | peak stretch | elongation capacity ×1.25 | max tension | working load ×3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (buoy6) | (-5.000, +1.830, -0.150) | (-0.911, +0.890, -0.150) | 4.038 m | 8.32 N/m (7.24–10.81) | 1.54 N | 0.17 m | 3.54 m | 4.42 m (110 % of L0) | 29.43 N | 88.3 N |
| 2 (buoy7) | (-5.000, +1.830, -0.150) | (-0.910, +0.304, -0.150) | 4.208 m | 8.32 N/m (7.24–10.81) | 1.54 N | 0.17 m | 3.47 m | 4.33 m (103 % of L0) | 28.83 N | 86.5 N |
| 3 (buoy10) | (-5.000, -1.830, -0.150) | (-0.910, -0.304, -0.150) | 4.208 m | 8.32 N/m (7.24–10.81) | 1.54 N | 0.17 m | 3.47 m | 4.33 m (103 % of L0) | 28.83 N | 86.5 N |
| 4 (buoy11) | (-5.000, -1.830, -0.150) | (-0.911, -0.890, -0.150) | 4.038 m | 8.32 N/m (7.24–10.81) | 1.54 N | 0.17 m | 3.54 m | 4.42 m (110 % of L0) | 29.43 N | 88.3 N |
| 5 (buoy1) | (+5.000, +1.830, -0.150) | (+0.911, +0.890, -0.150) | 4.038 m | 8.32 N/m (7.24–10.81) | 1.54 N | 0.17 m | 0.31 m | 0.39 m (10 % of L0) | 2.60 N | 7.8 N |
| 6 (buoy4) | (+5.000, +1.830, -0.150) | (+0.910, +0.304, -0.150) | 4.208 m | 8.32 N/m (7.24–10.81) | 1.54 N | 0.17 m | 0.31 m | 0.38 m (9 % of L0) | 2.55 N | 7.7 N |
| 7 (buoy13) | (+5.000, -1.830, -0.150) | (+0.910, -0.304, -0.150) | 4.208 m | 8.32 N/m (7.24–10.81) | 1.54 N | 0.17 m | 0.31 m | 0.38 m (9 % of L0) | 2.55 N | 7.7 N |
| 8 (buoy16) | (+5.000, -1.830, -0.150) | (+0.911, -0.890, -0.150) | 4.038 m | 8.32 N/m (7.24–10.81) | 1.54 N | 0.17 m | 0.31 m | 0.39 m (10 % of L0) | 2.60 N | 7.8 N |

| anchor (x, y, z) m | lines | anchor design load (sum of line maxima) | anchor working-load limit ×3 |
|---|---|---|---|
| (-5.000, +1.830, -0.150) | 1, 2 | 58.26 N | **174.8 N** |
| (-5.000, -1.830, -0.150) | 3, 4 | 58.26 N | **174.8 N** |
| (+5.000, +1.830, -0.150) | 5, 6 | 5.15 N | **15.5 N** |
| (+5.000, -1.830, -0.150) | 7, 8 | 5.15 N | **15.5 N** |

## 3. Hardware

- **Elastic shock cord, usable submerged: recommended for all three articles.**
  - It must hold its secant stiffness within −13 % / +30 % of the target over the working stretch
    (at-rest to peak).
  - Required elongation capacity: see the headline (×1.25 on the FloatSim peak stretch). If the
    cord is shorter than L0 (a stiff lead in series), its strain rises in proportion.
  - Its hysteresis is not modelled. It damps the slow modes and shifts the mean position by half
    the loading/unloading gap, which the pull test measures.
- **Pulley and counterweight (constant tension): rejected** (rev A §3). All its restoring is
  geometric, so it is far too soft in surge.

## 4. Tank validation targets (installed mooring, before any waves)

Check the installed mooring against these FloatSim predictions:
- the static pull: a horizontal pull through the article centre at the attachment depth, force
  against offset;
- free decays from ~0.3 m surge, ~0.1 m sway and ~5° yaw;
- **the calm static tilt** (headline): each buoy's tilt at rest, before and after attaching the
  lines.

The buoy's yaw decay is not predictable in FloatSim. Its yaw period comes from the collar
stiffness and the buoy's yaw inertia.

| | surge period | sway period | yaw period | surge pull K / F at 0.25, 0.5 m | sway pull K / F at 0.1 m | yaw pull K / M at 5° |
|---|---|---|---|---|---|---|
| buoy | 16.6 s | 26.4 s | 1.12 s | 7.31 N/m / 1.83, 3.65 N | 2.60 N/m / 0.260 N | 2.00 N·m/rad / 0.174 N·m |
| cluster | 23.1 s | 51.6 s | 24.78 s | 15.05 N/m / 3.72, 6.36 N | 2.64 N/m / 0.264 N | 3.06 N·m/rad / 0.268 N·m |
| platform | 23.0 s | 59.0 s | 33.40 s | 60.63 N/m / 12.65, 20.66 N | 8.11 N/m / 0.812 N | 28.96 N·m/rad / 2.547 N·m |

The surge and sway periods are the round-1 FloatSim decays scaled by √(K_decay / K) (same
effective mass).

## 5. Predicted envelope

**Cluster and platform** (FloatSim, the moored design, the conservative drift sum):

| article | case (drift sum) | min / max line tension | T_min / T0 | slack lines: peak re-tension | mean offset (range) | max tilt | max yaw | line above the local surface | validity |
|---|---|---|---|---|---|---|---|---|---|
| cluster | extreme: H 0.5 m, T 2.35 s | 0.04 / 15.23 N | 0.03 slack (allowed) | stay slack (≤ 0.05 N, their hanging weight): no re-tension in the wave train | 3.27 m (3.15–3.39) **> +1 m** | 15.3° | 7.0e-01° | breaks the surface by 0.19 m | indicative (tilt > 5.7°) |
| cluster | extreme: H 0.5 m, T 2.65 s | 0.05 / 12.01 N | 0.03 slack (allowed) | stay slack (≤ 0.05 N, their hanging weight): no re-tension in the wave train | 2.53 m (2.46–2.61) **> +1 m** | 12.9° | 4.7e-01° | breaks the surface by 0.18 m | indicative (tilt > 5.7°) |
| cluster | operational: H 0.12 m, T 1.4 s | 1.04 / 1.83 N | 0.73 PASS | none | 0.07 m (0.05–0.09) | 2.5° | 7.9e-04° | submerged (-0.09 m) | LEVEL1 valid |
| cluster | operational: H 0.12 m, T 2.84 s | 0.81 / 2.09 N | 0.56 PASS | none | -0.00 m (-0.04–0.04) | 17.6° | 2.1e-02° | submerged (-0.09 m) | indicative (tilt > 5.7°) |
| platform | extreme: H 0.5 m, T 2.35 s | 0.04 / 29.43 N | 0.03 slack (allowed) | stay slack (≤ 0.04 N, their hanging weight): no re-tension in the wave train | 3.41 m (3.30–3.53) **> +1 m** | 22.7° | 1.8e+00° | breaks the surface by 0.18 m | indicative (tilt > 5.7°) |
| platform | extreme: H 0.5 m, T 2.65 s | 0.04 / 24.70 N | 0.03 slack (allowed) | stay slack (≤ 0.05 N, their hanging weight): no re-tension in the wave train | 2.85 m (2.78–2.93) **> +1 m** | 19.4° | 1.4e+00° | breaks the surface by 0.17 m | indicative (tilt > 5.7°) |
| platform | operational: H 0.12 m, T 1.4 s | 0.87 / 1.99 N | 0.61 PASS | none | 0.06 m (0.05–0.07) | 2.2° | 3.2e-03° | submerged (-0.09 m) | LEVEL1 valid |
| platform | operational: H 0.12 m, T 2.9 s | 0.27 / 2.81 N | 0.19 PASS | none | 0.00 m (-0.05–0.06) | 17.5° | 3.6e-02° | submerged (-0.09 m) | indicative (tilt > 5.7°) |

**Single buoy** (FloatSim, the moored design, the conservative drift sum, dt = 0.0025 s, 210 s).
- *stable*: yaw stays at round-off.
- *yaw grows*: the model grows yaw, still < 1° at the end of the run. The loads hold over the run;
  a longer test grows further.
- *diverges*: no prediction.

| H | T | FloatSim | max tilt | max yaw | T_min / T0 | max tension | slack lines: peak re-tension | surge range |
|---|---|---|---|---|---|---|---|---|
| 0.04 m | 2.75 s | stable | 9.4° | 3.6e-11° | 0.97 | 2.46 N | none | -0.04–0.04 m |
| 0.12 m | 2.45 s | stable | 9.5° | 2.9e-11° | 0.94 | 2.55 N | none | -0.01–0.01 m |
| 0.12 m | 2.55 s | stable | 12.7° | 1.1e-09° | 0.93 | 2.57 N | none | -0.01–0.02 m |
| 0.12 m | 2.65 s | stable | 15.8° | 1.5e-10° | 0.93 | 2.57 N | none | -0.04–0.05 m |
| 0.12 m | 2.75 s | stable | 16.8° | 1.1e-09° | 0.95 | 2.53 N | none | -0.07–0.07 m |
| 0.2 m | 2.35 s | stable | 11.9° | 4.3e-11° | 0.88 | 2.69 N | none | 0.02–0.07 m |
| 0.2 m | 2.65 s | **yaw grows** | 20.9° | 8.2e-05° | 0.88 | 2.71 N | none | -0.01–0.10 m |
| 0.2 m | 2.75 s | **yaw grows** | 22.0° | 2.0e-03° | 0.91 | 2.62 N | none | -0.07–0.11 m |
| 0.2 m | 3.5 s | stable | 5.2° | 1.2e-11° | 0.93 | 2.57 N | none | -0.08–0.13 m |
| 0.35 m | 2.0 s | **yaw grows** | 10.9° | 2.2e-05° | 0.53 | 3.54 N | none | 0.40–0.52 m |
| 0.35 m | 2.35 s | **yaw grows** | 18.0° | 1.3e-05° | 0.60 | 3.38 N | none | 0.29–0.39 m |
| 0.35 m | 2.65 s | **diverges: no prediction** | — | — | — | — | — | — |
| 0.35 m | 3.0 s | **yaw grows** | 23.4° | 2.0e-06° | 0.85 | 2.76 N | none | -0.08–0.30 m |
| 0.35 m | 3.5 s | stable | 9.4° | 4.7e-11° | 0.80 | 2.88 N | none | -0.05–0.31 m |
| 0.5 m | 2.35 s | **diverges: no prediction** | — | — | — | — | — | — |
| 0.5 m | 2.65 s | **diverges: no prediction** | — | — | — | — | — | — |
| 0.5 m | 3.0 s | **yaw grows** | 29.5° | 2.5e-06° | 0.56 | 3.50 N | none | 0.22–0.68 m |
| 0.5 m | 3.5 s | stable | 14.0° | 4.8e-11° | 0.55 | 3.49 N | none | 0.15–0.65 m |

- **Operational band (H ≤ 0.12 m): no slack, for every article** (headline: the kinematic
  estimate and the moored FloatSim runs).
- ⚠ **Cluster and platform yaw in the extremes: 0.47–1.8°** (rev A: 0.04–0.23°).
  - Their heading-0 antisymmetric motion is forced by the 1e-5 asymmetry of the coupled
    radiation kernel (DESIGN-BASIS §C6).
  - Rev B's yaw is 8–11 × softer (yaw period 25–33 s against rev A's 8.8–10.2 s), with the
    down-flume lines slack.
  - Only the peak was kept, so whether the yaw is steady or growing is not established.
  - Three of the four exceed Phase D's proposed 0.5° flag.
- **The buoy's yaw:** the buoy is mirror-symmetric to round-off, so any yaw above round-off is
  the model's own (parametric) growth.
- **Small-angle validity:** FloatSim here is LEVEL1 (small angles, valid to 0.1 rad = 5.7°).
  - Every prediction near the tilt resonance, line loads included, is **indicative** across the
    whole operational band.
  - All the extremes are beyond it. The buoy's divergence is also beyond it: LEVEL2 decides
    whether it is physical.
- **Snap loads are not modelled.** FloatSim's line is quasi-static (no line inertia). The
  "peak re-tension" is the quasi-static tension when a slack line comes taut again.

## 6. Test procedure notes

- **Before waves:** measure each line's at-rest tension (or stretch) against the tables; the calm
  tilt; the pull stiffness and the decay periods against §4.
- **Extreme cases in increasing H** (0.2 → 0.35 → 0.5 m) at each period.
  - Monitor the offset live. Stop a case if the down-flume offset approaches the tracking limit
    (e.g. 0.9 m of the +1.0 m), or a line nears its working load.
  - Slack of the down-flume lines is allowed in the extremes.
  - Do NOT stiffen the lines to fix the field of view.
- **Buoy extremes:** run them LAST, with a height ramp (0.2 → 0.35 → 0.5 m). Watch yaw and roll;
  **stop on any growth of yaw**. The cases FloatSim marks *diverges* have no prediction.
- Heading 0 (waves along the flume axis) for every article. The articles are mirror-symmetric
  about it, so yaw is not wave-driven; a misalignment would drive it.
- Skip cases with H/λ > 0.08, and H = 0.5 m below T = 2.35 s.

## 7. Follow-ups before the test (deferred today)

| Item | What it could change |
|---|---|
| **Static tilt ≤ 1° (proposed criterion)** | Xabier to confirm. A looser limit allows more pretension on the cluster and platform; a tighter one less. |
| **The buoy's yaw vs tilt conflict** (a pitch-neutral collar, e.g. a cross-flume bar) | Could restore the buoy's H ≥ 0.35 m extremes near resonance without the tilt shift. Needs a design and a FloatSim check. |
| **BEM regeneration at NT = 36** (flume databases' 12-sided waterline; ~1.7 h platform BEM) | All heave/tilt resonances ~0.04–0.08 s shorter; the fine period band moves. The ζ/3 comparison holds (it is relative to the unmoored article). |
| **Full Phase D** (442 cases) | The criteria across the whole matrix, with rev B's lines. |
| **LEVEL2 and the resonant-subset rerun** | Resonant tilts, line loads and offsets, all indicative today. It also decides the buoy's divergence. |
| **Drift mechanism** | The sum used today adds FloatSim's own mean force to the recorded fixed-body splash-zone bound. A moving-body splash-zone model could lower the offsets and tensions. |
| **Submerged collar and line drag** | Not modelled. The slender collar's added mass (≈ 0.4 kg) and drag are small; line drag in the extremes, where the lines break the surface, is not included. |
| **HWRL confirmations F1–F7** | Any of them can force a redesign (page 1). |
