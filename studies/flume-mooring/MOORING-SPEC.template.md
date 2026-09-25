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
| F1 | Anchors can be fixed on the flume walls **under water, at each article's attachment depth** | anchors at x = ±5.0 m, y = ±1.83 m, **z = {{ZB}} m (buoy), {{ZC}} m (cluster), {{ZP}} m (platform)** | A higher or lower anchor slopes the line. Its vertical pull at the attachment adds calm tilt and tilt coupling, so the sweep must be re-run for the real anchor depth. (Rev A's question, anchors 0.72 m *above* still water, no longer arises.) |
| F2 | Wall anchor rating | per-anchor working-load limits (×3 on the FloatSim peak): **{{WLL}}** | Lower than these is unlikely to bind; confirm. |
| F3 | Wave height per period, steepness | H/λ ≤ 0.08 at 2.7 m depth: H 0.5 m only for T ≥ 2.01 s (and from 2.35 s by the 3° tilt rule), H 0.35 m for T ≥ 1.675 s | Cases outside the wavemaker envelope drop out. |
| F4 | Water depth | 2.7 m | The drift bound falls for deeper water; periods unchanged. |
| F5 | Test-section position vs the wavemaker and beach | a clean regular-wave window ≥ 2–3 min at the article, which sits on the flume centreline | Shorter windows do not settle the lightly damped resonances (ζ ≈ 3–4 %). |
| F6 | Tracking field of view | surge −0.3 … **+1.0 m** (down-flume), sway ±0.1 m, plus the article footprint (platform 2.06 m wide at 45°) | Rev B's lines are softer: in the extremes the articles pass +1.0 m (see §5). Those cases must stop earlier (see *Test procedure*). |
| F7 | **Standard soft-mooring hardware, usable submerged** | elastic shock cord: stiffness and elongation per the tables below (secant stiffness within −13 % / +30 % of target over the working stretch), submerged weight ≈ 0.02 N/m | Without it, the lines need farther anchors (longer lines, lower strain) or another soft element. Redesign. |

## Headline

{{HEADLINE}}

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
  stiffness, the tilt period shifts **{{PINSHIFT}}** (the multiple of the tolerance).
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
   ({{ZETA}}), so the limits are {{LIMS}}. A detuning δ scales the resonant response by
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

{{SWEEP}}

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
  - **single buoy:** a stiff radial collar of radius **r = 0.2 m** on the spar at z = {{ZB}} m, one
    line at the point facing each anchor. The collar is **submerged**, so it is a hydrodynamic
    appendage that the model does not include:
    - make it slender (e.g. a cross of two 0.4 m bars ≈ 25 mm in diameter): added mass
      ≈ 0.4 kg, drag area ≈ 0.02 m²;
    - a solid disk of r = 0.2 m would add ≈ (8/3)ρr³ ≈ 21 kg of heave added mass (the heave
      plate's is ≈ 7.9 kg): **do not use a disk**;
  - **cluster:** each spar at z = {{ZC}} m (a clamp or eye on the spar);
  - **platform:** the 8 up- and down-stream row spars at z = {{ZP}} m.
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

{{BUOY}}

### 2b. Cluster (4 buoys at 45°)

{{CLUSTER}}

### 2c. 4×4 platform (16 buoys at 45°)

{{PLATFORM}}

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

{{TANK}}

The surge and sway periods are the round-1 FloatSim decays scaled by √(K_decay / K) (same
effective mass).

## 5. Predicted envelope

**Cluster and platform** (FloatSim, the moored design, the conservative drift sum):

{{ENVELOPE}}

**Single buoy** (FloatSim, the moored design, the conservative drift sum, dt = 0.0025 s, 210 s).
- *stable*: yaw stays at round-off.
- *yaw grows*: the model grows yaw, still < 1° at the end of the run. The loads hold over the run;
  a longer test grows further.
- *diverges*: no prediction.

{{BUOYRUNS}}

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
