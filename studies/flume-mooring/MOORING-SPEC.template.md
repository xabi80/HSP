# Flume mooring specification — OSU Large Wave Flume (HWRL), 1:50 test articles

**CONSERVATIVE SPECIFICATION, 2026-09-25.** It was issued ahead of full verification: Phase D (the
full FloatSim case matrix) follows before the test (see *Follow-ups*). Built by `build_spec.py`
from FloatSim results (`spec_statics.json`, `spec_extremes.json`, `spec_extreme_buoy.json`,
`line_hardware.json`). Basis and derivations: `DESIGN-BASIS.md`.

## Page 1 — Facility assumptions to CONFIRM WITH HWRL BEFORE BUYING HARDWARE

None of these is a confirmed facility fact. The design rests on all of them.

| # | Assumption | Value this specification uses | If it proves false |
|---|---|---|---|
| F1 | Anchors can be fixed on the flume walls **0.72 m above still water** (the pin plane) | anchors at x = ±5.0 m, y = ±1.83 m, **z = +0.717 m** | The lines slope into the splash zone and the attachment must drop towards the waterline, where the pretension tilts the pinned buoys (calm-water tilt 4.9° cluster, 9.1° platform). Redesign. |
| F2 | Wall anchor rating | per-anchor working-load limits in the tables below: **311 N per platform anchor** (cluster 80 N, buoy 20 N) | Lower pretension or softer lines, with more offset. Redesign. |
| F3 | Wave height per period, steepness | H/λ ≤ 0.08 at 2.7 m depth: H 0.5 m only for T ≥ 2.01 s (and from 2.35 s by the 3° tilt rule), H 0.35 m for T ≥ 1.675 s | Cases outside the wavemaker envelope drop out. |
| F4 | Water depth | 2.7 m | The drift bound falls for deeper water; periods unchanged. |
| F5 | Test-section position vs the wavemaker and beach | a clean regular-wave window ≥ 2–3 min at the article, which sits on the flume centreline | Shorter windows do not settle the lightly damped resonances (ζ ≈ 3–4 %). |
| F6 | Tracking field of view | surge −0.3 … **+1.0 m** (down-flume), sway ±0.1 m, plus the article footprint (platform 2.06 m wide at 45°) | The extreme cases must stop earlier (see *Test procedure*). |
| F7 | **Standard soft-mooring hardware** | elastic shock cord: stiffness and elongation per the tables below (secant stiffness within −13 % / +30 % of target over the working stretch), dry weight ≈ 0.3 N/m | Without it, the lines need farther anchors (longer lines, lower strain) or another soft element. Redesign. |

## Headline

{{HEADLINE}}

Frame for all coordinates (the TRUE flume frame): origin at the article's calm centre on the flume
centreline at still water; x down-flume (the wave direction), y across, z up. Anchors and
attachments are at z = +0.717 m, the pin plane.

What is conservative in these numbers:
- the **drift sum**: the full recorded fixed-body drift bound, applied in-run at each spar's
  waterline, plus FloatSim's own mean wave force on the moving body;
- **{{SCALE}} pretension** on the cluster and platform, over the Phase C design (line stiffness k
  kept);
- **working loads ×3** and **elongation capacity ×1.25** on the worst-case FloatSim values;
- the worst cases are H = 0.5 m at T = 2.35 s (the largest admissible drift) and 2.65 s (the
  tilt/heave resonance).

## 1. Layout and line make-up (all articles)

- An X-spread of four soft lines to the flume walls, 20.1° off the flume axis. The platform's four
  lines are 2-leg bridles, 8 legs, to the two pins of each up- and down-stream half-row.
- Each line or leg is an **elastic shock cord**, with short stiff leads/shackles if needed, of
  unstretched length L0. It is pre-stretched to the pretension T0 when attached: tension at rest
  ≈ k × (chord − L0).
- Attachments are at the pin plane (+0.717 m):
  - **single buoy:** a stiff radial collar (crossbar) of radius **r = 0.2 m** on the spar top, one
    line at the point facing each anchor. Yaw period 0.86 s, below the wave band. It holds
    heading for tracking;
  - **cluster:** each spar's pin;
  - **platform:** the pins of the 8 up- and down-stream row spars.
- Pretension is set by L0. Check it at installation against the at-rest tension and stretch in the
  tables.

## 2. Per-article line tables

The **up-flume lines (anchors at x = −5 m) carry the drift** and set every peak value. The
down-flume lines relax towards T0 and below. **Buy and rig every line to its article's
up-flume values** (the headline) so the lines are interchangeable.

### 2a. Single buoy

{{BUOY}}

### 2b. Cluster (4 buoys at 45°)

{{CLUSTER}}

### 2c. 4×4 platform (16 buoys at 45°)

{{PLATFORM}}

Anchor working-load limits exceed the 250 N flag only if the platform row says so. **F2 must be
confirmed for those values.**

## 3. Hardware

- **Elastic shock cord: recommended for all three articles.**
  - It must hold its secant stiffness within −13 % / +30 % of the target over the working stretch
    (at-rest to peak).
  - Below −13 %, the extreme offset leaves the field of view. Above +30 %, the surge period falls
    to ~14 s and the slack margin vanishes.
  - Required elongation capacity: see the headline (×1.25 on the FloatSim peak stretch). If the
    cord is shorter than L0 (a stiff lead in series), its strain rises in proportion.
  - Its hysteresis is not modelled. It damps the slow modes and shifts the mean position by half
    the loading/unloading gap, which the pull test measures.
- **Pulley + counterweight (constant tension): rejected.** All its restoring is geometric, about
  20 × softer in surge (surge periods 75–108 s). Under the drift it moves the articles past the
  +1.0 m tracking limit from H = 0.12 m (cluster, platform) or 0.2 m (buoy), and 3–5 m in the
  extreme band. FloatSim cannot model it; the screen is in closed form (`line_hardware.py`).

## 4. Tank validation targets (installed mooring, before any waves)

Check the installed mooring against these FloatSim predictions:
- the static pull (a horizontal pull through the article centre at the pin plane, force against
  offset);
- free decays from ~0.3 m surge, ~0.1 m sway, ~5° yaw.

The buoy's yaw decay is not predictable in FloatSim; its yaw period comes from the collar
stiffness and the buoy's yaw inertia.

{{TANK}}

The surge and sway periods include the coupling to the buoys' tilt. Decay damping (FloatSim, round
1) is ζ ≈ 8 → 4 % per cycle in surge and 5 → 3 % in sway.

## 5. Predicted envelope

**Pretension against slack** (worst cases, FloatSim, drift sum; pass: T_min ≥ 0.15 T0):

{{SCAN}}

**Worst cases at the specified pretension** (FloatSim, the conservative drift sum; buoy at
T0 = 4.0 N on the collar; cluster and platform at {{SCALE}} T0):

{{ENVELOPE}}

- **Operational band (H ≤ 0.12 m).**
  - Mean offset ≤ ~0.03 m at the drift bound (static), and line tensions stay within ±10 % of T0
    statically.
  - At the tilt/pitch resonance the buoys tilt **9.2–9.6° at H = 0.04 m** and **~16–17° at
    H = 0.12 m** (buoy, cluster; FloatSim sweeps).
- **Small-angle validity:** FloatSim here is LEVEL1 (small angles, valid to 0.1 rad = 5.7°). Every
  prediction near the pitch/tilt resonance, line loads included, is **indicative across the whole
  operational band**, and all the worst cases above are beyond it. LEVEL2 is required for resonant
  predictions (follow-up).
- **Not predicted:** the single buoy's extremes near its moored pitch resonance (H = 0.5 m at
  T = 2.35 s; untested at 2.55 s). FloatSim diverges there at 28° pitch. No prediction until
  LEVEL2; see *Test procedure*.

## 6. Test procedure notes

- **Before waves:** measure each line's at-rest tension (or stretch) against the tables; the pull
  stiffness and decay periods against §4.
- **Extreme cases in increasing H** (0.2 → 0.35 → 0.5 m) at each period.
  - Monitor offset live. Stop a case if the down-flume offset approaches the tracking limit
    (e.g. 0.9 m of the +1.0 m), or a line nears slack or its working load.
  - Do NOT stiffen the lines to fix the field of view.
- **Dropped buoy extremes** (no FloatSim prediction until LEVEL2): run them LAST, with a height
  ramp (0.2 → 0.35 → 0.5 m). Watch roll, yaw and line tension; stop on any growth of roll or yaw.
- Heading 0 (waves along the flume axis) for every article. The articles are mirror-symmetric
  about it, so yaw is not wave-driven; a misalignment would drive it.
- Skip cases with H/λ > 0.08, and H = 0.5 m below T = 2.35 s.

## 7. Follow-ups before the test (deferred today)

| Item | What it could change |
|---|---|
| **BEM regeneration at NT = 36** (flume databases' 12-sided waterline; ~1.7 h platform BEM) | All heave/tilt resonances ~0.04–0.08 s shorter; the fine period band moves. Envelope loads change little (the peak height is damping-controlled). |
| **Full Phase D** (442 cases) | The criteria across the whole matrix, incl. operational-band dynamic tensions and offsets, and the criterion-6 interference check. Could flag cases, or revise T0 or anchor loads. |
| **LEVEL2 and the resonant-subset rerun** | Resonant tilts, line loads and offsets, all indicative today; the buoy's dropped extremes get a prediction. Could raise or lower the peak loads. |
| **Drift mechanism** | The sum used today adds FloatSim's own mean force (moving-body submerged drag, 20–70 % of the bound at H = 0.5 m) to the recorded fixed-body splash-zone bound. The mechanisms are distinct (FloatSim has no splash zone), so this is not the same force twice; it is conservative because the bound assumes a fixed body. A moving-body splash-zone model could lower the offsets and tensions. |
| **Static-solver false convergence** (tracker) | None for these results: every static solve is residual-checked. |
| **HWRL confirmations F1–F7** | Any of them can force a redesign (page 1). |
