# Does the operational mooring need pretension? — evaluation (2026-09-25)

**Evaluation only, for Xabier to decide. MOORING-SPEC rev C is NOT changed.**
- Built by `pretension_study.py` (FloatSim throughout: the moored decks, settles, modal analysis,
  rigid pulls and moored wave runs with the drift sum applied in-run).
- Every number below comes from `pretension_study.json`.

## Bottom line

**Recommendation: keep rev C's operational pretension (V0).** Xabier decides.

The operational cords do not NEED their pretension to measure the response:
- **The response is unchanged.** At the "taut minimum" V1, the wave-frequency response is within
  1.2 % (heave RAO) and 0.7° (max tilt) of V0, and the surge amplitude is unchanged.
- **The calm tilt falls** from 1.00° to 0.25° (cluster) / 0.33° (platform).
- **The tilt interference falls** from −0.73 / −0.72 % to −0.54 / −0.60 %.

But the gain is small, and it buys three costs:

1. **Cord cycling.**
   - V0's lines stay stretched in every operational case run (minimum 0.27 N).
   - V1's lines lose their stretch for 18–40 % of each cycle at the tilt resonance.
   - The down-flume lines hang loose for 82–100 % of the time under the H = 0.12 m drift.
   - A loose cord in waves is outside what FloatSim models (its line is quasi-static). Tank and
     model then see different moorings in exactly the resonant cases the tests are for.
2. **Fouling risk.** Quasi-statically, the loose cords droop 0.30–0.46 m but stay ≥ 0.47 m from
   any other spar and ≥ 1.14 m from any heave plate. How a near-neutral cord moves in the
   orbital flow is NOT modelled: that is a question for the installer.
3. **Position repeatability and installation control.**
   - The calm position is held by 0.36 / 0.45 N instead of 1.44 / 1.42 N.
   - Surge stiffness drops 26 % / 18 %, so the H = 0.12 m drift offsets grow ×1.49 / ×1.44.
   - The calm-tilt pretension gauge shrinks from 1° to 0.25–0.33°, too small to verify the
     tension.
   - The cluster's extreme set, re-tensioned to V1's tension, would leave the tracking window
     (1.054 m at H = 0.5 m, T = 2.35 s, FloatSim) and need re-sizing.

**The trade:** calm tilt (1.0° → 0.3°) and a 0.1–0.2 % smaller tilt shift, against cord cycling,
fouling risk, and weaker position repeatability and installation control.
- V0 already meets ζ/3, and its 1° calm tilt is confirmed acceptable. So the gain does not pay
  for the loss of taut, modelled lines.
- **The buoy keeps its pretension regardless** (see *Buoy*).

## 1. Line weight (checked first)

- **The record uses 0.02 N/m for the submerged cords, not the dry 0.3 N/m.** No correction is
  needed.
  - `floatsim_decks.W_LINE` = 0.02 N/m is used for every attachment below still water.
  - The 0.3 N/m dry weight was only for the rev A lines in air.
- **Itemised:**
  - 0.02 N/m is 6.7 % of the dry 0.3 N/m, which implies a cord density of 1.07 g/cm³.
  - A 1.1–1.2 g/cm³ cord gives 9–17 % (0.028–0.050 N/m): not quite "10–20 %".
  - A natural-rubber core alone (≈ 0.93 g/cm³) floats; the braided sheath makes the cord
    slightly negative.
- **Effect on rev C:** with the heaviest cord, the at-rest sag grows ×2.5 and the tension changes
  by ≤ 1.8 %. No material effect:

{{WEIGHT}}

- **At low pretension the weight matters.** V1's calm sag is 170 mm (cluster) / 98 mm (platform)
  with the record's 0.02 N/m, 307 / 190 mm with 0.05 N/m, and ~1 mm with a neutral cord.

## 2. The physics claims, checked (FloatSim)

1. **"Taut at zero tension: linear in each direction at ~k (one pair) instead of ~2k" —
   partly.**
   - At T0 = 0 the surge stiffness is symmetric (+/−), but it is **34 % of V0, not 50 %**
     (cluster 5.1 vs 15.0 N/m; platform 20.5 vs 60.6 N/m).
   - One pair engaging accounts for half. The rest is catenary compliance: at near-zero tension
     the submerged weight sags the soft cord, and the cord must straighten before it stretches.
   - **Sway drops more, to 26 % / 28 %**, because the pretension's geometric sway stiffness
     (T/L) is lost too.
   - So the surge period grows ×1.7 and the sway period ×1.9, not ×√2.
2. **"Drift offsets ~2×" — ×1.5 at V1.** V1 keeps both pairs engaged for small offsets:
   0.067 → 0.100 m (cluster) and 0.061 → 0.088 m (platform) at H = 0.12 m, T = 1.4 s.
3. **"Removes the calm static tilt" — yes.** 0.25° / 0.33° at V1, and 0 without tension.
4. **"Tilt interference nearly unchanged, because k governs it" — no. It falls with T0:**
   - cluster −0.73 → −0.54 (V1) → −0.23 % (T0 = 0);
   - platform −0.72 → −0.60 → −0.23 %.

   The pretension is part of the coupling. At low tension the catenary compliance softens the
   line, and at zero tension only one pair engages per half-cycle. That is a benefit of lower
   pretension, but a small one.
5. **"Exactly taut at zero tension" is not zero tension with a submerged-weight cord.**
   - The weight sags the soft cord and stretches it to ~0.2 N at rest (T0 = 0: 0.21 / 0.22 N;
     V2: 0.20 / 0.20 N).
   - Only a neutral cord is force-free.
   - For the same reason, V2's kinematic dead band is not force-free with the record's cord (see
     §3).

## 3. Variants

- **V1 = "taut minimum": T0 = k × (tolerance + creep).**
  - Tolerance: ±10 mm (Xabier).
  - Creep: natural rubber creeps ~2.4–4 % of its deflection per decade of time in tension, ~4 %
    when wet (Gent, *Engineering with Rubber*, ch. 7; "Long-time creep in a pure-gum rubber
    vulcanizate", PMC6728486).
  - A test day (1 min → 1 day) is 3.2 decades, so ~13 % of the loaded stretch. The loaded
    stretch is taken conservatively as V0's operational peak (0.50 m cluster, 0.34 m platform).
  - Result: **V1 = 0.306 N (cluster) / 0.439 N per leg (platform) nominal**, 0.357 / 0.450 N at
    rest.
  - With the dry creep rate (2.4 %/decade): 0.20 / 0.30 N.
  - The same relaxation applies to V0: its at-rest tension falls ~13 % over a test day
    (1.44 → ~1.25 N), and its calm tilt with it (1.00 → ~0.87°). **Flag for the rev C
    installation check:** allow for it, or re-check daily. No spec change made here.
- **T0 = 0:** exactly taut (L0 = chord); statics only.
- **V2:** installed 10 mm slack (L0 = chord + 10 mm); statics only.
  - **Dead band** (every line kinematically slack): **21 / 20 mm in surge and 64 / 57 mm in
    sway** (cluster / platform), about ±10 mm / ±30 mm. A neutral cord gives no restoring force
    in it.
  - With the record's 0.02 N/m cord, the weight-sagged lines still restore inside it: 4.5 /
    16.1 N/m in surge.

### 3a. Statics: cluster

{{STAT_CLUSTER}}

### 3b. Statics: 4×4 platform

{{STAT_PLATFORM}}

- Every pull is symmetric (+ equals −): with the article centred, each direction engages its
  mirror pair.
- "Small" is ±2 mm; "secant" is ±50 mm.
- Periods are the round-1 FloatSim decays scaled by √(K_decay / K).

## 4. Operational runs (FloatSim; V0 and V1)

- **Cases:** H = 0.04 and 0.12 m, at the tilt resonance (2.84 s cluster, 2.9 s platform) and at
  1.4 s, with the drift sum in-run.
- **V0 was re-run** at the same cases, because the rev C runs had no H = 0.04 m and no heave RAO.
- **"Slack"** means no elastic stretch along the chord (chord ≤ L0). The cord then hangs on its
  own weight, which is why FloatSim's minimum tension stays at 0.10–0.18 N rather than zero.
- **Surge amplitude** is the article-mean surge. **Heave RAO** is the hub (cluster) / deck
  (platform) heave amplitude over the wave amplitude.

{{RUNS}}

- **The wave-frequency response is unchanged by the cycling**:
  - heave RAO within −0.4 to −1.2 %;
  - surge amplitude equal to 1 mm;
  - max tilt 0.5–0.7° lower in V1.
- **Fouling geometry** (quasi-static catenary at each line's slackest instant, own spar
  excluded):
  - the loose V1 cords droop 0.30–0.46 m below their chord;
  - they stay ≥ 0.63 m (cluster) / ≥ 0.47 m (platform) from any other spar, and ≥ 1.14 m from
    any heave plate.
  - **Dynamic cord motion is not modelled.** A near-neutral cord that has lost its stretch for
    a third of every resonant cycle moves with the water. Whether it can reach a spar, a heave
    plate or a neighbouring line is a question for the installer, best answered with a cord in
    the tank.

## 5. Extreme set at V1's at-rest tension

The extreme set would match the operational at-rest tension. A statics estimate (FloatSim rigid
pulls) of its H = 0.5 m max surge, and FloatSim runs where it moves by more than 5 %:

{{EXTREME}}

- **The cluster's extreme set leaves the tracking window at V1's tension** (1.054 m at 2.35 s).
  It would need re-sizing (a stiffer cord, about ×5.5).
- The platform's stays within it (estimated 0.972 m).

## 6. Buoy (statics)

Its collar turns pretension into yaw stiffness (K_yaw ≈ 4·T0·r, plus a small line-geometry
part) and into part of its tilt shift. Rev C: 2.40 N, yaw period 1.12 s.

{{BUOY}}

- **The buoy keeps its pretension (Xabier's expectation, verified):**
  - Below ~2.0 N its yaw period moves into the T_p/2 parametric zone (1.22–1.37 s).
  - Below ~1.5 N it enters the wave band.
  - At V1 (0.468 N) it is 2.48 s: in the wave band and at the principal zone T_p ≈ 2.75 s.
  - The tilt-shift gain (−1.37 → −0.38 %) does not pay for that.
- **Itemised:** the zones Xabier quoted (2.55 / 5.1 / 1.28 s) are round 2's, for rev A's buoy
  (pitch 2.55 s). Rev C's moored pitch period is 2.72–2.75 s, so the zones are ≈ 2.75 / 5.5 /
  1.37 s. Each row above uses its own pitch period, with a ±15 % flag.

## 7. Disagreements, itemised (the record wins)

| Xabier's reasoning | The record |
|---|---|
| Zero pretension halves surge/sway stiffness (~k vs ~2k) | Surge 34 %, sway 26–28 % of V0: catenary compliance of the weight-sagged cord, and the lost geometric (T/L) stiffness |
| Periods ~√2 longer, drift offsets ~2× | T0 = 0: surge ×1.7, sway ×1.9. V1: offsets ×1.44–1.49 |
| Tilt interference nearly unchanged (k governs it, not T0) | It falls with T0: −0.73 → −0.54 → −0.23 % (cluster) |
| Lines exactly taut at zero tension | With a submerged-weight cord they self-tension to ~0.2 N through their sag. Only a neutral cord is force-free |
| Submerged weight roughly 10–20 % of dry | 9–17 % for 1.1–1.2 g/cm³. The record uses 0.02 N/m (6.7 %, ≈ 1.07 g/cm³), already submerged: no correction |
| Buoy parametric zones 2.55 / 5.1 / 1.28 s | ≈ 2.75 / 5.5 / 1.37 s for rev C's buoy (pitch 2.72–2.75 s) |
| (not raised) | Wet creep (~13 %/day) also relaxes V0's at-rest tension and calm tilt: 1.00 → ~0.87° over a test day |
