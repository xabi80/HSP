# Review of the HSFP 1:50 wave test matrix (9-26-2026) against the flume mooring study

Reviewed: `HSFP_1-50_Wave_Test_Matrix_9-26-2026.xlsx`, kept unchanged in this folder.

- The copy with FloatSim's pre-test predictions is `*_FloatSim.xlsx`:
  - Regular_Waves rows 24–26;
  - the new FloatSim_Predictions sheet;
  - it is built by `fill_predictions.py`.
- Every number below comes from the study's records (MOORING-SPEC rev C.1, DESIGN-BASIS,
  `viewer_rows_revc/`, the JSON files) or from the matrix itself.
- The standing rule applies: where the matrix and the record disagree, it is itemised here.

## Overall opinion

The matrix is well built:
- Inputs drive formulas, so it recalculates.
- It flags the facility limits and the flume's cross modes.
- Its sea states come from the site (NDBC 46005).
- Its run and tank-time accounting is explicit.
- It sequences each configuration as decay → regular → irregular → extremes.

It is a good planning tool. Its weak points are where it meets the models and the mooring:
1. One configuration cannot be run as specified.
2. The resonances are under-resolved.
3. Constant steepness puts the largest waves exactly at the resonances.
4. The mooring is verified for only part of the height range and not for irregular seas.
5. Several tests that a validation campaign needs are missing.

## 1. Blocking

**1.1 The Directional Wave Basin is shallower than the buoy's draft.**
- P3-E and P3-F (72 runs) are specified at the basin's maximum depth, 1.36 m.
- **Confirmed on OSU's facility page** (engineering.oregonstate.edu/research/wave-lab/directional-wave-basin, checked 2026-09-26): max water depth 1.36 m (4.46 ft), basin height 2.1 m.
- The OSU test buoy draws ≈ 1.40 m (CAD: the ballast frame bottom is 0.433 m below the pipe,
  which is 0.967 m deep).
- The model cannot float there. Options:
  - drop the basin tests;
  - find a deeper basin;
  - test oblique headings in the flume only where the width allows (single buoy, cluster; the
    platform's diagonal does not fit);
  - accept a shallower-draft variant, which is not representative.

**1.2 The mooring philosophy.**
- The matrix asks for "a truncated mooring with equivalent horizontal stiffness" (emulating the
  2,500 m site's taut mooring).
- The design basis's confirmed intent (2026-09-24) is a soft restraint that disturbs the
  measured motion as little as possible. MOORING-SPEC rev C.1 implements that.
- Opinion: keep the soft restraint for the response and validation tests.
  - An equivalent-stiffness mooring needs a full-scale mooring design, which does not exist yet.
  - Its stiffness would enter the measured response, which is what the validation must avoid.
  - Add an equivalent-mooring configuration later, when the prototype mooring is defined.
- **Decision for Xabier.**

## 2. Resonances: coverage, resolution and wave height

FloatSim pre-test predictions, converged mesh (the FloatSim_Predictions sheet):

| | heave, moored | tilt/pitch, moored (operational cords) | tilt ζ | tilt half-power band |
|---|---|---|---|---|
| single buoy | 2.51 s model / 17.7 s FS | 2.66 s / 18.8 s | 4.2 % | 0.22 s model |
| cluster | 2.54 s / 18.0 s | 2.77 s / 19.6 s | 3.6 % | 0.19 s |
| platform | 2.56 s / 18.1 s | 2.84 s / 20.1 s | 3.2 % | 0.17 s |

**2.1 The sweep stops at the platform's tilt resonance.**
- The matrix's upper bound is 20 s FS (2.83 s model). The platform's tilt resonance is ≈ 20.1 s
  FS, and its upper flank is not covered.
- **Extend the regular sweep to ≈ 22 s FS (3.1 s model).** Row 26 now carries 20.6 s FS, the
  platform peak + 0.5 s.

**2.2 The 1 s FS grid misses the tilt peaks.**
- The tilt resonances are sharp: ζ 3–4 %, half-power bands 0.17–0.22 s model.
- Sampling within 95 % of a peak needs a step ≤ 0.33 × the band ≈ 0.05 s model (0.35 s FS).
  The matrix steps 1 s FS (0.14 s model).
- **Add a fine band per configuration:** the predicted centre ± 0.20 s model in 0.05 s steps,
  9 periods each:
  - buoy 17.4–20.2 s FS;
  - cluster 18.2–21.0 s FS;
  - platform 18.7–21.5 s FS.
- Re-centre on the measured free decay before the sweep. The periods are listed on the
  FloatSim_Predictions sheet.
- The heave peak is broad (ζ ≈ 11 %, band ≈ 0.6 s model), and the 1 s grid already resolves it.
- In waves the heave RAO peaks later than its decay period, at ~18.5–19 s FS (FloatSim sweep).

**2.3 Constant steepness puts the biggest waves at the resonances.**
- At the resonance periods, the "linear" level s1 is already 0.17–0.22 m model; s2 is
  0.33–0.44 m and s3 0.67–0.88 m.
- FloatSim's resonant tilts: ≈ 9.5° at H = 0.04 m and ≈ 17° at H = 0.12 m (`viewer_rows_revc/`,
  MOORING-SPEC §6). Both are already beyond FloatSim's small-angle validity (5.7°).
- **At resonance, s1 is not a linear level.**
- **In the resonance band, add constant small heights:** 0.02 and 0.04 m model (1 and 2 m FS),
  with 0.08 / 0.12 m as the linearity check. Keep the steepness levels outside the band.
- H = 0.02 m would give ≈ 5° of resonant tilt, by scaling the 0.04 m result (drag makes it
  somewhat larger). That is the cleanest validation data FloatSim can get.

## 3. The mooring's verified envelope vs the matrix's heights

The mooring is verified in two places only:
- the operational cords, for H ≤ 0.12 m;
- the extreme cords, at H = 0.5 m and T = 2.35 / 2.65 s model.

Against that:
- **Heights of 0.12–0.5 m are unverified for both sets.** The matrix's s1 exceeds 0.12 m from
  16 s FS; s2 from 11 s FS; s3 from 8 s FS.
- **Heights above 0.5 m are outside the design.**
  - s3 reaches 0.53–0.83 m for T ≥ 16 s FS.
  - The extreme irregular seas give single waves up to ≈ 0.56 m.
  - The extreme set, the anchors (≥ 300 N; the platform's working load is already 267.4 N at
    H = 0.5 m) and the tracking window (max surge 0.95–0.97 m at H = 0.5 m) were all sized at
    H = 0.5 m.
  - FloatSim's H = 0.5 m tilts are already 22–33°, well past its validity.
- **Recommendation:**
  - cap the regular waves at H = 0.5 m model (25 m FS);
  - put survival above that into the focused-wave and irregular-extreme runs;
  - or re-size the extreme set and the anchors for 0.83 m (a Phase D-type evaluation).

## 4. Irregular seas (105 runs)

- **Slow drift is not modelled.**
  - Wave groups drive the moored slow modes: surge 16.6–23.1 s model on the operational cords
    (≈ 117–164 s FS), 9.6–10.5 s on the extreme cords (≈ 68–74 s FS).
  - FloatSim has no second-order (difference-frequency) forces. The design basis recorded this
    risk when the matrix was regular-waves-only.
  - The offsets and line loads of the irregular runs, especially D1–X100 on either cord set,
    have no prediction.
  - A second-order slow-drift estimate is needed before those runs.
  - Run the sea states with Hs ≥ 0.12 m model on the extreme cords.
- **Run length vs creep.** A 3 h FS sea state is ≈ 26 min of model time. The platform's
  operational cords hold their no-slack margin for only ~14 min after re-tensioning
  (MOORING-SPEC §5). Near-resonance irregular runs on that set need a re-tension before each run,
  or an accepted, smaller margin.

## 5. Tests that the matrix does not include

1. **Mass properties and hydrostatics, before any wave** (DESIGN-BASIS Phase I):
   - weigh each buoy with the lead, and mark the floating waterline;
   - CoG height;
   - pitch inertia (swing test in air);
   - an inclining test in water for C55.

   The model's mass, CoG and inertia are parametric or assumed, and they set the pitch
   resonance to ±3 %.
2. **Mooring checks** (MOORING-SPEC §5 and §7):
   - the static pull at installation, and **before every extreme series** (it is the only way to
     tell the two cord sets apart);
   - the daily calm-tilt / tension re-check (creep);
   - the cord-set swap between the operational and extreme series.

   Each baseline configuration becomes two mooring states, with their time.
3. **Longer and more complete decays.**
   - The matrix's decay run is 2 min, with heave, pitch and surge.
   - The moored sway and yaw periods are up to 59 s and 33 s model. Five cycles need up to
     ≈ 5 min (the FloatSim_Predictions sheet, table 2).
   - Add sway and yaw decays.
   - Release at 2–3 amplitudes: the damping depends on amplitude. The field video gave ζ ≈ 12–13 %
     at a 14 cm release; FloatSim gives 8–15 %.
4. **Wave calibration without the model:** every regular and irregular condition in the empty
   flume at the model position. It is standard practice (the ITTC guideline the matrix cites),
   and it is not in the tank-time budget.
5. **Repeatability runs.** Repeat a few regular cases at resonance, and one irregular sea, three
   times per phase. Criterion 6 compares moored and free within ±3 %, the same order as the
   ±3–5 % repeatability of a physical test.
6. **Blind FloatSim predictions of the final matrix.** DESIGN-BASIS Phase D (442 cases) was
   planned on the earlier matrix. It needs re-planning for these periods, heights and
   configurations, and running before the tests.
7. **An instrumentation plan**, if it is not already covered elsewhere:
   - 6-DOF tracking of every body (the per-buoy tilts are what the validation compares);
   - a load cell per mooring line;
   - the joint loads the matrix mentions;
   - the incident and cross-flume wave gauges.

## 6. Smaller items

- **Sloshing windows.**
  - Six periods are within ±5 % of the flume's cross modes (the matrix's own 3.7 m / 2.74 m
    values): 8, 8.5, 9, 11, 15 and 16 s FS.
  - The design basis excluded such windows. The matrix keeps them with flags.
  - For RAO validation, exclude them. Replacements: 8.3 and 9.3 s instead of 8.5 and 9;
    10.2 / 11.5 s instead of 11; 14.5 / 16.5 s instead of 15 and 16. Otherwise accept them
    explicitly with the cross-flume gauges.
- **Very small waves at short periods.**
  - s1 at 4–6 s FS is 8–19 mm model: close to the gauge and tracking resolution.
  - 4 s FS is below the wavemaker's comfortable minimum (the matrix flags it).
  - Suggest a 1–2 cm minimum height, or drop 4 s.
- **Platform footprint.**
  - The matrix says ~2.4 m square; the spar grid is 2.06 m wide (MOORING-SPEC).
  - At 2.4 m the wall clearance is 0.65 m, against the 0.6 m criterion (0.80 m in the design).
  - Confirm what the 2.4 m includes.
- **Flume dimensions: no real discrepancy.**
  - OSU's Large Wave Flume page (checked 2026-09-26) gives the width as 3.7 m (12 ft) and the
    maximum depth for wind/storm waves as 2.7 m (9 ft).
  - 12 ft is 3.658 m and 9 ft is 2.743 m, so the design's 3.66 m and the matrix's 2.74 m are
    the exact values of the same dimensions.
- **Wavemaker period range** (the same page): 0.8 to 12+ s.
  - 4 s and 5 s FS (0.57 and 0.71 s model) are below it. The matrix flags only periods below
    its assumed 0.7 s minimum.
  - The page's maximum wave is 1.7 m at 5 s in 2.7 m of water; the matrix uses 1.8 m. That
    does not limit any case (the largest is 0.88 m).
- **Orientation and variants.**
  - The mooring is designed for the cluster and platform in the orientation of the rev C
    figures.
  - These are outside the spec:
    - P2-D (cluster at 45° to the waves) needs its own attachment layout;
    - P3-C (alternate ballast) changes the CoG, and so the calm tilt and the pretension
      setting;
    - the fin variants (P1-B, P2-B) may conflict with the buoy's collar at z = −0.50 m;
    - the tow, transition and PTO configurations need their own rigging.
- **Tank time.** The additions above add time: calibration, pulls, swaps, longer decays, fine
  bands and repeats. The 6 effective hours per day should be checked against them.

## What was filled in

The workbook's natural-period placeholders (Regular_Waves rows 24–26) now hold FloatSim's
pre-test values:
- row 24: heave 17.9 s FS;
- row 25: tilt 19.6 s FS, the cluster; the notes give the buoy (18.8 s) and the platform
  (20.1 s);
- row 26: 20.6 s FS, the platform's upper flank.

Their Full/Reduced flags stay **N**, so the run counts and tank time are unchanged until the
matrix owner switches them on. The FloatSim_Predictions sheet has:
- the per-article natural periods, free and moored (both cord sets), on the flume BEM and
  converged;
- the damping and half-power bands;
- the slow mooring modes, with their decay record length;
- the recommended fine-step bands.
