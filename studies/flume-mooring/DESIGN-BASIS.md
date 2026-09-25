# Flume mooring design: basis, confirmed criteria and Phase C results

**Status (2026-09-24): Phases A–C done; Phase D (verification runs) NOT run.** Xabier confirmed
the design intent, criteria and test matrix on the Phase A/B report (`cba1921`). Phase C (FloatSim
fixes C1 and C2, record fixes, single-buoy redesign, resonance bandwidth, tank predictions) is
below. Phase D waits for Xabier on the case list (§D) and the single-buoy yaw option (§C2).

**Goal:** design the station-keeping mooring for the actual OSU Large Wave Flume (HWRL) test. It
is not an OrcaFlex comparison.

Numbers are re-derived from the record: the code, committed outputs, and the study scripts named
per section (FloatSim throughout). "⚠" marks a gap or inconsistency in the record.

---

## Facility ASSUMPTIONS (open; check against HWRL before any hardware is bought)

The design rests on these. None is a confirmed facility fact. **The final mooring specification
must list them on its first page.**

| # | Assumption | Value the design uses | If it proves false |
|---|---|---|---|
| F1 | **Anchors can be fixed at the pin plane, 0.72 m above still water**, at the walls ±5.0 m up/downstream, y = ±1.83 m | anchor z = +0.717 m | The pin-level design depends on it. With lower anchors the pin-level lines slope down (0.72 m over the ~5.3 m chord, ~8°), into the splash zone near the walls (criterion 5 fails: the dry-weight line is wave-loaded), and pull down on the pins. The attachment then has to drop towards the waterline, and **the pretension moment about the pins returns: calm-water tilt 4.9° (cluster) and 9.1° (platform)** at the old SWL attachment (FloatSim, `attachment_options.json`, `static:*:swl_spar`), against criterion 1's 0.1°. |
| F2 | Anchor load rating | design-state load per anchor, before the safety factor: 4.1 N (buoy line, record; the redesign changes it, §C2), 16.5 N (cluster), **66.1 N per platform anchor** (2-leg bridle, 33 N per leg). WLL ≥ 3× that: **≥ 200 N per platform anchor** | Lighter anchors need a softer or lower-pretension mooring (longer surge period, more offset). Phase D replaces these record values with FloatSim maxima. |
| F3 | Maximum wave height per period, and the steepness limit | **H/λ ≤ 0.08** (finite depth 2.7 m): H = 0.5 m needs T ≥ 2.01 s, 0.35 m needs T ≥ 1.675 s, 0.2 m needs T ≥ 1.266 s. The wavemaker's own H(T) envelope is unknown | Cases above the wavemaker's envelope drop out of the matrix; the drift bound and loads fall with them. |
| F4 | Water depth | 2.7 m ("storm-wave max"; `mooring_sizing.H_FLUME`) | The drift bound uses coth(kh); a deeper flume lowers it at long periods. (FloatSim's runs are deep-water throughout: BEM and wave kinematics.) |
| F5 | Test-section position relative to the wavemaker and the beach | Not in the record. The FloatSim cases settle for 60–135 s of regular waves; the tank needs a clean incident-wave window at least that long before beach reflections arrive, with the article on the flume centreline (the sidewall analysis assumes it) | Shorter clean windows shorten the settle, which matters most near the lightly damped tilt resonance (§C3). |
| F6 | Tracking field of view | Surge from about −0.3 m (upstream) to **+1.0 m** (downstream; criterion 4: mean offset + dynamic ≤ 1.0 m), sway ±0.1 m, plus the article's footprint (platform 2.06 m wide at 45°) | A smaller field of view needs a stiffer mooring (shorter surge period) or a lower H in the extreme band. |

Also assumed (hardware, not facility): line dry weight 0.3 N/m; linear springs with a working
stroke ≥ 1.25 × the maximum stretch.

---

## Confirmed decisions (Xabier, 2026-09-24)

- **Design intent (a): soft restraint** that disturbs the measured motions as little as possible.
- **Attachment at the PIN PLANE, at the pins** (each moored spar's pin; not the hub centre). The
  single buoy attaches at its spar top, the same height (+0.717 m).
- **Periods stop at 3.5 s.**
- **Criteria as proposed** (§B, now confirmed), including mean tilt ≤ 3° and a mooring-period ratio
  ≥ 4 (surge, sway, yaw ≥ 4 × 3.5 s = 14 s).
  - *Why 3° is acceptable:* the mean tilt is the drift moment about the pins over the tilt
    stiffness, and the drift grows at least as H² (drag drift ∝ A·U² ∝ H³ at a fixed period;
    potential drift ∝ H²). Over the operational band (H ≤ 0.12 m) the mean tilt at the recorded
    bound is **≤ 0.12°** (0.124° buoy, 0.117° cluster, 0.108° platform at H = 0.12 m, T = 1.4 s; 0.04° at H = 0.08 m).
    3° is reached only in the extreme band, where the runs are for loads.
- **Drift APPLIED IN-RUN** as a steady +x force at each spar's calm waterline equal to the
  recorded bound (`mooring_sizing.drift_per_spar` with H/λ ≤ 0.08), ramped with the excitation
  (`floatsim_decks.drift_force`). Not superposed afterwards.
  - The bound is an upper bound on the mean drift: fixed-body splash-zone drag plus the Havelock
    potential term, no shielding, no reduction for the body moving with the wave. So **the mean
    offsets and line tensions it produces are conservative.**
  - *No double counting:* FloatSim itself carries no drift on a fixed body (≈ 1e-14 N,
    `design_basis.json`). Its mean force on a MOVING moored body is a separate relative-motion
    effect. Phase D case D0 reports FloatSim's own mean force in one moored run without the
    applied drift, to show it is negligible next to the bound.
- **Criterion 6 (mooring interference) is evaluated in the operational band only.** The
  free-floating reference is held by a NUMERICAL restraint on surge, sway and yaw only (period
  ~60 s), with a two-stiffness insensitivity check (60 s and 120 s: the heave and tilt RAOs must
  agree within 0.5 %). It is not meaningful in the extreme band, where the unrestrained reference
  drifts (tracker `EXCITATION-FIXED-REFERENCE-VS-DRIFT`).
- **Regular waves primary, heading 0.** *Recorded risk:* irregular waves would excite slow
  second-order (difference-frequency) drift at the ~15 s surge mode. FloatSim has no second-order
  wave forces, so it cannot model that; the tank would see larger low-frequency surge than Phase D
  predicts.

### Confirmed test matrix (1:50)

The two height ranges in the record are two bands, not a conflict:

| Band | Heights | Periods | Purpose |
|---|---|---|---|
| **Operational** | H = 0.04, 0.08, 0.12 m (2–6 m full scale) | every period of the list (§D) | motions; criterion 6 |
| **Extreme** | H = 0.2, 0.35, 0.5 m | a subset (§D) | line and anchor loads; offsets |

- T = 1.4–3.5 s, fine steps across the heave/pitch/tilt resonances (step from §C3), every period
  ≥ 5 % clear of the flume's sloshing (transverse cut-on) periods 1.25 / 1.53 / 2.19 s: excluded
  windows 1.1875–1.3125, 1.4535–1.6065 and 2.0805–2.2995 s.
- **Steepness cap H/λ ≈ 0.08.** Cases above it are flagged, not run. H = 0.5 m at T = 1.4 s would
  be H/λ = 0.163 (λ = 3.059 m at 2.7 m depth), beyond the ~0.14 breaking limit (Miche 0.142 at
  this depth). The cap admits H = 0.5 m only for T ≥ 2.01 s, 0.35 m for T ≥ 1.675 s, 0.2 m for
  T ≥ 1.266 s (the whole list).
- Moored free decays: heave and pitch/tilt, plus surge, sway and yaw, and a static pull test (FloatSim
  predictions in §C4).

---

## Phase C — FloatSim fixes, record fixes, redesign, predictions

### C0. FloatSim fixes (separate PRs, gated)

- **C1: catenary robust start** (`b737d40`; `floatsim/mooring/catenary_analytic.py`). The cold start stays
  first, so every geometry it solves is bit-identical. Only when it fails, or converges onto the
  non-physical H ≤ 0 root, does the solve retry from:
  - the line's previous converged (H, V_A) (warm start, cached per line in the state force);
  - a hanging-chain estimate (slack lines);
  - a taut-elastic estimate from the chord.

  It raises only if every start fails. Gates:
  - byte-identity identical (`scripts/catenary_byte_identity.py`, 11 hashes);
  - the moored platform at T = 3.5 s, which stopped at t = 25.7 s, now runs: heave RAO 1.248
    moored vs 1.251 free;
  - the not-slow suite: 756 passed, 0 failed.

  The slack-line (H < 0 root) case surfaced in this study: 0.3 N/m lines in air at the H = 0.5 m
  design offset.
- **C2: true anchors for non-origin reference points** (`109699d`; `make_catenary_state_force(...,
  body_reference_points=)`, passed by `build_system`). The deck anchor is now the TRUE inertial
  point for every body; the fairlead sits at reference point + displacement + arm. Gates:
  - byte-identity for origin-referenced decks;
  - a unit and a driver test pin the non-origin geometry;
  - the flume decks give the same forces with true anchors as with the old `anchor − reference`
    workaround (worst relative difference 7.4e-13), and all eight cached settles hit with
    identical joint residuals;
  - the not-slow suite: 765 passed, 0 failed.
- ⚠ **Found, not fixed (tracker):** `solve_static_equilibrium` can return `converged=True` with a
  large residual. When hybr starts on an exact equilibrium it may stop at once and report xtol
  "success" (seen here: 0.214 N residual, returned state unchanged). With the default Tikhonov
  term it also stalls on the lone buoy's near-singular yaw row. This study's static solves use a
  linear-predictor start and an explicit residual check (≤ 1e-4 N).
- **Noted:** `LinearSpring` earth anchors do not share the C2 defect. They are built on
  displacement with zero rest offset, so the anchor position never enters the force, but the
  deck's `anchor_b_global` is silently ignored.

### C1. Record corrections (item 1)

- **`mooring_sizing.C55_BUOY` = 70.771 N·m/rad**, FloatSim's C[4, 4] of the flume buoy
  (`single_osu_open_psd.nc`, about the CoG). It replaces (10.2 + 5.03)(2π/2.11)², which came from
  the invalid 2.11 s pitch period (FloatSim: 2.76 s). It feeds only the superseded trim printout.
- **Platform mean offset, 0.75 m (README) vs 0.71 m (design table):** these are two different
  quantities, not an error in either.
  - 0.71 m is the sizing value, drift / Kx = 83.6 / 117.37 = 0.712 m.
  - 0.75 m is `mooring_verify.csv`'s coupled static solve for the then-proposed waterline
    bow-and-stern attachment (0.750 m). At the pin plane the same solve gives 0.712 m.
  - The confirmed design (pin plane, 0.3 N/m lines) gets its offsets from the FloatSim pull
    curves (§C4).
- ⚠ **New: the flume BEM's waterplane is 4.5 % small.** FloatSim's buoy C33 is 186.26 N/m, against
  ρgπD²/4 = 195.05 N/m (the record's `C33_BUOY` = 194.5). The mesh (`coupled_bem_osu.NT = 12`) has
  a 12-sided waterline, which holds 12·sin 30°/2π = 0.9549 of the circle's area, and
  195.05 × 0.9549 = 186.26 exactly.
  - Consequence: FloatSim's flume heave period (2.54–2.61 s) is about 2 % long
    (√(195.05/186.26) = 1.023).
  - The fine period band (§C3) spans it either way. The BEM is unchanged; a finer waterline (or
    the analytic C33) is a tracker item for any absolute heave-period claim.
- **Drift bound at the confirmed cap (H/λ ≤ 0.08, was 1/15):** per spar, over T = 1.4–3.5 s:

  | H (m) | 0.04 | 0.08 | 0.12 | 0.2 | 0.35 | 0.5 |
  |---|---|---|---|---|---|---|
  | max drift / spar (N) | 0.011 | 0.069 | 0.214 | 0.921 | 3.149 (T 1.68 s) | **6.368 (T 2.01 s)** |

  The H = 0.5 m maximum rises 22 % over the record's 5.22 N (cap 1/15, T 2.25 s), because the
  looser cap admits H = 0.5 m down to T = 2.01 s. **At T = 2.05 s it tilts the articles 3.1–3.6°**
  (above the 3° criterion). The first admissible period after the 2.08–2.30 s slosh exclusion is
  2.35 s: 4.87 N/spar, 2.5–2.8°, below the 5.22 N the cluster and platform lines were sized for.
  Hence §D starts H = 0.5 m at 2.35 s.

### C2. Single-buoy mooring redesign (item 2; `single_buoy_redesign.py`)

The design state is the record's: the mean offset under the drift bound (a FloatSim static solve,
with the bound applied at the spar waterline) plus A_w = H/2. The criteria are clearance of the
slack-side line to the 0.6H crest ≥ 0.10 m, and tension ≥ 0.15 T0.

**Sag.** FloatSim's calm sag of the record design (T0 = 2.23 N) is **0.365 m, not 0.30 m.** The
record's wL²/(8T0) used the unstretched length (4.21 m). The line's weight w·L0 hangs over the
stretched 5.3 m span, so sag ≈ w·L0·s/(8H) = 0.38 m.

**(a) Pretension.** k = 2.009 N/m is kept, which sets surge ≈ 15 s.

| T0 per line (N) | 2.23 (record) | 3.0 | **4.0** | 5.0 | 6.0 | 8.0 | 10.0 |
|---|---|---|---|---|---|---|---|
| surge / sway period (s) | 15.6 / 26.0 | 15.1 / 23.6 | **14.9 / 21.3** | 14.8 / 19.6 | 14.7 / 18.2 | 14.5 / 16.2 | 14.3 / 14.7 |
| clearance at H 0.35 / 0.5 (m) | −0.058 / −0.461 | +0.153 / −0.116 | **+0.297 / +0.144** | +0.372 / +0.259 | +0.418 / +0.318 | +0.470 / +0.379 | +0.499 / +0.409 |
| slack-side T_min/T0 at H 0.5 | 0.40 | 0.41 | **0.51** | 0.60 | 0.67 | 0.75 | 0.81 |

**Selected: T0 = 4.0 N per line**, the smallest that passes clearance and slack at every H, even at
the conservative 6.37 N drift. Its full state:

| | calm | H 0.12 | H 0.2 | H 0.35 | H 0.5 |
|---|---|---|---|---|---|
| mean offset (m) / mean tilt | 0 | 0.033 / 0.13° | 0.141 / 0.56° | 0.481 / 1.91° | 0.978 / 3.87° (at T 2.01 s) |
| line low point (m above SWL) / clearance to 0.6H | +0.546 (sag 0.171 m) | 0.542 / +0.470 | 0.534 / +0.414 | 0.507 / +0.297 | 0.444 / +0.144 |
| tension min / max (N) | 4.05 | 3.88 / 4.21 | 3.63 / 4.47 | 2.95 / 5.18 | 2.05 / 6.16 |
| spring stretch, max (m) | 2.01 | 2.10 | 2.22 | 2.58 | **3.07** |

Periods: surge 14.9 s (margin 6 % over 14 s), sway 21.3 s (52 %), heave 2.540 s.

- ⚠ **Hardware:** a 2 N/m line pretensioned to 4 N is stretched 2.0 m in calm water and 3.1 m at
  the H = 0.5 m design state, for a 3.3 m unstretched length (5.3 m chord). With criterion 7's
  1.25 factor it needs ~3.8 m of linear stroke, so effectively the whole line must be a soft
  elastic cord. Check the product's linearity to ~90 % strain, or lower the extreme H.

**(b) Yaw restraint.** The yaw inertia is FloatSim's deck I_zz = **0.063 kg·m²**
(`articulated_wall.IZZ`), and the axisymmetric added inertia is ≈ 0. Yaw stiffness from FloatSim
(T0 = 4 N):

| collar | r | K_yaw (N·m/rad) | yaw period, calm / at the H 0.5 offset |
|---|---|---|---|
| radial (lines face their anchors) | 0.5 mm | 0.0080 | 17.6 / 17.5 s |
| | 1 mm | 0.0161 | 12.4 / 12.4 s |
| | 10 mm | 0.161 | 3.9 s |
| | 40 mm | 0.648 | 1.96 s |
| | **80 mm (spar surface)** | 1.305 | **1.38 s: at the band edge** |
| | 120 mm | 1.972 | 1.12 s |
| | 200 mm | 3.337 | 0.86 s |
| pinwheel (tangential, balanced) | 20 mm | 0.0020 | 35 s / **unstable (K < 0)** |
| | 40 mm | 0.0079 | 17.7 s / **unstable (K < 0)** |
| | 80 mm | 0.0317 | 8.9 / 12.8 s |
| | 200 mm | 0.198 | 3.6 / 4.0 s |

- **Radial:** K_yaw = 16.1·r = 4·T0·r (the pretension's moment arm), so T_yaw ∝ 1/√(r·T0).
  **Yaw ≥ 14 s needs r ≤ 0.8 mm.** Any practical eye or shackle offset exceeds that.
- **Pinwheel:** K = 4·k_eff·r² at the calm position (T ∝ 1/(r√k)), but the mean offset turns the
  lines off tangency. At r ≤ 40 mm the yaw stiffness goes NEGATIVE at the H = 0.5 m offset.
  Where it stays positive (r ≥ 80 mm) the period is below 14 s.
- **So no collar meets yaw ≥ 14 s robustly.** Every r between ~1 mm and ~0.1 m puts the yaw
  period inside the 1.4–14 s band. A collar at the spar surface (1.38 s) is the worst case.
- **Options (Xabier to choose):**
  1. **Recommended: a swivel on the spar axis (r ≈ 0).** There is no yaw restoring and no yaw
     resonance. The buoy is axisymmetric, so yaw changes none of its hydrodynamics, and heading-0
     waves exert no yaw moment on it. The swivel stops line wind-up.
     - Cost: the buoy's heading drifts slowly (markers and instruments need no fixed heading).
     - Phase D's FloatSim runs need a numerical yaw restraint (the lone free buoy's LEVEL2 yaw
       instability above ~13° of pitch; §B caveat 2). It uses the criterion-6 restraint (60 s),
       and is not hardware.
  2. **A stiff radial collar, r ≥ 0.12 m** (yaw ≤ 1.12 s, below the band by a factor 1.25 in
     period). It holds the heading, but the restraint is stiff in yaw, against intent (a).
- **Itemised disagreement with the request's estimate:**
  - The request's scaling T_yaw ∝ 1/(r√T0) assumes K ∝ r²·T0, the geometric term T0·r²/ℓ only.
    For radial lines the moment-arm term T0·r dominates it by ℓ/r ≈ 50–5000.
  - For tangential lines the pretension term vanishes and K ∝ k·r², which does not depend on T0.
  - I_zz is 0.063 kg·m², not ~0.1.
  - r ~0.1 m gives yaw ≈ 1.2 s (radial) or ≈ 6–9 s (pinwheel), not ≥ 14 s.

### C3. Resonance bandwidth and the fine period step (item 3; `resonance_bandwidth.py`)

**Method.** FloatSim regular-wave sweeps of the moored design configuration (pin level, 0.3 N/m;
buoy at T0 = 4 N), with **wave-relative Morison drag** (STEP 5 PR1) and no applied drift:
- T = 2.30–3.10 s in 0.05 s steps;
- 120 s of settling after the 15 s ramp; amplitudes over the last 4 periods, stationary to
  ≤ 1.8 % (most ≤ 0.1 %).

Drag damping grows with amplitude, so the narrowest band is at the smallest operational height,
**H = 0.04 m**. H = 0.12 m was run for the buoy and cluster to show the widening.

The half-power band comes from a parabola in ω through 1/a² over the points with a ≥ a_max/2 (a
linear SDOF's 1/a² is exactly quadratic near resonance), checked by direct interpolation.

| Article, H | Mode | Peak (swept) | T_n (fit) | ζ_eq | Half-power band | ΔT (fit / direct) |
|---|---|---|---|---|---|---|
| buoy, 0.04 m | pitch | 9.57° at 2.55 s | 2.546 s | 4.2 % | 2.442–2.659 s | 0.217 / 0.239 s |
| | heave | RAO 2.26 at 2.65 s | 2.696 s | 11.1 % | 2.427–3.031 s | 0.604 / 0.686 s |
| cluster, 0.04 m | tilt | 9.60° at 2.60 s | 2.597 s | 3.6 % | 2.508–2.693 s | **0.185** / 0.212 s |
| | heave | RAO 2.28 at 2.70 s | 2.725 s | 11.3 % | 2.448–3.073 s | 0.625 s |
| cluster, 0.12 m | tilt | 16.6° at 2.60 s | 2.592 s | 6.4 % | 2.435–2.770 s | 0.335 / 0.368 s |
| | heave | RAO 1.53 at 2.90 s | 2.860 s | 14.9 % | 2.489–3.359 s | 0.870 s |
| platform, 0.04 m | tilt | 9.25° at 2.65 s | 2.648 s | 3.2 % | 2.567–2.734 s | **0.167** / 0.194 s |
| | heave | RAO 2.16 at 2.70 s | 2.753 s | 10.6 % | 2.489–3.079 s | 0.591 s |

- **Step rule.** The worst sampling of a peak is half a step off it, where the amplitude is
  1/√(1 + (step/ΔT)²) of the peak. Requiring ≥ 95 % gives **step ≤ 0.33 ΔT**.
  - The narrowest band is the platform's tilt at H = 0.04 m (ΔT = 0.167 s, so step ≤ 0.055 s), so the **fine step is 0.05 s** across the tilt and
    pitch resonances.
  - The heave bands are 0.6 s wide, so a 0.10 s step already meets the rule there. A 0.10 s coarse
    step is used elsewhere.
- **The moored buoy's pitch resonance is at 2.55 s** (free: 2.76 s). The pin-level lines at the
  spar top, pretensioned to 4 × 4 N, stiffen pitch.
- ⚠ **FloatSim cannot predict the single buoy's upper operational band near resonance with free
  yaw.**
  - At H = 0.12 m, T = 2.3–2.75 s, the moored buoy (lines on the spar axis) diverges: roll and yaw
    blow up (pitch 50–200°), and two cases crashed on the flipped geometry.
  - A numerical 60 s yaw spring does not stop it. With the stiff radial collar (r = 0.12 m, yaw
    1.12 s) the same cases are stable: pitch 16.6° / 14.6° at 2.55 / 2.65 s, roll and yaw exactly
    0.
  - This is the record's lone-buoy LEVEL2 instability above ~13° of pitch. Every buoy pitch at
    resonance in the operational band (9.6° at H = 0.04 m, ~17° at 0.12 m) exceeds LEVEL2's 0.1 rad
    anyway, so those results are indicative only.

### C4. Tank predictions: decays and pull curves (item 4; `tank_predictions.py`)

**Configuration.** Pin level, 0.3 N/m lines; buoy at T0 = 4 N. Calm water with the deck's drag.
Rigid initial offsets: surge 0.3 m, sway 0.1 m, yaw 5°.

**Periods** come from up-crossings of the article's mean displacement, low-passed below 0.2 Hz.
Rigid yaw of the articulated articles also swings the pinned buoys; their 2.3–2.9 s tilt modes
appear in the spectrum and would otherwise set the crossings.

**Pull curves** are the lines' restoring force against a rigid offset from the moored equilibrium.
Hydrostatics add nothing to these motions. A pull through the article centre at the pin plane
makes no moment about the pins.

| | buoy (T0 = 4 N) | cluster | platform |
|---|---|---|---|
| surge decay period / ζ per cycle | **16.5 s** / 8.3→3.8 % | **16.2 s** / 8.4→4.0 % | **16.1 s** / 8.3→3.7 % |
| sway decay period / ζ | **22.4 s** / 5.0→2.6 % | **26.8 s** / 5.0→3.1 % | **25.9 s** / 5.0→3.1 % |
| yaw decay period / ζ | none on the axis (K = 0); **collar r = 0.12 m: 1.12 s from K and I** (FloatSim decay not runnable, below) | **11.0 s** / ≈ 3 % | **12.5 s** / 5.3→3.2 % |
| pull stiffness surge / sway (N/m) | 7.38 / 3.60 | 30.57 / 9.78 | 124.4 / 42.1 |
| pull stiffness yaw (N·m/rad) | 0 (axis) / 1.97 (r = 0.12 m) | **15.6** | **206.4** |
| surge pull, F at 0.25 / 0.5 / 1.0 / 1.5 m | 1.85 / 3.68 / 7.32 / 10.80 N | 7.64 / 15.23 / 29.52 / 39.59 N | 31.1 / 62.1 / 122.5 / 162.7 N |
| slack-side T_min/T0 at 0.5 / 1.0 / 1.5 m | 0.77 / 0.56 / 0.36 | 0.57 / 0.21 / **0.10** | 0.55 / **0.13** / 0.04 |

- **The cluster's yaw is settled, and it FAILS criterion 2.**
  - The FloatSim decay is stable at 11.0 s (ζ ≈ 3 %). The negative coupled eigenvalue in
    `design_basis.py` was a linearisation artefact.
  - Rigid-rotation stiffness is 15.6 N·m/rad, not 2.18. The missing 13.4 N·m/rad is Σ p_b·F_b: the
    lines' pull at each pinned buoy times its arm, carried to the hub through the pins. That is the
    joint-reaction geometric stiffness that the per-body linearisation (and its Rayleigh quotient)
    omits.
  - 11.0 s is only 3.1 × 3.5 s. Heading-0 waves exert no yaw moment on the symmetric cluster, so
    the practical risk is small. Lengthening yaw to 14 s needs about 40 % less of the yaw stiffness
    that the pretension's moment arm supplies: lower T0 (costing clearance and slack) or fairleads
    closer to the centre (against the confirmed at-the-pins attachment). **Xabier to choose:
    accept 11 s for yaw, or trade.**
  - The platform behaves the same way. Its rigid-rotation stiffness is 206.4 N·m/rad against
    `design_basis.py`'s 62.8, and the FloatSim decay gives **12.5 s** (ζ 5 → 3 %), also below
    14 s (ratio 3.6).
- **Single-buoy yaw with the stiff collar cannot be decayed in FloatSim.** From 0.1° the yaw
  grows at exactly ζ = −0.028 per cycle until the geometry fails (from 1° and 5° it fails
  sooner). That is −ω·dt/2 (ω = 5.6 rad/s, dt = 0.01 s): the
  numerical damping of FloatSim's one-step-lagged state force. The spar's yaw drag, which is
  physically ≈ 0, cannot offset it. In heading-0 waves the yaw is never seeded, so the wave runs
  stay stable (§C3). The 1.12 s prediction is FloatSim's stiffness and inertia.
- **Criterion 3 in the pull curves.** The slack-side line falls below 0.15 T0 beyond about
  1.2 m of surge for the cluster (0.10 at 1.5 m) and 0.95 m for the platform (0.13 at 1.0 m).
  - With H = 0.5 m from 2.35 s (§D), the design states (bound mean offset + A_w, read off these
    curves) stay inside: ≈ 0.90 m, T_min ≈ 0.28 T0 (cluster); ≈ 0.88 m, ≈ 0.23 T0 (platform).
  - At the looser cap's 6.37 N the platform would reach ≈ 1.08 m and ≈ 0.12 T0, below the
    criterion.

---

## Phase D — final case list and cost (NOT run; `phase_d_plan.py` → `phase_d_plan.json`)

**Periods (24).** The fine band covers every article's H = 0.04 m tilt/pitch half-power band
(2.442–2.734 s) plus one step each side, at the §C3 step. Elsewhere the spacing is ~0.1 s, hugging
the slosh exclusions.

| | Periods (s) |
|---|---|
| coarse, below | 1.40, 1.45 · 1.65, 1.75, 1.85, 1.95, 2.05 · 2.30 |
| **fine, 0.05 s** | **2.35, 2.40, 2.45, 2.50, 2.55, 2.60, 2.65, 2.70, 2.75, 2.80** |
| coarse, above | 2.90, 3.00, 3.10, 3.20, 3.35, 3.50 |

"·" marks an excluded slosh window (1.4535–1.6065, 2.0805–2.2995 s).

**Cases per article** (buoy, cluster, platform):

| Kind | H (m) | Periods | Cases | Notes |
|---|---|---|---|---|
| operational, moored | 0.04, 0.08, 0.12 | all 24 | 72 | drift bound applied in-run at each spar's waterline |
| criterion-6 reference | 0.04, 0.08, 0.12 | all 24 | 72 | same article, no lines; numerical surge/sway/yaw restraint, 60 s |
| insensitivity check | 0.08 | 2.55, 3.00 | 2 | restraint at 120 s |
| extreme, moored (loads) | 0.2, 0.35, 0.5 | 1.40, 1.85, 2.05, 2.35, 2.55, 2.65, 2.75, 3.00, 3.50 | 23 | 27 minus the 4 flagged below; settle 180 s, mean over 30 s (two surge periods) |
| moored decays | — | — | 2 | heave; pitch/tilt (surge/sway/yaw done in §C4) |
| D0 (platform only) | 0.12 | 2.65 | 1 | NO applied drift: FloatSim's own mean force (double-count check) |

**Flagged, not run** (identical for the three articles):

| Case | Reason |
|---|---|
| H 0.35 at 1.40 s | H/λ 0.114 > 0.08 |
| H 0.5 at 1.40 s | H/λ 0.163 > 0.08 (and beyond breaking, 0.142) |
| H 0.5 at 1.85 s | H/λ 0.094 > 0.08 |
| H 0.5 at 2.05 s | H/λ 0.077 passes, but the bound (6.14 N/spar) tilts the buoy, cluster and platform by 3.56 / 3.35 / 3.10° (> 3°) |

H = 0.5 m therefore runs from 2.35 s. There the largest bound is 4.87 N/spar, below the 5.22 N the
cluster and platform lines were sized for, and the platform's slack-side line keeps ≈ 0.23 T0
(≥ 0.15) at its design state.

**Totals: 514 cases** (171 buoy, 171 cluster, 172 platform).

**Cost.** Measured FloatSim wall time per case in Phase C: same harness, wave-relative drag,
120 s settle, 17 cases in parallel on this 32-core machine. Per simulated second:

| | minutes per simulated second | minutes per 146 s case | case-minutes (all cases) | wall time at 17-way parallel |
|---|---|---|---|---|
| buoy | 0.0059 | 0.9 | 158 | 0.2 h |
| cluster | 0.033 | 4.8 | 878 | 0.9 h |
| platform | 0.40 | 58 | 10 694 | **10.5 h** |
| **total** | | | 11 730 | **≈ 11.5 h** |

Savings, if wanted:
- The criterion-6 references only at H = 0.04 and 0.12 m: −24 cases per article, −1.4 h.
- A 90 s settle outside the fine band: about −10 %.

The platform dominates. Run it first, 17 at a time; the buoy and cluster fit around it.

**Prerequisites before Phase D** (Xabier):
- The single-buoy yaw option (§C2). It changes the buoy deck:
  - the swivel runs with a numerical yaw restraint;
  - the collar runs as modelled.
- Whether cluster/platform yaw at 11–12.5 s is accepted (§C4).
- Hardware confirmation of the buoy's elastic-line stroke.

**Method** (`floatsim_decks.wave_setup` / `run_case` / `drift_force`):
- Wave-relative Morison drag with the same wave and ramp as the excitation.
- The drift bound is ramped with the excitation.
- Each case starts from the moored settle.
- Extreme cases start at the pull-curve offset for the case's mean drift, so that the lightly
  damped ~16 s surge mode need not settle from zero. The residual slow-surge amplitude is
  reported per case.
- Outputs per case:
  - wave-frequency heave and tilt amplitudes;
  - mean offset and mean tilt;
  - line tensions (min / max, and slack against 0.15 T0);
  - line clearance to the local crest (0.6 H);
  - excursion;
  - for the references, the criterion-6 ratios.

---

## Open decisions for Xabier (after Phase C)

1. **Single-buoy yaw** (§C2).
   - Yaw ≥ 14 s is not achievable robustly: a radial collar needs r ≤ 0.8 mm, and a pinwheel goes
     unstable under the drift offset.
   - Choose:
     - an axis swivel (free yaw; FloatSim diverges near resonance at H ≥ 0.12 m), or
     - a stiff radial collar, r ≥ 0.12 m (yaw ≤ 1.12 s; stable in FloatSim head seas).
   - **Recommendation: the stiff collar, r = 0.12–0.2 m.** It holds the heading for tracking, it
     keeps the FloatSim predictions runnable, and yaw is unexcited at heading 0. Its only cost is
     a stiff yaw restraint, which the soft-restraint intent did not target.
2. **Cluster and platform yaw at 11.0 s and 12.5 s** fail the ≥ 14 s separation (ratio 3.1 and
   3.6).
   - Accept them, since heading 0 exerts no yaw moment on the symmetric articles.
   - Or trade clearance and slack margin (lower T0) or attachment radius.
3. **H = 0.5 m below 2.35 s**, dropped by the 3° mean-tilt criterion (the case at 2.05 s). Confirm.
4. **Buoy line hardware:** a 2 N/m elastic cord pretensioned to 4 N needs ~3.8 m of linear
   stroke over a 3.3 m unstretched length.
5. **FloatSim limits to carry into Phase D:**
   - all resonant tilts in the operational band exceed LEVEL2's 0.1 rad (indicative only);
   - the flume BEM's 12-sided waterline (heave ~2 % long);
   - the static-solver false convergence (tracker).
6. **Facility assumptions F1–F6** (top of this document) go to HWRL before hardware is bought.

---

# Record: the Phase A/B report (cba1921), with pointers to what Phase C resolved

## Phase A — design basis

### A(a) Test matrix: what the record says

| Source | Heights | Periods | Notes |
|---|---|---|---|
| `README.md` §scope; `mooring_sizing.py` `H_LIST`, `T_WAVE` | **0.2, 0.3, 0.4, 0.5 m** | **1.4–4.0 s** (sizing sweep, 53 values; no discrete test periods) | "moderate matrix, max 0.5 m"; regular waves, head seas; H/L ≤ 1/15 cap |
| `studies/rao_cross_model_out/README.md` | 0.04–0.12 m ("realistic operational band", 2–6 m full scale at 1:50) | — | ⚠ a different scale/height basis |
| `docs/m11b-pr8-rao-closure.md` | 0.03–1.2 m | 1.2–3.3 s | OrcaFlex band; withdrawn with the comparison |
| `platform-12buoy/flume-wall-effect/REBUTTAL-sidewall.md` | — | avoid **2.19 / 1.53 / 1.25 s** | flume transverse cut-on (sloshing) periods |

- → **Resolved:** the matrix is confirmed; see *Confirmed test matrix* above.
- ⚠ **The record does not define the test matrix.** It gives a mooring-sizing envelope
  (0.2–0.5 m, 1.4–4.0 s), a smaller "operational" band for other studies (0.04–0.12 m), and no
  discrete period list. **Xabier to confirm**:
  - the heights and periods;
  - regular waves only, or irregular too (irregular seas add slow drift near the mooring periods);
  - headings (the record uses head seas, 45° orientation for the cluster and platform);
  - whether free decays run moored.
- **The steepness cap removes the short-period corners.** H/L ≤ 1/15 caps H = 0.3 m for
  T ≤ 1.6 s, 0.4 m for T ≤ 1.8 s and 0.5 m for T ≤ 2.2 s. H = 0.2 m is just under the cap at
  1.4 s.
- 2.19 s (a cut-on period) lies inside the sizing band.

### A(b) Flume geometry and constraints

| Item | Value | Source |
|---|---|---|
| Width | 3.66 m | `mooring_sizing.W_FLUME`, README |
| Water depth | 2.7 m (assumed; "storm-wave max") | `mooring_sizing.H_FLUME`; ⚠ HWRL Q7 unanswered |
| Length | 104 m | README |
| Article width at 45° (plates) / wall clearance per side | buoy 0.29 m / 1.69 m; cluster 0.88 m / 1.39 m; platform 2.06 m / **0.80 m** | re-derived: (3.66 − width)/2 |
| Anchors | at the walls, ±5.0 m up/downstream, y = ±1.83 m, at the pin plane **+0.717 m** | design; ⚠ availability at +0.72 m unknown: now **ASSUMPTION F1**. HWRL Q1 asked about the SWL and must be re-asked. |
| Wavemaker / beach distances, test-section position, gauge positions, tracking coverage | — | ⚠ **not in the record** (HWRL Q7, Q8 unanswered) |
| Transverse cut-on periods | 2.19, 1.53, 1.25 s | flume-wall study |

### A(c) Current mooring design, at pin level

- **Layout:** an X-spread of 4 lines. For the platform, each line is a 2-leg bridle to the two
  spars of a half-row, modelled as 2 FloatSim lines each carrying k/2 and T0/2.
- **Spread:** 20.1° from the flume axis (atan(1.83/5)).
- **Line stiffness:** from the design surge period, T_surge = 15 s:
  `Kx = (M + A11)(2π/15)²` and `k = Kx/(4cos²α)`.
- **Pretension:** `T0 = 1.2·k·cosα·(δ + A_w)`, where the mean offset `δ = N_spar·F_drift(0.5 m)/Kx`
  and `A_w = 0.25 m` is the design wave amplitude. The 1.2 keeps the slack-side line at ≥ 1/6 of T0
  at the design state (`mooring_sizing.xspread`).
- **Design wave height:** 0.5 m.

| | M + A11 | Kx (design) | k / T0 per line | fairleads (+0.717 m) | chord | L0 (unstretched) | EA = k·L0 | pre-stretch |
|---|---|---|---|---|---|---|---|---|
| 1 buoy | 40.4 kg | 7.09 N/m | 2.01 N/m / 2.23 N | spar top, on axis | 5.324 m | 4.212 m | 8.46 N | 1.11 m |
| 1 cluster | 166.9 kg | 29.28 N/m | 8.30 N/m / 9.01 N | each spar's pin (±0.295, ±0.295) | 4.950 m | 3.864 m | 32.1 N | 1.09 m |
| 4×4 platform | 668.9 kg | 117.4 N/m | bridle 33.27 N/m / 36.07 N (leg 16.64 / 18.03) | the 8 up/downstream-row pins (x = ±0.884) | 4.223 / 4.393 m | 3.140 / 3.309 m | 52.2 / 55.1 N | 1.08 m |

**Drift bound** (`mooring_sizing.drift_per_spar`): the mean drift on one FIXED surface-piercing
spar, as the sum of two terms:
- splash-zone Morison drag `F_d = (2/3π)·ρ·Cd·D·A·U²`, with `U = Aω·coth(kh)`, Cd = 1.2 and
  D = 0.1593 m;
- a Havelock small-ka potential bound `F_p = (5π²/16)·ρ·g·a·A²·(ka)³`.

There is no shielding credit, and H is capped at L/15. The envelope maximum per spar:

| H (m) | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|
| max drift / spar (N) | 0.92 (T 1.40 s) | 1.95 (1.70 s) | 3.30 (2.00 s) | **5.22 (2.25 s)** |

Article totals at H = 0.5 m: 5.2, 20.9 and 83.6 N. Mean offsets at the FloatSim pin-level
rigid-surge stiffness (6.75 / 30.57 / 124.5 N/m) are **0.77 / 0.68 / 0.67 m**.

The record's maximum line tensions at the design state (T0 + k·cosα·(δ + A_w)) are
**4.1 / 16.5 / 66.1 N** (platform: bridle; 33 N per leg).

⚠ Inconsistencies in the record (→ resolved in §C1):
- The README gives the platform offset at H = 0.5 m as 0.75 m; `mooring_design_table.csv` gives
  0.71 m.
- `mooring_sizing.C55_BUOY` is built from the invalid 2.11 s pitch period. It feeds only the
  superseded trim printout, not the line design.

### A(d) Mooring natural periods at pin level, 0.3 N/m (`design_basis.py`)

**Method:**
- **Mass:** M(ω) is FloatSim's M + A∞ with A∞ replaced by the BEM A(ω) at the mode frequency
  (iterated).
- **Stiffness:** K is FloatSim's C plus the linearised FloatSim catenary force at the settled
  equilibrium.
- **Primary value:** the rigid-pattern Rayleigh quotient. The pins do no work in a rigid motion.
- **Second value:** the constrained eigenmode, which lets surge couple into tilt through the
  pin-level lines.
- **Platform:** settles at 0.02 and 1.0 N/m bracket 0.3 N/m (the two differ by < 0.3 %).

| Mode | 1 buoy | 1 cluster | 4×4 platform |
|---|---|---|---|
| Surge | 15.6 s (coupled 17.1 s) | 14.7 s (coupled 16.2 s) | 14.6 s (coupled 16.1 s) |
| Sway | 26.0 s (26.9 s) | 26.0 s (26.9 s) | 25.0 s (25.9 s) |
| Yaw | **none: lines on the spar axis** | 22.7 s (rigid) — ⚠ see below | 19.0 s (rigid) to 29–30 s (coupled) — ⚠ |
| Heave (same analysis) | 2.55 s | 2.57 s | 2.59 s (record decay 2.607 s) |
| Pitch / tilt (record, drag-free decays) | pitch 2.76 s (free) | tilt 2.59 s at pin level (2.86 s free) | tilt 2.65 s at pin level (2.92 s free) |

**Wave band** (record): 1.4–4.0 s. The in-band resonances are heave 2.55–2.61 s, pitch/tilt
2.59–2.76 s, and cut-on 2.19 s.

**Yaw** is uncertain. (→ Settled in §C4. The cluster yaw is stable at 11.0 s and the platform's at 12.5 s; the rigid Rayleigh values below omit the joint-reaction stiffness. The single buoy is covered in §C2.)
- The single buoy has zero yaw restoring (all lines on its axis). Combined with the lone buoy's
  LEVEL2 yaw instability, **the single buoy needs a finite-radius collar or bridle**.
- For the cluster, the coupled eigen-analysis returns a negative-stiffness yaw-like mode
  (λ = −0.018). The rigid yaw stiffness is +2.18 N·m/rad, and the calm settle of the same
  configuration was stable. The likely cause is that the linearisation omits the geometric
  stiffness of the joint reactions, which carry the pretension through the pins.
- **A FloatSim yaw free decay is needed (Phase D).**

**Hub versus pin plane.** "Cluster center level" reads as the pin-plane height: the hub reference
point is at +0.717 m, the same height, so the static conclusions are identical. **Flag:** if the
hub itself was meant, all lines meet on the cluster axis. That gives zero yaw restoring (like the
single buoy) and routes the lines at spar-top height over the other spars' pins. The platform deck
sits at +0.90 m, not in the pin plane.

---

## Phase B — acceptance criteria (CONFIRMED 2026-09-24)

Criteria 2 and 6 rest on the soft-restraint intent.

**1. Static tilt.**
- Calm-water static tilt ≤ **0.1°** for every article. Pin level meets it: 0.000° for the
  cluster (0.3 N/m), platform (0.02 and 1.0 N/m) and buoy (balanced lines). The line acts
  through the pin, so pretension and line weight make no moment.
- ⚠ **Pin level does NOT give zero tilt under waves.** The mean drift acts at each spar's SWL,
  0.717 m below the line.
  - Tilt ≈ F_spar·0.717 / K_tilt, with K_tilt = 75.4 (cluster) or 81.4 N·m/rad (platform) from
    the pretension-tilt records, and C55 = 70.8 N·m/rad for the buoy.
  - At the drift bound, per H = 0.2 / 0.3 / 0.4 / 0.5 m:
    - cluster 0.50 / 1.06 / 1.80 / **2.85°**;
    - platform 0.46 / 0.98 / 1.67 / **2.64°**;
    - buoy 0.53 / 1.13 / 1.92 / **3.03°**.
  - Proposal: mean tilt ≤ **3°** at the maximum H, reported per case so the measured mean can be
    corrected.

**2. Separation.**
- Proposal: surge, sway and yaw periods ≥ **4 × T_wave,max**.
- Rationale: for a soft spring the wave-frequency horizontal response grows by
  1/(1 − (T/T_n)²). That is ≤ 6.7 % at a ratio of 4, 12.5 % at 3, 4.2 % at 5. The in-band
  resonances (2.55–2.9 s) are then separated ≥ 5×.
- At T_wave,max = 4.0 s this needs ≥ 16 s. Sway (25–27 s) and yaw (≥ 19 s, except the buoy's)
  pass. **Surge is marginal: 14.6–17.1 s is a ratio of 3.6–4.3.** If the matrix stops at 3.5 s it
  passes (≥ 4.2).

**3. No slack.**
- Minimum line tension ≥ **0.15·T0** in every case, including the drift-bound mean offset. This
  matches the design basis's built-in 1/6 margin.
- Every line must stay within its spring's linear range.

**4. Excursion.**
- Mean surge offset (drift bound) plus the dynamic surge amplitude ≤ **1.0 m**. Pending HWRL on
  tracking and gauge coverage; HWRL Q8 asked about 0.5–0.8 m.
- |mean sway| ≤ **0.10 m**, so the centred sidewall analysis (≤ 4 % wall effect at 0.80 m
  clearance) stays applicable.
- Body-to-wall clearance ≥ **0.6 m** at all times.

**5. Clearance, checked dynamically.**
- Line to any spar, pin or hub surface: ≥ **0.10 m**.
- **Lines in air:** line to the instantaneous local free surface ≥ **0.10 m**. Otherwise the
  dry-weight catenary is invalid and the waves load the line.
  - (→ FloatSim gives 0.365 m; the buoy is redesigned in §C2.) At 0.3 N/m the sag is `wL²/(8T0)` = **0.30 m for the single buoy** (low T0 = 2.23 N), 0.06 m
    for the cluster and 0.02 m for a platform leg.
  - The buoy's line low point is then ≈ 0.42 m above the SWL, against a crest of ≈ 0.27 m at
    H = 0.5 m. The fairlead also heaves with the buoy, so this is **marginal and must be checked
    per case.**

**6. Mooring interference.**
- Wave-frequency heave RAO moored vs free within **±3 %** (below the ±3–5 % repeatability of a
  physical decay test, as quoted in the sidewall rebuttal).
- Tilt amplitude within **±5 % or 0.5°**, whichever is larger. Across the test matrix.
- Re-derived from the committed H = 0.1 m viewer rows (SWL-spar attachment, calm-water drag; NOT
  the design configuration):
  - cluster 2.2 s: 0.6628 vs 0.6618 (+0.15 %) ✓;
  - but platform 2.9 s: 1.5067 vs 1.6064 (**−6.2 %**), and the other pairs range −2.9 % to
    +1.4 %.
- ⚠ The free reference drifts when unmoored (tracker `EXCITATION-FIXED-REFERENCE-VS-DRIFT`), so at
  large H the free runs are not stationary. **The free reference needs a decision.** → Decided: operational band only, with a numerical restraint (above).

**7. Line capacity (an output).**
- Required working load limit (WLL) per line or leg ≥ **3 × the maximum dynamic tension** over
  the matrix, drift bound included.
- The spring's working stroke must be ≥ **1.25 ×** its maximum stretch.
- Each wall anchor needs WLL ≥ 3 × the sum of its legs.
- Rationale for 3: the drag coefficients are unmeasured (Cd, Cd_n placeholders), the drift is a
  bound, near-slack snatch is possible, and the springs see fatigue cycling.
- Record-based preview (Phase D replaces it): 4.1 / 16.5 / 33 N per leg gives WLL ≥ 12 / 50 /
  100 N per leg, and ≥ 200 N per platform anchor.

### Caveat 1: the drift source

- **Potential drift is small.** ka is 0.027–0.164 for the spars (a = 0.080 m) over T = 1.4–4.0 s.
  The plates (a = 0.144 m at z = −1.38 m) are attenuated by e^(−2kd) = 0.003–0.39. The Havelock
  share of the recorded bound is ≤ 1.5 % for T ≥ 2.0 s, and 5–11 % at 1.4–1.6 s. **Drag drift
  dominates the bound**, as reasoned, except at the shortest periods.
- **FloatSim has no drift on a fixed body.** Its mean wave-relative drag on the FIXED cluster or
  platform is **zero** (≈ 1e-14 N, against the bound's 5.22 N per spar at H = 0.5 m,
  T = 2.25 s). Its Morison elements stop at the MWL and have no splash-zone (wetted-length)
  term, and the cycle mean of |u|u on a submerged element vanishes. FloatSim therefore carries
  **neither** recorded drift mechanism. Its drift on a moving body (for example 0.093 m/s on the
  unmoored M11b platform at H = 1.0 m) is a different, relative-motion effect.
- **The recorded bound governs the mean offset and the mean tension.** Proposal for Phase D:
  apply the bound as a steady force at each spar's SWL in the FloatSim runs, or superpose it on
  FloatSim's mean. → **Decided: applied in-run.**

### Caveat 2: validity (LEVEL2, 0.1 rad)

- Wave slope kA over the matrix, with H capped at L/15: ≥ 0.10 rad for all H ≥ 0.2 m at
  T ≤ 2.0 s, and for H = 0.5 m down to T ≈ 3.5 s.
- Record tilt/slope ratios (FloatSim, free): 1.1–2.6 off resonance (2.2 and 3.5 s), and 3–13
  across 2.4–3.3 s (peak 12.9 at the buoy's 2.76 s).
- **Hence nearly the whole matrix is beyond 0.1 rad.** Only H = 0.2 m at T ≥ 3.5 s (estimated
  0.08–0.094 rad) is inside. Tilts, and tilt-driven hub motions and line loads there, are
  indicative only.
- At pin level the tilt resonance moves to 2.59 s (cluster) and 2.65 s (platform).

---
