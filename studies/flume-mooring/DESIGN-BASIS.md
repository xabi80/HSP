# Flume mooring design: basis, confirmed criteria and Phase C results

**Status (2026-09-25): the mooring specification is at REV C.1 (rev C, ACCEPTED with record items G5, plus the creep allowance H.1) (Phase G: two cord sets on the same anchors and attachment points, with figures; `MOORING-SPEC.md`). Rev B (Phase F) is its operational set, unchanged; rev A (Phase E, pin plane, T0 +60 %) is withdrawn.** Earlier status: Phases A–C done, including Phase C round 2 (Xabier's decisions 1–7 on `6c93a80`, §C5–C9); Phase D (verification runs) NOT run.** It waits for Xabier on the BEM regeneration (§C8), the line element and the drift double counting (§C7), and the other open decisions at the end of Phase D.

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
| F1 | **Anchors can be fixed on the flume walls UNDER WATER, at each article's attachment depth** (rev B, Phase F), at ±5.0 m up/downstream, y = ±1.83 m | anchor z = the attachment depth: −0.50 m (buoy), −0.15 m (cluster, platform) | A higher or lower anchor slopes the line; its vertical pull at the attachment adds calm tilt and tilt coupling, so the sweep (`attachment_sweep.py`) must be re-run for the real anchor depth. *Superseded question (rev A): anchors at the pin plane, 0.72 m above still water. It no longer arises.* |
| F2 | Anchor load rating | **Rev C values: `MOORING-SPEC.md` summary and line tables (they supersede the ones here).** Record before rev B: design-state load per anchor, before the safety factor: 4.1 N (buoy line, record; the redesign changes it, §C2), 16.5 N (cluster), **66.1 N per platform anchor** (2-leg bridle, 33 N per leg). WLL ≥ 3× that: **≥ 200 N per platform anchor** | Lighter anchors need a softer or lower-pretension mooring (longer surge period, more offset). Phase D replaces these record values with FloatSim maxima. The round-2 extreme runs (§C7) reach 41.8 N per platform leg, so ~84 N per anchor and WLL ≥ 250 N (indicative). |
| F3 | Maximum wave height per period, and the steepness limit | **H/λ ≤ 0.08** (finite depth 2.7 m): H = 0.5 m needs T ≥ 2.01 s, 0.35 m needs T ≥ 1.675 s, 0.2 m needs T ≥ 1.266 s. The wavemaker's own H(T) envelope is unknown | Cases above the wavemaker's envelope drop out of the matrix; the drift bound and loads fall with them. |
| F4 | Water depth | 2.7 m ("storm-wave max"; `mooring_sizing.H_FLUME`) | The drift bound uses coth(kh); a deeper flume lowers it at long periods. (FloatSim's runs are deep-water throughout: BEM and wave kinematics.) |
| F5 | Test-section position relative to the wavemaker and the beach | Not in the record. The FloatSim cases settle for 60–135 s of regular waves; the tank needs a clean incident-wave window at least that long before beach reflections arrive, with the article on the flume centreline (the sidewall analysis assumes it) | Shorter clean windows shorten the settle, which matters most near the lightly damped tilt resonance (§C3). |
| F6 | Tracking field of view | Surge from about −0.3 m (upstream) to **+1.0 m** (downstream; criterion 4: mean offset + dynamic ≤ 1.0 m), sway ±0.1 m, plus the article's footprint (platform 2.06 m wide at 45°) | A smaller field of view needs a stiffer mooring (shorter surge period) or a lower H in the extreme band. |
| F7 | **Standard soft-mooring hardware available at HWRL** | **Rev C values: `MOORING-SPEC.md` summary and line tables (they supersede the ones here).** Record before rev B: Elastic cord (§C7):<br>• secant stiffness within −13 / +30 % of 2.0 / 8.3 / 16.6 N/m over the working strain;<br>• rated elongation ≥ 131 % / 85 % / 100 % if the cord is the whole unstretched line (3.1 / 3.9 / 3.1–3.3 m);<br>• ~0.3 N/m dry weight.<br>Wall anchors and fittings for 7 / 22 / 84 N design loads (platform: per 2-leg anchor, extreme runs). | Without such a cord, the lines need farther anchors (longer lines, lower strain) or a different stiffness element; the design changes. |

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

## Confirmed decisions on the Phase C report (Xabier, 2026-09-24, on `6c93a80`)

1. **Single-buoy yaw: a stiff radial collar at r = 0.2 m** (yaw 0.86 s). The swivel is rejected:
   free yaw diverges in FloatSim near resonance, and a fixed heading helps tracking and keeps
   body-frame pitch and roll unmixed. Verified in §C5.
2. **Cluster and platform yaw at 11.0 s and 12.5 s: accepted.** Criterion 2 is refined below.
   Condition (a), mirror symmetry, is verified in §C6. Condition (b) is a Phase D output.
3. **H = 0.5 m from T = 2.35 s: confirmed.** Drift is 4.87 N/spar, inside the 5.22 N sizing, and
   the tilt is below 3°.
4. **Line hardware** (§C7): the soft-element choice is Xabier's.
5. **Flume BEM waterline** (§C8): regenerate if any resonance moves by more than half a fine step
   (0.025 s), reporting the cost first. The BEM files are committed.
6. **Phase D scope with the saving:**
   - criterion-6 free references at H = 0.04 and 0.12 m only;
   - D0 kept;
   - an explicit residual check on every static solve (`floatsim_decks.checked_static_equilibrium`).
7. **Small-angle validity** is recorded plainly (§C9). LEVEL2 is proposed as the next FloatSim
   milestone; it is not started.

**Criterion 2, refined.** The ≥ 4 × T_wave,max separation applies to the modes the waves DRIVE.
At heading 0, an article that is mirror-symmetric about the wave axis has no first-order or mean
wave forcing in yaw, sway-antisymmetric roll or sway. Its yaw period may then sit below 14 s,
provided:
- (a) the symmetry holds on FloatSim's own assembled system (§C6);
- (b) every Phase D case reports its maximum yaw, and flags any case where it is not small. The
  threshold is yaw > 0.5° or > 5 % of the case's maximum pitch.

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

## Phase C, round 2 — decisions 1–7 (`buoy_yaw_collar.py`, `symmetry_check.py`, `line_hardware.py`, `bem_waterline.py`)

### C5. Single-buoy yaw collar, r = 0.12 m vs the decided r = 0.2 m (decision 1)

**Statics** (FloatSim, T0 = 4 N):

| | r = 0.12 m | **r = 0.2 m** |
|---|---|---|
| yaw stiffness / period | 1.972 N·m/rad / 1.123 s | **3.337 N·m/rad / 0.863 s** |
| surge / sway / heave | 14.9 / 21.1 / 2.540 s | 14.9 / 21.0 / 2.539 s |
| H = 0.5 m, conservative drift (6.37 N): clearance to 0.6H, T_min/T0, T_max, stretch | +0.164 m, 0.51, 6.15 N, 3.06 m | **+0.177 m, 0.51, 6.15 N, 3.06 m** |
| H = 0.5 m, matrix (from 2.35 s, 4.87 N) | +0.192 m, 0.59, 5.76 N, 2.87 m | **+0.203 m, 0.59, 5.76 N, 2.87 m** |

- Every static clearance holds at r = 0.2 m. The lines leave the collar outward, 0.12 m clear of
  the spar surface.
- Dynamically (§C7, H = 0.5 m, T = 2.65 s) the lowest line point stays **0.21 m** above the local
  incident surface.

**Yaw response near the moored pitch resonance** (T = 2.45 / 2.55 / 2.65 s; H = 0.04 and 0.12 m):
- **Unseeded (heading 0):** yaw is never forced. Only round-off appears: ≤ 1e-13 rad at r = 0.12 m,
  ~5e-11 rad at r = 0.2 m after 60 s.
- **Seeded (1e-8 rad yaw and roll):** the growth rate equals FloatSim's numerical K·dt/(2I) to
  within 3 %.
  - The physical part is σ = −0.03 … +0.007 s⁻¹ (Richardson over dt = 0.01 / 0.005 s), an
    equivalent |ζ| ≤ 0.1 %.
  - So pitch does not pump yaw at either radius at H ≤ 0.12 m. The largest value is r = 0.2 m at
    H = 0.12 m, T = 2.55 s (pitch 16.5°, already beyond LEVEL1), where any real yaw damping of
    0.1 % cancels it.
- **At H = 0.5 m, T = 2.65 s:**
  - **r = 0.12 m goes unstable**: yaw reaches 7° within 120 s, growing at 0.33 s⁻¹ against its
    numerical 0.078 s⁻¹.
  - **r = 0.2 m does not**: after dt extrapolation, ~0.007 s⁻¹ above the numerical rate.
  - **The decision is supported.**
  - *Itemised note on the reasoning.* A heading-0 symmetric buoy's yaw cannot be forced by pitch at
    any order; it can only grow parametrically. For pitch-modulated coefficients the principal
    parametric zones are T_yaw = T_p and 2T_p (2.55 and 5.1 s); T_p/2 (≈ 1.28–1.33 s) is a
    secondary zone. FloatSim shows that zone widening enough at H = 0.5 m to catch 1.12 s but not
    0.86 s, so the conclusion matches Xabier's.
- **Numerical consequence for Phase D.** The collar's yaw restoring comes through FloatSim's
  one-step-lagged state force, and the spar's yaw drag is ≈ 0, so round-off grows at
  K·dt/(2I) = 0.265 s⁻¹ at dt = 0.01 s. A 146 s case would reach ~0.4 rad.
  - The buoy runs at **dt = 0.005 s** (operational; ≤ 1e-4 rad) and **0.0025 s** (extreme 234 s
    runs); measured cost ×2.3 and ×3.6.
  - Tracker `STATE-FORCE-LAG-NEGATIVE-DAMPING`.
- ⚠ **At H = 0.5 m, T = 2.35 s the buoy diverges regardless of dt** (0.26–0.31 s⁻¹ at dt = 0.005 →
  0.00125 s, converging to ~0.22 s⁻¹ of model growth), with or without drift.
  - Its pitch there is **28°**. The growth falls with H: ~0.07 s⁻¹ at H = 0.35 m (24°), ~0.01 s⁻¹ at
    0.2 m (18°).
  - This is the small-angle kinematics beyond validity (§C9). Those extreme buoy cases are flagged
    as needing LEVEL2.

### C6. Mirror symmetry about the wave axis (decision 2, condition (a))

These are FloatSim's own assembled operators under the mirror y → −y (body permutation; DOF signs
+, −, +, −, +, −):

| relative residual | buoy (collar 0.2 m) | cluster (45°) | platform (45°) |
|---|---|---|---|
| M + A∞ | 9e-17 | 5e-16 | 2e-15 |
| C | 6e-18 | 6e-18 | 6e-18 |
| radiation kernel K(t) | 1e-13 | **1.0e-5** | **1.5e-5** |
| heading-0 excitation | 7e-16 | 4e-16 | 1e-15 |
| state force (catenary + wave-relative drag) | 3e-16 | 3e-15 | 1e-15 |
| joint-projector | — | 1e-15 | 2e-15 |

- **All three articles are mirror-symmetric, including the platform at its 45° rotation.** The one
  exception is Capytaine's coupled radiation solution, which is symmetric to 1e-5.
- **Heading-0 runs at the tilt resonance** (H = 0.04 m, 120 s after the ramp) show antisymmetric
  motion that is **steady, not growing**. This is a forced response to that 1e-5, amplified at
  resonance:

  | | roll (per 30 s window) | sway | yaw | pitch |
  |---|---|---|---|---|
  | cluster | 0.041–0.042° | 1.2 mm | 0.0011° | 9.8° |
  | platform | 0.084–0.085° | 2.3 mm | 0.021° | 9.7° |
  | buoy | round-off only | | | |

- Proposed "not small" threshold for Phase D's per-case yaw flag: yaw > 0.5° or > 5 % of the
  case's maximum pitch. The extreme runs reach 0.08–0.40° of yaw.

### C7. Soft-line hardware (decision 4; `line_hardware.py`)

**Required stroke.** FloatSim extreme runs at H = 0.5 m, T = 2.35 and 2.65 s:
- wave-relative drag;
- the drift bound applied in-run ("drift"), or not at all (D0, FloatSim's own mean force only);
- the buoy on its collar at dt = 0.0025 s.

Clearance is the lowest line point above the LOCAL incident surface (21 points per line along
FloatSim's catenary profile).

| | unstretched L0 | pre-stretch T0/k | static design max (matrix) | **dynamic max, drift** (2.35 / 2.65 s) | dynamic max, D0 | stroke needed (× 1.25) | max strain if the cord is all of L0 |
|---|---|---|---|---|---|---|---|
| buoy (k 2.009, T0 4.0) | 3.13 m | 2.01 m | 2.87 m | diverged / **3.28 m** | diverged / 2.90 m | **4.1 m** | **105 %** |
| cluster (k 8.30, T0 9.01) | 3.86 m | 1.09 m | 1.94 m | **2.63** / 2.49 m | 2.13 / 2.08 m | **3.3 m** | 68 % |
| platform leg (k 16.6, T0 18.0) | 3.14 / 3.31 m | 1.08 m | 1.94 m | **2.52** / 2.49 m | 1.97 / 2.08 m | **3.2 m** | 80 % |

"Diverged" is the buoy at T = 2.35 s: the small-angle instability of §C5.

**The same runs against the other criteria.** Indicative (§C9): H = 0.5 m at resonance is far
beyond LEVEL1.

| | T_min/T0 (≥ 0.15) | mean offset (surge range) | clearance to local surface (≥ 0.10 m) | max line tension |
|---|---|---|---|---|
| buoy 2.65 s, drift / D0 | 0.42 / 0.58 | 0.71 m (0.57–0.86) / 0.11 m | +0.21 / +0.19 m | 6.6 / 5.8 N |
| cluster 2.35 s, drift / D0 | **0.09** / 0.17 | **1.28 m (1.14–1.39)** / 0.45 m | **−0.13** / +0.26 m | 21.8 / 17.7 N |
| cluster 2.65 s, drift / D0 | **0.10** / 0.19 | 0.84 m (0.72–0.95) / 0.21 m | **+0.035** / +0.26 m | 20.7 / 17.3 N |
| platform 2.35 s, drift / D0 | **0.04** / 0.20 | **1.16 m (1.02–1.29)** / 0.33 m | **+0.06** / +0.35 m | 41.8 / 32.8 N per leg |
| platform 2.65 s, drift / D0 | **0.04** / **0.11** | 0.84 m (0.69–0.98) / 0.22 m | +0.14 / +0.34 m | 41.4 / 34.5 N per leg |

- ⚠ **D0 shows the in-run drift is double counted.** FloatSim's own mean force (wave-relative drag
  on the MOVING body, a different mechanism from the fixed-body bound) is a large fraction of the
  bound:

  | | offset without the applied bound | as a fraction of the bound |
  |---|---|---|
  | buoy | 0.11 m | ~20 % |
  | cluster | 0.21–0.45 m | 38–70 % |
  | platform | 0.22–0.33 m | 42–53 % |

  Applying the bound on top therefore roughly doubles the offset. With it, the cluster and
  platform at H = 0.5 m, T = 2.35 s leave the field of view (surge to 1.29–1.39 m), and slack
  (0.04–0.10 T0) and clearance fail. Without it, they pass everything but the platform's slack at
  2.65 s (0.11). **Decision needed** (below): keep the sum as a conservative envelope, or apply
  only the part of the bound FloatSim does not already carry.
- ⚠ **Dynamic slack.** The static design state predicted T_min/T0 ≥ 0.22. The dynamic runs reach
  0.04–0.19 at resonance, because the tilt moves the pins. If confirmed beyond LEVEL1, the cluster
  and platform need more pretension, as the buoy got in §C2.
- Anchor load: the platform leg reaches 41.8 N, so ~84 N per 2-leg anchor (record: 66 N), which
  gives WLL ≥ 250 N (F2).

**(a) Elastic shock cord.**
- *In FloatSim:* the catenary represents it only as a linear elastic line of axial stiffness k
  (EA = k·L0, uniform weight). FloatSim represents neither the cord's nonlinearity (stiff start, a
  softer plateau, stiffening near the rated elongation) nor its hysteresis (the catenary is
  quasi-static).
- *Nonlinearity, bounded* by FloatSim runs at the secant stiffness × 0.7 / 1.0 / 1.3 at the same
  T0:

  | secant k | surge period | H = 0.5 m design offset | max stretch | T_min/T0 at H = 0.5 m | clearance at H = 0.5 m |
  |---|---|---|---|---|---|
  | × 0.7 | 19.1–19.4 s | **1.14–1.17 m (outside the field of view)** | 2.66–3.96 m | 0.29–0.63 | +0.24–0.36 m |
  | × 1.0 | 16.1–16.4 s | 0.88–0.91 m | 1.94–2.87 m | 0.22–0.59 | +0.16–0.32 m |
  | × 1.3 | 14.1–14.6 s | 0.74–0.77 m | 1.55–2.27 m | 0.17–0.56 | +0.10–0.29 m |

  **The cord's secant stiffness over its working strain must stay within −13 % / +30 % of k.**
  Below −13 % the extreme offset leaves the field of view; above +30 % surge falls to 14 s and the
  slack and clearance margins vanish.
- *Hysteresis* is not represented. It adds damping to the slow modes, so tank decays will settle
  faster than predicted, and it offsets the mean position by half the loading/unloading gap, which
  the tank pull test measures.

**(b) Pulley + counterweight (constant tension W = T0).**
- *In FloatSim: not representable.* The catenary's unstretched length L0 = chord − T0/k must stay
  positive, so its axial stiffness cannot fall below T0/chord (the zero-length-spring limit), and
  FloatSim has no constant-tension connector. Screened in closed form (exact for the ideal
  element), with FloatSim's article masses:

  | | surge / sway stiffness | surge / sway period | mean offset + H/2 at H = 0.04 / 0.08 / 0.12 / 0.2 / 0.35 / 0.5 m |
  |---|---|---|---|
  | buoy | 0.37 / 2.75 N/m | 75 / 26 s | 0.05 / 0.23 / 0.63 / **2.04** / **3.46** / **4.36** m |
  | cluster | 0.70 / 6.58 N/m | 108 / 33 s | 0.08 / 0.43 / **1.18** / **2.92** / **4.19** / **5.10** m |
  | platform | 2.86 / 30.6 N/m | 107 / 30 s | 0.08 / 0.42 / **1.13** / **2.69** / **3.75** / **4.47** m |

  All restoring is geometric: ~20 × softer in surge than the springs. **It fails criterion 4 from
  H = 0.12 m** (cluster, platform) or 0.2 m (buoy), and the extreme band would push the articles
  3–5 m. Its 75–108 s surge mode would also take several minutes of clean waves to settle (F5).
  Its merits (no slack, constant sag, no stroke) do not rescue it.

**Recommendation (the hardware choice is Xabier's): elastic cord (a) for all three articles.**
- Secant stiffness within −13 / +30 % of 2.0 / 8.3 / 16.6 N/m over the working range.
- Rated elongation ≥ 131 % (buoy), ≥ 85 % (cluster), ≥ 100 % (platform) if the cord makes up the
  whole unstretched line. A longer cord in series with a stiff rope lowers the strain only if the
  total line length grows, i.e. farther anchors (F1).
- Dry weight ~0.3 N/m (the design value).
- Counterweight lines (b) are rejected.

### C8. Flume BEM waterline: effect on the resonances, and the regeneration cost (decision 5; `bem_waterline.py`)

**The mesh artefact is in pitch too, not only heave.** Capytaine hydrostatics of one buoy against
the spar/plate panels round (NT):

| NT | 12 (committed) | 24 | 36 | 48 | 96 | circle |
|---|---|---|---|---|---|---|
| C33 (N/m) | 186.26 | 192.83 | 194.06 | 194.49 | 194.91 | 195.05 |
| C44 = C55 (N·m/rad) | 70.771 | 73.282 | 73.753 | 73.919 | 74.078 | ≈ 74.13 |
| displaced volume (m³) | 0.019643 | 0.020336 | 0.020466 | 0.020512 | 0.020556 | — |

C55 is dominated by ρgV(z_B − z_G). The polygon hull's volume is short by the same 0.9549, so the
pitch and tilt restoring are 4.5 % low as well.

**Resonance periods.** FloatSim modal periods: K = C + the linearised catenary; M + A(ω) iterated
at the mode; the joints' null space. The single buoy uses single-buoy databases built at each NT
by the same pipeline (`BEM_NT=<n> python bem_cluster.py single`). The cluster and platform use
their NT = 12 coupled databases patched with the single-buoy change: each buoy's C block
replaced, its self A(ω) and A_inf corrected; interaction blocks and B kept. Values are
NT = 12 → NT = 96 (shift):

| | heave | pitch / tilt | worst residual at NT 24 / 36 / 48 |
|---|---|---|---|
| buoy, free | 2.561 → 2.519 s (**+0.041**) | 2.761 → 2.684 s (**+0.077**) | 0.023 / 0.011 / 0.006 s |
| buoy, moored (design, r = 0.2 m) | 2.539 → 2.499 s (+0.040) | 2.506 → 2.444 s (+0.062) | 0.019 / 0.009 / 0.005 s |
| cluster | 2.573 → 2.532 s (+0.041) | 2.604 → 2.540 s (+0.064) | 0.020 / 0.010 / 0.006 s |
| platform | 2.604 → 2.561 s (+0.043) | 2.659 → 2.599 s (+0.060) | 0.017 / 0.008 / 0.004 s |

- **Every resonance moves 0.040–0.077 s, 1.6–3 × the 0.025 s threshold. Regenerate before
  Phase D.**
- **NT = 36 is enough:** worst residual 0.011 s, under half a fine step with margin. NT = 24
  leaves 0.023 s.
- **Corroboration:** the converged free buoy (heave 2.519 s, pitch 2.684 s) matches the separate
  fine-mesh OSU buoy model (heave 2.52 s, pitch 2.69 s). The flume mesh artefact explains the
  difference that model had with the flume numbers.

**Regeneration cost (measured Capytaine time per frequency; 48 frequencies + ∞):**

| database | panels at NT 36 | s / frequency | total | memory |
|---|---|---|---|---|
| single buoy | 936 | 0.3 | **done** (`single_osu_open_nt36_psd.nc`, 0.8 min) | small |
| cluster (4 buoys) | 3 744 | ~5 (estimate: 0.5 s at NT 12, 2.1 s at NT 24) | ~5 min | < 2 GB |
| platform (16 buoys) | 14 976 | 123.7 (NT 12: 20.9, NT 24: 49.0) | **~1.7 h** | ~11 GB of 64 |

Then the FloatSim consequences, before Phase D:
- the three moored settles (cluster ~5 min, platform ~30 min);
- the H = 0.04 m bandwidth sweeps, to re-centre the fine band (≈ 1 h wall, platform-dominated);
- the item-4 decays (≈ 1 h).

The fine band will shift ~0.06 s shorter, to about 2.30–2.75 s. 2.30 s is already a listed
period.

**Total: ~2 h of BEM plus ~2–3 h of FloatSim, then Phase D.** Commands:
```
cd studies/flume-mooring
BEM_NT=36 python bem_cluster.py single                    # done
BEM_NT=36 python bem_cluster.py                           # cluster_osu_open_rot45_nt36(_psd).nc
cd ../platform-12buoy/flume-wall-effect
PLAT_ROT_DEG=45 BEM_NT=36 python coupled_bem_osu.py open full 48   # coupled_osu_open_rot45_nt36.nc
python psd_project.py coupled_osu_open_rot45_nt36.nc
```
The harness then has to point at the `_nt36` files (floatsim_decks BUOY_NC / CLUSTER_NC /
PLATFORM_NC).

**The BEM files the design rests on are now committed:**
- `single_osu_open_psd.nc` (49 KB);
- `cluster_osu_open_rot45_psd.nc` (0.49 MB);
- `platform-12buoy/flume-wall-effect/coupled_osu_open_rot45_psd.nc` (7.4 MB);
- the single-buoy NT = 24/36/48/96 `_psd.nc` files used above.

The raw (pre-PSD) files are reproducible by the commands above with `BEM_NT` unset (NT = 12):
`python bem_cluster.py single`, `python bem_cluster.py`, and
`PLAT_ROT_DEG=45 python coupled_bem_osu.py open full 48` followed by `psd_project.py`. Toolchain:
Python 3.13.11, Capytaine 2.3.1, NumPy 2.4.0, SciPy 1.17.1. The raw databases have small negative
B eigenvalues at high ω (single buoy −0.80 against a max of 28.5, from irregular frequencies of
the surface-piercing spar), which `psd_project` clips; the same holds at every NT.

### C9. Small-angle validity (decision 7), stated plainly

- The moored buoy's pitch at resonance is **9.6° at H = 0.04 m, the SMALLEST operational wave**,
  against LEVEL1's 0.1 rad (5.7°) validity limit. The cluster and platform tilt are 9.6° and 9.3°.
- **Every FloatSim prediction near the pitch/tilt resonance is therefore indicative, across the
  whole operational band.** That includes the line loads: the tilt moves the attachment point.
- Phase D flags every fine-band case as indicative, and each case's output flags max |tilt| >
  0.1 rad (`phase_d_plan.py`).
- **LEVEL2 is now REQUIRED for resonant predictions.** It is proposed as the next FloatSim
  milestone (tracker `LEVEL2-INTEGRATOR-UNWIRED`) and is not started.

---

## Phase D — final case list and cost (NOT run; `phase_d_plan.py` → `phase_d_plan.json`)

**Round-2 changes** (decisions 6 and 7; `phase_d_plan.py` re-run):
- Criterion-6 references at H = 0.04 and 0.12 m only.
- The buoy at dt = 0.005 / 0.0025 s.
- Every fine-band case flagged indicative.
- D0 kept.
- Every static solve residual-checked.

The new totals: **442 cases** (147 buoy, 147 cluster, 148 platform), of which **186 are flagged
indicative**. About **10.2 h** wall at 17-way parallel (buoy 0.4 h, cluster 0.7 h, platform
9.1 h).

**Before Phase D** (all Xabier's):
1. Regenerate the cluster and platform BEMs at NT = 36 (§C8, ~2 h), then re-settle, re-sweep the
   bandwidth and re-centre the fine band (~2–3 h FloatSim).
2. Choose the line element (§C7).
3. Decide the drift double counting (§C7).
4. Decide the cluster/platform pretension against dynamic slack (§C7).
5. Beyond small-angle validity, the extreme buoy cases near resonance (H ≥ 0.35 m around 2.35 s)
   diverge in FloatSim (§C5). Run them flagged, or drop them until LEVEL2.

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

**Round-1 totals (superseded by the round-2 totals above): 514 cases** (171 buoy, 171 cluster, 172 platform).

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

**Round-1 prerequisites** (answered by decisions 1–7; the round-2 list above replaces them):
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

## Phase E (2026-09-25): conservative mooring specification issued — `MOORING-SPEC.md` / `.pdf`

**WITHDRAWN, same day: superseded by Phase F (rev B).** Its +60 % pretension is REVERTED: the pin-plane lines shift the buoys' tilt resonance 6.5–8.6 × more than the resonance tolerates (Phase F). Kept below as the record.

Xabier's priority change: a CONSERVATIVE specification today. Full verification (BEM
regeneration, Phase D, LEVEL2, drift mechanism) follows before the test and is listed in the
spec's follow-ups. What was used:
- the drift SUM (bound + FloatSim's own mean force);
- the buoy at T0 = 4.0 N on the r = 0.2 m collar;
- targeted FloatSim worst cases (H = 0.5 m, T = 2.35 / 2.65 s) for the cluster and platform
  (`run_spec_extremes.py`);
- statics from `mooring_spec.py`.

The spec itself is generated by `build_spec.py`.

- ⚠ **The instructed +20 % pretension FAILED the slack pass** (T_min ≥ 0.15 T0): cluster
  0.09 / 0.11, platform 0.04 / 0.05. Itemised disagreement: the tension swing is NOT independent
  of T0.
  - The slack-side line goes slack when its pin travels towards the anchor by more than the
    pre-stretch T0/k. The pin travel is the mean offset (1.1–1.2 m with the sum) plus the dynamic
    surge and the pin motion from 20–30° of tilt.
  - Past slack, the line hangs at its own weight (~1 N), whatever T0.
  - Scan: +40 % still fails (cluster 0.11 / 0.17, platform 0.10 / 0.12). **+60 % passes**
    (cluster 0.19 / 0.26, platform 0.21 / 0.23) and is specified, with k kept.
  - Surge stays 16.0–16.1 s. Yaw falls to 8.8 s (cluster) and 10.2 s (platform); decision 2
    accepts it for symmetric articles.
- ⚠ **Flags:**
  - The platform anchor working-load limit is **311 N (> 250 N)**: F2 must be confirmed for it.
  - The platform legs need **158 % rated elongation** (L0 shrinks to 2.49 m at +60 %).
  - The cluster and platform at T = 2.35 s reach **1.29 / 1.23 m of surge, beyond the +1.0 m
    tracking limit**. The lines are not stiffened for it; the tank procedure runs the extremes in
    increasing H with live offset monitoring.
  - Every worst case is at 20–30° of tilt, so the results are indicative (LEVEL1).

## Phase F (2026-09-25): attachment height — the rev B specification (`attachment_sweep.py`, `buoy_stability_check.py`)

Xabier approved the attachment-height comparison, with a decision rule so that it ends in a
specification. Rev A (Phase E) is withdrawn and its +60 % pretension REVERTED
(`mooring_spec.T0_SCALE` removed). `MOORING-SPEC.md` / `.pdf` are regenerated as rev B.

### F1. The reason for the change (recorded as instructed)

- The pin plane was chosen to remove calm-water tilt (criterion 1). **Its ~9 % tilt stiffening
  was accepted without comparing it to the resonance bandwidth.**
- The comparison, now made: at the pin plane with the design T0 and k, the tilt period shifts
  −9.19 / −8.83 / −9.04 % (buoy / cluster / platform). The tolerance is ζ/3 = 1.42 / 1.19 /
  1.05 %, so the pin plane is **6.5–8.6 × over**.
- The static pretension moment acts about the PIN. The dynamic coupling (≈ k·h², h = the
  attachment's height above the tilt mode's rotation centre) acts about that centre, near the
  CoG (M11a F1). No single height satisfies both unless the pretension is low.

### F2. Xabier's numbers, re-derived (standing rule)

- **ζ at H = 0.04 m** (`resonance_bandwidth.json`, tilt): 4.25 / 3.56 / 3.15 %. Xabier's
  4.2 / 3.6 / 3.2 % is confirmed. The sweep uses the unrounded values: limits 1.416 / 1.188 /
  1.050 %.
- **δ ≤ ζ/3 → response within 5 %:** 1/√(1 + 1/9) = 0.949. ✓
- ⚠ **"Tilt-period shift ≤ ~1 %":** the limits are 1.05–1.42 %. The buoy's is 1.42 %, not ~1 %.
- **Expectation "passing designs attach ~0.1–0.4 m below SWL with much lower pretension":**
  - the cluster and platform confirm it: −0.15 m, with T0 at 17 % and 8.5 % of the design;
  - ⚠ **the buoy does not:** it attaches at −0.50 m (F4).

### F3. Method

- The sweep grid, the modal and static evaluation and the kinematic slack estimate are as
  documented in `attachment_sweep.py` and the spec's *Design rule* section.
- **The tie-breaks:**
  - The rule selects the most pretension, then the attachment closest to the waterline.
  - A third tie-break was added: **the stiffest surge**, i.e. the least excursion, from the
    confirmed criterion 4. It decides only the buoy (F4).
  - An intermediate version used "least tilt shift" instead. It picked the buoy's k ×0.5. Its
    FloatSim battery is kept as `buoy_stability_check_k0p5.json`: 5 of 18 cases grow yaw
    (up to 0.3°) and 3 diverge, and its predicted H = 0.5 m offsets reach 1.95 m.
  - k ×1 halves the offsets (≤ 0.68 m, inside the field of view) and the cord strain. Its
    stability is not better: 6 cases grow yaw (all < 0.002° at 210 s) and 3 diverge, in
    different cases.
- **Static tilt:**
  - The per-buoy tilt stiffness estimate (75.36 / 81.45 N·m/rad) under-predicts FloatSim's
    settle at z = −0.15 m, by 1.4 % (cluster: 1.014°) and 6.4 % (platform: 1.064°).
  - **The settle wins:** T0 was trimmed by the settled ratio, then re-settled (step `trim`).
  - Result: 0.999° (cluster) and 1.0001° (platform). The platform's 1e-4° excess is inside the
    settle's averaging resolution; the next trim (0.01 % of T0) is immaterial and was not run.
- ⚠ **The drift sum's sign at the tilt resonance.**
  - FloatSim's own mean force is up-wave there. The restrained H = 0.12 m runs give −0.04 /
    −0.14 / −0.26 N (buoy / cluster / platform) at resonance, against +0.07 / +0.14 / +0.11 N
    at 1.4 s.
  - So at resonance the "sum" is smaller than the bound alone.
  - Kinematic operational T_min/T0 with the negative term dropped: cluster 0.539, platform
    0.157, against 0.565 / 0.179 with it. Both pass; the platform's margin is thin either way.
    The buoy's margin is large (≥ 0.93 in its moored runs).

### F4. The chosen designs

| | buoy | cluster | platform |
|---|---|---|---|
| attachment z | -0.50 m | -0.15 m | -0.15 m |
| k (×design) | 2.01 N/m (×1) | 4.15 N/m (×0.5) | 8.32 N/m (×0.5) |
| T0 per line/leg (×design) | 2.400 N (×0.6000) | 1.494 N (×0.1658) | 1.540 N (×0.0854) |
| tilt period free → moored (shift; limit ζ/3) | 2.761 → 2.723 s (-1.37 %; 1.42 %) | 2.856 → 2.835 s (-0.73 %; 1.19 %) | 2.924 → 2.902 s (-0.72 %; 1.05 %) |
| heave period free → moored (shift) | 2.561 → 2.548 s (-0.50 %) | 2.586 → 2.584 s (-0.08 %) | 2.606 → 2.606 s (-0.03 %) |
| calm static tilt: estimate / FloatSim settle | 0.000° / 0.000° | 0.985° / 0.999° | 0.939° / 1.000° |
| surge / sway / yaw (sweep) | 16.6 / 26.4 / 1.12 s | 23.1 / 51.2 / 25.12 s | 23.0 / 58.2 / 32.95 s |
| surge / sway / yaw (spec statics) | 16.6 / 26.4 / 1.12 s | 23.1 / 51.6 / 24.78 s | 23.0 / 59.0 / 33.40 s |
| op. T_min/T0 kinematic (≥ 0.15) | 0.940 | 0.565 | 0.179 |
| op. T_min/T0 moored FloatSim (≥ 0.15) | 0.930 (5 runs) | 0.562 (2 runs) | 0.189 (2 runs) |
| at-rest stretch / L0 | 1.195 / 3.930 m | 0.346 / 4.589 m | 0.171 / 4.038 m |
| pull K surge | 7.31 N/m | 15.05 N/m | 60.63 N/m |

**The single buoy.**
- Its pretension is capped by the collar, not by static tilt. The collar balances the lines, so
  the static tilt is 0.
- The collar's pretension stiffens pitch in proportion to T0·r, at any depth. The attachment
  depth only sets the k·h² part.
- **The conflict:** yaw stiffness is ∝ T0·r too. Every buoy design passing ζ/3 has yaw
  ≤ ~2.0 N·m/rad, a period ≥ 1.12 s: the zone where §C5 found the r = 0.12 m collar
  parametrically unstable at H = 0.5 m. FloatSim, the chosen design (18 cases, the drift sum):

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

- The operational band is stable, with no slack.
- "Stable" means yaw at round-off (≤ 1e-8 rad). The buoy is mirror-symmetric to round-off
  (§C6), so any larger yaw is the model's own parametric growth.
- Of the extreme band:
  - **3 diverge:** H = 0.35 m at 2.65 s and H = 0.5 m at 2.35 / 2.65 s. Rev A's buoy (yaw 0.86 s)
    was stable at H = 0.5 m, 2.65 s and diverged near 2.35 s (§C5).
  - **6 grow yaw slowly**, all < 0.002° at 210 s: H = 0.2 m at 2.65 / 2.75 s, H = 0.35 m at
    2.0 / 2.35 / 3.0 s, H = 0.5 m at 3.0 s.
  - The rest is stable.
  - Every predicted case is inside the field of view (surge ≤ 0.68 m).
- All are beyond LEVEL1: LEVEL2 decides whether the divergence is physical.
- Remedy (not designed): a pitch-neutral collar, e.g. a cross-flume bar, which gives yaw
  stiffness without the pitch term.

**The submerged collar** (z = −0.50 m, r = 0.2 m) is a hydrodynamic appendage the model does not
include:
- slender cross, 2 × 0.4 m bars ≈ 25 mm in diameter: added mass ≈ ρπ(D/2)²L ≈ 0.4 kg, drag area
  ≈ 0.02 m²;
- a solid disk would add (8/3)ρr³ ≈ 21 kg (the heave plate's is ≈ 7.9 kg), so it must not be a
  disk.

### F5. Step 5 — the targeted extremes and the moored operational checks (cluster, platform)

All on the chosen design, with the conservative drift sum:

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

- **Operational band:** no slack, confirmed directly in FloatSim. This supersedes the kinematic
  estimate.
- **Extremes:**
  - The down-flume lines go slack (allowed) and **stay slack through the wave train**: the mean
    offset (2.5–3.4 m) far exceeds their pre-stretch (0.35 / 0.17 m). Their peak is ≤ 0.05 N,
    their hanging weight.
  - So FloatSim shows **no re-tension load** in the steady state. They re-tension only as the
    article drifts back after the waves stop (not run). FloatSim's line is quasi-static, so
    snap dynamics are not modelled either way.
  - Peak line tension (the up-flume lines): 15.2 N (cluster), 29.4 N (platform, per leg).
  - ⚠ **Yaw 0.47–1.8°** (rev A: 0.04–0.23°).
    - It is forced by the coupled-kernel asymmetry (§C6), and rev B's yaw is 8–11 × softer.
    - Steady or growing is not established (only the peak was kept).
    - Three of the four exceed Phase D's proposed 0.5° flag.
  - ⚠ **The offsets are far beyond the +1.0 m field of view**, 2–3 × rev A's. The k ×0.5 lines
    are soft and the slack side stops pulling.
  - The rule has no excursion criterion. The cluster's k ×1 alternative (−0.30 m, T0 ≈ 1.3 N)
    has about 2 × the surge stiffness; the platform has no passing k ×1 design. **Xabier to
    decide whether excursion joins the rule.**
- The submerged lines break the local surface in the H = 0.5 m troughs (column "line above
  local surface"). Line drag is not modelled.

### F6. Open for Xabier (rev B)

1. **Static tilt ≤ 1°:** proposed; confirm. It sets the cluster and platform pretension.
2. **Excursion in the rule** (F5): the rule's pick puts the H = 0.5 m extremes 2.5–3.3 m
   down-flume.
3. **The buoy's yaw vs tilt conflict** (F4): accept the diverging extremes, or design a
   pitch-neutral collar.
4. **Underwater wall anchors** (F1 of the spec) to HWRL.

## Phase G (2026-09-25): two cord sets and the figures — the rev C specification (`extreme_set.py`, `mooring_figures.py`)

Xabier's decisions on rev B (`fb6ba5b`):
- The operational design stands, and calm static tilt ≤ 1° is **CONFIRMED**.
- Add an extreme cord set on the same anchors and attachment points.
- Add figures generated from the committed geometry.

**Context for the record** (Xabier): the rev B decision rule omitted the confirmed excursion
criterion (Phase B criterion 4). That was the advisor's error; the third tie-break partly
compensated. The conflict is real: minimal interference needs soft lines, and station-keeping in
extreme waves needs stiff ones. It is resolved with TWO cord sets, not by compromising one
design.

### G1. Xabier's reasoning, re-derived (standing rule)

1. **"Calm equilibrium depends on T0, not k; no new settle is needed" — TRUE for the at-rest
   tension, FALSE for the code's nominal T0.**
   - The nominal T0 is the tension at the design (untilted) chord. The calm 1° tilt moves each
     attachment δ ≈ 15 mm towards its anchor, so the at-rest tension is T0 − k·δ.
   - With the same nominal T0, the extreme cords sit at 1.21 N (cluster) and 0.87 N (platform)
     at rest instead of 1.44 / 1.42 N: a different, less tilted equilibrium.
     FloatSim's settle of the cluster's extreme set at the nominal T0 (cache `cluster:ext_k5`)
     lands at 0.88° calm tilt, against 1.00°.
   - With the at-rest tension matched (nominal T0 = T_rest + m·(T0 − T_rest)), the line forces at
     the calm geometry are identical, so the operational settle IS the extreme set's calm
     equilibrium (the same 1° tilt) and no new settle is needed. FloatSim's joint-projected static residual at the operational settle: cluster 0.0077 N with the operational lines, 0.0073 N with the extreme lines (nominal T0 instead: 0.13 N); platform 0.0024 N with the operational lines, 0.0022 N with the extreme lines (nominal T0 instead: 0.24 N) (extreme_set_residuals.json).
   - The spec specifies the at-rest tension; the tank sets it by tension or calm tilt anyway.
   - The k scan used the nominal T0 (offsets differ by ~1 cm). The final runs of the chosen k use
     the at-rest-matched pretension.
2. **"Offset scales ~1/k, so ~3× the rev B k" — not borne out: cluster ×5 (criterion 4) / ×4.5 (mean only), platform ×6 (criterion 4) / ×5 (mean only).**
   - The pretension (geometric) stiffness does not scale with k.
   - The down-flume lines go slack under the drift, so only half the lines restore.
   - The buoys' tilt carries the attachments further down-flume.
   - A quasi-static rigid-pull estimate put it at ~×3.5–4 (`extreme_set.py predict`). FloatSim
     needs ~20 % more.
3. **"Choose the SMALLEST k whose MEAN offset ≤ 1.0 m" — itemised.**
   - The confirmed criterion 4 is mean + dynamic amplitude ≤ 1.0 m. The record wins, so the
     spec selects on the run's maximum surge.
   - The mean-only multiples are reported alongside: cluster ×4.5, platform
     ×5.
4. **"Run the buoy's extreme set if its rev B offsets exceed 1.0 m" — they do not.**
   - The predicted H = 0.5 m cases reach 0.68 m of surge at most (mean ≤ 0.46 m).
   - For the 3 diverging cases, the drift bound alone gives 0.19–0.67 m of static offset. The
     FloatSim own-force share there is unknown.
   - The buoy keeps one cord set.
5. **The extreme set breaks the confirmed criterion 2 (surge ≥ 14 s)** — not in Xabier's list,
   itemised: cluster +13 % (T_surge 10.5 s) / platform +15 % (T_surge 9.6 s) of wave-frequency surge amplification at T = 3.5 s.
   - Criteria 2 and 4 cannot both hold at H = 0.5 m (≥ 14 s caps k at ~×2.8, leaving ≳ 1.5 m of offset, extrapolated from the k scan).
   - **ACCEPTED as declared** (Xabier, 2026-09-25; G5).
6. **"Raising T0, keeping calm tilt ≤ 3°, the confirmed mean-tilt allowance" — itemised.**
   - The confirmed criterion 1 is the MEAN tilt ≤ 3° at the maximum H under waves, and it
     already contains the drift-induced tilt.
   - So a 3° calm tilt plus the drift tilt would exceed it. It is not a calm-tilt allowance.
   - Result: Not run, because there is nothing to cut. In the final runs the down-flume lines go slack and **stay slack through the wave train** (their peak is their hanging weight). Their pre-stretch is far below the mean offset, and it stays so even at the 3° calm-tilt cap (T0 ≈ 3× the at-rest tension: pre-stretch cluster 208 mm against a mean offset ≥ 0.61 m; platform 85 mm against a mean offset ≥ 0.62 m). Raising T0 would only add to the up-flume peak.

### G2. The extreme set

| article | k (× operational) | T = 2.35 s: mean / max surge | T = 2.65 s: mean / max surge | peak line tension | slack lines: peak re-tension | verdict (max surge ≤ 1.0 m, both periods) |
|---|---|---|---|---|---|---|
| cluster | ×3.5 (14.53 N/m) | 1.18 / 1.30 m | 0.87 / 1.02 m | 20.02 N | 0.07 N | fails |
| cluster | ×4 (16.60 N/m) | 1.04 / 1.17 m | 0.77 / 0.92 m | 20.80 N | 0.08 N | fails |
| cluster | ×4.5 (18.68 N/m) | 0.93 / 1.07 m | 0.69 / 0.84 m | 21.48 N | 0.09 N | fails (mean alone passes) |
| cluster | ×5 (20.75 N/m) | 0.83 / 0.98 m | 0.62 / 0.78 m | 22.12 N | 0.09 N | **CHOSEN** |
| cluster | ×5.5 (22.83 N/m) | 0.76 / 0.91 m | 0.57 / 0.73 m | 22.76 N | 0.10 N | pass |
| cluster | ×6 (24.90 N/m) | 0.69 / 0.85 m | 0.53 / 0.68 m | 23.44 N | 0.12 N | pass |
| cluster | **×5 FINAL** (at-rest-matched T0) | 0.82 / 0.97 m | 0.61 / 0.77 m | 22.12 N | 0.10 N | **PASS: the spec** |
| platform | ×3.5 (29.11 N/m) | 1.30 / 1.41 m | 1.00 / 1.15 m | 38.60 N | 0.05 N | fails |
| platform | ×4 (33.27 N/m) | 1.16 / 1.29 m | 0.89 / 1.05 m | 40.09 N | 0.06 N | fails |
| platform | ×4.5 (37.43 N/m) | 1.05 / 1.18 m | 0.80 / 0.97 m | 41.44 N | 0.06 N | fails |
| platform | ×5 (41.59 N/m) | 0.95 / 1.09 m | 0.74 / 0.90 m | 42.72 N | 0.07 N | fails (mean alone passes) |
| platform | ×5.5 (45.75 N/m) | 0.88 / 1.02 m | 0.68 / 0.85 m | 43.99 N | 0.07 N | fails (mean alone passes) |
| platform | ×6 (49.91 N/m) | 0.81 / 0.96 m | 0.63 / 0.81 m | 45.28 N | 0.07 N | **CHOSEN** |
| platform | **×6 FINAL** (at-rest-matched T0) | 0.80 / 0.95 m | 0.62 / 0.79 m | 45.27 N | 0.08 N | **PASS: the spec** |

- **Chosen** (criterion 4, max surge ≤ 1.0 m at T = 2.35 and 2.65 s): cluster ×5,
  platform ×6.
- **Declared** (not a criterion): cluster tilt -3.77 %, heave -0.09 %, surge 10.4 s, platform tilt -4.65 %, heave -0.03 %, surge 9.4 s.
- **Stiffness band:** −0 / +30 %. A softer cord moves H = 0.5 m out of the window.
- **Pre-stretch:** only 69 mm / 28 mm. Set T0 by tension or by the calm tilt, not by length.

| article | set | case (drift sum) | min / max line tension | slack lines: peak re-tension | mean offset (range) | max tilt | max yaw | line above the local surface | validity |
|---|---|---|---|---|---|---|---|---|---|
| cluster | extreme, k ×5 | H 0.5 m, T 2.35 s | 0.06 / 22.12 N | stay slack (≤ 0.07 N, their hanging weight): no re-tension in the wave train | 0.82 m (0.66–0.97) | 21.6° | 0.92° | breaks the surface by 0.18 m | indicative (tilt > 5.7°) |
| cluster | extreme, k ×5 | H 0.5 m, T 2.65 s | 0.06 / 20.28 N | stay slack (≤ 0.10 N, their hanging weight): no re-tension in the wave train | 0.61 m (0.45–0.77) | 30.5° | 0.81° | breaks the surface by 0.17 m | indicative (tilt > 5.7°) |
| platform | extreme, k ×6 | H 0.5 m, T 2.35 s | 0.05 / 45.27 N | stay slack (≤ 0.06 N, their hanging weight): no re-tension in the wave train | 0.80 m (0.64–0.95) | 27.9° | 2.55° | breaks the surface by 0.17 m | indicative (tilt > 5.7°) |
| platform | extreme, k ×6 | H 0.5 m, T 2.65 s | 0.05 / 44.69 N | stay slack (≤ 0.08 N, their hanging weight): no re-tension in the wave train | 0.62 m (0.44–0.79) | 32.9° | 2.32° | breaks the surface by 0.16 m | indicative (tilt > 5.7°) |

### G3. Figures (`mooring_figures.py`) and their spot-check

- Every figure is generated from committed data:
  - the decks, for the geometry;
  - `spec_statics*.json`, for anchors, attachments and T0;
  - the FloatSim settle and `line_hardware.line_states`, for the line profiles;
  - `build_spec.sweep_selection`, the same rows as the spec's sweep table, for the rationale;
  - the run JSON, for the offsets.
- The spot-check reads back from the drawn figures one anchor coordinate and the attachment
  depth per article, and compares them with the spec's line tables:

- buoy: table line 1 anchor (-5.000, +1.830, -0.500), attachment z -0.501 m | plan label found True, plan marker at it True; elevation markers [-0.501], depth label 'attachment z = -0.50 m: 0.50 m bel…' -> MATCH
- cluster: table line 1 anchor (-5.000, +1.830, -0.150), attachment z -0.150 m | plan label found True, plan marker at it True; elevation markers [-0.15], depth label 'attachment z = -0.15 m: 0.15 m bel…' -> MATCH
- platform: table line 1 anchor (-5.000, +1.830, -0.150), attachment z -0.150 m | plan label found True, plan marker at it True; elevation markers [-0.15], depth label 'attachment z = -0.15 m: 0.15 m bel…' -> MATCH

### G4. The 3D viewer

- The "Flume Mooring Motion" artifact is republished with the rev C arrangement: a static calm
  equilibrium per article and cord set, with provenance labels.
- Its earlier animations used a superseded mooring (lines at the still-water line). They are
  withdrawn. No rev C wave run has saved frames, so nothing is animated. The 0.1 rad validity
  gate stays.

### G5. Rev C accepted: record items (Xabier, 2026-09-25)

1. **Criterion 2 for the extreme set: ACCEPTED as DECLARED**, on the same basis as the tilt shift.
   - The criterion limits how much the mooring changes the wave-driven response. For the extreme
     set that change is declared, not a criterion: those tests measure loads and survival, and
     the tank and FloatSim see the same mooring.
   - Recorded: surge periods cluster 10.5 s / platform 9.6 s;
     wave-frequency surge amplification cluster +13 % / platform +15 %
     at T = 3.5 s.
   - Reason: criteria 2 and 4 cannot both hold at H = 0.5 m, and criterion 4 (the tracking
     window) governs for this set.
2. **Cord-set identification** (the installation checks and the summary procedure).
   - Both sets share the same at-rest tension and the same 1° calm tilt, so the calm tilt CANNOT
     tell which set is installed.
   - A static pull CAN: cluster 15.0 → 73.2 N/m (×4.9; threshold 33 N/m), platform 60.6 → 349.3 N/m (×5.8; threshold 146 N/m).
   - The pull is a REQUIRED check before every extreme series, with the expected stiffness of
     each set tabulated in the spec §5. Verified: the ratio is 4.9–5.8×, consistent with
     Xabier's 5–6×.
   - The consequence of getting it wrong, from the rev B runs: extreme waves (H = 0.5 m) on the
     operational cords drift the cluster / platform 3.39 m (mean 3.27 m) / 3.53 m (mean 3.41 m).
   - The same runs show a further consequence, added to the spec: cluster peak tension 15.2 N against their working load 6.3 N, stretch 3.67 m against their elongation capacity 0.63 m; platform peak tension 29.4 N against their working load 8.4 N, stretch 3.54 m against their elongation capacity 0.42 m.
3. **Anchor rating: every wall anchor rated ≥ 300 N** (the platform's working load is
   267.4 N). F2 is updated in the spec; the build asserts that no working load
   exceeds the rating.

## Phase H (2026-09-25): does the operational mooring need pretension? (evaluation; NO spec change)

Xabier's question, answered in `PRETENSION-EVAL.md` (`pretension_study.py`, `pretension_study.json`).
Rev C stands; Xabier decides.

- **V1 ("taut minimum")**: T0 = k (10 mm tolerance + ~13 %/day wet creep of the loaded stretch)
  = 0.31 / 0.44 N per line/leg (cluster / platform).
  - The calm tilt falls from 1.00° to 0.25 / 0.33°.
  - The tilt shift falls from −0.73 / −0.72 % to −0.54 / −0.60 %.
  - The wave-frequency response is unchanged: heave RAO within 1.2 %, max tilt 0.5–0.7° lower.
  - The lines lose their stretch for 18–40 % of each resonant cycle, which FloatSim's
    quasi-static line does not model.
  - The H = 0.12 m drift offsets grow ×1.44–1.49.
  - The calm-tilt gauge shrinks to 0.25–0.33°.
  - The cluster's extreme set would leave the window (1.054 m).
  - **Recommendation: keep V0.**
- **Physics:**
  - At T0 = 0, surge is 34 % and sway 26–28 % of V0, not 50 %: the weight-sagged cord's
    catenary compliance.
  - The tilt interference falls with T0 rather than staying unchanged.
  - A submerged-weight cord self-tensions to ~0.2 N.
- **Line weight:** the record already uses the submerged 0.02 N/m. No correction.
- **Buoy:** it keeps its pretension. V1 puts its yaw at 2.48 s, in the wave band and at the
  principal parametric zone.
- **Flag for rev C's installation check:** wet creep relaxes V0's calm tilt from 1.00 to ~0.87°
  over a test day.

**DECISION (Xabier, 2026-09-25): KEEP V0, rev C's operational pretension.**

- **The decisive reason:** V0 keeps every operational cord taut, so the tank and FloatSim see the
  same, modelled mooring. V1's slack/taut cycling in the resonant cases (18–40 % of each cycle)
  is unmodelled dynamics in exactly the cases the validation depends on.
- Pretension is not needed for wall avoidance. **It is kept for model fidelity and installation
  control.**
- The creep finding is carried into the spec as rev C.1: the calm-tilt check, a daily re-check
  and the pull-test band allow for creep.

### H.1 Spec amendment rev C.1 (the creep allowance; MOORING-SPEC §4, §5, §7)

- **Creep assumption, recorded:** wet natural rubber, ~4 % of the stretch per decade of time (dry
  ~2.4–4 %). Sources: Gent, *Engineering with Rubber*, ch. 7; PMC6728486. It is an assumption to
  confirm against the purchased cord's datasheet.
- **At the calm geometry it relaxes the at-rest tension:** 96.0 / 92.9 / 87.4 % left at 10 min /
  1 h / 1 day. The calm tilt goes 1.00 → 0.96 / 0.93 / 0.87°.
- **The pull-test band** moves its low edge with the tension left, for both cord sets.
- **Daily re-check** before the first run: calm tilt (cluster, platform) or tension (buoy).
  Re-tension below:

  | Article, set | Threshold | Basis | Reached after |
  |---|---|---|---|
  | Cluster, operational | 0.52° | no-slack margin | never |
  | Cluster, extreme | 0.72° | tracking window, FloatSim slope 80 mm per N | never |
  | Platform, operational | **0.95°** | no-slack margin | **~14 min** |
  | Platform, extreme | none | window margin exceeds the whole tension | — |
  | Buoy | **2.23 N** | yaw 15 % clear of the T_p/2 zone | **~54 min** |

- ⚠ **Found while implementing, for Xabier:**
  - The platform's operational margin (T_min / T_rest = 0.19 at H = 0.12 m near resonance) is
    thinner than a day's creep. The spec therefore re-tensions it before the first run AND before
    its H = 0.12 m near-resonance cases.
  - Its lines stay taut far longer: the taut limit is 0.81°, about 37 days of creep.
  - The buoy is likewise re-tensioned before its extreme series.
  - Alternative: accept a reduced margin on the platform (T_min / T_rest ≈ 0.07 at the end of a
    day, still taut).

## Open decisions for Xabier (after Phase C round 2; STOP before Phase D)

1. **BEM regeneration at NT = 36** (§C8). Every resonance moves 0.040–0.077 s (> 0.025 s). The cost
   is ~1.7 h for the platform BEM, ~5 min for the cluster (the single buoy is done), then ~2–3 h of
   FloatSim re-verification. Approve before Phase D.
2. **Line element** (§C7): elastic cord recommended for all three; counterweight lines fail the
   field of view. The hardware choice is yours; F7 goes to HWRL.
3. **Drift double counting** (§C7). FloatSim's own mean force is 20–70 % of the bound at
   H = 0.5 m. Choose:
   - keep bound + FloatSim (conservative; at T = 2.35 s the cluster and platform then fail the
     field of view, slack and clearance); or
   - apply only the part of the bound FloatSim does not carry.

   Decision 1's D0 check is answered: there IS overlap.
4. **Dynamic slack of the cluster/platform lines** at H = 0.5 m near resonance (0.04–0.19 T0 against
   0.15). Indicative (LEVEL1), but it points to more pretension, as for the buoy.
5. **Extreme buoy cases near the pitch resonance** (H ≥ 0.35 m around 2.35 s) diverge in FloatSim
   (pitch 24–28°). Run them flagged, or drop them until LEVEL2.
6. **The yaw "not small" threshold** for Phase D's flag: 0.5° or 5 % of max pitch (proposed).
7. **LEVEL2 as the next FloatSim milestone** (decision 7): proposed, not started.
8. **Facility assumptions F1–F7** to HWRL before any hardware is bought.

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
