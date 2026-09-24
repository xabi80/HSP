# Flume mooring design: basis and proposed criteria (Phases A and B)

**Status (2026-09-24): FOR REVIEW.** Xabier confirms the test matrix, the design intent and the
criteria before any FloatSim fix (C1, C2) or verification run (Phase D).

**Goal:** design the station-keeping mooring for the actual OSU Large Wave Flume test. It is not
an OrcaFlex comparison.

**Decisions in force:**
- pin-level attachment;
- line dry weight 0.3 N/m (assumed, pending hardware);
- design intent: a SOFT restraint that disturbs the measured motions as little as possible.

Numbers are re-derived from the record: the code, committed outputs, and `design_basis.py` /
`design_basis.json` (FloatSim matrices, no time-domain runs). "⚠" marks a gap or inconsistency
in the record.

---

## Phase A — design basis

### A(a) Test matrix: what the record says

| Source | Heights | Periods | Notes |
|---|---|---|---|
| `README.md` §scope; `mooring_sizing.py` `H_LIST`, `T_WAVE` | **0.2, 0.3, 0.4, 0.5 m** | **1.4–4.0 s** (sizing sweep, 53 values; no discrete test periods) | "moderate matrix, max 0.5 m"; regular waves, head seas; H/L ≤ 1/15 cap |
| `studies/rao_cross_model_out/README.md` | 0.04–0.12 m ("realistic operational band", 2–6 m full scale at 1:50) | — | ⚠ a different scale/height basis |
| `docs/m11b-pr8-rao-closure.md` | 0.03–1.2 m | 1.2–3.3 s | OrcaFlex band; withdrawn with the comparison |
| `platform-12buoy/flume-wall-effect/REBUTTAL-sidewall.md` | — | avoid **2.19 / 1.53 / 1.25 s** | flume transverse cut-on (sloshing) periods |

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
| Anchors | at the walls, ±5.0 m up/downstream, y = ±1.83 m, at the pin plane **+0.717 m** | design; ⚠ availability at +0.72 m unknown. HWRL Q1 asked about the SWL and must be re-asked. |
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

⚠ Inconsistencies in the record:
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

**Yaw** is uncertain.
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

## Phase B — proposed acceptance criteria (for Xabier to confirm)

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
  - At 0.3 N/m the sag is `wL²/(8T0)` = **0.30 m for the single buoy** (low T0 = 2.23 N), 0.06 m
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
  large H the free runs are not stationary. **The free reference needs a decision.**

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
  FloatSim's mean. For Xabier to choose.

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

## Questions for Xabier (before C1/C2 and any run)

1. **Test matrix:** heights, periods, regular or irregular, headings, moored decays. Plus the
   1:50 / 0.04–0.12 m versus 0.2–0.5 m discrepancy.
2. **Design intent** (soft restraint) and the criteria values above, including the 3° mean-tilt
   allowance and the ratio of 4.
3. **Drift in Phase D:** apply the recorded bound as a mean force, or superpose it.
4. **Free reference for criterion 6** at large H (unmoored runs drift).
5. **Single-buoy yaw:** a collar or bridle radius, since pin-level lines on the axis give no yaw
   restoring. And confirm pin-plane rather than hub attachment.
6. **HWRL:** re-ask anchors at +0.72 m (not the SWL), anchor WLL, test-section position, and
   tracking coverage.
