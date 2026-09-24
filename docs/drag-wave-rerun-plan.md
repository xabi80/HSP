# Rerun cycle after STEP 5 PR1 (wave-relative Morison drag): plan and cost

**Status: PLAN ONLY, not run.** Written 2026-09-24, after STEP 5 PR1 (`14518a3`, `bf15e9a`,
`1ea8662`). The OrcaFlex comparison stays HELD until the M11b fan (§2.1) is rerun.

## 1. The rule

- **Calm-water results are unaffected.** In still water the fluid velocity is zero, so absolute
  and relative velocity are the same; the calm path is byte-identical after PR1 (all 12 drag decks
  plus two integrations, `scripts/drag_calm_byte_identity.py`). This covers every free decay:
  M5–M11 decay gates, spar-fin / OSU / cluster decays, the flume pitch and tilt-mode decays, and
  the flume static equilibria.
- **Every wave-response result gets one rerun**, with `build_system(..., drag_wave=wave,
  drag_wave_ramp=ramp)`, passing the same wave and ramp as the excitation.
- **Frequency-domain drag tools rerun too.** `drag_fd.py`, `pvr_fd.py` and `mode_shape_fd.py`
  evaluate "the exact FloatSim calm-water drag closure". They need the wired closure evaluated at
  the cycle times after the ramp.

## 2. Prerequisite decisions (before any wave-response rerun)

1. **Station-keeping for unmoored wave runs** (tracker `EXCITATION-FIXED-REFERENCE-VS-DRIFT`).
   - Position-aware drag couples a body's mean drift to the wave phase, while the linear BEM
     excitation stays at the start position.
   - The M11b platform at H = 1.0 m, T = 3.141 s drifts 44 m in 508 s (0.093 m/s). Its heave
     envelope then beats at λ/v = 166 s (measured about 170 s), quasi-periodic and undamped, so
     it can never meet the single-frequency settle criterion.
   - Recommended: (a) a soft surge mooring matching the tank design (T_surge ≈ 15 s), applied
     identically in OrcaFlex. The alternatives are (b) advecting the excitation reference (a core
     change) or (c) running unmoored and reporting drift and phase slip per case.
2. **LEVEL2 validity.** Every tilt above 0.1 rad is reported as indicative only
   (`LEVEL2-INTEGRATOR-UNWIRED`). The lone free buoy's yaw goes unstable above ~13° of pitch.
   The corrected spar-fin pitch period (3.26 s) now sits inside the fan band, so its cases at
   3.141–3.3 s will be the large-pitch ones.
3. **Catenary cold start (STEP 5 PR4).** The moored flume platform at T = 3.5 s failed in
   `solve_catenary`. Rerun it after the warm-start PR, or keep it free-only.

## 3. Tier 1: the three named reruns

### 3.1 M11b 12-buoy fan (the OrcaFlex deliverable)

- **35 cases** (7 periods × 5 heights, Cd_n = 5.0) through `platform_rao_pilot.run_case`, with the
  §2.1 station-keeping.
- **Off-resonance collapse test (explicit).** Drag excitation is quadratic in wave amplitude, so
  away from resonance the RAO must GROW with H. That reverses the calm-water fan, where RAO falls
  with H everywhere.
  - Cases: the fan's off-resonance periods T = 2.0, 2.5, 2.8 s at all 5 heights (already in the
    35), plus T = 1.6 and 1.8 s × 5 heights (**+10 cases**) below the band, where the calm-water
    RAO was < 0.03.
  - Pass criterion: RAO strictly increasing in H at each of T = 1.6, 1.8, 2.0, 2.5 s, and a
    fitted exponent p > 0 in RAO ∝ H^p (p → 1 where drag excitation dominates the linear
    excitation).
  - Measured anchor: T = 2.0 s, H = 0.30 went 0.0019 → 0.0521.
- **The H = 1.0 m, T = 3.141 s case** (and every other capped case). Record drift speed, λ/v and
  the stroboscopic section per case (`pr8_h1_response.py`). With (a) the drift is bounded and the
  case should settle; if it still does not, that is a new finding, not a cap to raise.
- Cost: 45 cases × ~7 min (measured 5.4–8.7 min per settled case; 35 min for a capped one) at
  6 in parallel ≈ **1–1.5 h**.

### 3.2 Spar-fin single-buoy RAOs (with the pitch-block fix, `f9f84d9`)

- `sparfin_rao` 70 cases (Cd 5 and Cd 1, 5 heights × 7 periods), fin fan 250 (5 configurations
  × 50), refined periods and no-fin ≈ 35. About **355 cases**.
- Cost: ~0.3 min each (single body) at 8 in parallel ≈ **15–25 min**.
- Watch for the corrected pitch resonance (3.26 s) in-band: LEVEL2 flags, and possible yaw
  instability at large pitch (the pre-fix check blew up at H = 0.12, T = 3.3 s).

### 3.3 Flume viewer rows and pitch check

- 17 viewer rows (buoy, cluster and platform, moored and free, H = 0.1 m, T = 2.2 / 2.9 / 3.5 s),
  plus the failed moored platform T = 3.5 s after STEP 5 PR4.
- `pitch_check.py` forced sweep: 44 buoy cases + 2 cluster cases. Its decays are unaffected.
- Cost: platform 6 × ~8 min, cluster 6 × ~1 min, buoy 6 × ~0.2 min, pitch sweep ~46 × 0.3 min;
  in parallel ≈ **30 min**.

**Tier 1 total ≈ 2–2.5 h wall.**

## 4. Tier 2: the rest of the wave-response inventory

| Study | Cases | Per case | Wall (parallel) |
|---|---|---|---|
| M11b Cd_n = 1.0 fan, pilot, H = 0.3 period sweeps, low-H fan, Cd checks | ~115 | ~7 min | ~2.5 h (6×) |
| 12-buoy fin sweep (`platform_fin_fan`) | 275 | ~7 min | ~5.5 h (6×) |
| 16-buoy fin sweep (`platform16_fin_fan`) | 400 | ~10 min (1.4× the 12-buoy) | ~11 h (6×) |
| M10 cluster RAO + cluster fin fan | 70 + 300 | ~1 min | ~50 min (8×) |
| Pin-vs-rigid (TD + drag scan) | ~36 | ~10 min | ~1 h (6×) |
| Flume-wall TD cross-check (open/walled × 0°/45° × 4 T) | 16 | ~9 min | ~25 min (6×) |
| FD drag tools (`drag_fd`, `pvr_fd`, `mode_shape_fd`) | small | minutes | < 30 min |
| **Tier 2 total** | **~1,200** | | **≈ 22 h wall** |

- Memory: platform processes hold 2–6 GB (tracker `CONSTRAINED-INTEGRATOR-SWEEP-MEMORY`), so use
  one process per case and at most 6 platform processes at a time on this 64 GB machine.
- **Downstream to regenerate afterwards:**
  - cross-model plots and 4-model surfaces;
  - the fin & array-size report;
  - the OSU deck's pin-vs-rigid slides;
  - the flume-wall technical deck and the TEAMER sidewall rebuttal;
  - the flume mooring README, `RESPONSE-mooring.md` and deck (still held for Decision 1).

## 5. Order

1. Decide §2.1 (station-keeping) and land STEP 5 PR4 (catenary warm start).
2. Tier 1 §3.1 (fan + collapse test): this unblocks the OrcaFlex comparison.
3. Tier 1 §3.2 and §3.3.
4. Tier 2, then regenerate the downstream materials.
