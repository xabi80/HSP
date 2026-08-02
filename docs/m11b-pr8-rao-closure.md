# M11b PR8 closure — 12-buoy platform heave RAO + buoy acceleration

**Deliverable.** Platform-heave RAO and per-buoy heave-acceleration
outputs for Xabier's **external OrcaFlex comparison** of the 12-buoy
articulated platform. This PR **PRODUCES comparable outputs; it does
NOT judge agreement** — there is no pass/fail tolerance here. The
correction recorded in §3 below is the substantive result: the
provisional "no resonance" reading from the single-amplitude STEP A/B
sweep was wrong, and this document supersedes it.

**System.** 4 clusters × 3 buoys, `yaw_locked` joints
buoy→hub→platform: **17 bodies / 102 DOF / 64 constraint rows**.
Coupled N-body BEM (shared RAO database, heading 0 only), Cummins
convolution, velocity-level KKT joints. Morison drag: plate
`Cd_n = 5.0` (KNOWN heave-plate broadside, M11a PR4), `Cd_t = 1.5`,
`r = 0.215 m`, `z_body = −0.2617 m`; spar distributed cylinder
`D = 0.1682 m`, `Cd = 1.2`, 10 segments.

**Wave conditions.** Operational band H ∈ [0.03, 1.2] m,
T ∈ [1.2, 3.3] s. **RAO = |response amplitude| / A**, A = H/2 (per
wave amplitude). Below T ≈ 2 s the platform-heave RAO is < 0.03
(sub-resonant, negligible); the informative band is 2.8–3.3 s.

---

## 1. Baseline / record verification (merge-gate reconciliation)

Full-suite pre-merge regression, full scope incl. slow
(`python -m pytest -q -p no:cacheprovider`, 2:40:09):
**800 passed / 50 skipped / 20 xfailed / 1 failed.**

Reconciled against the record (re-derived from commit messages, not
memory):

| point | passed/skip/xfail/fail | commit | source |
|-------|------------------------|--------|--------|
| M11a PR3 | 764 / 50 / 20 / 1 | — | quoted in `6a8b855` |
| M11a close | **771** / 50 / 20 / 1 | `6a8b855` | "+7 (the PR4 gates), ZERO regressions" |
| M11b PR7 | **779** / 50 / 20 / 1 | `5adcfbc` | "+8 vs PR4's 771 (3 PR6 + 5 PR7); ZERO regressions" |
| **M11b PR8** | **800** / 50 / 20 / 1 | this PR | **779 + 21 new PR8 tests**, ZERO regressions |

- **Label correction (itemized per "the record wins").** The 779 run
  is **M11b PR7 (`5adcfbc`)**, not "M11a PR7" — M11a has no PR7
  (it ran PR1–PR4, then the tank-gated PR5/status). The *figures*
  771/50/20/1 and 779/50/20/1 are exactly correct; only the milestone
  label differed.
- **The single `/1`** is the carried F2 hypothesis red
  (`test_property_F_ref_equals_T_pullback_of_F_attach`), tracker
  entry `F2-HYPOTHESIS-TOLERANCE-EMPIRICAL`. It imports none of PR8's
  three changed core files (`retardation.py`, `driver.py`,
  `newmark.py`); `connector.py` and the test are byte-identical to
  `main`. **NOT a PR8 regression.** This is now the third full-suite
  run to cost a triage cycle (M9 PR2 → M11b PR7 → M11b PR8); the
  standing recommendation is to clear it in the carried `fix-` branch
  via a magnitude-scaled / condition-number bound, not a third bare
  `rtol` loosening (tracker updated).

The 21 new PR8 tests: kernel-decay-exemption (10),
restoring-PSD-gate (7), newmark-stop-check (4).

---

## 2. The natural period is settled and drag-invariant (STEP A)

Undamped heave natural period from the assembled coupled system,
`T_n = 2π√((M + A_inf)/K)`:

- **T_n = 3.143 s** (STEP A, condensed heave line of the 102-DOF
  assembly).
- **Drag-invariant by construction, not merely empirically.** The
  Morison drag enters the integrator as a velocity-dependent RHS
  `state_force`; it does **not** modify the LHS mass `M + A_inf` or
  restoring `K`. `build_system` produces an identical LHS / kernel /
  constraint set for every `Cd_n` (only the `state_force` is swapped
  via `dataclasses.replace`), so `T_n = 3.143 s` is identical across
  all `Cd_n`. This is the physics note ("damping does not enter
  M/K") upheld at the code level.
- **No period gap vs a single cluster.** Decomposition ratios
  platform-vs-4×cluster: A33 = 1.0223, C33 = 1.0000, M = 1.0255 — all
  ≈ unity; cluster reference values 98.01 / 64.07 / 663.24 reproduced
  exactly. The 12-buoy assembly does not shift the heave natural
  period away from the cluster building block.

---

## 3. The correction — the platform DOES have a heave resonance

### 3a. What STEP A/B concluded, and why it was wrong (superseded)

STEP B swept the H = **0.30 m** period response over 1.2–6.0 s at the
operational `Cd_n = 5.0` and found:

- **no overshoot** anywhere (platform-heave RAO < 1 at every period);
- RAO rising monotonically toward the long-wave limit, reaching
  **0.9915 at T = 6.0 s**.

The provisional conclusion recorded in commit `958c19f` and the first
`rao_3d.png` title was: *"no resonance peak — the 3.14 s heave mode is
drag-damped-out."* **This is superseded.** It was an artifact of
sampling a **single** wave amplitude (H = 0.30 m) that already sits in
the drag-damped regime. Nothing about H = 0.30 is special; it simply
does not probe the amplitudes where the resonance is visible.

### 3b. What the amplitude fan measured (35 runs, Cd_n = 5.0)

Sweeping **7 periods × 5 heights** at the operational `Cd_n = 5.0`
(same build, adaptive settle) — platform-heave RAO:

| T (s) | H=0.05 | H=0.15 | H=0.30 | H=0.60 | H=1.00 |
|-------|--------|--------|--------|--------|--------|
| 3.000 | **1.190** | 0.706 | 0.499 | 0.339 | 0.245 |
| 3.141 | **1.393** | 0.808 | 0.564 | 0.379 | 0.277 |
| 3.257 | **1.485** | 0.873 | 0.611 | 0.412 | 0.306 |
| 3.300 | **1.493** | 0.893 | 0.628 | 0.426 | 0.317 |

- **Overshoot (RAO > 1) appears ONLY at H = 0.05** in the fan; by
  H = 0.15 quadratic drag has already pushed the near-resonance RAO
  below 1.
- **Amplitude gating**, at T = 3.257: RAO **1.485 (H=0.05) →
  0.306 (H=1.0), a 4.86× drop**; at T = 3.3: 1.493 → 0.317, 4.71×.
- Buoy-heave RAO tracks higher — buoy7 (cluster C) reaches **1.72** at
  H = 0.05, T = 3.257.

### 3c. Corrected design statement

> **The platform has a heave resonance near its 3.14 s natural
> period. It is AMPLITUDE-GATED by quadratic (Morison) drag: the
> response overshoots (RAO > 1) only at small wave amplitude
> (H ≲ 0.1 m at Cd_n = 5.0) and is progressively damped out as
> amplitude grows.** There is no single "platform heave RAO" — the RAO
> is a function of amplitude as well as period.

---

## 4. Generalized lesson — single-amplitude sweeps cannot characterize a quadratically-damped system

For a Morison-damped body the effective damping scales with response
amplitude, so the **RAO is itself amplitude-dependent**. A sweep at one
amplitude measures the system only on one contour of that surface and
**cannot** be extrapolated to others — as STEP B's H = 0.30 sweep
demonstrated by hiding a real resonance entirely.

**Operational implication for the tank campaign.** The test matrix
**must span wave amplitude**, not just period. A period-only matrix at
a single (or moderate) height will mis-rank — or miss — the heave
resonance, exactly as STEP B did in simulation. This is CLAUDE.md §13
Item 19 ("synthetic/partial-scenario validation is necessary but not
sufficient") in its amplitude form: a partial-amplitude sweep is a
partial scenario.

---

## 5. RAO-vs-absolute inversion — the worst cases are at opposite ends (measured)

RAO (dimensionless) and absolute acceleration peak at **opposite ends**
of the amplitude range. Measured steady-window heave-acceleration
amplitude (from the committed fan case CSVs, not derived):

| quantity | H = 0.05 (peak-RAO end) | H = 1.00 (low-RAO end) |
|----------|-------------------------|------------------------|
| platform-heave RAO | **1.49** (T=3.3) | **0.31** (T=3.3) |
| platform-heave accel amp | 0.16 m/s² | **1.00 m/s²** |
| buoy7 heave accel amp | 0.16 m/s² | 0.85 m/s² |
| buoy7 heave accel **peak \|·\|** | 0.17 m/s² | **1.11 m/s²** |

- **Highest RAO occurs at the amplitude with the LOWEST absolute
  acceleration, and vice versa.** As amplitude rises H=0.05 → 1.0
  near resonance, RAO **falls ~4.8×** while peak \|acc\| **rises
  ~6.5×** (0.17 → 1.11 m/s²). The design-driving sea state cannot be
  read off the RAO peak; absolute acceleration must be evaluated at
  the large-amplitude end.
- **Nonlinearity signature.** peak-\|acc\| / half-range = **1.31** at
  H = 1.0 (strongly non-sinusoidal — quadratic-drag harmonics) vs
  **1.06** at H = 0.05 (near-sinusoidal). The acceleration harmonic
  content grows with amplitude, corroborating the quadratic-damping
  mechanism independently of the RAO magnitude.

---

## 6. Independent validation — the long-wave limit

As ω → 0 a floating body must follow the long wave exactly, so
**RAO → 1** regardless of the model. Measured (H = 0.30 sweep):

| T (s) | 3.8 | 4.0 | 4.5 | 5.0 | 6.0 |
|-------|-----|-----|-----|-----|-----|
| platform-heave RAO | 0.807 | 0.851 | 0.922 | 0.959 | **0.9915** |

RAO climbs monotonically to **0.9915 at T = 6.0 s** (0.85% below
unity, still rising). **No modeling error can fake RAO → 1** — this is
a model-independent check that the entire assembled pipeline (Cummins
kernel + 64-row KKT joints + single-body-tiled hydrostatics +
excitation) reproduces the correct rigid long-wave asymptote. The fan
corroborates: RAO approaches 1 from below at long T across all
amplitudes. This is the strongest single validation in the deliverable
and it is amplitude-robust.

---

## 7. Findings — throughput and the two enabling fixes

- **Adaptive settle (~6× throughput).** The opt-in
  integrate-until-window-converges early-stop
  (`stop_check`/`stop_check_interval` in `integrate_cummins`; strictly
  additive, byte-identical when off — verified) cut per-case cost from
  the fixed ~420 s cap to **60–79 s** actual (all 35 fan cases hit
  `converged_early`). The 35-case fan ran in ≈ 2.5 h vs ≈ 14 h at the
  fixed cap. This is what made the amplitude fan affordable within one
  session; without it the correction in §3 would not have been found
  under a period-only budget.
- **Two prerequisites that unblocked the platform run (permanent).**
  (i) STEP 4 replaced the corrupt Capytaine *multibody*
  `compute_hydrostatic_stiffness` (which injected spurious per-block
  cross-DOF coupling, giving an **indefinite** assembled C) with a
  **single-body block tiled ×12** (composite C33 = 2653; block eig
  `[0,0,0,161.28,161.28,221.08]`, PSD). (ii) STEP 3 added a
  build-time restoring-PSD gate over the constraint-feasible subspace
  `null(G)`, so an indefinite C now fails fast at build rather than as
  a zero-norm-quaternion divergence mid-integration. See tracker
  `PLATFORM-HYDROSTATIC-C-INDEFINITE` (resolved) and
  `CAPYTAINE-MULTIBODY-HYDROSTATIC-COUPLING` (external-tool finding).

---

## 8. Guidance for the OrcaFlex comparison

- **Highest-information corner: small amplitude (H ≈ 0.05 m) near
  T ≈ 3.2–3.3 s.** There the resonance is barely gated, so the RAO is
  **most sensitive to Cd** (small drag errors move the peak the most),
  **and** the response is nearly linear (minimal drag harmonics), so a
  FloatSim-vs-OrcaFlex RAO mismatch there **isolates the linear
  impedance** (M + A, C, B_rad, excitation) from the drag model. This
  corner carries the most diagnostic information per run.
- **Cd-sensitivity of the peak (item-2 measurement, H = 0.30).** The
  driven peak sharpens and its RAO grows as drag is reduced:

  | Cd_n | driven peak RAO | driven peak T | in-band overshoot |
  |------|-----------------|---------------|-------------------|
  | 5.0 (operational) | none in 1.2–6.0 s (→ 0.9915 @ 6 s) | — | none at H≥0.15 |
  | 2.5 | ≥ 1.058 (still rising at 4.0 s) | > 4.0 s | begins at T=3.6 |
  | 1.0 | **1.400** | **3.464 s** | T ≥ 3.0 s |

  **The M/K natural period stays fixed at 3.14 s across all Cd** (§2);
  the *driven* peak PERIOD moves to **longer** T as Cd increases
  (3.46 s → >4.0 s → none). That shift is a damped-driven-response
  effect — the response peak sits **below** ω_n and moves further
  below as damping grows (ω_r = ω_n√(1−2ζ²)), reinforced by the rising
  long-wave shelf — **NOT** a change in the natural period. So compare
  OrcaFlex at **matched Cd**, and read the peak location as a joint
  diagnostic of Cd **and** the excitation shape, not of the natural
  period.
- **Heading caveat.** Outputs are for **heading 0**, where cluster A
  (+x) is **downwave** and cluster C (−x) is **upwave** — this
  contradicts the pinned "cluster A upwave" note because the coupled
  RAO database ships heading 0 only. **RAO magnitudes are
  heading-label-independent**; only the acceleration *role* labels
  (which cluster is upwave) would flip under a heading-180 BEM
  re-solve (~190 min, deferred).

---

## 9. Artifacts

- `studies/platform-12buoy/pr8_pilot_out/` — 9-run pilot, H=0.30
  period sweep (base + ext), `cd_check.csv`, `cd_peak_pin.csv`,
  `rao_3d.png`.
- `studies/platform-12buoy/pr8_fan_out/` — 35-run amplitude fan
  (per-case CSVs with heave + acceleration channels), `rao_summary.csv`,
  `manifest.json`, `rao_fan_3d.png`.
- Superseded: the "no resonance / drag-damped-out" reading in
  `958c19f` and the original `rao_3d.png` title. §3 above is the
  operative conclusion.
