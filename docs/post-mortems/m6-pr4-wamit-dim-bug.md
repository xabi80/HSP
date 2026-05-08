# Post-mortem — WAMIT dimensionalisation latent bug

**Discovered:** 2026-05-06 during M6 PR4 Pre-3 dual-path RAO
verification.
**Fixed in:** branch `fix-wamit-dimensionalisation`, merged to
`main` before M6 PR4 implementation begins.
**Severity:** Phase 1 latent bug, *known and documented* in code
comments from M5 PR1 onward but never tracked or fixed. Latent
through five subsequent PRs because free-decay validation is
mass-and-stiffness dominated (BEM `A` is ≤ 0.1 % of rigid-body `M`
in the natural-period band for OC4). Surfaced at the first PR
exercising the F_exc-dominated regime (RAO extraction).

## TL;DR

`floatsim/hydro/readers/wamit.py` returned WAMIT public-format
`.1` / `.3` / `.hst` values verbatim, treating them as
SI-dimensional. WAMIT v7's default output is **non-dimensional**
(manual §4.2; HydroDyn user guide §6 follows the same convention).
Each coefficient must be multiplied by `ρ · g · ULEN^k` (or the
appropriate variant per coupling type) to recover SI units. For
OC4 marin_semi (ρ = 1025, ULEN = 1) the missing factor is ~1000
on translation-translation pairs and ~1000 × g ≈ 10 000 on
excitation forces.

The bug was **explicitly known**: a code comment in
`tests/validation/test_oc4_pitch_period_buoyancy_only_c.py` from
M5 PR1 onward read

> "the WAMIT reader does NOT currently apply ULEN-based dimensional
>  rescaling — that's a separate latent bug, out of scope for this
>  fix."

Diagnosis correct, deferral correct, **but no tracked follow-up
entry was created**. Five subsequent PRs (M5 PR2, M6 PR1, M6 PR2,
M6 PR3, fix-pr2-cmzt, fix-radiation-kernel, fix-s3-wavemod) all
built on the broken reader. The bug surfaced at M6 PR4 Pre-3 —
the first place the F_exc-dominated regime was exercised.

## Why the bug stayed latent through M5–M6 PR3

For free-decay validation the linearised period is
`T = 2π · √((M + A(ω_n)) / C)`, where `M` is the dimensional
rigid-body mass (Robertson 2014 published value, in code as
`OC4_PLATFORM_MASS_KG = 1.347 × 10⁷ kg`), `A(ω_n)` is the
frequency-dependent BEM added mass at the natural frequency, and
`C` is the dimensional Robertson hydrostatic stiffness.

For OC4 in heave: `M ≈ 1.35 × 10⁷ kg`, `A(ω_n) ≈ 1.5 × 10⁷ kg`
(dim) or `1.5 × 10⁴ kg` (non-dim, missing the 1000× ρ factor).
`C = 3.836 × 10⁶ N/m`. Computing the period both ways:

```
With dim A:    T = 2π·√((1.35e7 + 1.5e7) / 3.836e6) = 17.1 s   (Robertson published ~17.3)
With non-dim A: T = 2π·√((1.35e7 + 1.5e4) / 3.836e6) = 11.78 s
```

But in the M6 PR3 free-decay test, the *measured* FloatSim heave
period was 11.78 s (matching the broken-reader expectation), and
the *measured* OpenFAST period was also ~17 s (since OpenFAST
applies the rescaling internally). Yet PR3 and earlier tests
asserted *period* against an internally-consistent FloatSim
band, not against the OpenFAST reference — so the discrepancy was
masked.

For OC4 pitch: `M_55_at_SWL = 9.27 × 10⁹ kg·m²` (Robertson
parallel-axis dim), `A(ω_n) ≈ 7.7 × 10⁹` dim or `7.5 × 10⁶`
non-dim. The pitch period is far less sensitive to `A` than heave
because `M ≫ A` even at the natural frequency. M6 PR3 free-decay
got 25.67 s vs OpenFAST 26.83 s, attributed to F1-residual
(combined-deck mass distribution). With the dimensional fix `A`
becomes much larger and the period shifts — the F1-residual
classification needs reanalysis (this is the "send re-measurement
results before merging" step the user locked in).

## How M6 PR4 Pre-3 surfaced it

Pre-3's dual-path RAO verification computes the heave RAO two
ways:

- **Path A** — WAMIT impedance: read `marin_semi.{1,3,hst}`,
  build `Z(ω) = -ω²(M+A) + iωB + C`, solve `ξ = Z⁻¹ F_exc`.
- **Path B** — OpenFAST `.outb` time-series lstsq fit of
  `heave / wave_elev` at the IFFT-quantised wave frequency
  (per Item 21).

At WaveTp = 25 s (long-wave limit): Path B gave heave RAO ≈ 1.09
m/m (correct: the body follows a long wave like a cork). Path A
gave 7.4 × 10⁻⁵ m/m. Ratio 1 / 14 700 ≈ 1 / (ρ · g · ULEN²) =
1 / 10 055 — the missing dimensionalisation factor on `F_exc`.

Once the discrepancy was reduced to a single ρ·g·ULEN² factor,
the cause was unambiguous: WAMIT non-dim → dim rescaling
missing. Confirmed by inspecting the `.3` file's heave entries
at long period (`F_exc[3] ≈ 380` non-dim → `380 × 1025 × 9.81 =
3.82 × 10⁶ N/m wave amplitude` ≈ `C[2,2]`, which gives RAO ≈ 1
in the long-wave limit ✓).

## The "known but not tracked" failure mode

The motivating comment was correct in every detail:

- It identified the bug.
- It scoped the deferral correctly (the fix is non-trivial
  parser surgery; out of scope for the M5 PR1 audit).
- It linked the symptom to the cause.

What it lacked: **a tracked follow-up that would force the
question "is this still latent?" at every subsequent PR**. With
no tracked entry, the comment functioned as a tombstone — read
once at write-time, never re-read.

Five PRs later, the M6 PR4 Pre-3 audit re-encountered the same
diagnosis from scratch (via dual-path verification) rather than
"oh, this is the F2 we noted in M5". That's wasted effort and
also a near-miss: had Pre-3 not run, PR4's RAO assertions would
have failed at the same magnitude, and the diagnosis would have
landed at PR4 implementation time — by which point fixture
regenerations and convention-doc additions would be entangled
with the WAMIT fix.

**Standing rule (conventions doc Item 23):** any code comment of
the form "this is a separate bug, out of scope for this PR" must
be paired with a named entry in
`docs/openfast-cross-check-report.md`'s Named follow-ups section
(or the analogous tracker for non-M6 work). Comments rot;
tracked items force decisions.

## Item 19 prediction validation

The "code-path exercise principle" was codified in M6 PR4 Pre-2
after four findings (hydrostatic-gravity, asymmetric-CoG,
radiation kernel, WaveMod) all fitted the same shape: a code
path correct in synthetic / unit / partial-scenario tests was
silently wrong under production-quality inputs and full-scenario
activation.

Finding #5 (this WAMIT bug) was *predicted* by Item 19's
framing: a known-latent code path (the WAMIT reader's missing
dimensionalisation) wasn't exercised by any production-data
test until PR4. The prediction held: PR4 was the first place
the bug surfaced.

This finding is the strongest validation yet of the audit
pattern. It also exposes a refinement: Item 19 says "code paths
that consume external data … need a real-data exerciser
somewhere in the suite". This bug had M5 PR1's
`test_marin_semi_trimmed_*` reader-level tests (real WAMIT
data!), but those tests asserted only *order-of-magnitude*
sanity (`1e3 < surge_aa < 1e5` non-dim). The order-of-magnitude
gate is too loose to catch a 1000× factor — what was needed was
a **published-value comparison** (`A_inf_heave ≈ 1.45 × 10⁷ kg`
per Robertson Table 3-1). The fix branch added that test.

**Refinement to Item 19**: real-data tests should compare against
*published reference values* where available, not just
order-of-magnitude sanity. Order-of-magnitude tests catch
sign-flip / bookkeeping errors but miss 10× / 1000× scaling
errors that fall within the loose bands.

## F1-residual reclassification (locked, post-fix-wamit-dim)

PR3's free-decay test asserts the OC4 pitch period and reported
a -4.29 % rel-err vs OpenFAST pre-fix, classified as
"F1-residual" (combined-deck distributed-inertia approximation).
With the WAMIT dim-fix, the BEM-derived `A_pitch(ω_n)` becomes
1000× larger and the period shifts to **+20.54 % rel-err**:

| State | FloatSim period | OpenFAST | rel-err |
|-------|----------------:|---------:|--------:|
| Pre-fix-wamit-dim | 25.67 s | 26.83 s | -4.29 % |
| Post-fix-wamit-dim (locked) | 32.34 s | 26.83 s | **+20.54 %** |

The pre-fix gap was **mostly the WAMIT bug**, in the OPPOSITE
direction. The post-fix +20.54 % gap is **unexplained** and
tracked as **F1-revised / KD-2-revised** in
`docs/openfast-cross-check-report.md`. Original distributed-
inertia hypothesis is falsified — distributed inertia would
shorten the period (more rotational mass), not lengthen it.

Possible new causes (none yet investigated):
1. Mass bookkeeping: combined-CoG aggregation of `I_55` may
   double-count or under-count platform-with-ballast moment.
2. Hydrostatic-stiffness off-diagonal sign / factor (Robertson
   publishes only diagonals; FloatSim zeroes off-diagonals).
3. BEM frequency interpolation at the natural frequency
   (likely small effect: < 1 %).
4. Cummins reference-point handling for `M+A(ω_n)` vs `C`.

The xfail-strict marker on PR3's period assertion stays — only
the reason string and rel-err magnitude updated.

## Item 24 / Item 25 — additional findings on this branch

The Pre-3 phase-residual diagnosis (Decision 1) and the
three-check kernel gate refactor (Decision E + Decision 3)
surfaced two additional concerns on this same branch:

- **Item 24 (LEAD vs LAG)**: phase reporting between an
  impedance-domain RAO computation (`arg(xi_hat)` under +i
  convention = LEAD) and a `cos+sin` lstsq fit
  (`atan2(B, A)` = LAG) requires negating exactly one to get
  consistent phases. Pre-3 dual-path verification surfaced this
  as a 12.7° gap at WaveTp = 10 s that initially looked like a
  WAMIT phase-convention bug. See
  `docs/diagnostics/m6-pr4-pre3-phase-residual-diagnosis.md`.
- **Item 25 (three-check kernel gate)**: the post-WAMIT-dim BEM
  pushed surge / sway / yaw at marin_semi to ~ 1.7 % of peak at
  `omega_max` — below the asymptote-regime threshold but above
  the pre-fix 1 % gate. Decision E refactored the gate into
  three separate checks: Check 1 (input proxy, advisory at 5 %),
  Check 2 (asymptote consistency, hard at 0.10), Check 3
  (post-extension kernel decay, hard at 0.001). The marin_semi
  reference now passes all three with margin.

## Files produced

- `floatsim/hydro/readers/wamit.py` — `assume_dimensional`
  kwarg + `rho_water_kg_m3`, `g_m_s2`, `ulen_m` factors;
  strengthened `_maybe_warn_nondimensional` based on rotational
  added-mass scale; module docstring updated for LEAD vs LAG
  convention warning (Item 24).
- `floatsim/hydro/retardation.py` — three-check kernel gate
  refactor (Items 25). Check 1 demoted to soft warning (5 %
  threshold); Check 2 unchanged; Check 3 NEW post-extension
  hard error (0.1 % decay gate).
- `tests/unit/test_wamit_reader.py` — three new regression tests
  (Robertson `A_inf_heave`, Robertson `A_inf_pitch`, strengthened
  heuristic). Updated existing tests to pass
  `assume_dimensional=True` on the synthetic_simple fixture.
- `tests/unit/test_retardation_kernel.py` — three Decision-3
  unit tests (Check 1 warning, Check 3 raises, marin_semi
  all-three-checks-clean regression).
- `tests/unit/test_retardation_kernel_extension.py` — t_max
  bumps to clear Check 3 on existing tests.
- `tests/validation/test_oc4_heave_free_decay.py` — kernel
  t_max 60 → 200 s; period now 17.19 s (Robertson 17.3 s, +0.6 %).
- `tests/validation/test_m6_openfast_free_decay.py` — xfail
  reason updated to F1-revised / KD-2-revised (32.34 s,
  +20.54 %).
- `docs/openfast-cross-check-conventions.md` — Items 22, 23, 24,
  25.
- `docs/openfast-cross-check-report.md` — KD-2 marked falsified;
  KD-2-revised added with diagnostic hypothesis list.
- `docs/diagnostics/m6-pr4-pre3-rao-verification.md` — Pre-3
  pre-fix observations.
- `docs/diagnostics/m6-pr4-pre3-phase-residual-diagnosis.md` —
  Decision 1 LEAD-vs-LAG diagnosis.
- `docs/diagnostics/m6-pr4-pre3-surge-kernel-quality.png` —
  Decision E calibration evidence.
- `docs/diagnostics/m6-pr4-pre3-phase-residual-diagnosis.png` —
  sliding-window phase / amp / residual diagnostic plot.
- `scripts/m6_pr4_pre3_rao_verification.py` — dual-path
  verification (Path A returns LAG = -arg(xi_hat) per Item 24).
- `scripts/m6_pr4_pre3_surge_kernel_quality.py` — per-DOF
  kernel-decay diagnostic (Decision E calibration).
- `scripts/m6_pr4_pre3_phase_residual_diagnosis.py` —
  Decision 1 sliding-window diagnostic.
- `tests/support/rao_extraction.py` — sinusoidal lstsq RAO
  extractor (independent of any reader bug; usable unchanged).
- `CLAUDE.md` §13 — pattern lock updated to five findings.
- This document.
