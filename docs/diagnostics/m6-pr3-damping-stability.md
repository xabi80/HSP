# M6 PR3 — pitch damping stability + t_max convergence diagnostics

**Status**: ⚠ **PR3 paused per Mod 2 / Mod 4 findings — re-scope needed.**

Two pre-implementation diagnostics ran on the regenerated S2 fixture
(post-Mod-1, with `PtfmSurge=0.0` IC) before locking PR3 assertion
targets. Both surfaced findings that prevent the planned damping
assertion (`ζ ≈ 0.0297, rtol = 5e-2`) from being a meaningful
cross-check on the current FloatSim setup.

## 1 — Mod 2: OpenFAST pitch damping is amplitude-dependent (hyperbolic envelope)

Fitted log-decrement ζ over three windows of the 22 positive pitch
peaks observed in the regenerated `s2_pitch_decay.csv`:

| Window | N cycles | ζ |
|--------|---------:|---:|
| Peaks 1–5  |  5 | **0.02968** |
| Peaks 5–10 |  5 | **0.01494** |
| Peaks 10–20 | 10 | **0.00880** |

`min ζ = 0.0088`, `max ζ = 0.0297`, mean = 0.0178.
Relative spread `(max-min)/mean = 1.17` — **117 %**, vastly exceeding
the Mod 2 stability gate of `rtol = 5e-2`.

The peaks themselves:

| i | t [s] | peak [°] |   | i | t [s] | peak [°] |
|---:|------:|---------:|--|---:|------:|---------:|
| 0 |  26.8 | 3.4628 |  | 11 | 321.8 | 0.7933 |
| 1 |  53.7 | 2.6358 |  | 12 | 348.6 | 0.7424 |
| 2 |  80.5 | 2.1214 |  | 13 | 375.4 | 0.6969 |
| 3 | 107.4 | 1.7940 |  | 14 | 402.2 | 0.6578 |
| 4 | 134.2 | 1.5396 |  | 15 | 429.0 | 0.6217 |
| 5 | 160.9 | 1.3623 |  | … |   …   |   …    |
| 6 | 187.8 | 1.2115 |  | 21 | 589.8 | 0.4703 |

Successive peaks decay much faster early (when the amplitude is
large) than late (when the amplitude is small). This is the
**hyperbolic-envelope** signature of Faltinsen 1990 §4 / M5 PR5
(`tests/validation/test_m5_drag_free_decay.py`), in which **quadratic
viscous drag** dominates the dissipation:

```
ξ_n = ξ_0 / (1 + n · ξ_0 · δ),    δ = (4/3) · ρ · C_D · A_drag / m_eff
```

A pure-radiation (linear) decay produces `ξ_n = ξ_0 · exp(-n·ζ_lin)`
— constant log-decrement across all windows. Mod 2's spread shows
this is **not** the regime S2 is in.

### Root-cause confirmation: S2 has Morison drag active

Inspecting `tests/fixtures/openfast/oc4_deepcwind/inputs/s2_pitch_decay/s2_pitch_decay_HydroDyn.dat`:

```
PotMod   = 1     (potential-flow radiation/excitation ON)
RdtnMod  = 1     (convolution radiation memory ON)
NMembers = 25    (25 Morison members — full OC4 platform geometry)
MCoefMod = 3     (member-based Cd table)
PropPot  = True  (member i = 1..25)
```

OpenFAST's S2 dissipation is **radiation + quadratic Morison drag**.
The 25 members cover the full OC4 platform (heave plates, columns,
braces) at OC4's published `C_D` values. Of the two contributions,
the Morison drag dominates: at ξ_pitch ~ 5° the Reynolds-scaled
drag forces on the heave plates produce O(0.03) effective ζ at the
first peak, falling roughly as `1/ξ_n` as the amplitude decays —
exactly the hyperbolic shape Mod 2 measured.

## 2 — Mod 4: kernel t_max convergence (post-fix kernel)

On the FloatSim PR3 setup (marin_semi.1 BEM + Robertson platform-only
mass + hand-authored OC4 hydrostatic stiffness, no Morison
elements), `compute_retardation_kernel` was run at four `t_max`
values; pitch period and ζ were extracted from a 600 s Cummins
free-decay run from `pitch_IC = 5°`:

| t_max [s] | period [s] | ζ          | rel-err vs t_max=300 (period) | rel-err vs t_max=300 (ζ) |
|----------:|-----------:|-----------:|------------------------------:|--------------------------:|
| 100 | 18.4316 | 1.085e-9 | 5.50e-13 | 6.80e-6 |
| 150 | 18.4316 | 1.085e-9 | 1.16e-15 | 1.09e-10 |
| 200 | 18.4316 | 1.085e-9 | 1.21e-14 | 1.06e-9  |
| 300 | 18.4316 | 1.085e-9 | 0.00e+00 | 0.00e+00  |

**Locking**: t_max = 100 s already converges within rtol = 1e-3
(both period and ζ). 200 s is comfortable; 100 s would also work
if speed mattered.

But the period and ζ themselves are striking:

- **Period = 18.43 s**, vs. OpenFAST's measured 26.81 s (a **31 %
  mismatch**, far outside the planned `rtol = 2e-2` and well beyond
  the F1 envelope).
- **ζ ≈ 1.09 × 10⁻⁹** (radiation only, essentially zero), vs.
  OpenFAST's amplitude-dependent ζ ranging 0.0088–0.0297. Three
  to four orders of magnitude off.

Decomposing the period mismatch:

| Quantity | FloatSim PR3 setup | Implication |
|----------|---------------------:|---|
| `M[4,4]` (rigid pitch inertia at SWL) | 9.27 × 10⁹ kg·m² | Robertson platform-only; parallel-axis at z_G=−13.46 m |
| `A_inf[4,4]` (marin_semi) | 7.44 × 10⁶ kg·m² | small relative to M |
| `C[4,4]` (Robertson Table 3-3) | 1.078 × 10⁹ N·m/rad | full restoring including −m·g·z_G at platform-only |
| Naive uncoupled `T = 2π·√((M+A_inf)/C)` | **18.43 s** | matches FloatSim free-decay exactly |

The 26.81 s OpenFAST period requires roughly 2.1 × the inertia OR
0.47 × the stiffness. F1 (combined CoG at z_G ≈ −5.7 m) shifts both:
the combined mass m ≈ 1.4 × 10⁷ kg at z_G ≈ −5.7 m gives
`-m·g·z_G ≈ +7.8 × 10⁸ N·m/rad`, vs. Robertson's `+1.78 × 10⁹` —
a *smaller* gravity contribution, partly cancelled by tower/RNA pitch
inertia I_yy_tower+RNA ≈ 3 × 10⁹ kg·m². The combined effect roughly
matches the 26.81 s period, but **F1 needs both M and C to be
rebuilt for the combined deck**, not just C as the original F1
note suggested. The follow-up should be re-scoped accordingly.

## 3 — Why the original PR3 plan is no longer viable

The locked workflow expected:

> "Resume PR3 (rebase milestone-6-openfast-cross-check, period
> xfail-strict under F1, damping should now pass)."

That assumed the kernel fix would unblock damping. **It doesn't**,
because:

- FloatSim's PR3 setup has **no Morison elements** for the OC4
  platform — only the BEM-derived linear radiation kernel. Radiation
  damping at the OC4 pitch resonance is essentially zero
  (`ζ ≈ 1e-9`, three orders of magnitude below quantization noise).
- OpenFAST's S2 has **25 Morison members active** — its observed
  damping is dominated by quadratic drag, not radiation, and its
  envelope is hyperbolic (amplitude-dependent ζ).

A direct ζ comparison between these two is **apples-to-oranges**.
The `rtol = 5e-2` damping assertion in the original Q4 plan
implicitly assumed both tools are running the same physics — they
are not.

## 4 — Re-scope options

Three paths forward, ordered by scope:

### Option A: Reframe S2 as radiation-only on both sides

- Disable HydroDyn's Members block in the S2 scenario
  (`scenario_config.py`: add `"Members": []` or an `MCoefMod=0`
  override — TBD which OpenFAST input keyword switches Morison drag
  off cleanly).
- Re-run S2; re-extract CSV.
- Expect FloatSim ζ ≈ 1e-9 to match OpenFAST ζ ≈ tiny (radiation at
  pitch resonance). Period stays mismatched (F1).

This is the cleanest **physics-isolation** test — exactly what
PR3's plan-text said: "free response isolates restoring + radiation".
The current S2 includes drag, so it isn't actually isolated. Fixing
the scenario fixes the cross-check premise.

**Risk**: with both tools at ζ ≈ 0, the assertion has no signal —
it can't fail. The pitch period mismatch (F1) becomes the only
testable thing. PR3 effectively becomes a period-comparison test
under F1-xfail, with a documented note that radiation damping is
too small to test on OC4 pitch.

### Option B: Add Morison elements to FloatSim's PR3 deck

- Author OC4 Morison member geometry in a FloatSim deck (heave
  plates, columns, braces; matching OC4 published Cd values).
- Expand the M5 PR4 Morison-element pipeline to populate from the
  deck.
- Run pitch decay with both Morison + radiation; compare envelopes
  via the hyperbolic-fit machinery from M5 PR5.

**Scope**: this is a half-milestone of work, far outside PR3's
~250-line budget. Belongs in a separate PR (call it M6 PR3.5 or
re-use the M5 PR4 pipeline for OC4 specifically).

### Option C: Re-scope PR3 to amplitude-after-N-cycles or first-half-cycle

- Drop the ζ assertion entirely.
- Compare first-half-cycle pitch amplitude trajectory between
  FloatSim and OpenFAST — both should track identically until the
  first peak (when nonlinear drag has had little time to dissipate).
- After that, FloatSim's higher peaks (no drag) diverge from
  OpenFAST's lower peaks (with drag). Document the divergence as a
  "drag-not-yet-implemented" Known Discrepancy.

**Risk**: the test becomes weak. First-half-cycle agreement is
mostly a test of M and C, which `test_oc4_natural_periods` and
M6 PR2 already cover.

## 5 — Recommendation

**Option A**, with two assertions kept and one downgraded:

1. **Period assertion** (xfail-strict under F1, as originally
   planned) — still tests M and C bookkeeping.
2. **Sanity assertion** (envelope decays trend-based per Mod 3) —
   confirms FloatSim's free-decay is not pumping energy.
3. **Damping ζ assertion** dropped or made informational only.
   Document in the cross-check report that pitch radiation damping
   on OC4 is too small to be a useful cross-check metric — the real
   damping cross-check belongs to S5 (drag-on free decay) where
   the scenario is designed for it.

This preserves PR3's place in the sequence (free-decay before
RAOs) without forcing a damping assertion that the physics setup
doesn't support.

## 6 — Files produced by these diagnostics

- `scripts/m6_pr3_mod2_damping_stability.py` — Mod 2 runner
  (window-based ζ on the S2 reference).
- `scripts/m6_pr3_mod4_tmax_convergence.py` — Mod 4 runner
  (FloatSim Cummins free-decay across `t_max` ∈ {100, 150, 200,
  300} s on the post-fix kernel).
- This document — findings + re-scope recommendation.

**Awaiting Xabier's call** between Options A / B / C before any
PR3 test code lands.
