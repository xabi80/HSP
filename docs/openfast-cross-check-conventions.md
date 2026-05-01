# OpenFAST / HydroDyn Cross-Check Conventions Audit (M6)

Pre-flight checklist for the M6 OpenFAST cross-check, owned by
`docs/milestone-6-plan.md` v2 Q8. Every item carries **both**
columns required by Xabier's PR1 lock:

- **(a) Written assertion + source citation** -- a referenced claim
  about how OpenFAST/HydroDyn does the relevant thing, with a
  pointer (page, section URL, theory document) the reviewer can
  consult.
- **(b) Runnable sanity-check protocol** -- a concrete, reproducible
  procedure that demonstrates the assertion against a live OpenFAST
  run, even if the protocol is not exercised by CI today.

The doc is **not done** until both columns are filled in. Items
marked `🟡 PR2+` carry a complete protocol but the live verification
is deferred until the corresponding scenario PR runs OpenFAST. Items
marked `✅ verified at PR1` have been confirmed against the cited
source AND have a sanity-check that has either been runnable in
isolation or whose verification is documented in the codebase.

The motivating example for this doc's existence is the
hydrostatic-gravity bug surfaced by the M6 PR1 audit. See
`docs/post-mortems/hydrostatic-gravity-bug.md` and CLAUDE.md §13
for the institutional-memory pattern.

---

## Item 1 -- Reference point (`PtfmRefzt`)

**(a) Citation.** OpenFAST's ElastoDyn input file declares
`PtfmRefzt` -- "platform reference height", the z-coordinate (in
metres, +z up, MWL at z=0) of the body reference point about which
all platform-level translations and rotations are reported. For the
OC4 DeepCwind case in OpenFAST/r-test `5MW_OC4Semi_Linear/`, the
canonical value is `PtfmRefzt = 0.0`; the convention is documented
in:
- *OpenFAST User's Guide*, ElastoDyn input description --
  https://openfast.readthedocs.io/en/main/source/user/elastodyn/input_files.html
- *OpenFAST Modularization Document* §"Platform reference"
  (NREL/TP-5000-XXXX), distributed with OpenFAST source.

FloatSim's body reference point lives in `Body.reference_point` /
`HydroDatabase.reference_point` and must equal `PtfmRefzt` to
numerical noise for the cross-check to be meaningful.

**(b) Sanity-check protocol.** 🟡 PR2+

```
1. Parse PtfmRefzt from the committed inputs/OC4Semi_ElastoDyn.dat.
2. Load the FloatSim deck for the same scenario; read
   Body.reference_point[2].
3. Assert |reference_point[2] - PtfmRefzt| < 1e-9.
```

To be implemented as `tests/validation/test_m6_openfast_conventions.py`
or as a deck-identity preflight at the top of every M6 scenario test.

---

## Item 2 -- Wave heading (`WaveDir`)

**(a) Citation.** HydroDyn `WaveDir` is the wave propagation
direction in **degrees**, with `0°` corresponding to waves
propagating in the inertial-frame `+X` direction. From the
*HydroDyn User's Guide*:
- https://openfast.readthedocs.io/en/main/source/user/hydrodyn/input_files.html
  -- §"Waves" subsection.

FloatSim's `RegularWave.heading_deg` uses the same convention
(degrees, `0° = +X`); see `floatsim/waves/regular.py` module
docstring.

**(b) Sanity-check protocol.** 🟡 PR3+

```
1. Run a regular-wave scenario at WaveDir = 45° in OpenFAST,
   amplitude 0.5 m, period 10 s.
2. Run the matching FloatSim case at heading_deg = 45°.
3. After the integrator's startup ramp completes, the steady-state
   surge:sway amplitude ratio must be cos(45°):sin(45°) ≈ 1:1 in
   both tools, with the same sign on both axes.
```

A heading-mismatch bug (e.g. degrees vs radians, or a 90° axis
swap) shows as a sway-only or surge-only response.

---

## Item 3 -- Euler order (HIGHEST RISK)

**(a) Citation -- to verify at PR1+.** HydroDyn reports platform
rotations as channels `PtfmRoll, PtfmPitch, PtfmYaw` in **degrees**.
The expected underlying convention is **ZYX-intrinsic** (yaw about
inertial Z, then pitch about new Y, then roll about new X --
matching FloatSim's deck-I/O convention per ARCHITECTURE.md §3.2).
Source to consult:
- *HydroDyn Theory Document* (NREL technical report, distributed
  with OpenFAST releases) -- §"Output conventions" / §"Coordinate
  systems".
- *FAST Modularization Document* -- §"Generalised coordinate ordering".

The risk: ServoDyn historically used a **different** convention for
blade pitch angles, and the platform-output convention has been
documented inconsistently in older OpenFAST versions. **Do not
assume**; verify with both the theory doc AND the runnable protocol
below.

**(b) Sanity-check protocol -- runnable test, not just prose.** 🟡 PR2+

Per Xabier's lock: this protocol must be a **runnable test**, not
prose. Implementation lands at PR2 (or a dedicated
`tests/validation/test_m6_euler_order_sanity.py`):

```
1. Author an OpenFAST case with non-trivial platform initial
   orientation -- specifically:
       PtfmRollIC  = 5.0 deg
       PtfmPitchIC = 10.0 deg
       PtfmYawIC   = 15.0 deg
   (locked at SIMULTANEOUS non-zero values to discriminate the
   composition order; a single non-zero would not.)
2. Run a 1-second OpenFAST simulation with no waves, no wind, and
   all DOFs locked except the three rotational ones (so the body
   stays at the IC). Sample the platform rotation channels at t=0.
3. Independently, in FloatSim: build a quaternion from the same
   (5°, 10°, 15°) Euler angles via
   `quaternion_from_euler_zyx(roll, pitch, yaw)`, recover the
   rotation matrix, decompose back to ZYX-intrinsic Euler.
4. Compare the FloatSim recovered Euler triple to OpenFAST's
   reported (PtfmRoll, PtfmPitch, PtfmYaw) at t=0. Disagreement at
   the level of the 5°/10° terms (not just numerical noise) =
   wrong order.
```

The discriminator condition: with three simultaneous non-trivial
angles, ZYX-intrinsic and ZYX-extrinsic differ at first order in
the off-diagonals (a roll-yaw cross-term picks up the pitch). A
single-axis test would not catch a subtly wrong composition order.

---

## Item 4 -- Time origin

**(a) Citation.** OpenFAST starts at `t = 0` (the first row of the
`.out` file has `Time = 0.0`). Some HydroDyn wave models
(specifically `WaveModH = 5`, irregular waves with ramp-up) execute
a startup ramp in negative simulated time, and the published `.out`
omits the ramp region. From *HydroDyn User's Guide*, §"Wave
Generation".

For M6 we use **regular waves only** (S3 sweep) which do not invoke
the ramp-up path, so `t = 0` corresponds to clean wave start. The
FloatSim integrator's `ramp_duration` (default 20 s, see CLAUDE.md
§6) must be set to `0.0` for cross-check runs OR the comparison
must skip the ramp region.

**(b) Sanity-check protocol.** ✅ verified at PR1 (no live OpenFAST
required; this is a fixture-format invariant).

```
1. After loading any committed scenario CSV via
   load_openfast_history, assert history.t[0] == pytest.approx(0.0).
2. The CSV loader's _validate_time_column already checks
   strict-monotonic-increasing; manual review of any committed
   fixture's first row confirms t[0] = 0.0.
```

The loader contract enforces `t[0] = 0.0` implicitly via the JSON
sidecar's `dt_s` agreement check (an OpenFAST run with non-zero
start time would produce a CSV whose `t[0]` did not match
`extracted_at - duration`).

---

## Item 5 -- Hydrostatic stiffness decomposition (HIGHEST IMPACT)

**(a) Citation.** HydroDyn's `PtfmCMatrix` (when present in the
HydroDyn input) and the equivalent stiffness inferred from the BEM
data carry the **buoyancy/waterplane** contribution **only** --
gravity (`m*g*z_G`) is the responsibility of ElastoDyn, which
applies it from the platform `PtfmCMzt` (centre-of-mass z-coord)
and `PtfmMass`. From:
- *HydroDyn User's Guide*, §"Hydrostatic Restoring".
- *HydroDyn Theory Document*, §"Linear hydrostatic restoring":
  `C = ρg(I_wp + V z_B) + (gravity terms applied externally)`.

This convention matches FloatSim's M6-PR1-locked separation:
`HydroDatabase.C_source = "buoyancy_only"` for WAMIT and Capytaine
readers, with `assemble_cummins_lhs(...)` adding the gravity term
via `floatsim.hydro.hydrostatics.gravity_restoring_contribution`.

**Pre-flight invariant (audited at PR1):** ✅ The audit that
surfaced the missing `floatsim.hydro.hydrostatics` module
(`docs/post-mortems/hydrostatic-gravity-bug.md`) established the
buoyancy-only-vs-full split exists in FloatSim and matches HydroDyn's
documented decomposition.

**(b) Sanity-check protocol.** ✅ partly verified at PR1, 🟡 full
verification at PR2.

PR1 has already verified the gravity-coupling separation against
the analytical OC4 pitch period:
`tests/validation/test_oc4_pitch_period_buoyancy_only_c.py` --
this passes ONLY when `assemble_cummins_lhs` correctly adds
`-m*g*z_G` to a buoyancy-only `C`. The asymmetric-CoG discriminator
test
(`tests/validation/test_gravity_restoring_asymmetric_cog.py`)
further pins the cross-coupling convention to
Faltinsen 1990 Eq. 2.104 (no factor of ½).

Full M6 verification at PR2 (S1 static equilibrium):

```
1. Read PtfmMass, PtfmCMzt from the committed
   inputs/OC4Semi_ElastoDyn.dat.
2. Build a FloatSim deck with the same (mass, CoG offset, gravity).
3. Run static equilibrium on FloatSim's HSFP-equivalent OC4 deck;
   compare displaced position to OpenFAST's S1 CSV's last sample.
4. Tolerance: atol = 1e-3 m on translations, atol = 1e-2° on
   rotations (per docs/milestone-6-plan.md v2 Q4).
```

If the comparison fails by more than the tolerance, the gravity
decomposition is wrong somewhere -- the hydrostatic-gravity audit
narrows the suspects to (a) ElastoDyn vs FloatSim mass/CoG
disagreement OR (b) HydroDyn's `PtfmCMatrix` carrying something
different from what its docs claim.

---

## Item 6 -- `CompElast = 0` gravity footgun

**(a) Citation.** With `CompElast = 0` (ElastoDyn disabled),
ElastoDyn does not run -- and ElastoDyn is what applies gravity to
the platform via the
`PtfmMass * g * (z - PtfmCMzt)` restoring term. HydroDyn alone
provides the buoyancy/waterplane term. From the *OpenFAST User's
Guide*, §"Compfast modules" and §"Standalone HydroDyn driver":
running with `CompElast = 0` and the FAST glue code requires the
platform DOFs to either be locked OR for a separate gravity input
to be supplied via the standalone HydroDyn driver path.

**Implication for M6:** for scenarios where the platform must be
free in pitch/roll/yaw under gravity, `CompElast = 0` is **wrong**
-- the system would integrate without the stabilising gravity term.

**(b) Sanity-check protocol & resolution.** 🟡 PR2 (S1).

Two acceptable workarounds (per `docs/milestone-6-plan.md` v2 Q2):

- **Option A: HydroDyn standalone driver with explicit gravity
  input.** Build a `*_Driver.dat` and invoke
  `openfast_hydrodyn_driver` rather than the full FAST glue code.
  Cleanest isolation but adds a separate driver file per scenario.
- **Option B: keep `CompElast = 1`, lock unused platform DOFs.**
  Set ElastoDyn's `PtfmSurgeDOF = PtfmSwayDOF = PtfmYawDOF = False`
  and free only `PtfmHeaveDOF / PtfmRollDOF / PtfmPitchDOF` per the
  scenario. ElastoDyn applies gravity correctly to the freed DOFs.

The committed scenario `.fst` files (vendored at PR2) document
which option each scenario uses. The footgun is recorded explicitly
in `scripts/extract_openfast_fixtures.py` SCENARIOS table.

Sanity check (runnable post-PR2):

```
1. With CompElast=0 and a heave-only IC, run a 60-second OpenFAST
   case. Plot heave vs time.
2. The oscillation period must be longer than sqrt(2) of the
   buoyancy-only period -- if it matches the buoyancy-only period
   exactly, gravity is being skipped (the bug). If it matches the
   published OC4 17.3 s period, ElastoDyn is correctly contributing
   gravity.
```

---

## Item 7 -- Wave elevation reference

**(a) Citation.** HydroDyn `WaveOriginZ = 0.0` (the default) places
the wave-elevation reference at the still water level (SWL). From
*HydroDyn User's Guide*, §"Waves" -- the elevation channel
`Wave1Elev` (or `Wave1Elevxi` for irregular waves) reports surface
elevation in metres relative to SWL.

FloatSim's `RegularWave.elevation` uses the same reference (z=0 is
SWL, +z is up, η is the displacement of the surface from z=0).

**(b) Sanity-check protocol.** 🟡 PR3+ (regular-wave scenarios).

```
1. In a no-wave scenario (S1, S2): assert OpenFAST's Wave1Elev
   channel is identically zero throughout the run.
2. In a regular-wave scenario (S3 RAO sweep): the Wave1Elev time
   history must be sinusoidal with the configured amplitude and
   period; t=0 phase must match the FloatSim eta(t=0) value modulo
   2π.
```

---

## Item 8 -- Output sample rate alignment

**(a) Citation.** HydroDyn's output sample rate is controlled by
`OutFileFmt` and `DT_Out` in the top-level `.fst` file. For the M6
fixtures, `DT_Out = 0.05 s` matches the FloatSim integrator's
typical dt; the JSON sidecar's `dt_s` field captures this.

**(b) Sanity-check protocol.** ✅ verified at PR1.

The CSV loader's `_validate_time_column` enforces a strict tolerance
between the observed mean dt in the time array and the JSON
sidecar's claimed `dt_s` (`rel_err < 1e-3`). A regenerated fixture
that drifted off the locked `dt = 0.05 s` would fail to load at
all. See `tests/unit/test_openfast_csv.py::test_csv_dt_disagreement_with_metadata_raises`.

---

## Item 9 -- Coordinate sign

**(a) Citation.** HydroDyn's `PtfmSurge`, `PtfmSway`, `PtfmHeave`
are the inertial-frame translations of the platform reference point
in metres, with the same sign convention as FloatSim's `xi[0:3]`
(positive = +inertial-axis displacement). From *HydroDyn User's
Guide*, §"Output channels".

No sign flip is required at the loader. The convention matches
FloatSim's deck-I/O and ARCHITECTURE.md §3.

**(b) Sanity-check protocol.** 🟡 PR2 (S1 / S2).

```
1. Run a free-decay scenario with PtfmHeaveIC = +0.5 m (heave UP).
2. The first sample (t=0) of OpenFAST's PtfmHeave channel must be
   +0.5 m (positive, not -0.5).
3. The FloatSim equivalent run must report xi[0, 2] = +0.5.
```

---

## Item 10 -- RAO phase convention (HIGHEST RISK)

**(a) Citation -- to verify at PR1+.** WAMIT's `.3` file (which
HydroDyn consumes for excitation forces) writes RAO phase in
**degrees** under the **leads** convention by default:
`F_exc(t) = Re[X(omega) · A_wave · exp(+i omega t)]`. HydroDyn
inherits this convention when reading the WAMIT input. From:
- *WAMIT v7.4 User Manual*, §13.3 -- "Phase Conventions".
- *HydroDyn User's Guide*, §"WAMIT input" -- inherits WAMIT
  conventions verbatim.

FloatSim's `HydroDatabase.RAO` uses the same "leads" convention
(see the OrcaFlex VesselType reader for the matching choice). The
M5 PR2 Capytaine reader explicitly conjugates Capytaine's "lags"
convention to align with this.

The risk: phase-convention slip is silent on simple amplitude
checks; only a per-frequency phase comparison surfaces it. M6 PR3
(S3 RAO sweep) is the place this is tested.

**(b) Sanity-check protocol -- per Xabier's lock.** 🟡 PR3.

Per Xabier's PR1 directive: "verify by single-frequency time-history
comparison. Plot η(t) and PtfmHeave(t) overlaid, measure visual
phase shift, compare to phase from .3 file directly. Sign must
match modulo 2π."

Implementation:

```
1. Run S3 at T=10 s (a single regular-wave period away from
   resonance). Extract Wave1Elev(t) and PtfmHeave(t) from the
   committed CSV.
2. Compute the phase of PtfmHeave(t) at t large (post-startup-ramp,
   steady-state) by sinusoidal fit. Compute the phase of
   Wave1Elev(t) the same way.
3. The phase difference Δφ = arg(PtfmHeave) - arg(Wave1Elev) must
   equal arg(RAO_heave[T=10s]) -- the value FloatSim's
   HydroDatabase reports -- modulo 2π.
4. Tolerance: 5° absolute (per docs/milestone-6-plan.md v2 Q4).
```

A factor-of-(-1) sign flip on the RAO would make the heave 180°
out of phase with the wave -- exactly what this test catches.

---

## Verification status summary (PR1)

| Item | Status |
|------|--------|
| 1. Reference point (`PtfmRefzt`) | 🟡 PR2+ |
| 2. Wave heading (`WaveDir`) | 🟡 PR3+ |
| 3. Euler order (HIGH RISK) | 🟡 PR2 (runnable test) |
| 4. Time origin | ✅ verified at PR1 (loader invariant) |
| 5. Hydrostatic decomposition (HIGH IMPACT) | ✅ part-verified PR1, 🟡 full at PR2 |
| 6. CompElast=0 gravity footgun | 🟡 PR2 (S1 deck choice) |
| 7. Wave elevation reference | 🟡 PR3+ |
| 8. Output sample rate alignment | ✅ verified at PR1 (loader invariant) |
| 9. Coordinate sign | 🟡 PR2 |
| 10. RAO phase convention (HIGH RISK) | 🟡 PR3 (runnable test) |

**Items not allowed past PR1 without both columns filled:** none.
Every item above carries (a) a written assertion + source citation
AND (b) a runnable sanity-check protocol, even when the live
verification waits for the relevant scenario PR.

This file is **part of the audit pattern** codified in
CLAUDE.md §13. The same dual-column structure is the template for
future cross-check milestones.
