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

## Item 11 -- OpenFAST output channel naming and access

**(a) Citation.** When `.outb` (binary) outputs are read via the
`openfast_io.FAST_output_reader.FASTOutputFile` class:

- Channel names are accessed via `output.info["attribute_names"]`.
  The alternative `output.channels` attribute is **not** reliable
  across `openfast_io` versions; do not depend on it.
- Channel units are at `output.info["attribute_units"]`.
- Sample data is at `output.data` (shape `(n_samples, n_channels)`).

Source: `openfast_io` source code
(`openfast_io/FAST_output_reader.py`), verified empirically against
the M6 baseline run by `scripts/openfast_setup/quick_sanity.py`
(Xabier, 2026-05-01).

For OpenFAST channel **names** themselves:

- Platform DOFs are exposed without a module prefix:
  `PtfmSurge`, `PtfmHeave`, `PtfmRoll`, `PtfmPitch`, `PtfmYaw`,
  plus body-frame velocities `PtfmTVxt`, `PtfmRVxt`, etc. NOT
  `ED.PtfmSurge` -- the ElastoDyn module prefix is dropped at
  the OpenFAST glue-code level.
- MoorDyn line tensions are `FAIRTEN{1,2,3}` and `ANCHTEN{1,2,3}`
  (FORTRAN-uppercase) **inside the separate `*.MD.out` text file**
  that MoorDyn writes alongside the main `.outb`. They are NOT in
  the `.outb` channel list. The same conceptual names also appear
  in HydroDyn-side documentation as `FairTen{1,2,3}` /
  `AnchTen{1,2,3}` (capitalised-at-word-starts) -- the
  `_RENAME_TABLE_CI` in `extract_openfast_fixtures.py` matches
  case-insensitively so both spellings map to the same canonical
  column. The S4 baseline OutList in MoorDyn's input deck already
  emits these by default; no MoorDyn-side edits required.

  Time alignment: MoorDyn skips the t=0 sample (its first row is
  at t=dt). The merge step in `_merge_moordyn_into_canonical`
  linearly interpolates each tension column onto the main `.outb`
  time grid, filling out-of-range samples with the nearest
  available value (`np.interp` with `left=col[0]`, `right=col[-1]`).

**(b) Sanity-check protocol.** ✅ verified at PR1.1 (post-baseline run).

```
1. Run extract_openfast_fixtures.py --mode read-only --scenario all
   on the committed .outb set. Confirm it returns successfully and
   that the produced CSV files contain the canonical SI columns
   (surge_m, sway_m, heave_m, roll_rad, pitch_rad, yaw_rad).
2. For S4 only, additionally confirm fair_ten_line{1,2,3}_n and
   anch_ten_line{1,2,3}_n columns are present and non-zero
   (~1.10 MN at fairlead, ~0.90 MN at anchor per the M6-PR1.1
   baseline sanity-check report).
3. Spot-check by parsing one .outb manually:

       from openfast_io.FAST_output_reader import FASTOutputFile
       out = FASTOutputFile("inputs/s1_static_eq/s1_static_eq.outb")
       assert "PtfmHeave" in out.info["attribute_names"]
       assert "ED.PtfmHeave" not in out.info["attribute_names"]
```

The `_RENAME_TABLE` in `extract_openfast_fixtures.py` is the
single source of truth for the OpenFAST -> canonical SI mapping;
adding new channels requires only a new entry there.

---

## Item 12 -- Static-equilibrium scenarios use last-30-s time-averages

**(a) Citation.** Empirical observation from the M6 baseline
sanity-check (Xabier, 2026-05-01):

- S1 (no waves, no mooring): heave equilibrium ~0.65 m with
  `last10%_std = 0.13 m`. The natural heave period (~17 s) and the
  light radiation damping in still water mean full settling
  requires impractically long simulations (TMax = 200 s gives
  ~12 cycles; would need ~40 cycles for `std < 0.01`).
- S4 (moored, no waves): MoorDyn took ~48 s of init time, leaving
  only ~152 s of usable sim. PtfmSurge `last10%_std = 0.71 m`.
  Tensions converged faster (line stiffness >> hydrostatic) but
  inherit the surge oscillation envelope.

Both scenarios are **physically settling but not converged**; the
reference value is therefore the **time-average over the last 30 s**
of each channel, NOT the instantaneous final-sample value.

For the dynamic scenarios (S2 free decay, S3 RAO sweep, S5 drag
decay) the cross-check metric is per-cycle peak amplitude
extraction, not a steady-state mean -- this Item 12 applies to S1
and S4 only.

**(b) Sanity-check protocol.** 🟡 PR2 (S1) and PR5 (S4).

```
1. In the scenario test (e.g. tests/validation/test_m6_s1_static_eq.py):
   load the committed CSV via load_openfast_history.
2. Compute mean over the last 30 s of simulated time:

       t = history.t
       mask = t >= (t[-1] - 30.0)
       reference_value = float(np.mean(channel[mask]))

3. The instantaneous final value (`channel[-1]`) is NOT the
   reference -- it carries the residual oscillation that the
   averaging window suppresses.
4. Tolerance must accommodate the residual oscillation amplitude
   (see Item 13).
```

**Equilibrium reference is the strict last-30-s mean of the
decimated CSV.** Eyeballed last-value or last-10% values used
in earlier sanity reports are NOT the reference. The PR1.1 vs
PR1.2 disagreement on the S1 heave equilibrium (~0.65 m vs
0.475 m) traces to this -- they were different measurements of
the same time series. Cross-check tests must compute the
last-30-s mean from the committed CSV directly (per the protocol
above) so the reference is reproducible from the artifact rather
than from a one-off observation.

This protocol is also documented in `docs/milestone-6-plan.md` v2
Q4's tolerance table.

---

## Item 13 -- Cross-check tolerances must accommodate residual oscillation

**(a) Citation.** Following from Item 12: in still-water
quasi-static scenarios, the reference value's underlying
time-history still oscillates with amplitude comparable to the
settling envelope. The cross-check tolerance must be **at least
the oscillation amplitude**, not the typical analytical-comparison
tolerance.

Concrete locks per the M6 baseline sanity-check (2026-05-01,
amended 2026-05-04 after the S1 TMax=600 re-extraction):

| Scenario | Channel | Tolerance |
|----------|---------|-----------|
| **S1 (unmoored static equilibrium): cross-check heave, roll, pitch only.** Surge/sway/yaw have zero hydrostatic stiffness in the unmoored OC4 configuration; neutrally stable, no defined equilibrium. Validated in S4 (moored) and S3 (wave-excited) instead. See Item 14 for the general principle. | | |
| S1 | `heave_m` (equilibrium) | ≥ ±0.15 m absolute |
| S1 | `roll_rad` (equilibrium) | ≥ ±0.5° absolute (~8.7e-3 rad) |
| S1 | `pitch_rad` (equilibrium) | ≥ ±0.5° absolute (~8.7e-3 rad) |
| S4 | `surge_m` (offset) | ≥ ±0.7 m absolute |
| S4 | `fair_ten_line{1,2,3}_n` | ±5% relative |
| S4 | `anch_ten_line{1,2,3}_n` | ±5% relative |

These supersede the tighter tolerances in
`docs/milestone-6-plan.md` v2 Q4 (which were drafted before live
OpenFAST data was available).

**(b) Sanity-check protocol.** 🟡 PR2 (S1) and PR5 (S4).

```
1. Use the last-30-s mean per Item 12 as the reference value.
2. Optionally compute the OpenFAST-side residual standard
   deviation over the same window for diagnostic context, but do
   NOT widen the tolerance to "3 sigma" or similar -- the
   tolerance is locked above by physics (settling envelope), not
   by sample-statistics.
3. If FloatSim's prediction sits within the tolerance band of the
   reference, declare match. The asymmetry (FloatSim and OpenFAST
   each settle at slightly different mean values around the same
   physical equilibrium) is absorbed by the tolerance.
```

If the FloatSim equilibrium sits *outside* the band by more than
the tolerance, the failure mode is one of: deck-identity
mismatch (mass/inertia/restoring), gravity-decomposition error
(Item 5 regression), or a real M2/M3-era integration bug -- treat
as a debugging starting point, not as evidence the tolerance
itself is wrong.

---

## Item 14 -- Static equilibrium cross-checks are valid only on restored DOFs

**(a) Citation.** Static equilibrium is the configuration ``ξ*``
where ``F_total(ξ*) = 0``. For the linearised Cummins assembly,
this reduces to ``C·ξ* = F_external`` for time-independent
``F_external``. **A DOF whose row/column of ``C`` is all zero
has no defined equilibrium**: any value of that ``ξ`` component
satisfies ``F_total = 0``. The system is rank-deficient on that
DOF; the equilibrium is a manifold (a line, plane, or
higher-dimensional subspace), not a point.

For the OC4 DeepCwind unmoored configuration:

- **Restored** (non-zero ``C[i, i]``): heave (``C_33 = ρgA_wp``),
  roll (``C_44 = ρgI_xx_wp + buoyancy/gravity coupling``), pitch
  (``C_55``).
- **Unrestored** (``C[i, i] = 0``): surge, sway, yaw. No waterplane
  contribution; no gravity coupling at first order; no mooring.

Concrete consequence:

- FloatSim's :func:`floatsim.solver.equilibrium.static_equilibrium_solver`
  applies a small diagonal regularisation ``λ·I`` (default
  ``λ ≈ 1e-8 · max|C_ii|``) to make the system invertible. For
  unrestored DOFs this regularised solution returns ``ξ_i ≈ 0``.
  This is the correct and only well-defined behaviour.
- OpenFAST's free time-domain integrator does *not* regularise.
  Without restoring, surge/sway/yaw drift slowly (a few mm/s for
  OC4) under residual numerical noise, integrating to non-zero
  but physically meaningless offsets over a few hundred seconds.
- A naive ``rtol=5e-2`` cross-check on every DOF would flag the
  drift offset as a FloatSim-vs-OpenFAST mismatch -- but it
  measures **numerical drift in the reference**, not physics.

**Rule.** Each scenario PR must explicitly enumerate the restored
DOFs and assert only on those. Unrestored DOFs are tested in
the scenarios where they *are* restored (S3 wave-excited;
S4 mooring-restored) or skipped entirely.

**(b) Sanity-check protocol.** ✅ verified at PR2 (S1).

```
1. For each scenario, identify the restored DOFs by inspecting
   diag(C) where C is the FloatSim-assembled hydrostatic matrix
   (after gravity term per Item 5). DOFs with ``C[i, i] == 0``
   are unrestored.
2. Cross-check assertions iterate only over restored DOFs.
3. The test docstring documents which DOFs are tested and why
   the others are excluded, citing this Item.
```

For the M6 set:

| Scenario | Restored DOFs | Unrestored DOFs |
|----------|---------------|-----------------|
| S1 (unmoored statics) | heave, roll, pitch | surge, sway, yaw |
| S2 (free decay) | n/a -- dynamic test, period-fitting not equilibrium |
| S3 (RAO sweep) | all 6 (excited by waves) | n/a |
| S4 (moored statics) | all 6 (mooring restores horizontals) | n/a |
| S5 (drag decay) | n/a -- dynamic test, peak-fitting not equilibrium |

---

## Item 15 -- Static equilibrium under Cummins linearisation

**(a) Citation.** The Cummins formulation linearises the platform's
equation of motion about a chosen reference point -- conventionally
the BEM solver's hydrostatic origin (where ``PtfmVol0`` was
computed). Inside FloatSim's
:func:`floatsim.solver.equilibrium.solve_static_equilibrium`, the
residual is

    r(xi) = C * xi - F_external(t=0, xi, xi_dot=0)

For a deck whose total mass and displaced volume balance exactly at
the BEM reference (``m_total * g = rho * V0 * g``), ``F_external = 0``
and the equilibrium is at ``xi = 0`` -- the linearisation point
itself. **A FloatSim deck whose mass/buoyancy balance does not
coincide with the BEM reference will not show the imbalance as an
equilibrium offset (xi=0 always); the imbalance must be applied as
an external static force.**

OpenFAST handles this differently: its nonlinear time-domain
integrator settles into the offset directly, with the unbalanced
weight pushing the platform up (or down) until the buoyancy
restores at the new draft. The two formulations are equivalent in
the small-displacement limit -- the OpenFAST equilibrium offset
is ``F_residual / C_diag`` for the dominant DOFs -- but they
report different ``xi_eq`` for the same deck.

**Implication for cross-checks.** M6 cross-checks therefore apply
OpenFAST's deck residual as ``F_external`` in FloatSim
(:func:`tests.support.openfast_deck.compute_openfast_deck_residual`)
rather than asserting equilibrium-offset agreement. This validates
the linearised assembly + the gravity decomposition (Item 5)
without forcing a deck-identity refit between the two tools.

**(b) Sanity-check protocol.** ✅ verified at PR2 (S1).

```
1. Confirm with no F_external, FloatSim's solve_static_equilibrium
   returns xi=0 (test_zero_external_force_returns_zero_xi).
2. Compute F_residual = compute_openfast_deck_residual(deck_dir).
3. Apply F_residual via the state_force callable; solve.
4. Assert resulting xi_eq matches the OpenFAST CSV's last-30-s
   mean within Item 13 tolerances on RESTORED DOFs only (Item 14).
```

The test in
``tests/validation/test_m6_openfast_static_eq.py`` exercises this
end-to-end against the committed S1 (unmoored) reference; six
assertions, all pass at PR2.

---

## Item 16 -- Damping cross-check tolerance depends on dissipation regime

**Source.** M6 PR3 Mod 2 diagnostic
(`docs/diagnostics/m6-pr3-damping-stability.md`).

**Rule.** Damping cross-check tolerance depends on the dominant
dissipation mechanism in the scenario, not on a single project-wide
rtol. Three regimes:

1. **Quadratic-drag-dominated** (Morison drag active). Free-decay
   envelope is hyperbolic (Faltinsen 1990 §4): successive peaks
   follow ``ξ_n = ξ_0 / (1 + n · ξ_0 · δ)`` with
   ``δ ∝ ρ · C_D · A_drag / m_eff``. Log-decrement ζ extracted from
   short windows decreases with amplitude — there is no single
   linear ζ to assert against. Cross-checks must compare
   **per-peak amplitudes** against the hyperbolic-envelope
   reference (per-peak rtol, not log-decrement).
2. **Linear-radiation-dominated** (Morison disabled, but kernel
   damping at ω_n is non-trivial). Envelope is exponential; ζ from
   log-decrement is well-defined. Standard `rtol = 5e-2` per Q4 of
   the M6 plan applies.
3. **Radiation-only on a low-damping eigenmode**. Some BEM-defined
   eigenmodes have radiation damping at the natural frequency that
   sits at the numerical-noise floor (e.g., OC4 pitch:
   ``B_55(ω_n=0.34 rad/s)`` ≈ 1.85 × 10⁴ N·m·s/rad gives ``ζ ~ 1.6 × 10⁻⁵``).
   On these modes neither tool produces a meaningful damping value
   — peaks barely decay over the simulation horizon. Tight ζ
   assertions have **no signal**. The only testable property is
   **non-negativity** (radiation must dissipate, not inject energy)
   — this is what catches kernel pathologies like the M6 PR3
   pre-fix bug.

**Decision rule when scoping a damping cross-check on a new
scenario:**

- Is the dominant dissipation mechanism (radiation, viscous drag,
  mooring) **matched** in both tools? If not, disable the unmatched
  mechanism in the reference (per Option A of the M6 PR3 re-scope:
  S2 disabled Morison drag) or move the comparison to a different
  scenario.
- After matching, measure ζ in the reference over windows
  ``peaks 1-5 / 5-10 / 10-20``. If the windows agree to within
  rtol = 5e-2, lock the tight assertion. If they don't, the
  scenario is not in regime 2 — fall back to the matching
  regime's protocol (per-peak hyperbolic for regime 1,
  non-negativity-only for regime 3).

**Verification status.** Applied to S2 in M6 PR3:
``tests/validation/test_m6_openfast_free_decay.py`` runs the
non-negativity assertion (regime 3); the diagnostic-log test
emits ζ over the three windows for the cross-check report. Future
S5 (drag-on heave decay, M6 PR6) will exercise regime 1.

---

## Item 17 -- `z_G` must be consistent with the mass `M` and stiffness `C`

**Source.** M6 PR4 Pre-1 audit
(`docs/diagnostics/m6-pr4-pre1-cmzt-audit.md`); fix landed on the
`fix-pr2-cmzt` branch.

**Rule.** ``z_G`` (the vertical CoG coordinate used in any force,
moment, or stiffness computation) must be paired consistently
with the mass ``M`` and the stiffness ``C`` referenced in the
same equation. The pairings come from the **same source** at the
**same level of system completeness**:

- **Robertson 2014 Table 3-1 OC4 platform-with-ballast pair**:
  ``M = 1.3473 × 10⁷ kg``, ``z_G = -13.46 m``. This is the with-
  ballast mass and CoG; the ballast water inside OC4's offset
  columns and centre column is included in both.
- **OpenFAST `*_ElastoDyn.dat` steel-only pair**: ``PtfmMass =
  3.852 × 10⁶ kg``, ``PtfmCMzt = -8.66 m``. Steel-only structure;
  the ballast water is treated separately by HydroDyn
  ``FillGroups``.
- **Combined-system pair** (platform + tower + RNA): aggregated
  by the parser, with each component contributing its own
  ``(m_i, z_i)`` from a consistent source.

**Mixing one mass with the other CoG is a bookkeeping error.** It
may not surface in a specific assertion (the M6 PR4 Pre-1 audit
showed F-vector elements are independent of ``z_G`` for
axisymmetric on-axis-CoB decks like OC4), but it leaks the moment
the test reaches a configuration where vertical lever arms
matter — non-axisymmetric mass distribution, a moored case where
mooring-line anchor positions feed into a moment with vertical
arms, or a body geometry where the CoB doesn't sit on the
symmetry axis.

**Verification status.** Pinned by:

- Two unit tests in ``tests/unit/test_openfast_deck.py``:
  ``test_F_residual_invariant_to_platform_cog_z`` (algebraic
  invariant for axisymmetric decks); and
  ``test_default_platform_cog_z_pairs_with_default_mass``
  (default ``(M, z_G)`` consistency for OC4).
- The pre-fix behaviour was inconsistent (parser used
  ``PtfmCMzt = -8.66 m`` with ``M = 1.347 × 10⁷ kg``); the fix
  replaced ``PtfmCMzt`` reads with the explicit
  ``OC4_PLATFORM_COG_Z_M = -13.46 m`` constant and exposed
  ``platform_cog_z_m`` as a kwarg paired with
  ``platform_total_mass_kg``.

**The same protocol applies to any future scenario PR.** When a
new component / coefficient enters the residual computation,
verify that its ``(m, z_G)`` pair comes from a single, internally
consistent source. The two unit tests above are the runnable
sanity check.

---

## Item 18 -- Wave-mode value must match intent

**Source.** M6 PR4 Pre-2 audit (`docs/diagnostics/m6-pr4-pre2-steady-state-check.md`); fix landed on the `fix-s3-wavemod` branch.

**Rule.** Wave-mode value must match intent.
``WaveMod = 1`` is regular Airy (deterministic monochromatic wave at
`WaveTp`); ``WaveMod = 2`` is JONSWAP / Pierson-Moskowitz irregular
spectrum (random sea state with `WaveTp` interpreted as the spectral
peak). RAO cross-checks require ``WaveMod = 1``; spectral cross-checks
require ``WaveMod = 2``. **Misconfiguration is silent at deck
generation time** (OpenFAST happily accepts either value with the
same `WaveTp` field) and only surfaces in scenario response analysis
— typically when an FFT or per-cycle-amplitude diagnostic shows the
labelled wave period is not the dominant frequency in the wave-elev
channel. Pinned by the deck-generation regression test in
`openfast_setup/tests/test_scenario_decks.py` which asserts that the
generated `*_SeaState.dat` `WaveMod` value matches what
`seastate_edits` declared for every scenario.

**Failure mode that masked the original bug**: in S3 the comment
read `# regular Airy` while the value was `2` (JONSWAP). The deck
generated cleanly; OpenFAST ran cleanly; CSVs extracted cleanly.
The first place the misconfiguration would have surfaced was the
RAO assertion at PR4 — by which time we'd have committed the
fixtures, the test, the report. Pre-2's FFT diagnostic caught it
upstream.

---

## Item 19 -- The code-path exercise principle

**Source.** Generalisation of the M6 audit pattern across four
findings: hydrostatic-gravity (M5), asymmetric-CoG factor
(convention-audit), radiation-kernel (M6 PR3 pre), WaveMod (M6 PR4
Pre-2). All four share the same structural shape: a code path that
was correct in synthetic / unit / partial-scenario tests, but
silently wrong under production-quality inputs and full-scenario
activation.

**Rule.** A code path correct in synthetic, unit, or
partial-scenario tests is NOT validated to be correct under
production-quality inputs and full-scenario activation. Therefore:
any code path that consumes external data (BEM databases, deck
files, time-history fixtures) or that activates conditionally on
configuration values must have at least one cross-check or
production-data test in the suite. Synthetic-only validation is
necessary but not sufficient.

**PR-scoping implication.** When scoping a PR, ask:
*what code paths does this PR newly activate?* — and ensure each
gets a real-data test. Examples from the four findings:

- **Hydrostatic-gravity (M5)**: BEM readers shipped buoyancy-only
  `C` with a docstring caveat that the gravity contribution must
  be added downstream. The `assemble_cummins_lhs` path was
  unit-tested with synthetic full-`C` HDBs (which already had the
  gravity contribution baked in by hand). The bug surfaced at
  M6 PR1's audit when the code path "real BEM `C` flows through
  the unwritten-but-promised `Body` assembly" was first
  exercised.
- **Asymmetric-CoG factor**: a sign / factor-of-2 ambiguity in
  the gravity-restoring decomposition was invisible while all
  test fixtures had on-axis CoG (where the asymmetric term
  vanishes). Surfaced when the audit pattern (CLAUDE.md §14)
  required a discriminator test with off-axis CoG.
- **Radiation kernel (M6 PR3 pre)**: trapezoidal-cosine sum was
  unit-tested with constant-`B` synthetics (which masks both the
  truncation and Nyquist pathologies). The bug surfaced when
  the kernel code path was first exercised with realistic
  marin_semi.1 BEM data on a free-decay extraction at variable
  `t_max`.
- **WaveMod (M6 PR4 Pre-2)**: S1 / S2 ran `WaveMod = 0` (still
  water), bypassing wave generation entirely. S3 was the first
  PR to need wave forcing; the misconfig sat latent for two
  scenario PRs.

**Test-suite implication.** Cross-check tests at the
production-data level (M6 PR2 → PR6) are necessary; the
synthetic/unit suite alone has demonstrated four classes of
silent failure across this codebase. Conversely, the cross-check
pattern is uneconomical for every code path — the right
discipline is the one captured here: at PR-scoping time,
enumerate newly-activated code paths and ensure each has at
least one real-data exerciser somewhere in the suite (cross-check
PR, validation-tier test, or property-based test against a
non-trivial fixture).

---

## Item 20 -- RAO extraction from finite-time regular-wave runs requires frequency-selective filtering

**Source.** M6 PR4 Pre-2 step 4 smoke test (WaveTp = 10 s on
post-WaveMod=1-fix S3 reference). With wave generation now clean,
the pitch response showed a residual ~ 8 % secondary peak at
T ≈ 28.6 s — the OC4 pitch natural-period transient — even after
1000 s of simulation.

**Rule.** RAO extraction from finite-time regular-wave simulations
must use frequency-selective filtering at the wave frequency to
reject persistent natural-frequency transients ("ringing").
Extending simulation time is not a remediation: the transient
persists for many natural-period decay constants in lightly-damped
DOFs (OC4 pitch radiation damping at the natural frequency is
~ 10⁻⁹ in regime 3 per Item 16; with small-amplitude Morison drag
included it remains O(0.001) for the pitch responses S3 produces).
This is the standard practice in production RAO tools (OrcaFlex
post-processing, AQWA's `RAOSPECTRA`, WAMIT's harmonic post-
processors), and is documented in offshore-engineering literature
(Faltinsen 1990 §4 on transient responses; ITTC procedures for
regular-wave seakeeping tests).

**Implementation choice (M6 PR4)**: sinusoidal least-squares fit
at the wave frequency over the steady-state window. NOT a
band-pass filter. Reasoning: band-pass filters have
design-dependent phase shifts that bias the RAO phase output;
lstsq is unambiguous, one line of NumPy, and produces a useful
fit-residual diagnostic that flags frequencies where the wave-
frequency fit fails to capture most of the response variance
(typical signature: low-amplitude DOF responses where the
natural-frequency transient and the wave-driven response are
comparable in magnitude).

**Verification status.** Implemented in
`tests/validation/test_m6_openfast_regular_wave.py` (M6 PR4) for
every (DOF, wave frequency) pair. Per-pair fit residual recorded
as a diagnostic-only output; pairs with residual > 0.10 are
treated as advisory (logged but not asserted-on).

---

## Item 21 -- OpenFAST quantises WaveTp to the nearest IFFT bin

**Source.** M6 PR4 Pre-2 step 6 (post-WaveMod=1 fix verification).

**Rule.** OpenFAST's regular-wave generator constructs the wave
train via IFFT on a frequency grid with spacing
``WaveDOmega = 2 * pi / WaveTMax``. The requested ``WaveTp`` is
silently snapped to the nearest grid bin
``omega_k = k * WaveDOmega``, with ``k = round(WaveTMax / WaveTp)``.
For ``WaveTp`` values that don't evenly divide ``WaveTMax``, the
actual generated wave period differs from the labelled one by up
to half a bin-width ratio.

For the M6 S3 sweep with ``WaveTMax = 600 s`` the quantised periods
are:

| WaveTp [s] | k | T_actual [s] | rel-err |
|-----------:|--:|-------------:|--------:|
|  4 | 150 |  4.000 | 0     |
|  5 | 120 |  5.000 | 0     |
|  6 | 100 |  6.000 | 0     |
|  7 |  86 |  6.977 | -0.33 % |
|  8 |  75 |  8.000 | 0     |
| 10 |  60 | 10.000 | 0     |
| 12 |  50 | 12.000 | 0     |
| 14 |  43 | 13.953 | -0.33 % |
| 16 |  38 | 15.789 | -1.32 % |
| 18 |  33 | 18.182 | +1.01 % |
| 20 |  30 | 20.000 | 0     |
| 22 |  27 | 22.222 | +1.01 % |
| 25 |  24 | 25.000 | 0     |
| 30 |  20 | 30.000 | 0     |

**Implication for RAO extraction (Item 20).** The lstsq fit basis
at the body-response wave frequency MUST use the quantised
``omega = 2 * pi / T_actual``, NOT ``2 * pi / T_label``. Fitting at
the labelled frequency on a non-divisor-of-WaveTMax scenario
produces basis-frequency mismatch that inflates the fit residual
by O(few %) -- enough to fail a 5 % residual gate that the
underlying signal would otherwise pass cleanly. Pre-2's first
pass on the regenerated S3 reference flagged WaveTp = 16, 18, 22
as FAIL on the residual gate; switching to the quantised
frequency reduced residuals to < 0.0002 on all 14 scenarios.

**Implementation in PR4.** `tests/validation/test_m6_openfast_regular_wave.py`
computes the quantised period from the per-scenario SeaState
``WaveTMax`` at fit time. The same helper is used by
`scripts/m6_pr4_pre2_steady_state.py`. Pinned by the per-scenario
amplitude assertion: each scenario's wave_elev fitted at its
quantised omega must match WaveHs/2 to within rtol = 2e-2 (as
verified by Pre-2 step 6).

**Workaround (not used)**: setting WaveTMax to a multiple of every
labelled WaveTp would eliminate the quantisation, but the
required value (LCM of all labels) is impractically long. The
fit-at-quantised-omega convention is the standard remediation.

---

## Item 22 -- WAMIT files are non-dimensional by default; readers must apply ``rho * g * ULEN^k`` rescaling

**Source.** M6 PR4 Pre-3 finding
(`docs/diagnostics/m6-pr4-pre3-rao-verification.md`); fix landed
on the `fix-wamit-dimensionalisation` branch. Pre-3 surfaced the
bug via dual-path RAO verification (FloatSim WAMIT-impedance
versus OpenFAST time-series-lstsq); the impedance-path heave RAO
came out ~10⁴× too small at WaveTp = 25 s.

**Rule.** WAMIT public-format text files are **non-dimensional by
default** (WAMIT v7 manual §4.2; HydroDyn user guide §6 follows
the same scheme). Any reader consuming `.1` / `.3` / `.hst` must
apply the per-DOF dimensional rescaling factors:

| File | Coefficient | Non-dim → dim factor (mode i, mode j) |
|------|-------------|----------------------------------------|
| `.1` | `A_ij(omega)`, `A_inf_ij` | ``rho * ULEN^k_ij`` |
| `.1` | `B_ij(omega)` | ``rho * omega * ULEN^k_ij`` |
| `.3` | `F_exc_i` (per unit wave amp) | ``rho * g * ULEN^l_i`` |
| `.hst` | `C_ij` | ``rho * g * ULEN^k_ij'`` |

with the integer powers:

- `k_ij = 3 + (1 if i in {4,5,6}) + (1 if j in {4,5,6})` for
  `.1` (added mass / damping).
- `l_i = 2 + (1 if i in {4,5,6})` for `.3` (excitation).
- `k_ij' = 2 + (1 if i in {4,5,6}) + (1 if j in {4,5,6})` for
  `.hst` (hydrostatic stiffness).

OC4 marin_semi: `rho = 1025 kg/m^3`, `g = 9.80665 m/s^2`,
`ULEN = 1.0 m`. With `ULEN = 1`, the only factors that matter are
`rho` (= 1025) and `g` (= 9.80665).

**FloatSim implementation**: ``floatsim.hydro.readers.wamit``
exposes ``assume_dimensional`` (default ``False``),
``rho_water_kg_m3``, ``g_m_s2``, ``ulen_m`` kwargs on the four
public readers (``read_wamit``, ``read_added_mass_and_damping``,
``read_excitation_force``, ``read_hydrostatic_stiffness``).
Callers reading actual WAMIT files use the defaults; callers
reading hand-crafted dimensional fixtures (e.g.
`tests/fixtures/bem/wamit/synthetic_simple.*`) pass
``assume_dimensional=True``.

**Strengthened heuristic.** The pre-fix
``_maybe_warn_nondimensional`` used a single magnitude threshold
(``max|A| < 10``) which missed the marin_semi case (surge `A_inf
= 8527` non-dim is above 10). Post-fix uses a *rotational* added-
mass threshold (`A_inf[3:, 3:]` < `1e8 kg*m^2`) — semi-submersible
pitch added mass is O(1e9) dim, so 100× below that is
unambiguously non-dim. Fires loudly if the caller asserts
``assume_dimensional=True`` on data that doesn't pass the
rotational test.

**Verification status.** Pinned by:

- `tests/unit/test_wamit_reader.py::test_dot1_marin_semi_dimensional_A_inf_heave_matches_robertson`
  asserts dimensional A_inf_heave matches Robertson 2014
  Table 3-1 published 1.45 × 10⁷ kg to within 5 %.
- `test_dot1_marin_semi_dimensional_A_inf_pitch_matches_robertson`
  asserts dimensional A_inf_pitch matches the published value
  to within 5 %.
- `test_dimensionality_heuristic_catches_high_amplitude_nondim`
  asserts the strengthened heuristic catches the marin_semi
  pattern that the pre-fix threshold missed.

---

## Item 23 -- Deferred-known-bugs must be tracked, not just commented

**Source.** M6 PR4 Pre-3 retrospective.

**Rule.** When a code comment of the form "this is a separate
bug, out of scope" is written, it must be paired with a **named,
tracked Phase-N follow-up entry** in the project's report
document or equivalent tracker. Comments rot; tracked items
force decisions.

**Failure mode this captures.** The WAMIT dimensionalisation bug
was known and explicitly documented in
`tests/validation/test_oc4_pitch_period_buoyancy_only_c.py`'s
docstring from M5 PR1 onward:

> "the WAMIT reader does NOT currently apply ULEN-based dimensional
>  rescaling — that's a separate latent bug, out of scope for this
>  fix."

The comment was correct, the diagnosis was right, but no tracked
follow-up entry was created. Subsequent PRs (M5 PR2, M6 PR1,
M6 PR2, M6 PR3, fix-pr2-cmzt, fix-radiation-kernel,
fix-s3-wavemod) all built on the broken reader. The bug surfaced
at M6 PR4 — five PRs later — when RAO extraction first
exercised the F_exc-dominated regime.

**Standing rule going forward.** Any code comment of the form
"this is a separate bug, out of scope for this PR" must be
accompanied by:

1. A named entry in `docs/openfast-cross-check-report.md`'s
   Named follow-ups section (or the analogous tracker for non-M6
   work), with a unique identifier (e.g., `F2-foo`, `KD-N`).
2. A short description of the symptom and the deferred fix.
3. The PR-scoping implication: which downstream PRs are blocked
   on this fix landing.

This pairs with Item 19 (code-path exercise principle): the
*reason* deferred bugs persist is that the latent code path
isn't exercised until a much later PR; the tracker forces the
question "what does this comment defer to?" at the moment the
comment is written.

**Verification status.** The four known-but-deferred items as of
2026-05-06 (F1-residual, KD-2, KD-3, and this WAMIT-dim bug
itself) are tracked in `docs/openfast-cross-check-report.md`.
Future deferred-bug comments must be paired with an entry there.

---

## Item 24 -- LEAD vs LAG: phase reporting between impedance and lstsq paths

**Source.** M6 PR4 Pre-3 phase-residual diagnosis,
``docs/diagnostics/m6-pr4-pre3-phase-residual-diagnosis.md``.

**Rule.** When comparing an impedance-domain RAO phase against a
time-domain ``cos + sin`` lstsq fit (as Pre-3's dual-path
verification does), the two reporting conventions differ by a
sign:

- **Impedance path** (Path A): ``arg(xi_hat)`` under the ``+i``
  convention is the LEAD of the response (negative-of-lag).
- **lstsq path** (Path B): ``atan2(B, A)`` on the
  ``A cos(omega t) + B sin(omega t)`` basis is the LAG of the
  signal.

For the same physical motion under the ``+i`` convention they
satisfy ``Path A = -Path B``. To compare them at a 1° phase
gate, **negate exactly one** before subtracting. The Pre-3
verification script negates Path A's ``arg(xi_hat)`` so both
paths report LAG.

**Failure mode this captures.** Pre-3 dual-path verification on
M6 PR4 Pre-3, post-WAMIT-dim-fix, showed a 12.7° phase gap at
WaveTp = 10 s (Path A −6.17°, Path B +6.55° — mirror reflection
around 0°). Initial diagnosis attributed this to a WAMIT
``exp(-i omega t)`` vs FloatSim ``exp(+i omega t)`` time-
convention mismatch; conjugating F_exc on read narrowed the
gap (12.7° → 2.7° at 10 s) but did not close it, *and* widened
the 25 s gap (0.42° → 0.92°). The conjugation hypothesis is
falsified by the asymmetric move at the two frequencies.

The actual residual is a phase-reporting convention: under the
``+i`` convention,

- ``x(t) = Re[x_hat * exp(+i omega t)] = Re(x_hat) * cos - Im(x_hat) * sin``
- ``= |x_hat| * cos(omega t + arg(x_hat))``

so ``arg(x_hat)`` is the LEAD. A ``cos + sin`` fit returns
``atan2(B, A)`` — equivalent to writing
``R cos(omega t - atan2(B, A))`` — which is the LAG.

For the same physical motion: ``LAG = -LEAD``. Negating Path A
collapses 12.7° → 0.38° at 10 s and 0.42° → 0.39° at 25 s —
both within the 1° gate.

**Standing rule going forward.** Any phase comparison between
an impedance-domain complex amplitude and a time-domain cos+sin
fit must explicitly state the reporting convention (LEAD or
LAG) and negate exactly one side. Mixing conventions silently
produces sign-flipped phase residuals that scale with the
magnitude of the imaginary part of the impedance solution —
small near LF / HF asymptotes (where Im(xi) is small) and
large in resonant or diffraction-dominated regimes.

**Verification status.** Pre-3 dual-path verification at
WaveTp = 10 s and 25 s passes with Path A LAG = −arg(xi_hat).
Future RAO consumers (M6 PR4's 84-assertion sweep, future
multi-body PRs) inherit this discipline via the shared
``tests/support/rao_extraction.py`` module — Path A consumers
must negate at the call site or wrap the impedance solver to
return LAG by default.

---

## Item 25 -- Retardation-kernel three-check gate structure (post-fix-wamit-dim refactor)

**Source.** M6 PR4 Pre-3 / fix-wamit-dimensionalisation Decision E
(2026-05-07). The pre-fix gate was a single hard error on the
input proxy ``|B_ii(omega_max)| / max|B_ii| > 1 %``. Post-WAMIT-
dim-fix the marin_semi BEM is dimensional and the surge / sway /
yaw entries land at ~ 1.7 % of peak — below the asymptote-regime
threshold but well above 1 %. The pre-fix gate would have blocked
these DOFs at fixture construction even though their kernels
decay cleanly. Decision E refactored the gate into three separate
checks, each testing something different.

**Rule.** ``floatsim.hydro.retardation.compute_retardation_kernel``
runs three independent checks on every retardation-kernel
construction:

1. **Check 1 — input proxy (advisory)**: computes
   ``|B_ii(omega_max)| / max|B_ii|`` per diagonal. **Soft warning**
   if any diagonal exceeds ``5 %``. Indicates the BEM grid is
   under-resolved relative to the typical asymptotic-regime
   cutoff. Does NOT raise — the post-extension check below is
   authoritative.
2. **Check 2 — asymptote consistency (hard error)**: computes
   ``std(B*omega^4) / mean(B*omega^4)`` over the last 10 grid
   samples per entry. Raises ``ValueError`` if any diagonal entry
   exceeds ``0.10`` (the per-entry ``1/omega^4`` tail fit is
   unreliable otherwise). Off-diagonal failures fall back to
   zero-tail-contribution rather than raising.
3. **Check 3 — post-extension kernel decay (hard error)**: after
   the kernel is built (Filon-trapezoidal cosine quadrature on
   the BEM grid + ``1/omega^4`` tail extension on
   ``[omega_max, 5*omega_max]``), computes
   ``|K_ii(t_max)| / max|K_ii(t)|`` per diagonal. Raises
   ``ValueError`` if any exceeds ``0.001`` (= 0.1 %). This is the
   authoritative gate: an un-decayed kernel produces sustained
   oscillation in the Cummins convolution and corrupts the
   simulation.

**Calibration.** The marin_semi reference passes all three
checks with margin: heave/roll/pitch < 1 % at omega_max (Check 1
silent); surge/sway/yaw at ~ 1.7 % (Check 1 silent at the 5 %
threshold); std/mean ratios well under 0.10 across all diagonals
(Check 2 silent); post-extension decay to < 6e-5 of peak by
t = 200 s (Check 3 ~ 50× margin from the 0.1 % gate). See
``docs/diagnostics/m6-pr4-pre3-surge-kernel-quality.png``.

**Standing rule going forward.** Every kernel construction
inherits the three-check discipline automatically. Tests that
need to bypass Check 3 (e.g., synthetic narrow-Gaussian B(omega)
that decays slowly) must use a t_max large enough to clear the
gate, or accept the ``ValueError`` as the test's expected
outcome. Pinned by:

- ``tests/unit/test_retardation_kernel.py::test_kernel_check1_warns_on_high_b_at_omega_max_but_does_not_raise``
- ``tests/unit/test_retardation_kernel.py::test_kernel_raises_check_3_when_decay_is_too_slow``
- ``tests/unit/test_retardation_kernel.py::test_marin_semi_passes_all_three_checks_cleanly``

**Why the refactor.** The pre-fix single-check structure
conflated three different concerns into one gate: "does the BEM
grid extend into the asymptotic regime" (Check 1's question),
"is the per-entry tail fit well-defined" (Check 2's question),
and "does the resulting kernel actually decay" (Check 3's
question). Different DOFs can satisfy different subsets at
different threshold values, and the right answer is the
intersection of all three rather than a single proxy. The
refactor also documents which check should fire on which
pathology, so future contributors know whether the right fix is
"widen the BEM grid" (Check 1 + Check 3 both fire), "the BEM
data is corrupt or sub-asymptotic" (Check 2 fires), or "increase
t_max" (only Check 3 fires).

**Verification status.** Pinned by the three unit tests above.

---

## Item 26 -- MoorDyn dynamic damping is not captured by analytic catenary

**Source.** M6 PR4 implementation (2026-05-08 G3 narrowing). The
PR4 time-domain dual-path test surfaced that OC4 unmoored
radiation-only heave damping is ζ ≈ 0.057 % (verified from
``B_33(omega_n) / (2 sqrt((M+A_33(omega_n)) * C_33))`` at
omega_n_heave = 0.3635 rad/s; e-folding time ~ 81 minutes; 1 %
decay time ~ 6.2 hours). OpenFAST's reference simulation at
``TMax = 1200 s`` reaches a clean steady state on heave only
because **MoorDyn provides dynamic mooring damping** that
FloatSim's analytic catenary connector does not capture.

**Rule.** Forced-response time-domain cross-checks of lightly-
damped DOFs fail when the OpenFAST reference includes mooring
damping that FloatSim cannot reproduce. The lstsq fit on the
FloatSim time-series is contaminated by the un-decayed free-decay
transient, while the OpenFAST fit is clean. Phase 1 cross-checks
must use either:

  (a) **impedance-domain validation**, which does not require
      transient settling (the impedance is purely algebraic); or
  (b) **scenarios where the dominant damping mechanism is matched
      in both tools** (e.g., S5 drag-on heave decay, where Morison
      drag dominates radiation in both tools).

**Failure mode this captures.** The M6 PR4 plan was originally
time-domain (integrate Cummins forward at each WaveTp, lstsq-fit,
compare against OpenFAST's lstsq). Decision A's structural
sub-check (time-domain ≅ impedance at WaveTp = 10 s, 25 s) caught
the disagreement. After F2 extended the FloatSim duration to 1200 s
to match OpenFAST, the disagreement persisted -- confirming that
duration-matching alone is insufficient when the damping mechanism
is missing on one side.

**Verification status.** PR4 narrowed to impedance-only Path A
(verified across 14 wave periods); time-domain dual-path test
preserved as xfail-strict pending two named follow-ups
(F-WAVE-FORCE-CONV + F-DAMP-MATCH); see
``docs/openfast-cross-check-report.md`` PR4 entry. Future work:
(a) move time-domain RAO validation to S5 where Morison drag
matches; (b) wire MoorDyn-equivalent dynamic mooring damping into
FloatSim's catenary connector (out of Phase 1 scope).

---

## Item 27 -- Free-decay vs forced-response damping tolerance

**Source.** Same as Item 26 (M6 PR4 G3 narrowing).

**Rule.** Free-decay tests are *tolerant* of low damping -- the
transient IS the signal, the test asserts on its period and
envelope. Forced-response RAO tests are *not tolerant* of low
damping -- the transient *contaminates* the wave-frequency lstsq
fit, biasing both amplitude and phase. Test design must consider
which regime applies before scoping a cross-check.

**Quick rule of thumb.** For a DOF with damping ratio ζ:

  - free-decay test: any ζ > 0 is fine (you measure ζ from the
    envelope); the transient damping rate IS the assertion.
  - forced-response RAO test: needs ``simulation_duration >>
    -log(target_residual) / (ζ ω_n)`` BEFORE the lstsq window
    opens, otherwise the wave-frequency fit picks up the
    free-decay transient as off-frequency content. For OC4 heave
    (ζ = 0.057 %, ω_n = 0.364 rad/s), reaching 1 % residual takes
    ~ 1050 s of simulation TIME; reaching 0.1 % takes ~ 1575 s.

**Failure mode this captures.** Item 26's PR4 time-domain
disagreement was *partly* missing-MoorDyn-damping (the dominant
mechanism in OpenFAST that FloatSim lacks) but also reflects this
test-design distinction: a 1200 s OpenFAST simulation with light
radiation-only damping would have the same transient contamination
as FloatSim. OpenFAST's saving grace is MoorDyn; remove that, and
both tools fail the forced-response test design at low ζ.

**Verification status.** Codified at M6 PR4 G3; applied at
test design time for any future forced-response cross-check.

---

## Item 28 -- F-RESONANCE-PEAK-FRAGILITY: lightly-damped resonance peaks are not bug-suitable for tight cross-checks

**Source.** M6 PR4 implementation (2026-05-09 H1 marker
refinement) after the post-PR4 sweep showed heave RAO disagreement
in a band around ``T_n_heave = 17.286 s``. Empirically confirmed
by ``scripts/m6_pr4_resonance_fragility.py``: at exactly
``omega_n_heave`` the peak amplitude varies by 9.3 % across
linear / cubic / nearest interpolation schemes for ``B(omega)`` --
the schemes are identical 5 % off-resonance, but at the peak
itself the choice of interpolation produces a non-trivial spread.

**Rule.** RAO at resonance scales as ``|F_exc(omega_n)| /
(omega_n · B(omega_n))``. When ``B(omega_n)`` is small (ζ < 1 %),
small differences in interpolation produce 10-20 % differences in
peak amplitude. **Off-resonance the steep impedance slope
``|Z(omega)| = |C - omega^2 (M+A) + i omega B|`` magnifies small
``(M+A, C)`` differences into large RAO disagreements that taper
smoothly with offset rather than cutting at any specific band
edge.** This is a property of the comparison, not a property of
either tool.

**Cross-check action.** Within a band around any DOF's natural
frequency -- **currently ±25 % of omega_n empirically**, calibrated
to capture observed PR4 fragility patterns including the heave
14 s phase tail at +24 % offset -- cross-checks must use either
widened tolerance (``rtol = 20 %``) or be excluded via xfail-strict
with an F-RESONANCE-PEAK-FRAGILITY reason. The principled criterion
is the impedance-magnitude band where ``|Z(omega)|`` is within a
factor of K of its minimum value at omega_n; this is tracked as
**TODO-FRAGILITY-BAND-CRITERION** for a future refinement that
replaces the empirical ±25 % with a mechanism-derived band.

**Per-metric calibration in PR4.** The ±25 % rule applies
uniformly, but xfail markers are calibrated to the empirically-
failing periods (the rule predicts fragility, but accidental passes
within the band do not flag as XPASS-strict). For heave in PR4:

  - amp xfail-strict: WaveTp = 16, 18 s
  - phase xfail-strict: WaveTp = 14, 16, 18 s
    (14 s amp passes; the impedance-slope tail produces phase err
    of −7.79° at +24 % offset but amp gap is only −0.80 %.)

**Verification status.** Pinned by
``tests/validation/test_m6_openfast_regular_wave.py`` per-metric
xfail-strict markers; calibration evidence in
``scripts/m6_pr4_resonance_fragility.py`` and
``docs/diagnostics/m6-pr4-rao-sweep-results.md``.

---

## Item 29 -- F-LOW-SNR: cross-check has an SNR floor; skip rather than xfail

**Source.** M6 PR4 implementation (2026-05-09 H1 marker
refinement). At super-resonant wave periods (WaveTp << T_n) the
body's response amplitude is at the numerical noise floor in both
tools; the OpenFAST lstsq fit at the wave frequency has
``resp_resid > 0.10`` (response is dominated by off-frequency
content, not the wave drive).

**Rule.** Frequencies where the OpenFAST response at the wave
frequency is below the lstsq fit's noise floor (``resp_resid >
0.10``) cannot be cross-checked meaningfully. **Skip such
frequencies with a documented reason; do NOT use xfail.** xfail
implies a known cause of failure (a named follow-up that, when
closed, will make the test pass); a low-SNR comparison is not
"expected to fail" -- it's "not meaningfully comparable." The
cross-check ratio of two near-zeros is dominated by whichever
tool's noise structure happens to have larger projection onto
the wave-frequency lstsq basis.

**Threshold rationale.** ``resp_resid > 0.10`` means the wave-
frequency lstsq fit captures less than 90 % of the signal
variance; the remaining 10 %+ is at other frequencies (free-decay
transient, low-frequency drift, super-frequency noise). A 5 % rtol
on amp or 5° atol on phase from a fit that itself has 10 %+ off-
frequency content is not a meaningful test.

**Verification status.** Pinned by
``_maybe_skip_low_signal`` in
``tests/validation/test_m6_openfast_regular_wave.py``.
Empirically, the threshold catches heave at WaveTp = 4-6 s, pitch
at WaveTp = 4-8 s, and pitch at WaveTp = 30 s (where the long-wave
limit + small body excitation produces low pitch RAO).

---

## Item 30 -- HydroDyn joint axial drag uses 1/4 factor, not standard Morison 1/2

**Source.** M6 PR6 Step A/B/C investigation (Outcome (a)) — read of
`OpenFAST/modules/hydrodyn/src/Morison.f90` lines 3085 (init) and 4742
(runtime) plus first-principles verification against the S5 OC4
hyperbolic-envelope measurement.

**HydroDyn's axial drag formula at a joint** (verified from Morison.f90):

```fortran
! init (line 3079-3085):
p%An_End(:,i) = An_drag      ! Σ_members sgn · k · π · R²  (full disc area vector)
Amag_drag = Dot_Product(An_drag, An_drag)
p%DragConst_End(i) = JAxCd · ρ / (4 · Amag_drag)

! runtime (line 4729 + 4742):
vmag = vrel · An_End                                 ! scalar
F_D_End(i, j) = An_End(i) · DragConst_End(j) · |vmag| · vmag
```

Algebraic reduction at a joint with a single attached vertical
member of diameter `D` (`An_End = ±π R² ẑ`, `A_x = πR² = πD²/4`):

```
F_z = -(1/4) · ρ · A_x · JAxCd · v_z · |v_z|
```

**Compare to "standard Morison" axial drag**:

```
F_z_naive = -(1/2) · ρ · A_x · Cd_axial · v_z · |v_z|
```

So HydroDyn's `JAxCd` is **implicitly a "two-face combined" disc
coefficient**: HydroDyn applies it with a 1/4 factor instead of the
standard Morison 1/2. Per-face equivalent (for matching against
single-face heave-plate Cd tables): `Cd_per_face = JAxCd / 2`. To
reproduce HydroDyn's drag in a FloatSim Morison-equivalent
aggregation:

```
R_axial_joint = 0.25 · ρ · A_x · JAxCd · cos³(θ_axis_from_vertical)
```

**Verification status.** Pinned by:

- `scripts/m6_pr6_drag_aggregation.py` — aggregates OC4 marin_semi 25
  cylindrical members + 3 axial-drag joints with the 1/4 factor;
  predicts δ_hyperbolic = 0.3130 1/m vs OpenFAST S5 measured 0.3090
  1/m (1.28 % rel-err, within 5 % gate).
- `docs/diagnostics/m6-pr6-drag-aggregation.md` — derivation and
  per-member contribution table.
- `tests/validation/test_m6_openfast_drag_decay.py` (PR6) — full
  FloatSim Morison run with the aggregate, hyperbolic-envelope
  log-decrement assertion against OpenFAST.

**Standing rule going forward.** Any FloatSim simulation that needs
to reproduce HydroDyn axial-drag effects must use the 1/4 factor
(or equivalently, halve the HydroDyn `JAxCd` before passing to
FloatSim's standard Morison module). The FloatSim Morison module
itself uses the standard 1/2 factor (per `floatsim/hydro/morison.py`
docstring); the conversion lives at the test / driver layer.

**Lessons learned.** The factor-of-2 disagreement was suspiciously
clean and traced cleanly to the HydroDyn source within a 30-minute
budget. The "AxCd is a two-face combined coefficient" interpretation
is the most physically sensible read but is not documented in the
HydroDyn User's Guide (which gives input-file syntax but does not
state the formula's factor). Reading the source was the only way
to confirm. Future HydroDyn-equivalence work should default to
reading `Morison.f90` for any quantitative match — the User's Guide
gives the inputs, not the formula.

---

## Item 31 -- MoorDyn FairTen / AnchTen are positive scalar tension magnitudes

**Source.** M6 PR5 Step A — inspection of the OC4 S4 MoorDyn deck
(`tests/fixtures/openfast/oc4_deepcwind/inputs/s4_moored_eq/`) plus
direct check of the OF CSV column structure.

**Rule.** MoorDyn's `FairTen<i>` and `AnchTen<i>` output channels are
**positive scalar magnitudes** (no sign convention to track), units
of N. They represent the total cable tension at each end of line `i`,
NOT a 3-component vector projection. No coordinate-frame
specification is required — tension magnitude is frame-invariant.

For the **touchdown regime** (line lies on the seabed at its anchor
end, common in OC4 catenary mooring with ~ 17 m slack on 835 m
lines): the vertical tension at the touchdown point is zero by
definition of touchdown, so `AnchTen = H` (the horizontal tension
component, which is constant along the line). For the **fully
suspended regime**: `AnchTen = sqrt(H² + V_anchor²)` where
`V_anchor` is the vertical tension component at the anchor.

**FloatSim's `CatenarySolution` matches directly.** Its
`T_fairlead = sqrt(H² + V_fairlead²)` property is the exact analogue
of MoorDyn's `FairTen`. `T_anchor` is not exposed as a property but is
trivially `sqrt(H² + V_anchor²)` from the existing fields (and
reduces to `H` in touchdown regime — `V_anchor = 0` by the dataclass
field semantics).

**No sign or coordinate-frame mismatch between tools.** This is the
simplest cross-check convention encountered so far in M6: both tools
emit positive scalars in the same units.

**Verification status.** Pinned by:

- The OC4 S4 fixture's CSV columns
  (`fair_ten_line{1,2,3}_n`, `anch_ten_line{1,2,3}_n`) all positive
  in N.
- M6 PR5's pre-flight prediction (`scripts/m6_pr5_mooring_prediction.py`)
  computing `T_fairlead` and `T_anchor` from FloatSim's
  `CatenarySolution` and comparing to OpenFAST's reported tensions
  within Step C tolerance.

---

## Item 32 -- MoorDyn line MassDen is air mass; submerged weight needs cross-section buoyancy subtraction

**Source.** M6 PR5 Step B/C pre-flight diagnostic. First-pass
catenary prediction used the MoorDyn-deck `MassDen` value
(`113.35 kg/m` for OC4's 76.6 mm chain) directly as
`w = MassDen · g = 1112 N/m`, producing per-line tensions
4.2 % higher than the OpenFAST + MoorDyn reference. The 4.2 %
discrepancy was clean enough to suggest a missing physical
correction; investigation traced it to MoorDyn's convention:
**`MassDen` is the AIR mass per unit length, not the submerged
weight**.

**Rule.** For a catenary line submerged in water with density
`rho_water`, the **submerged weight per unit length** is:

```
w_submerged = (MassDen_air - rho_water * A_cross) * g
```

where `A_cross = pi * D^2 / 4` (for a cylindrical chain of
hydrodynamic diameter `D`) is the cross-sectional area that
displaces water. For OC4's chain:

```
A_cross = pi * 0.0766^2 / 4 = 4.61e-3 m^2
w_sub   = (113.35 - 1025 * 4.61e-3) * 9.80665
        = (113.35 - 4.72) * 9.80665
        = 1065.4 N/m                          (4.2 % less than air weight)
```

**FloatSim implementation.** `floatsim.mooring.CatenaryLine`'s
`weight_per_length` field is defined in the docstring as
"**submerged** weight per unit unstretched length, in N/m"
(`docstring excerpt from catenary_analytic.py:64`). Callers
reading MoorDyn decks must apply the correction at the
deck-parsing boundary.

**Generalisation.** This correction matters whenever a marine
line has non-negligible cross-section relative to its mass density.
The 4.2 % on OC4 chain happens because chain steel density
(~ 7800 kg/m³) is well above water but the chain has gaps; for
a solid synthetic rope or fibre line, the correction can be
larger or smaller depending on material vs water density. For
neutrally-buoyant lines the submerged weight is zero and the
analytic catenary degenerates (the solver requires
`weight_per_length > 0`); use a connector that handles
neutrally-buoyant cables explicitly.

**Verification status.** Pinned by:

- `tests/validation/test_m6_openfast_moored_eq.py` — uses the
  corrected `w_sub` in its `_LINE_W_SUB_N_PER_M` constant; all
  6 PR5 assertions pass at sub-0.15 % rel-err on tensions.
- `scripts/m6_pr5_mooring_prediction.py` — pre-flight derivation
  + comparison to OpenFAST showing 4.2 % discrepancy with naive
  `m_air · g` and 0.1 % agreement with the corrected `w_sub`.

---

## Item 33 -- Moored surge averaging window must cover >= 2 natural periods

**Source.** M6 PR5 pre-flight diagnostic (R1b TMax=1200s
re-extraction). After bumping the S4 fixture from 200 s to
1200 s, surge oscillation was still present at ~ 1 m amplitude
at the simulation end: OC4 moored surge has a ~ 100 s natural
period with very slow damping (radiation + MoorDyn line drag
are the only dissipation mechanisms in still water).

**Issue.** A short averaging window (e.g., the PR2 30-s
precedent for unmoored static equilibrium) samples one
half-cycle of a slowly-damped mode and is **biased by the
oscillation phase**, not the true equilibrium:

```
S4 surge over various last-N-second windows (TMax=1200s):
  last 30s:  -0.861 m  (biased by phase of oscillation)
  last 60s:  -0.396 m
  last 120s: -0.066 m
  last 200s: -0.0004 m  ✓ true equilibrium (by 3-fold symmetry)
  last 400s: -0.073 m
  last 600s: -0.088 m
```

The 200-s window covers 2 full natural periods and washes out
the oscillation phase to give the true mean. Heave + line
tensions remain clean over either window because their modes
are well-damped at this regime.

**Rule.** Moored cross-checks must use an averaging window
**at least 2× the longest under-damped natural period** of
the system. For OC4 (surge T_n ~ 100 s), this is 200 s. For
other floaters, derive from the slowest under-damped DOF.

**FloatSim implementation.** PR5's test
(`tests/validation/test_m6_openfast_moored_eq.py`) uses
`_SURGE_AVG_WINDOW_S = 200.0` for surge and the standard
`_HEAVE_AVG_WINDOW_S = 30.0` for heave / tensions. The
asymmetry is documented in the test fixture's reference-
load function.

**Generalisation.** Any future moored-equilibrium cross-check
must select averaging windows per DOF based on the under-
damped mode structure. The PR2 30-s precedent is appropriate
only for fully-damped modes (the static-eq case where mooring
isn't even present).

**Verification status.** Pinned by PR5 test's per-DOF window
selection + this convention's explanation. Pre-flight
diagnostic in `scripts/m6_pr5_mooring_prediction.py`.

---

## Verification status summary (PR2)

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
| 11. Channel naming + `out.info["attribute_names"]` access | ✅ verified at PR1.1 |
| 12. Last-30-s averaging for S1 / S4 | ✅ verified at PR2 (S1), 🟡 PR5 (S4) |
| 13. Tolerances accommodate residual oscillation | ✅ verified at PR2 (S1), 🟡 PR5 (S4) |
| 14. Static equilibrium tests assert only on restored DOFs | ✅ verified at PR2 (S1) |
| 15. Static equilibrium under Cummins linearisation | ✅ verified at PR2 (S1) |
| 16. Damping tolerance depends on dissipation regime | ✅ verified at PR3 (S2 regime 3); 🟡 PR6 (S5 regime 1) |
| 17. z_G consistency with mass M and stiffness C | ✅ verified at fix-pr2-cmzt (PR4 Pre-1 audit) |
| 18. Wave-mode value must match intent | ✅ verified at fix-s3-wavemod (PR4 Pre-2); pinned by deck-gen regression test in openfast_setup/tests/ |
| 19. Code-path exercise principle | ✅ documented; implication captured in PR-scoping guidance |
| 20. RAO extraction requires frequency-selective filtering | 🟡 PR4 (lstsq fit in test_m6_openfast_regular_wave.py) |
| 21. OpenFAST quantises WaveTp to nearest IFFT bin | ✅ verified at fix-s3-wavemod (PR4 Pre-2); fit at quantised omega closes the residual gate on all 14 S3 scenarios |
| 22. WAMIT files are non-dimensional; readers must rescale | ✅ verified at fix-wamit-dimensionalisation (PR4 Pre-3) |
| 23. Deferred-known-bugs must be tracked, not just commented | ✅ codified at fix-wamit-dimensionalisation; backfilled tracker entries for F1-residual, KD-2/3, this fix |
| 24. LEAD vs LAG -- phase reporting between impedance and lstsq paths | ✅ verified at fix-wamit-dimensionalisation Pre-3 (Path A negated) |
| 25. Retardation-kernel three-check gate structure | ✅ verified at fix-wamit-dimensionalisation Decision E (3 unit tests) |
| 26. MoorDyn dynamic damping vs analytic catenary | ✅ codified at M6 PR4 G3 narrowing; impedance-only PR4 path documented |
| 27. Free-decay vs forced-response damping tolerance | ✅ codified at M6 PR4 G3 narrowing |
| 28. F-RESONANCE-PEAK-FRAGILITY (±25% omega_n band, empirical) | ✅ verified at M6 PR4 H1 (scripts/m6_pr4_resonance_fragility.py + per-metric xfail markers); TODO-FRAGILITY-BAND-CRITERION tracks principled refinement |
| 29. F-LOW-SNR skip threshold (resp_resid > 0.10) | ✅ verified at M6 PR4 H1 (_maybe_skip_low_signal in test_m6_openfast_regular_wave.py) |
| 30. HydroDyn joint axial drag uses 1/4 factor (not standard Morison 1/2) | ✅ verified at M6 PR6 Step A/B/C from Morison.f90 source + first-principles check |
| 31. MoorDyn FairTen / AnchTen are positive scalar magnitudes; touchdown AnchTen = H | ✅ verified at M6 PR5 Step A from MoorDyn deck inspection + CSV channel check |
| 32. MoorDyn line MassDen is air mass per length; submerged weight = (m_air - rho_water * A_cross) * g | ✅ verified at M6 PR5 Step B/C (4.2 % tension correction matched empirically) |
| 33. Moored surge averaging window must cover >= 2 natural periods | ✅ verified at M6 PR5 pre-flight (last-30s biased; last-200s converges) |

**Items not allowed past PR1 without both columns filled:** none.
Every item above carries (a) a written assertion + source citation
AND (b) a runnable sanity-check protocol, even when the live
verification waits for the relevant scenario PR.

This file is **part of the audit pattern** codified in
CLAUDE.md §13. The same dual-column structure is the template for
future cross-check milestones.
