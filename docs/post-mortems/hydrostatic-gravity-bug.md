# Post-mortem — missing ``m·g·z_G`` gravity contribution to ``C``

**Discovered:** 2026-04-29 during the M6 OpenFAST cross-check audit.
**Fixed in:** branch `fix-hydrostatic-gravity-coupling` (this commit).
**Severity:** Phase 1 latent bug, not yet user-visible (no Phase 1 test
exercised the buggy combination of inputs); would have produced
catastrophically wrong dynamics on the OC4 cross-check the moment
M6 PR2 fired its first assertion.

## TL;DR

The BEM readers (WAMIT, Capytaine) produce a **buoyancy-only**
hydrostatic restoring matrix ``C`` and explicitly document that "the
gravity contribution ``m·g·z_G`` must be added by downstream
:class:`floatsim.bodies.Body` assembly." That downstream assembly was
never written. ``floatsim/hydro/hydrostatics.py`` — listed as a planned
module in ARCHITECTURE.md §4 — did not exist. ``assemble_cummins_lhs``
consumed ``hdb.C`` verbatim. For OC4 DeepCwind the missing term is
``+1.78e9 N·m/rad`` (positive, stabilising) — comparable in magnitude
to the buoyancy contribution itself, often even larger. A
buoyancy-only-``C`` OC4 case has **negative** pitch restoring without
the fix; ``natural_periods_uncoupled`` returns ``NaN`` and any
time-domain run integrates to instability.

## Symptom that surfaced the bug

Audit during the M6 plan: `floatsim/hydro/hydrostatics.py` is listed
in ARCHITECTURE.md §4 as "Restoring matrix, gravity/buoyancy balance"
but does not exist on disk. A grep for any code that adds ``m·g·z_G``
to a restoring matrix returned nothing. The only invocation site,
``assemble_cummins_lhs``, was passing ``hdb.C`` straight through.

Cross-checking with the readers' docstrings turned up the
contradiction. The WAMIT reader (``floatsim/hydro/readers/wamit.py``):

> The WAMIT ``.hst`` file contains only the **buoyancy / waterplane**
> contribution to hydrostatic restoring — it is *not* the full
> restoring matrix. ... Downstream :class:`floatsim.bodies.Body`
> assembly is expected to add the gravity contribution from the body's
> mass and CoG.

Capytaine reader (`capytaine.py`) carries the same disclaimer. The
M5 PR1 unit test ``test_marin_semi_trimmed_C_heave_is_positive``
explicitly notes "Roll/pitch may be negative — that is expected,
because the gravity restoring contribution m\*g\*z_G must be added by
the body assembly downstream." The author had clearly understood the
issue but the downstream code was never written.

## Numerical impact for OC4 DeepCwind

Robertson 2014 NREL/TP-5000-60601 Table 3-3 (full restoring):
``C_55 = 1.078e9 N·m/rad``.

Gravity-only contribution at OC4 mass ``m = 1.347e7 kg`` and
CoG depth ``z_G = -13.46 m``:
``-m·g·z_G = -1.347e7 · 9.80665 · (-13.46) ≈ +1.78e9 N·m/rad``.

Buoyancy-only residual: ``1.078e9 - 1.78e9 ≈ -7.0e8 N·m/rad``
(negative — a buoyancy-only OC4 has a **statically unstable** pitch
restoring; the platform sits stably only because gravity adds the
larger positive contribution).

The bug therefore halves to inverts the pitch restoring depending on
``z_G``. Heave is gravity-insensitive (``C_33 = ρgA_wp`` only); surge,
sway, yaw are unrestored anyway.

## Why the bug stayed invisible through M5

No existing test exercised the combination
**(reader-supplied buoyancy-only ``C``) ∧ (non-trivial mass/CoG) ∧
(pitch/roll DOFs).**

| Test | Path | Why it passed |
|------|------|----------------|
| M1 ``test_oc4_natural_periods`` | Hand-coded synthetic full-restoring ``C`` from Robertson 2014 Table 3-3 | Bypasses readers; no reader→assembly path involved |
| M2 ``test_oc4_heave_free_decay`` | OrcaFlex VesselType reader (``platform_small.yml``) → assembly → heave-only IC | Heave is ``C_33 = ρgA_wp``; gravity contributes 0. *Also*: OrcaFlex VesselType actually returns FULL restoring (see "Convention notes"), so the bug was double-invisible on this path |
| M2 ``test_cummins_free_decay_analytical`` | Synthetic heave-only ``C`` | Heave only |
| M3 ``test_m3_regular_wave_steady_state`` | Synthetic full-``C`` fixture, heave-dominated | No reader, no pitch IC |
| M4 multi-body validations | OrcaFlex VesselType + heave-coupling tests | Same as M2: heave-dominated AND OrcaFlex returns full |
| M5 reader unit tests (PR1, PR2) | Read-only — never call ``assemble_cummins_lhs`` | Bug not exercised |

The combination first surfaces in M6 PR2 (S1 OC4 static equilibrium),
where pitch and roll are exercised against a reader-supplied
buoyancy-only ``C``. The audit caught it before that PR fired its
first assertion.

The pre-existing test ``test_marin_semi_trimmed_C_heave_is_positive``
in `tests/unit/test_wamit_reader.py` is interesting: it asserts
``C[2,2] > 0`` (heave) but says nothing about pitch/roll, with a
docstring that **acknowledges** roll/pitch may be negative because the
gravity term has not yet been added. Institutional memory of the
missing-step survived, but the missing step was never built.

## The fix

1. **``floatsim/hydro/hydrostatics.py``** — new module with
   :func:`gravity_restoring_contribution`. Returns the symmetric 6×6
   ``ΔC_grav`` to be added to the buoyancy-only BEM ``C``:

   ```
   ΔC[3, 3] = ΔC[4, 4] = -m·g·z_G                  (dominant)
   ΔC[3, 5] = ΔC[5, 3] = ½·m·g·x_G                 (cross-coupling)
   ΔC[4, 5] = ΔC[5, 4] = ½·m·g·y_G                 (cross-coupling)
   all other entries: 0
   ```

   Convention is rotation-vector (consistent with FloatSim's
   quaternion-internal storage); see "Convention notes" below for the
   factor-of-½ rationale.

2. **``HydroDatabase.C_source: Literal["buoyancy_only", "full"]``** —
   new mandatory field (no default; every reader must declare).
   Validated in ``__post_init__``. The flag forces every BEM reader
   to declare what its ``C`` contains, removing the silent assumption
   that bit us.

3. **``assemble_cummins_lhs(... mass, cog_offset_from_bem_origin,
   gravity)``** — three new optional kwargs. Policy:
   - If ``hdb.C_source == "buoyancy_only"`` AND any of the three is
     ``None``: raise ``ValueError`` with explicit guidance. Cannot
     silently produce wrong physics.
   - If ``hdb.C_source == "buoyancy_only"`` AND all three provided:
     add ``gravity_restoring_contribution(...)`` to ``C``.
   - If ``hdb.C_source == "full"`` AND any of the three is provided:
     warn (likely double-counting); do not add.
   - If ``hdb.C_source == "full"`` AND none provided: pass ``C``
     through unchanged. Existing M2/M3/M4 tests that go through the
     OrcaFlex VesselType reader (which is ``"full"``) keep working
     unchanged.

4. **Reader source declarations:**
   - ``floatsim/hydro/readers/wamit.py`` → ``C_source="buoyancy_only"``
   - ``floatsim/hydro/readers/capytaine.py`` → ``C_source="buoyancy_only"``
   - ``floatsim/hydro/readers/orcaflex_vessel_yaml.py`` →
     ``C_source="full"``. OrcaFlex's VesselType bundles the body's
     mass distribution into the same record as the BEM output, so its
     exported ``HydrostaticStiffness`` block IS the full linearised
     stiffness. This is empirically verified against
     ``platform_small.yml`` (OC4 fixture): pitch ``C_55 ≈ 9.97e8 N·m/rad``,
     close to the Robertson 2014 full value ``1.078e9``; the
     buoyancy-only contribution would be ``-7e8`` (negative).

5. **Tests:**
   - ``tests/validation/test_oc4_pitch_period_buoyancy_only_C.py``
     (regression test) — pitch period from buoyancy-only ``C`` plus
     OC4 mass/CoG must fall in the published 22–32 s range.
     **Fails on un-fixed main with NaN; passes after fix.**
   - ``tests/unit/test_hydrostatics.py`` (8 tests) — diagonal terms,
     cross-coupling against a hand-derived reference, symmetry,
     translation/yaw zero-block, and four argument-validation cases.

## Convention notes — the rotation-vector factor of ½

For an offset CoG (``x_G``, ``y_G`` non-zero), the cross-coupling
``C[3, 5]`` and ``C[4, 5]`` depend on the rotation-vector
parameterisation chosen for the linearisation. The textbook
Faltinsen (1990, *Sea Loads*, Eq. 2.104) gives ``ΔC_46 = m·g·x_G`` (no
factor of ½) under the ZYX-intrinsic Euler convention. The
rotation-vector convention used here gives ``½·m·g·x_G``. Both agree
on the diagonal terms ``ΔC[3,3] = ΔC[4,4] = -m·g·z_G``.

FloatSim uses rotation-vector throughout the linearised assembly
(consistent with the quaternion-internal storage and the integrator
advancing quaternions via ``integrate_quaternion`` rather than
Euler-angle composition). The rotation-vector ½ is therefore the
convention-consistent choice.

For OC4 DeepCwind and any axisymmetric platform with ``x_G = y_G = 0``,
the cross-coupling vanishes and the convention choice is moot. It
matters only for non-symmetric mass distributions (e.g. ballasted
asymmetrically), which Phase 1 does not target. If a future Phase 2
case needs the Faltinsen convention specifically (because the BEM tool
producing the input uses ZYX-Euler), revisit by adding a
``rotation_parameterisation`` flag to
:func:`gravity_restoring_contribution`.

## What this teaches us

- **Trust but verify the docstring contract.** Three reader docstrings
  said "downstream assembly will add gravity." The downstream
  assembly didn't exist. The contradiction sat in plain sight for
  several milestones because no test exercised the buggy combination.
  The lesson: when a docstring claims a downstream invariant, follow
  the trail to the downstream code. If the code isn't there, that's a
  bug today, not "TODO."
- **A test author who anticipates the gap should also write the
  failing-test-now.** ``test_marin_semi_trimmed_C_heave_is_positive``'s
  docstring noted that roll/pitch may be negative because gravity
  hasn't been added. A companion test asserting the gravity term gets
  added (failing on day one until the missing module landed) would
  have surfaced the bug at M5 PR1 instead of M6 audit.
- **Mandatory dataclass fields force declarations.** ``C_source`` has
  no default. Every reader was forced to think about what its ``C``
  contains. The existing OrcaFlex assertion ("buoyancy_only" — wrong)
  was caught the moment the existing OC4 pitch test ran with the new
  flag check; "default to safer assumption if not — over-correction
  surfaces in the OC4 pitch test" was Xabier's prescient flag.
- **Cross-coupling is parameterisation-sensitive.** Implementing the
  diagonal-only term first (which is unambiguous) and adding the
  cross-coupling later with explicit convention notes preserves
  optionality. The unit test pins the convention so any future change
  is visible.

## References

- Faltinsen, O.M., 1990. *Sea Loads on Ships and Offshore Structures*,
  Cambridge University Press, Eq. 2.104.
- Newman, J.N., 1977. *Marine Hydrodynamics*, MIT Press, §6.16.
- Robertson, A. et al., 2014. *Definition of the Semisubmersible
  Floating System for Phase II of OC4*. NREL/TP-5000-60601, Table 3-3.
- WAMIT manual, §13.6 (``.hst`` file format and content).
- HydroDyn theory document (currently published with OpenFAST),
  §"Hydrostatic restoring".
