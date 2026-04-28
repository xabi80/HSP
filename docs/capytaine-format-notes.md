# Capytaine Format Notes

User-facing reference for `floatsim.hydro.readers.capytaine.read_capytaine`
and the `load_hydro_database(..., format="capytaine")` dispatch entry
point. Capytaine writes diffraction/radiation results as an
xarray-backed NetCDF file; this page documents what FloatSim consumes,
what it ignores, and the unit / sign conversions performed at the
boundary.

## Files read

| Variable                          | Dims                                            | Status |
| --------------------------------- | ----------------------------------------------- | ------ |
| `added_mass`                      | `(omega, radiating_dof, influenced_dof)`        | Required |
| `radiation_damping`               | `(omega, radiating_dof, influenced_dof)`        | Required |
| `excitation_force`                | `(complex, omega, wave_direction, influenced_dof)` | Preferred |
| `Froude_Krylov_force`, `diffraction_force` | same as above                          | Fallback (summed if `excitation_force` is absent) |
| `hydrostatic_stiffness`           | `(radiating_dof, influenced_dof)`               | Optional (zero if absent) |

QTFs, time-domain pressures, and free-surface elevation outputs are out
of scope for Phase 1.

## DOF ordering

Capytaine accepts arbitrary DOF labels (string-typed). FloatSim
requires the standard six labels (case-insensitive):

```
Surge, Sway, Heave, Roll, Pitch, Yaw
```

The reader reorders the 6x6 matrices and 6-vectors into FloatSim's
canonical (surge, sway, heave, roll, pitch, yaw) ordering regardless of
the order in which the labels appear on disk.

## Time / phase convention — Capytaine LAGS, FloatSim LEADS

Capytaine uses

```
x(t) = Re[X * exp(-i * omega * t)]   (LAGS)
```

FloatSim's `HydroDatabase.RAO` uses

```
F(t) = Re[X * A_wave * exp(+i * omega * t)]   (LEADS)
```

(matching the OrcaFlex VesselType reader). The reader applies the
translation **`RAO_floatsim = conj(excitation_force_capytaine)`** before
storing the RAO. If you compare Capytaine output to FloatSim output by
hand, remember the conjugation.

## Wave direction — Capytaine radians, FloatSim degrees

Capytaine stores `wave_direction` in radians. FloatSim stores
`heading_deg` in degrees. The reader converts at read time. If you
inspect the NetCDF file directly with `xarray`, expect radians.

## A_inf handling

Capytaine does not separately store an "infinite-frequency added mass"
matrix. Two valid sources for `A_inf`:

1. **The dataset includes a sample at `omega = +inf`.** Capytaine
   accepts `omega=infinity` as a problem condition; the resulting added
   mass is the high-frequency limit. The reader extracts that sample,
   stores it as `A_inf`, and strips the `+inf` row from the
   finite-frequency `A` and `B` arrays.
2. **The caller passes an explicit `a_inf` keyword.** Useful when the
   BEM run did not include the infinite-frequency case but the user
   knows `A_inf` from another source (analytical formula, prior run).

If neither is available, `read_capytaine` raises `ValueError`.
Guessing `A_inf` from the largest finite sample produces a subtly
wrong retardation kernel — the M2 PR4 regression is the cautionary
tale (overestimated `A_inf` shifts the heave natural period by
several percent and biases free-decay damping by tens of percent).

## Hydrostatic stiffness — same gravity gotcha as WAMIT

Capytaine's `hydrostatic_stiffness` covers the **buoyancy / waterplane**
contribution exclusively. Gravity terms (`m * g * z_G` on roll and
pitch) are absent — Capytaine does not know the mass distribution.
Downstream `floatsim.bodies.Body` assembly adds the gravity term, same
as for WAMIT. The OrcaFlex VesselType reader, by contrast, returns the
full restoring.

When `hydrostatic_stiffness` is absent from the NetCDF, the reader
substitutes a 6x6 zero matrix and assumes the body's restoring is
provided elsewhere (a hand-tuned `C` in the deck, or a separate
`floatsim.bodies.hydrostatics` calculation). This is the only field
that defaults silently — see the reader docstring for the rationale.

## Reference point

Capytaine's NetCDF does not carry a single canonical reference point
for moments. The default is `(0, 0, 0)` — the origin in the inertial
frame the BEM problem was solved in. Override via the
`reference_point` kwarg if the BEM run's origin was not at the
intended FloatSim reference point.

This matters for cross-checking against WAMIT (which writes its
reference point in the `.gdf` geometry file, propagated into the `.hst`
header) and OrcaFlex (which carries `OrigZ` on the VesselType).
The three readers must reference the same point for a clean
cross-check; pad the difference with `floatsim.bodies.shift_reference_point`
if not.

## Build dependencies

`netCDF4` (>= 1.6) and `xarray` (>= 2024.1) are baseline FloatSim
dependencies (added in M5 PR1, see `pyproject.toml`). Capytaine itself
is **not** a runtime dependency — FloatSim only consumes its NetCDF
output. Generating sphere fixtures requires Capytaine to be installed
in a separate environment; `scripts/build_sphere_capytaine_fixture.py`
is the tooling that runs Capytaine and writes the test NetCDF.

## Fixture and round-trip test

`tests/fixtures/bem/capytaine/sphere_submerged.nc` is the fully
submerged sphere fixture (Q4 of the M5 plan, depth ≥ 5R, no
free-surface effects). Analytical reference: `A_ii = (2/3) * pi * rho * R^3`
on all three translational DOFs (Lamb 1932 §92), `B_ii = 0`, `C = 0`.

The round-trip test in `tests/unit/test_capytaine_reader.py` exercises:
- DOF reordering: NetCDF written with non-canonical DOF order is
  reordered correctly into surge/sway/heave/roll/pitch/yaw.
- Phase translation: `excitation_force` from Capytaine is conjugated
  before storage as `RAO`.
- `A_inf` extraction from `omega=+inf` sample, including stripping
  the row from finite-frequency arrays.
- `hydrostatic_stiffness` defaults to zero when absent.
- `load_hydro_database(..., format="capytaine")` returns a
  `HydroDatabase` matching the analytical sphere reference to BEM
  panel-density tolerance.
