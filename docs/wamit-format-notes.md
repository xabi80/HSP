# WAMIT Format Notes

User-facing reference for `floatsim.hydro.readers.wamit.read_wamit` and the
`load_hydro_database(..., format="wamit")` dispatch entry point. Covers
which fields FloatSim consumes, which it ignores, and the sign / units
conventions translated at the boundary. The reader's docstring carries
the same content in code-comment form; this page is the "I have a WAMIT
case, what do I need to know" reference.

## Files read

| File   | Field                                       | Status |
| ------ | ------------------------------------------- | ------ |
| `.1`   | Added mass `A(omega)` and damping `B(omega)`| Required |
| `.1`   | `A_inf` (from `PER == -1` row)              | Required |
| `.1`   | Zero-frequency `A(0)` (from `PER == 0`)     | Read and discarded |
| `.3`   | Excitation force `X(omega, beta)` (complex) | Required |
| `.hst` | Hydrostatic stiffness `C` (6x6)             | Required |
| `.4`   | Motion RAO                                  | Optional cross-check (`read_motion_rao`) |

`.2` (mean drift), `.8`/`.9`/`.12` (QTFs), and `.pat` (panel pressures)
are out of scope for Phase 1.

## DOF ordering

WAMIT's DOF order is the FloatSim canonical order:

```
1 = surge   2 = sway   3 = heave   4 = roll   5 = pitch   6 = yaw
```

No reordering is performed; the matrices and force vectors flow through
unchanged.

## Time / phase convention

WAMIT's default is the "leads" phase convention:

```
F_exc(t) = Re[X(omega) * A_wave * exp(+i * omega * t)]
```

This matches `HydroDatabase.RAO` exactly, so `X` is stored as written.
Phase angles in `.3` and `.4` are degrees on disk and converted to
radians when assembled into the complex coefficient.

## Periods, omega, and the sentinel rows

- WAMIT writes period `PER` in seconds; the reader converts to
  `omega = 2 * pi / PER`.
- `PER == -1` is the **infinite-frequency** sentinel. The corresponding
  row carries `A_inf`. WAMIT writes no damping column for this row;
  some variants emit a zero column, which is also tolerated.
- `PER == 0` is the **zero-frequency** sentinel. `A(0)` is read and
  immediately discarded — FloatSim does not consume the
  zero-frequency added mass.
- Both sentinels are matched against `_PER_INFINITE_FREQ` and
  `_PER_ZERO_FREQ` with `atol = 1e-9`, so writers that emit
  `-1.0000000` or similar are tolerated.

## Symmetrization

WAMIT writes both `(i, j)` and `(j, i)` entries for off-diagonal
coefficients. They must agree to numerical precision. The reader
averages them and emits `UserWarning` if the relative disagreement
exceeds `1e-6` (`_SYM_RTOL`) — typically a sign that the WAMIT panel
mesh is too coarse for the chosen frequencies, not a parser bug.

## Hydrostatic stiffness — the gravity-term gotcha

The `.hst` file carries only the **buoyancy / waterplane** contribution
to hydrostatic restoring. WAMIT does not know the body's mass
distribution, so the gravity terms `m * g * z_G` on roll and pitch are
**absent**. Downstream `floatsim.bodies.Body` assembly adds the gravity
contribution from the body's mass and centre-of-gravity.

This contrasts with the OrcaFlex VesselType YAML reader, which returns
the **full** restoring (OrcaFlex bundles mass into the VesselType).
M5 PR3's three-reader cross-check applies this offset explicitly when
comparing the two outputs.

## Dimensional output requirement

WAMIT writes nondimensional output by default. FloatSim requires
dimensional output. Set the run-control flags to enable this — see the
HydroDyn user manual §6 for the canonical recipe (typically
`IPLTDAT=15` with `IPER` so `PER` is in seconds and `IFORCE` so
excitation is in N or N·m).

The reader emits a heuristic warning if the magnitudes look
nondimensional (`max |A| < 10` on the diagonal, the kind of value
nondimensional output produces for a platform-scale body).

## Multi-body / multi-draught

PR1 supports a single body at a single draught. WAMIT files with
multi-body output blocks or multi-draught sections are out of scope and
will fail the parser. If multi-body or multi-draught support is needed,
file a feature request — the reader can be extended to accept a body
index and a draught index.

## Fixture and round-trip test

`tests/fixtures/bem/wamit/marin_semi_trimmed.{1,3,hst,4}` is a trimmed
OC4 DeepCwind reference. See [`docs/wamit-fixture-attribution.md`](./wamit-fixture-attribution.md)
for source, license, geometry citation, and the trim recipe.

The round-trip test in `tests/unit/test_wamit_reader.py` exercises:
- `.1` parser: `A_inf`, zero-frequency, finite-frequency rows match expected values to `rtol=1e-12`.
- `.hst` parser: 6x6 `C` matches expected.
- `.3` parser: complex `F_exc(omega, beta)` magnitude and phase match.
- `.4` parser: motion RAO at one frequency reproduces the assembled
  prediction from `(M, A_inf, B, C, F_exc)` to `rtol=1e-3`.
- `load_hydro_database(..., format="wamit")` returns a `HydroDatabase`
  with the right shape; symmetry of `A`, `B`, `C` is validated.
