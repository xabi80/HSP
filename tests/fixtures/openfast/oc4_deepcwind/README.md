# OpenFAST OC4 DeepCwind Fixture Set (M6)

The committed-fixture vendoring + regeneration recipe for the M6
OpenFAST/HydroDyn cross-check. Owned by `docs/milestone-6-plan.md` v2.

## What lives here

```
tests/fixtures/openfast/oc4_deepcwind/
├── README.md                  # this file
├── inputs/                    # vendored OpenFAST input files
│   ├── OC4Semi_S1_static_eq.fst
│   ├── OC4Semi_S2_free_decay.fst
│   ├── OC4Semi_S3_rao_T08.fst       # ... + T10, T12, T14, T16, T18
│   ├── OC4Semi_S4_moored_eq.fst
│   ├── OC4Semi_S5_drag_decay.fst
│   ├── OC4Semi_HydroDyn.dat
│   ├── OC4Semi_ElastoDyn.dat
│   ├── OC4Semi_ServoDyn.dat         # off (CompServo=0) but referenced
│   ├── OC4Semi_AeroDyn.dat          # off
│   ├── OC4Semi_InflowWind.dat       # off
│   ├── OC4Semi_MAP.dat              # mooring (S4 only)
│   └── HydroData/
│       └── marin_semi.{1,3,hst,4}   # full OC4 BEM (not the trimmed
│                                    # subset under tests/fixtures/bem/wamit)
├── outputs/                   # pre-extracted SI-canonical CSVs (committed)
│   ├── s1_static_eq.csv       # see tests/support/openfast_csv.py for schema
│   ├── s1_static_eq.json
│   ├── ... (one CSV+JSON per scenario)
└── (none yet -- this is the M6 PR1 scaffolding state)
```

**M6 PR1 scaffolding note:** at PR1 time the `inputs/` directory is
empty -- the OpenFAST input files have not yet been vendored. PR2
will land them alongside the first scenario it activates. The
vendoring procedure below is the agreed-upon recipe; PR2 (and any
subsequent PR re-vendoring) follows it verbatim.

## Vendoring procedure (inputs)

The OpenFAST inputs derive from the OpenFAST/r-test repository:

- **Upstream:** [OpenFAST/r-test](https://github.com/OpenFAST/r-test)
- **Path in upstream:** `glue-codes/openfast/5MW_OC4Semi_Linear/`
- **License:** [Apache 2.0](https://github.com/OpenFAST/r-test/blob/main/LICENSE)
  -- safe to redistribute with attribution.

Step 1 -- shallow-clone the upstream:

```bash
git clone --depth 1 https://github.com/OpenFAST/r-test.git /tmp/r-test
```

Step 2 -- copy the linear-restoring OC4 case verbatim:

```bash
cp -r /tmp/r-test/glue-codes/openfast/5MW_OC4Semi_Linear/* \
      tests/fixtures/openfast/oc4_deepcwind/inputs/
cp /tmp/r-test/glue-codes/openfast/5MW_Baseline/HydroData/marin_semi.* \
   tests/fixtures/openfast/oc4_deepcwind/inputs/HydroData/
```

Step 3 -- per-scenario `.fst` overrides. The upstream r-test ships
one `.fst` per case; the M6 cross-check needs five hydrodynamics-only
variants (S1-S5 per `docs/milestone-6-plan.md` v2 Q2) that share the
same submodule files but differ in `CompElast/CompAero/CompInflow/CompServo`
flags, initial conditions, wave parameters, and durations. PR2 commits
the per-scenario `.fst` files alongside its first failing assertion;
the rest of the inputs stay verbatim from upstream.

The decision to **disable the wind turbine** (`CompElast=CompAero=
CompInflow=CompServo=0`) for hydrodynamics-only isolation is locked
in `docs/milestone-6-plan.md` v2 Q2. **Footgun** -- with `CompElast=0`
ElastoDyn does not apply gravity, so HydroDyn alone provides only the
buoyancy-referenced restoring (no `m*g*z_G` term). For the static
equilibrium scenario (S1) we use the workaround documented in
`docs/openfast-cross-check-conventions.md` (CompElast=1 with
locked-DOF approach OR HydroDyn standalone driver with explicit
gravity input).

## Output regeneration

The SI-canonical CSV+JSON pairs under `outputs/` are produced by:

```bash
# Ensure 'openfast' is on PATH; download from
# https://github.com/OpenFAST/openfast/releases
python scripts/extract_openfast_fixtures.py --scenario all
```

The script is the **single point of unit conversion** between
OpenFAST's native output (degrees, mixed units) and FloatSim's
canonical SI fixtures (radians, metres, Newtons). See
`tests/support/openfast_csv.py` module docstring for the canonical
CSV schema and the JSON sidecar contract.

CI does **not** regenerate; CSVs are committed as artefacts and
loaded directly via `tests.support.openfast_csv.load_openfast_history`.

## Re-vendor when

- OpenFAST major version bump that changes output channel naming
  (the rename table in `scripts/extract_openfast_fixtures.py
  _convert_to_canonical_si` may need updating).
- Geometry tweak in the upstream r-test fixture (rare; the OC4
  marin_semi case is multi-tool-validated and stable).
- Initial-condition / scenario-parameter change that requires a fresh
  `.fst` file.

Re-vendoring updates the JSON sidecars' `openfast_version` and
`extracted_at` so an audit can detect the bump.

## Geometry citation

Robertson, A. et al., 2014. *Definition of the Semisubmersible
Floating System for Phase II of OC4*. NREL/TP-5000-60601.
[PDF](https://www.nrel.gov/docs/fy14osti/60601.pdf)

The same case is also cited from `docs/wamit-fixture-attribution.md`
(M5 PR1's trimmed `marin_semi.*` subset) and is the canonical
floating-platform reference deck across M5-M6.
