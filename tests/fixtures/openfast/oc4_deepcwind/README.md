# OpenFAST OC4 DeepCwind Fixture

Vendored from the OpenFAST/r-test repository
([github.com/OpenFAST/r-test](https://github.com/OpenFAST/r-test),
Apache 2.0) at the version pinned in `inputs/manifest.json`. The
case is `5MW_OC4Semi_Linear` per CLAUDE.md §12; see
`docs/milestone-6-plan.md` v2 for the M6 cross-check scope.

## Layout

```
oc4_deepcwind/
├── README.md                      this file
├── baseline/                      OpenFAST 5MW_Baseline + reference case
│   ├── 5MW_Baseline/              wind-turbine + WAMIT data shared by all decks
│   │   ├── HydroData/             marin_semi.{1,3,hst,ss,ssexctn} (full BEM)
│   │   ├── AeroData/              wind-turbine airfoil tables (unused, CompAero=0)
│   │   ├── Airfoils/              same
│   │   └── *.dat                  ElastoDyn / BeamDyn / InflowWind defaults
│   └── case/                      reference baseline run (sanity-check artifact)
│       ├── 5MW_OC4Semi_WSt_WavesWN.fst
│       └── *.dat                  baseline subdeck files
└── inputs/
    ├── manifest.json              canonical scenario manifest (18 entries)
    ├── s1_static_eq/              S1: static equilibrium, no waves, no mooring
    ├── s2_pitch_decay/            S2: 5° pitch IC, free decay, 600 s sim
    ├── s3_rao_sweep/              S3: regular-wave RAO at 14 periods (Tp=4..30s)
    │   ├── WaveTp_004p0/
    │   ├── WaveTp_005p0/
    │   ├── ...
    │   └── WaveTp_030p0/
    ├── s4_moored_eq/              S4: moored equilibrium with MoorDyn (3 lines)
    └── s5_drag_decay/             S5: heave free decay with Morison drag
```

Each scenario directory holds the full OpenFAST input set (`.fst`
driver, `_ElastoDyn.dat`, `_HydroDyn.dat`, `_SeaState.dat`,
`_BeamDyn*.dat`, etc.) **and** the OpenFAST run log (`run_log/`
with stdout/stderr/exit_code captured at extraction time).

## What is NOT committed

The repository root ignores OpenFAST run artifacts that are too
large for git, via `tests/fixtures/openfast/.gitignore`:

- `*.outb` — OpenFAST binary output (50–100 MB per scenario,
  ~1 GB across the M6 set).
- `*.chkp` — OpenFAST checkpoint files (~350 MB per scenario).

These are **regenerable** from the committed inputs by re-running
OpenFAST. The downstream artifacts the M6 scenario tests actually
consume are the small canonical CSV + JSON pairs that
`scripts/extract_openfast_fixtures.py` writes (~50 KB per
scenario; committed under each scenario's directory once produced).

## Manifest-driven configuration

`inputs/manifest.json` is the canonical scenario list:

- `openfast_version_required` — pinned OpenFAST version that
  produced the reference data. Cross-check tests should error if
  the local OpenFAST version (when re-extracting) differs.
- `r_test_tag_required` — corresponding OpenFAST/r-test tag.
- `scenarios[]` — one entry per scenario, with:
    - `scenario_name`, `deck_dir`, `purpose` (one-line)
    - `output_channels` — exact OpenFAST channel names used.
      MoorDyn outputs `FairTen{1,2,3}` and `AnchTen{1,2,3}`;
      platform DOFs are `PtfmSurge`/`PtfmHeave`/etc. without
      module prefix (per the conventions doc, Item 11).
    - `fst_edits`, `elastodyn_edits`, `hydrodyn_edits`,
      `seastate_edits` — overrides applied to the upstream
      r-test deck to scope the run to the M6 scenario.
    - `sweep_value` — for S3 only, the wave period in seconds.

## Reproduction

To regenerate the .outb binaries from scratch (e.g. after an
OpenFAST version bump or a deck-edit):

```
# Install OpenFAST locally (download at
# https://github.com/OpenFAST/openfast/releases or build from source).
# Ensure 'openfast' is on PATH.

# Install the .outb reader (used by extract_openfast_fixtures.py).
pip install openfast-toolbox

# Run the extraction script -- it iterates over manifest.json,
# invokes OpenFAST against each scenario's .fst driver, and writes
# the canonical CSV + JSON pair next to each scenario.
python scripts/extract_openfast_fixtures.py --scenario all

# To process already-existing .outb files (skip OpenFAST
# re-execution, e.g. after a fresh checkout where the binaries
# were regenerated locally but the CSVs were not committed):
python scripts/extract_openfast_fixtures.py --scenario all --read-only
```

The script's `--read-only` mode skips the OpenFAST invocation and
just reads the existing `*.outb` per the manifest, applies the SI
unit conversions, and writes the canonical CSV+JSON pairs. Used
when the OpenFAST run is owned by someone else (e.g. on a
license-restricted machine).

## Attribution

The committed inputs are derived from the OpenFAST/r-test
distribution under the Apache 2.0 license. The wind-turbine
files (AeroData, Airfoils, `_BeamDyn*`) are present because the
`.fst` driver references their paths even though the M6
scenarios disable wind (`CompAero=CompInflow=CompServo=0`).

OC4 DeepCwind geometry citation: Robertson, A. et al., *Definition
of the Semisubmersible Floating System for Phase II of OC4*,
NREL/TP-5000-60601 (2014).
