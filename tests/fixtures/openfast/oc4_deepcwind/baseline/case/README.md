# OpenFAST Setup Scripts (M6 cross-check)

Three scripts to generate, run, and (eventually) extract OpenFAST
fixtures for the M6 cross-check milestone.

## Files

| File | Purpose |
|------|---------|
| `scenario_config.py` | Frozen dataclass definitions of the 5 M6 scenarios. Single source of truth. |
| `generate_scenario_decks.py` | Vendors OC4 baseline from r-test, applies per-scenario edits, writes decks under `tests/fixtures/openfast/oc4_deepcwind/inputs/`. Does NOT run OpenFAST. |
| `run_scenarios.py` | Reads the manifest produced by the generator, runs OpenFAST on each deck, captures logs. Does NOT convert outputs to fixtures (that's a separate `extract_openfast_fixtures.py` Claude Code lands in M6 PR1). |

## Prerequisites

1. **OpenFAST v4.1.2** installed and on PATH (or pass `--openfast-bin`).
2. **r-test** clone at the **v4.1.2 tag**:
   ```bash
   git clone https://github.com/OpenFAST/r-test.git
   cd r-test && git fetch --tags && git checkout v4.1.2
   ```
3. **openfast_io** Python package (note: NOT `openfast-toolbox` and NOT `openfast_toolbox`; the actual published package on PyPI is named `openfast_io`):
   ```bash
   pip install openfast_io
   ```
   This is a *scripting* dependency, NOT part of FloatSim's runtime
   baseline (`CLAUDE.md` §9). Use a separate environment if you want
   strict isolation.

## How the editing works (openfast_io API)

`openfast_io` reads an entire OpenFAST deck — the `.fst` plus every
referenced module file — into one nested dictionary, `fst_vt`, with
sub-dicts keyed by module name (`Fst`, `ElastoDyn`, `HydroDyn`,
`MoorDyn`, `ServoDyn`, etc.). The generator:

1. Reads the baseline once into an `InputReader_OpenFAST`.
2. For each scenario, deep-copies `fst_vt`, mutates the right sub-dicts
   per the scenario edits, then hands the modified dict to
   `InputWriter_OpenFAST` along with an output directory. The writer
   emits the whole deck (`.fst` + module files) into that directory.

Sweep scenarios (S3 RAO) re-use the same baseline read but produce one
output directory per swept value.

Edits target keys that **must already exist** in the baseline. If a
key is missing the generator raises `KeyError` immediately with a
sample of available keys — this catches typos and version skew at
generation time rather than letting OpenFAST fail later with a less
informative error.

## Workflow

### Step 1: Generate decks

```bash
python generate_scenario_decks.py \
    --r-test-root C:/work/r-test \
    --hsp-root C:/work/HSP \
    --clean
```

Produces:
```
HSP/tests/fixtures/openfast/oc4_deepcwind/
├── baseline/                       # vendored from r-test (committed)
│   ├── case/                       # 5MW_OC4Semi_WSt_WavesWN/
│   └── 5MW_Baseline/               # only files OC4 references
└── inputs/                         # generated per-scenario decks
    ├── s1_static_eq/
    ├── s2_pitch_decay/
    ├── s3_rao_sweep/
    │   ├── WaveTp_004p0/
    │   ├── WaveTp_005p0/
    │   ├── ...                     # 14 frequencies
    │   └── WaveTp_030p0/
    ├── s4_moored_eq/
    ├── s5_drag_decay/
    └── manifest.json               # for run_scenarios.py
```

### Step 2: Sanity-check one deck

Before running all 18 decks, run S1 alone to verify the edits worked:

```bash
python run_scenarios.py \
    --hsp-root C:/work/HSP \
    --openfast-bin C:/OpenFAST/fast_x64.exe \
    --scenarios s1_static_eq
```

Expected: `OK (~30s)`. If FAIL, inspect `<deck_dir>/run_log/stdout.log`
for the OpenFAST error message.

### Step 3: Run everything

```bash
python run_scenarios.py \
    --hsp-root C:/work/HSP \
    --openfast-bin C:/OpenFAST/fast_x64.exe \
    --continue-on-error
```

Total runtime: roughly `5 × 60s + 14 × 120s ≈ 30 minutes` on a modern
laptop. (S3 RAO sweep is the long pole — 1200s simulation each.)

### Step 4: Hand outputs to Claude Code

Once all decks have run successfully, the `.out` files inside each
`<deck_dir>/` are the raw OpenFAST output. Conversion to the FloatSim
fixture format (CSV + JSON metadata) is done by
`extract_openfast_fixtures.py` — Claude Code lands the skeleton in
M6 PR1, you fill in the channel mapping based on what `OutList`
actually wrote.

## Known issues / future work

1. **OutList not edited programmatically.** Output channel selection
   is in each module's `OutList` block — multi-line, not a key=value
   field. The generator does NOT touch `OutList`; the baseline's
   defaults are used. If a needed channel is missing, edit `OutList`
   once in the vendored baseline (`baseline/case/*.dat`) before
   regenerating.

2. **Sweep parameter directory naming.** Names like `WaveTp_010p0` use
   `p` for the decimal point to avoid filesystem issues with multiple
   dots. Cosmetic.

3. **No parallelization.** Decks run serially in `run_scenarios.py`.
   For S3's 14-deck sweep this is wasteful but simple. Future
   enhancement: GNU parallel or `concurrent.futures`.

4. **No OpenFAST version check at runtime.** Script assumes v4.1.2 but
   doesn't verify the binary's version matches. Future: parse
   `--version` and assert.

5. **Edit-target dispatch is hardcoded.** `EDIT_TARGETS` maps
   `fst_edits/elastodyn_edits/hydrodyn_edits` to `fst_vt` keys. To
   edit MoorDyn, ServoDyn, or other modules, add a `*_edits` field to
   `Scenario` in `scenario_config.py` and an entry in `EDIT_TARGETS`.

## Reference

- M6 plan: `docs/milestone-6-plan.md`
- Conventions doc: lands in M6 PR1
- HSP architecture: `ARCHITECTURE.md`
- HSP working agreement: `CLAUDE.md`
- openfast_io: <https://github.com/OpenFAST/openfast_io>
