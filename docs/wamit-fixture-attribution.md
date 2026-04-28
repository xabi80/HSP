# WAMIT Fixture Attribution

The trimmed WAMIT files committed under `tests/fixtures/bem/wamit/` named
`marin_semi_trimmed.{1,3,hst}` are derived from the
**OC4 DeepCwind semi-submersible** WAMIT outputs distributed with the
OpenFAST regression-test repository.

## Source

- **Upstream repo:** [OpenFAST/r-test](https://github.com/OpenFAST/r-test)
- **Path in upstream:**
  `glue-codes/openfast/5MW_Baseline/HydroData/marin_semi.{1,3,hst}`
- **Upstream license:** [Apache 2.0](https://github.com/OpenFAST/r-test/blob/main/LICENSE)
- **Geometry reference:** Robertson, A. et al.,
  *Definition of the Semisubmersible Floating System for Phase II of OC4*,
  NREL/TP-5000-60601, 2014.
  [https://www.nrel.gov/docs/fy14osti/60601.pdf](https://www.nrel.gov/docs/fy14osti/60601.pdf)

## Why this case

The OC4 marin_semi WAMIT case is multi-tool-validated — OpenFAST/HydroDyn,
ProteusDS, WEC-Sim, and AQWA all consume it. This makes parser
disagreements with the published outputs unambiguous bugs in the parser,
not reference uncertainty. It exercises every WAMIT format edge case
FloatSim cares about:

- Both the `PER == -1` infinite-frequency row and the `PER == 0`
  zero-frequency row are present.
- Off-diagonal coupling entries appear in both upper and lower triangle
  with small (panel-method-noise) asymmetries — see the docstring of
  `floatsim.hydro.readers.wamit._resolve_6x6_from_dict`.
- Excitation forces span 37 wave headings (full coverage at 10° spacing).

## Trim recipe

The full upstream files are ~10 MB combined — committing them verbatim
would be wasteful. The trim script keeps:

- **`.1`:** rows for `PER == -1`, `PER == 0`, and three finite periods
  closest to 5 s, 8 s, and 12.5 s. The exact periods written depend on
  the upstream frequency grid (typically `4.987 s`, `7.953 s`,
  `12.566 s`).
- **`.3`:** same three finite periods, restricted to wave heading
  `BETA = 0°`.
- **`.hst`:** unchanged — the file is already small (1 KB).

## Reproducing the trim

```bash
# Clone the upstream r-test (sparse-checkout HydroData if you prefer):
git clone --depth 1 https://github.com/OpenFAST/r-test.git /tmp/r-test

# Run the trim script (idempotent):
python scripts/trim_marin_semi_fixture.py \
    --src /tmp/r-test/glue-codes/openfast/5MW_Baseline/HydroData \
    --dst tests/fixtures/bem/wamit
```

The script lives at `scripts/trim_marin_semi_fixture.py` and is
documented inline. Output filenames are prefixed `marin_semi_trimmed.*`
to make the derivation obvious in directory listings.

## Conventions inherited from the source

The committed `marin_semi_trimmed.hst` reflects the WAMIT convention:
hydrostatic restoring covers the **buoyancy/waterplane** contribution
only. Gravity terms (`m·g·z_G` on roll and pitch) are not present —
WAMIT does not know the body's mass distribution. Downstream
`floatsim.bodies.Body` assembly is expected to add the gravity term.
The OrcaFlex VesselType YAML reader, by contrast, returns the full
restoring (OrcaFlex bundles mass into the VesselType). Tests that
cross-compare the two readers (M5 PR3) must apply this offset
explicitly.

## License compliance

OpenFAST is published under the Apache License 2.0. Redistributing
trimmed copies of the upstream files within this repository is permitted
under the Apache 2.0 grant of patent and copyright rights, with this
attribution file serving as the required notice. No modifications to the
file format have been made — the trim only drops rows; numeric values
in retained rows are bit-for-bit identical to the upstream.
