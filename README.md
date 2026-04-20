# FloatSim

Time-domain simulator for floating platforms (multi-body, 6-DOF each), consuming BEM
hydrodynamic databases from OrcaWave, WAMIT, or NEMOH/Capytaine. Internal engineering tool.

The architectural contract lives in [`ARCHITECTURE.md`](./ARCHITECTURE.md). The working
agreement for contributors (human or AI) lives in [`CLAUDE.md`](./CLAUDE.md). Read both
before changing code.

## Status

Phase 1, Milestone 0 — repository skeleton. No physics implemented yet.

## Install (development)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run checks locally

```bash
bash scripts/ci.sh
```

Runs `ruff`, `black --check`, `mypy --strict` on `floatsim/`, and `pytest -q`.

## Layout

See `ARCHITECTURE.md` §4.
