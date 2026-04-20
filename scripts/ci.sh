#!/usr/bin/env bash
# Local CI runner. Mirrors what a hosted CI would run on every PR.
# Fails fast — first failing step exits non-zero.
#
# Usage:
#   bash scripts/ci.sh            # run all checks
#   bash scripts/ci.sh --fix      # auto-fix ruff/black where possible, then re-check

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FIX=0
if [[ "${1:-}" == "--fix" ]]; then
    FIX=1
fi

echo "==> ruff"
if [[ $FIX -eq 1 ]]; then
    ruff check --fix .
else
    ruff check .
fi

echo "==> black"
if [[ $FIX -eq 1 ]]; then
    black .
else
    black --check .
fi

echo "==> mypy (strict on floatsim/)"
mypy

echo "==> pytest (excluding @pytest.mark.slow)"
pytest -q -m "not slow"

echo "==> all checks passed"
