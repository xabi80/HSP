"""OrcaWave `.owr` direct reader -- placeholder, unvalidated end-to-end.

Status (as of M6 PR1, 2026-04)
------------------------------
This reader is **not implemented** and **no test fixture is currently
available**. Calling :func:`read_orcawave` raises :exc:`NotImplementedError`
with a redirect to the three verified BEM-reader paths.

The OrcaWave/OrcaFlex licenses needed to (a) regenerate `.owr` test
fixtures and (b) run `OrcFxAPI` for direct binary parsing are not
currently available -- see `docs/milestone-6-plan.md` v2 (post-OrcaFlex-
license-loss pivot) and the M6 plan's Q9. The module remains in the
tree as institutional memory for future re-introduction:

- If a full OrcaWave / OrcFxAPI license becomes available again, this
  module is the natural home for direct `.owr` binary parsing (the
  "fallback" path described in CLAUDE.md §7's original note).
- If a community sphere fixture (matching the locked M5 PR3 spec)
  lands as `.owr`, it can be wired up here and validated against the
  Lamb 1932 §92 added-mass reference.

Verified BEM-reader paths
-------------------------
For OrcaWave-derived BEM data, use one of these instead:

1. :mod:`floatsim.hydro.readers.orcaflex_vessel_yaml` -- the M1.5
   reader that consumes OrcaFlex's VesselType YAML export. This is
   the human-readable serialisation produced when OrcaWave `.owr` is
   imported into OrcaFlex and saved as YAML; it carries the full set
   of computed coefficients FloatSim needs.
2. :mod:`floatsim.hydro.readers.wamit` -- the M5 PR1 reader for the
   WAMIT plain-text outputs that OrcaWave can also export
   (`.1`/`.3`/`.hst`).
3. :mod:`floatsim.hydro.readers.capytaine` -- the M5 PR2 reader for
   Capytaine NetCDF, useful when running an open-source BEM solver
   on the same geometry for cross-comparison.

Why a placeholder rather than nothing
-------------------------------------
ARCHITECTURE.md §4 lists `floatsim/hydro/readers/orcawave.py` in the
module layout. A future contributor searching the codebase for
"orcawave" should find this docstring and the redirect; deleting the
file would force rediscovery of the design intent and the licensing
constraints. CLAUDE.md §7 carries the same status note; the two are
intentionally redundant.

This module is **not** exported from :mod:`floatsim.hydro.readers`
(it is absent from `__init__.py`'s `__all__` and the
:func:`load_hydro_database` dispatcher). Direct import remains
possible for diagnostic purposes; calling :func:`read_orcawave`
raises :exc:`NotImplementedError`.
"""

from __future__ import annotations

from pathlib import Path

from floatsim.hydro.database import HydroDatabase

_REDIRECT_MESSAGE = (
    "floatsim.hydro.readers.orcawave.read_orcawave is not implemented "
    "(no working test fixture, no licensed access to OrcFxAPI). For "
    "OrcaWave-derived BEM data, use one of: read_orcaflex_vessel_yaml "
    "(VesselType YAML export), read_wamit (.1/.3/.hst), or read_capytaine "
    "(.nc). See the module docstring of "
    "floatsim/hydro/readers/orcawave.py and CLAUDE.md §7."
)


def read_orcawave(path: str | Path) -> HydroDatabase:
    """Placeholder; raises :exc:`NotImplementedError`.

    Intended future signature parses an OrcaWave ``.owr`` binary file
    via OrcFxAPI (when licensed) or a companion YAML/text export
    (preferred) into a :class:`HydroDatabase`. Neither path is
    currently wired up.
    """
    raise NotImplementedError(_REDIRECT_MESSAGE)
