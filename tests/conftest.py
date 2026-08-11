"""Shared pytest configuration for the FloatSim test suite.

Registers and activates a deterministic Hypothesis profile.

Why this file exists
--------------------
Hypothesis defaults to ``derandomize=False``, drawing a fresh seed on every
run, so property tests explore a different set of examples each time. Two runs
of this suite are therefore not comparable, and a property test that passes
today can fail tomorrow with no change to the code.

That is tolerable for ordinary development. It is not tolerable for the
FloatFEA coupling, whose gate **G1.5** requires this suite to produce
*bit-identical* results on the export branch and at the reference tag: any
difference is supposed to mean the solve path was touched. A randomised
strategy guarantees differences that have nothing to do with the export branch,
and the pressure would then be to relax G1.5 to "close enough" — which is
tolerance drift wearing a packaging costume.

Compounding it, ``.hypothesis/`` is gitignored (``.gitignore:40``) and no
example database is tracked, so a falsifying example found on one machine does
not reproduce on a fresh checkout. Determinism has to come from the seed, not
from a local cache that travels with nobody.

This is a test-harness file only. It adds no lines to the integrator or the
force model and does not touch the solve path.

Note that Hypothesis has **no pyproject.toml loader** — a ``[tool.hypothesis]``
table would parse silently and do nothing. Registering the profile here is the
only mechanism that takes effect.
"""

from __future__ import annotations

from hypothesis import HealthCheck, Verbosity, settings

settings.register_profile(
    "floatsim",
    derandomize=True,
    print_blob=True,
    # Numerical property tests can be slow without being wrong; a deadline
    # would turn machine load into a spurious failure and defeat the point.
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "debug",
    parent=settings.get_profile("floatsim"),
    verbosity=Verbosity.verbose,
)

settings.load_profile("floatsim")
