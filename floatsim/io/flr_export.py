"""Solve-state export to the FloatFEA `.flr` load interchange record.

Module 1 of 3 in the FloatFEA export set (solve state; then strip/patch; then
panel pressures). Schema: FloatFEA `docs/load-interchange-v1.md` **v1.2**.

Additive by construction
------------------------
This module is a **pure consumer** of :class:`~floatsim.solver.newmark.IntegrationResult`
and the assembled setup. It adds no lines to the integrator or the force model,
imports nothing that runs during a solve, and cannot alter a result. FloatFEA
gate **G1.5** requires HSP's regression suite to be bit-identical with and
without the export branch; nothing here can move a number.

The radiation memory term is not exported by the integrator
-----------------------------------------------------------
``mu(t)`` is a loop local in ``newmark.py`` (created ``:391``, updated ``:449``,
consumed ``:421``) and appears in no return value. Rather than change the solve
path to emit it, :func:`recompute_mu` **replays the convolution offline** over
the stored velocity history, using the same
:class:`~floatsim.hydro.retardation.RadiationConvolution` and the same push
order the integrator uses. That reproduces the applied values exactly and keeps
the additive-only property intact.

FloatFEA gate **G1.6** compares a reconstructed per-panel radiation field
against the body-level radiation force actually applied. Without ``mu`` that
comparison cannot be made at all, which is why this is a required channel.

Why the integrator block is exported
------------------------------------
Generalized-alpha does not form its balance at a timestep; it forms it at
*alpha-weighted* states (``newmark.py:415-422``)::

    (1-alpha_m) M a_{n+1} + alpha_m M a_n
      + (1-alpha_f) C x_{n+1} + alpha_f C x_n
      + mu_n                                   <- LAGGED, NOT BLENDED
      = (1-alpha_f) F_{n+1} + alpha_f F_n

So forces are exported **per source at index n**, together with the blend
parameters, and the consumer forms the blend itself. Exporting a pre-blended
``F_alpha`` would destroy the per-source decomposition G1.6 depends on, and a
blended sum cannot be taken apart again.

``alpha_m`` is exported alongside ``alpha_f`` because the **inertia** term blends
with a different parameter (0.42105 against 0.47368 at the default
``rho_inf = 0.9``). A consumer given only ``alpha_f`` would form the right force
blend against the wrong acceleration blend, and that residual would look exactly
like an FE mapping error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Final

import h5py
import numpy as np
from numpy.typing import NDArray

from floatsim.hydro.retardation import RadiationConvolution, RetardationKernel
from floatsim.solver.newmark import _generalized_alpha_coefficients

if TYPE_CHECKING:
    from floatsim.solver.newmark import IntegrationResult

SCHEMA_VERSION: Final[str] = "1.2"

# The convolution is lagged one step and never blended -- newmark.py:391, 421,
# 459, and the docstring at :48. Declared so a consumer does not blend it like
# the other terms and introduce an error while trying to remove one.
MU_TREATMENT: Final[str] = "lagged_unblended"


@dataclass(frozen=True)
class Provenance:
    """Record identity. A record without provenance is not analysable.

    FloatFEA gate **G1.3** rejects rather than warns on a missing field here.
    """

    hsp_git_sha: str
    hsp_dirty: bool
    floatsim_version: str
    run_id: str


def integrator_block(rho_inf: float, dt: float) -> dict[str, object]:
    """The ``/meta/integrator`` block required at schema v1.2.

    A **partial** block is rejected by the reader: a consumer holding
    ``alpha_f`` but not ``alpha_m`` would silently blend the inertia term
    wrongly, which is worse than a missing block because it still produces
    numbers.
    """
    alpha_m, alpha_f, gamma, beta = _generalized_alpha_coefficients(rho_inf)
    return {
        "scheme": "generalized_alpha",
        "rho_inf": float(rho_inf),
        "alpha_m": float(alpha_m),
        "alpha_f": float(alpha_f),
        "beta": float(beta),
        "gamma": float(gamma),
        "dt": float(dt),
        "mu_treatment": MU_TREATMENT,
    }


def recompute_mu(
    kernel: RetardationKernel,
    xi_dot: NDArray[np.floating],
    *,
    from_run_start: bool,
) -> tuple[NDArray[np.float64], int]:
    """Replay the radiation convolution to recover ``mu[n]`` as applied.

    Reproduces ``newmark.py``'s sequence exactly: the buffer is seeded with
    ``xi_dot[0]`` before the loop, ``mu[0]`` is **zero** by the startup
    convention (ARCHITECTURE.md §9.3 -- the buffer-evaluated value there is an
    O(dt) artifact the integrator deliberately skips), and thereafter each step
    pushes ``xi_dot[n+1]`` and then evaluates.

    Verified faithful, not argued: instrumenting ``integrate_cummins`` on a
    throwaway branch to record the in-solve ``mu`` and diffing against this
    replay gave **bit-identical** results, 0 of 4806 elements differing. The
    instrumentation was never merged, so the exporter stays additive.

    The warm-up region
    ------------------
    ``mu[0] = 0`` is correct **only at a true run start**, where the solver's own
    buffer was also empty. Given a *truncated window* with prior history the
    zero-padding is simply wrong, and it stays wrong until the buffer refills --
    so the leading samples are **invalid, not merely approximate**.

    ``from_run_start`` is a required keyword with no default, because a wrong
    default here produces plausible numbers rather than an error. Callers must
    state which case they are in.

    Parameters
    ----------
    kernel
        The retardation kernel the run used. Its ``dt`` must match the run's.
    xi_dot
        ``(N, n_dof)`` generalized velocity history from the same run.
    from_run_start
        ``True`` when ``xi_dot[0]`` is the first sample of the run, so the
        zero-padded buffer matches the solver's own startup. ``False`` for any
        window with history before it.

    Returns
    -------
    ``(mu, valid_from)`` -- ``(N, n_dof)`` float64 and the first index at which
    ``mu`` is trustworthy. ``valid_from`` is 0 at a run start and
    ``kernel.n_lags`` otherwise. **If ``valid_from >= N`` the entire array is
    invalid**, which is the common case for a short window against a long
    kernel: the 12-buoy platform runs a 60 s kernel (6000 lags) while
    ``run_case`` returns roughly 1955 samples.
    """
    v = np.asarray(xi_dot, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"xi_dot must be 2-D (N, n_dof); got shape {v.shape}")
    n_samples, n_dof = v.shape
    if n_dof != kernel.n_dof:
        raise ValueError(
            f"xi_dot has {n_dof} DOFs but the kernel has {kernel.n_dof}; "
            "the velocity history and kernel must come from the same run."
        )

    buffer = RadiationConvolution(kernel)
    buffer.push(v[0])

    mu = np.zeros((n_samples, n_dof), dtype=np.float64)
    # mu[0] stays zero: the continuous startup value is 0 and the integrator
    # uses that for its first RHS rather than the buffer artifact.
    for n in range(n_samples - 1):
        buffer.push(v[n + 1])
        mu[n + 1] = buffer.evaluate()

    valid_from = 0 if from_run_start else int(buffer.n_lags)
    return mu, valid_from


def write_solve_state(
    path: str | Path,
    *,
    result: IntegrationResult,
    kernel: RetardationKernel,
    body_name_to_index: dict[str, int],
    provenance: Provenance,
    rho_inf: float,
    gravity: float,
    water_density: float,
    water_depth: float,
    scale: str = "model",
    from_run_start: bool,
) -> None:
    """Write the solve-state half of a ``.flr`` record.

    Writes ``/meta``, ``/time``, ``/kinematics/<body>`` and, when the run
    carried constraints, ``/joints/<id>/lam``. Strip, patch and panel groups
    belong to the other two export modules.

    ``gravity`` is written as the vector ``(0, 0, -g)`` using **FloatSim's**
    value, not standard gravity -- FloatSim runs at 9.81 and a mismatch would
    surface downstream as an unexplained mass error.
    """
    t = np.asarray(result.t, dtype=np.float64)
    if t.size < 2:
        raise ValueError(f"need at least 2 samples to export; got {t.size}")
    dt = float(t[1] - t[0])

    meta: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "floatsim_version": provenance.floatsim_version,
        "hsp_git_sha": provenance.hsp_git_sha,
        "hsp_dirty": bool(provenance.hsp_dirty),
        "run_id": provenance.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "units": {"length": "m", "mass": "kg", "time": "s", "force": "N", "angle": "rad"},
        "gravity": [0.0, 0.0, -float(gravity)],
        "water_density": float(water_density),
        "water_depth": float(water_depth),
        "scale": scale,
        "assumptions": [],
        "integrator": integrator_block(rho_inf, dt),
    }

    mu, mu_valid_from = recompute_mu(
        kernel, result.xi_dot, from_run_start=from_run_start
    )
    if mu_valid_from >= t.size:
        raise ValueError(
            f"the entire mu history would be invalid: the kernel carries "
            f"{mu_valid_from} lags of memory but only {t.size} samples were "
            "supplied. Export from the full run rather than a truncated window, "
            "or carry at least one kernel length of pre-history."
        )
    meta["mu_valid_from"] = int(mu_valid_from)
    meta["window"] = {
        "from_run_start": bool(from_run_start),
        "t_first": float(t[0]),
        "t_last": float(t[-1]),
        "n_samples": int(t.size),
    }

    with h5py.File(str(path), "w") as h:
        h.attrs["meta"] = json.dumps(meta)

        time_grp = h.create_group("time")
        time_grp.create_dataset("t", data=t)
        time_grp.attrs["dt"] = dt
        time_grp.attrs["n_samples"] = int(t.size)

        for name, idx in sorted(body_name_to_index.items(), key=lambda kv: kv[1]):
            sl = slice(6 * idx, 6 * idx + 6)
            g = h.create_group(f"kinematics/{name}")
            g.create_dataset("position", data=np.asarray(result.xi)[:, sl][:, 0:3])
            g.create_dataset("rotation", data=np.asarray(result.xi)[:, sl][:, 3:6])
            g.create_dataset("velocity", data=np.asarray(result.xi_dot)[:, sl][:, 0:3])
            g.create_dataset("angular_velocity", data=np.asarray(result.xi_dot)[:, sl][:, 3:6])
            g.create_dataset("acceleration", data=np.asarray(result.xi_ddot)[:, sl][:, 0:3])
            g.create_dataset(
                "angular_acceleration", data=np.asarray(result.xi_ddot)[:, sl][:, 3:6]
            )
            # Per-channel, not global: FloatSim interprets xi[3:6] three ways
            # across its own modules and they agree only to first order. See
            # FloatFEA docs/conventions.md sec. Rotations.
            g.attrs["rotation_parameterisation"] = "zyx_intrinsic_euler"

            r = h.create_group(f"loads/{name}/radiation")
            r.create_dataset("mu", data=mu[:, sl])
            r.attrs["time_alignment"] = "state_n"
            r.attrs["mu_treatment"] = MU_TREATMENT
            # Samples before this index saw a zero-padded buffer that the solver
            # did not. The validator rejects any case screened inside it.
            r.attrs["valid_from"] = int(mu_valid_from)

        if result.lam is not None:
            j = h.create_group("joints")
            j.create_dataset("lam", data=np.asarray(result.lam, dtype=np.float64))
            # lam is expressed against the constraint Jacobian rows in each
            # joint's own basis -- NOT a force/moment 6-vector -- and the
            # Jacobian is evaluated at the step midpoint (newmark.py:436).
            j.attrs["jacobian_evaluation"] = "step_midpoint"
