"""Rigid-body condensation map ``T`` (6N x 6) with the Q5 label contract.

M8 PR4 instantiates the plan-Q5 **label-mapping contract** on the
condensation path: body -> block mapping is **by label, never
positional**. The other half of Q5 -- ``build_system``'s coupled
assembly path (shared N-body database declaration) -- is **deferred to
M9** per the M8 scope exclusion ("kernel + excitation INGESTION only;
the consuming machinery is exercised by the condensation scripts, not
by new solver features"). When M9 wires that path, THIS module is the
reference implementation of the contract it must adopt: index maps
built from labels, with hard raises on mismatch, missing, and
duplicate labels.

Rationale (plan Q5): block misalignment at ``6N x 6N`` produces
plausible-looking wrong answers that pass smoke tests -- the failure is
silent and the result is dimensionally valid. Positional mapping cannot
detect it; label mapping makes it impossible.

Kinematics (M8 Phase-1 Measurement A convention, validated at 0.0000 %
by the MA-gate): for a rigid assembly moving with composite translation
``X`` and rotation ``Theta`` about the condensed reference point
``r_c``, body ``k`` at reference point ``r_k`` moves

    x_k     = X + Theta x (r_k - r_c)      (small rotations)
    theta_k = Theta

so the per-body 6x6 block is ``[[I, -skew(r_k - r_c)], [0, I]]`` and
``T`` stacks the blocks in ``body_labels`` order. Condensed quantities:
``M_c = T^T M_18 T``, ``F_c = T^T F_18``, etc.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _skew(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cross-product (skew-symmetric) matrix of a 3-vector."""
    return np.array(
        [
            [0.0, -r[2], r[1]],
            [r[2], 0.0, -r[0]],
            [-r[1], r[0], 0.0],
        ],
        dtype=np.float64,
    )


def build_rigid_condensation_map(
    body_labels: tuple[str, ...],
    body_reference_points: dict[str, NDArray[np.float64]],
    condensed_reference_point: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build the ``(6N, 6)`` rigid-body condensation map ``T`` by label.

    Parameters
    ----------
    body_labels
        The N-body database's ``body_labels`` tuple (block order is the
        DATABASE's order -- the map is built to match it BY LABEL, so a
        caller supplying positions in any other order cannot misalign
        blocks).
    body_reference_points
        Mapping label -> ``(3,)`` reference point of that body (the
        point its 6-DOF block is expressed about; for the M8 fixtures,
        each hull's rotation centre = its CoG at cluster draft), metres,
        inertial frame.
    condensed_reference_point
        ``(3,)`` reference point of the condensed 6-DOF system (the
        composite CoG for the M8 gates), metres, inertial frame.

    Returns
    -------
    NDArray[np.float64]
        ``(6N, 6)`` map ``T`` with body ``k``'s block at rows
        ``[6k : 6k+6]`` in ``body_labels`` order.

    Raises
    ------
    ValueError
        - If ``body_labels`` contains a duplicate (a duplicate makes
          the label -> block map ambiguous; the Q5 contract forbids
          it).
        - If the label set and ``body_reference_points`` keys mismatch
          (missing label or unknown extra label) -- mapping silently by
          position instead is exactly the failure mode the contract
          exists to prevent.
    """
    if len(set(body_labels)) != len(body_labels):
        dups = sorted({lb for lb in body_labels if body_labels.count(lb) > 1})
        raise ValueError(
            f"duplicate body label(s) {dups} in body_labels {body_labels}; "
            "the label->block map is ambiguous (Q5 label contract)"
        )
    labels_set = set(body_labels)
    points_set = set(body_reference_points)
    if labels_set != points_set:
        missing = sorted(labels_set - points_set)
        extra = sorted(points_set - labels_set)
        raise ValueError(
            "body_reference_points does not match body_labels "
            f"(missing: {missing}; unknown: {extra}). The Q5 label "
            "contract forbids positional fallback -- every database "
            "label must be supplied exactly once."
        )

    r_c = np.asarray(condensed_reference_point, dtype=np.float64)
    n = len(body_labels)
    t_map = np.zeros((6 * n, 6), dtype=np.float64)
    for k, label in enumerate(body_labels):
        r_k = np.asarray(body_reference_points[label], dtype=np.float64)
        block = np.zeros((6, 6), dtype=np.float64)
        block[:3, :3] = np.eye(3)
        block[:3, 3:] = -_skew(r_k - r_c)
        block[3:, 3:] = np.eye(3)
        t_map[6 * k : 6 * k + 6, :] = block
    return t_map
