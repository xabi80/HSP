"""Per-panel pressure-field export (module 3 of 3).

Schema: FloatFEA `docs/load-interchange-v1.md` v1.2, group ``/panels/``.

This module **exports the field**. Mapping it onto an FE mesh is FloatFEA gate
**G4.4** in F4 and is deliberately not done here -- keeping that boundary clean
is what makes G1.6 (does the exported field reproduce what FloatSim applied?) and
G4.4 (does transferring it onto a non-matching mesh lose anything?) independent
failure modes rather than one blurred one.

Four decisions taken before this module was written
---------------------------------------------------
**1. Froude-Krylov and diffraction are exported SEPARATELY** (schema §7.1). The
v1.0 justification for merging them was factually wrong: Capytaine carries them
as distinct quantities, and they distribute over the hull by different fields --
FK is the incident pressure on the wetted surface, diffraction the scattered
field. The split also future-proofs the decision most likely to be revisited,
since FK is the term that would move to the instantaneous wetted surface if
G4.6's mean-surface constraint is ever reopened.

**2. Radiation sums over ALL 72 radiating DOF, not the local 6.** The database is
a genuine 12-body coupled solve -- cross-body added mass reaches 2.7% of
own-body at the rotational mode -- so the field on one hull depends on every
hull's motion. Note 72, not 102: only the twelve buoys carry hydrodynamics, the
hubs and platform being structural.

**3. Complex coefficients are stored, not time samples.** In steady periodic
motion the reconstruction is *exact* at the fundamental, so coefficients lose
nothing while collapsing the volume by the window length -- 1.73 GB of
time-domain field becomes ~232 MB of coefficients. Sizing arithmetic in schema
§5.0.2.

**4. Every field carries its validity window.** The BEM computes the field for a
hull at its **reference position**, and linear theory assumes small motion about
it; the platform translates ~2.3 spar diameters over a run. So the reference pose
and the actual pose at every exported instant are both recorded, and the
validator refuses snapshots beyond a stated bound. Third application of the
pattern in `docs/instrumentation.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

# Only the twelve buoys carry hydrodynamics; hubs and the platform are
# structural. An implementation summing over the 102-DOF global vector would
# index past the BEM data or pick up DOF with no radiation field.
N_HYDRO_DOF: Final[int] = 72

PANEL_SOURCES: Final[tuple[str, ...]] = ("froude_krylov", "diffraction", "radiation")


@dataclass(frozen=True)
class PanelGeometry:
    """Hull discretisation the pressure field is defined on.

    Exported alongside the field because a pressure without the area and normal
    it acts through cannot be integrated, and FloatFEA's mesh is not this one.
    """

    centroid: NDArray[np.float64]   # (P, 3), body frame
    area: NDArray[np.float64]       # (P,)
    normal: NDArray[np.float64]     # (P, 3), body frame, outward

    def __post_init__(self) -> None:
        p = self.centroid.shape[0]
        if self.area.shape != (p,) or self.normal.shape != (p, 3):
            raise ValueError(
                f"panel arrays disagree: centroid {self.centroid.shape}, "
                f"area {self.area.shape}, normal {self.normal.shape}"
            )

    @property
    def n_panels(self) -> int:
        return int(self.centroid.shape[0])


def extract_froude_krylov(problem: Any, geometry: PanelGeometry) -> NDArray[np.complex128]:
    """Per-panel **incident-wave** pressure -- the Froude-Krylov field.

    Uses Capytaine's ``airy_waves_pressure``, which is the incident potential
    evaluated on the hull. It needs no solved result: the incident wave does not
    depend on the body, which is precisely why FK is separable from diffraction
    at source.
    """
    from capytaine.bem.airy_waves import airy_waves_pressure

    return np.asarray(
        airy_waves_pressure(geometry.centroid, problem), dtype=np.complex128
    )


def extract_scattered(solver: Any, result: Any, geometry: PanelGeometry) -> NDArray[np.complex128]:
    """Per-panel pressure from a solved diffraction or radiation problem.

    Requires the problem to have been solved with ``keep_details=True`` -- the
    source distribution is what ``compute_pressure`` reconstructs from, and it is
    discarded otherwise. A solve without it raises rather than silently returning
    an incident-only field, which would look plausible and be wrong.
    """
    return np.asarray(
        solver.compute_pressure(geometry.centroid, result), dtype=np.complex128
    )


def integrate_panel_pressure(
    pressure: NDArray[np.complexfloating],
    geometry: PanelGeometry,
) -> NDArray[np.complex128]:
    """Integrate a panel pressure field to its 3-component force resultant.

    ``F = -sum_p  p_p * n_p * A_p`` -- the minus sign because the panel normal
    points **outward into the fluid**, so a positive pressure pushes the hull
    inward (`multibody-conventions.md` Item 5).

    This is the operation both G1.6 guards rest on, and the two guards must not
    be conflated:

    **(a) The G1.6 gate** compares the **sum** of the FK and diffraction fields
    against FloatSim's *combined applied excitation*. It must NOT be satisfiable
    by checking the halves separately -- FloatSim never applied the halves to
    anything, only their sum.

    **(b) A panel-extraction diagnostic** compares each field against
    **Capytaine's own** resultants. Useful, and it is what catches an error in
    the extraction itself -- but it tests agreement with Capytaine, not agreement
    with the simulation. An extraction matching Capytaine perfectly could still
    disagree with what the simulator applied, which is the failure (a) exists to
    catch.
    """
    p = np.asarray(pressure, dtype=np.complex128)
    if p.shape[-1] != geometry.n_panels:
        raise ValueError(
            f"pressure has {p.shape[-1]} panels but the geometry has "
            f"{geometry.n_panels}; they must come from the same mesh."
        )
    weighted = geometry.normal * geometry.area[:, None]
    return -np.tensordot(p, weighted, axes=([-1], [0]))


def displacement_from_reference(
    body_pose: NDArray[np.floating],
    reference_pose: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Translational distance of each exported instant from the BEM reference.

    The quantity the validity bound is applied to. The BEM field is computed for
    a hull at ``reference_pose``; linear theory assumes small motion about it,
    and the platform drifts ~2.3 spar diameters over a run, so a screened
    snapshot far from the reference is being loaded with a field that does not
    describe where the hull is.
    """
    bp = np.asarray(body_pose, dtype=np.float64)
    rp = np.asarray(reference_pose, dtype=np.float64)
    if bp.ndim != 2 or bp.shape[1] < 3:
        raise ValueError(f"body_pose must be (K, >=3); got {bp.shape}")
    return np.linalg.norm(bp[:, 0:3] - rp[0:3], axis=1)
