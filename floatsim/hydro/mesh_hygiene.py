"""BEM mesh panel-normal validation (M7.5 PR3).

Standalone utility for validating and correcting panel-normal
orientation on any BEM input mesh. Not wired into any FloatSim
reader; users invoke this in their own pre-BEM-solve scripts
(e.g. the spar-fin study's ``capytaine_run.py``). See
``docs/m7.5-reader-hygiene-plan.md`` §Q2 (final amendment
2026-07-03) for the algorithm history and
``docs/multibody-conventions.md`` Item 5 for the underlying
BEM convention.

Two-tier design (Q2 final disposition, 2026-07-03):

- **Tier 1 — orientation.** :func:`validate_panel_normals` and
  :func:`fix_panel_normals` use **per-panel ray-parity**: for
  each panel, cast a ray from ``centroid + epsilon`` along the
  as-stored normal; count intersections against all other
  panels via Möller-Trumbore triangle-fan tests; odd = inward
  = reversed. Assumption-free: no manifold requirement, no
  connected-component analysis, no parity-class propagation.
  Edge-adjacency machinery is retained only for T-junction
  detection (edge shared by more than two panels → hard raise;
  see tracker ``PANEL-NORMAL-NONCONVEX-BODIES``), open-boundary
  detection (edge adjacent to a single panel → ``UserWarning``
  documenting the ray-parity false-negative mode on open
  shells; see tracker ``BEM-MESH-THIN-SURFACE-ORIENTATION``),
  and degenerate-quad bookkeeping.

- **Tier 2 — physics screen.** :func:`check_hydrostatic_volume`
  computes displaced volume via the divergence-theorem
  triangle-fan sum over all panels as-stored. Reversed panels
  corrupt this number directly, independent of any topology
  assumption. This is the load-bearing protection against the
  ``1.30-vs-21.11 kg`` A_inf failure class that motivated
  M7.5. When ``mass`` is supplied, the report includes the
  buoyancy-vs-weight residual fraction.

Algorithm history (superseded predecessors, kept in the plan
doc for the audit trail):

1. Centroid-outward test (original Q2 lock 2026-06-30):
   inverted on concave features like the plate top annulus.
2. Edge-consistency + signed volume for closed manifolds
   (first amendment 2026-07-02): failed on the multi-shell
   fixture (96 open edges at plate-spar junction).
3. Per-component flood-fill + ray-parity for open components
   (second amendment 2026-07-02): failed on thin plates where
   strip panels are traversal-inconsistent with faces even in
   the correct state.

The Step 0 diagnostic gate (per-panel ray-parity measurement
on the terminal fixtures, 2026-07-03) established that the
prior red-gate target of 192 was the footprint of the
spar-fin study's z-band heuristic blind spot, not a mesh
property. Per-panel ray-parity was adopted as the final
algorithm; the recalibrated red gate documents the study
fixture's 24 objectively-misoriented strip panels in-suite.

Origin: the spar-fin study (``studies/spar-fin-decay``)
surfaced the reversed-normal pathology on 2026-06-27: 192
plate-face panels had inward-facing normals in the OrcaWave
GDF export, producing ``A_inf(heave) = 1.30 kg`` vs the
analytical ~30 kg. The study's ``fix_mesh_normals.py`` used a
z-band + radius + horizontal-face detection criterion
specific to the plate geometry; ray-parity generalises to any
BEM mesh, and detects that the study's z-band heuristic left
24 strip panels misoriented (invisible to Capytaine's added
mass calculation per the tier-2 measurement).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

_PathLike = str | Path

# Vertex-hash tolerance: coordinates are hashed after rounding
# to ``_VERTEX_HASH_TOL_RATIO * bounding_box_diagonal``. Used
# only for T-junction detection (the per-panel ray-parity core
# does not require vertex identification).
_VERTEX_HASH_TOL_RATIO: Final[float] = 1.0e-9


@dataclass(frozen=True)
class GdfMesh:
    """WAMIT GDF quad-panel mesh.

    Attributes
    ----------
    header_lines
        4-line WAMIT GDF header preserved verbatim:
        (title, ``"ULEN GRAV"``, ``"ISX ISY"``, ``"NPAN"``).
    panels
        ``(n_panels, 4, 3)`` float64 array of quad panel
        vertices. Each panel is 4 vertices in the winding
        order as written to the GDF file.
    """

    header_lines: tuple[str, ...]
    panels: NDArray[np.float64]

    def __post_init__(self) -> None:
        if len(self.header_lines) != 4:
            raise ValueError(
                f"GdfMesh header_lines must have exactly 4 entries; "
                f"got {len(self.header_lines)}."
            )
        if self.panels.ndim != 3 or self.panels.shape[1:] != (4, 3):
            raise ValueError(
                f"GdfMesh panels must have shape (n_panels, 4, 3); "
                f"got {self.panels.shape}."
            )

    @property
    def n_panels(self) -> int:
        return int(self.panels.shape[0])


@dataclass(frozen=True)
class OrientationReport:
    """Report from :func:`validate_panel_normals` under
    ``return_report=True``.

    Attributes
    ----------
    n_panels
        Total number of panels in the mesh.
    inward_indices
        ``(n_inward,)`` int64 array of panel indices whose
        as-stored normal points inward per ray-parity.
    indeterminate_indices
        ``(n_indeterminate,)`` int64 array of panel indices
        whose ray-parity result was ambiguous (grazing hits on
        both the original ray and the deterministic-retry
        perturbation, or a degenerate-normal panel that could
        not be cast from). Not counted as inward.
    n_degenerate_panels
        Number of panels whose ``_panel_normal`` returned None
        (both fan-triangle cross products near zero); these
        are a subset of ``indeterminate_indices``.
    n_open_edges
        Number of unique edges (frozenset of two vertex keys)
        adjacent to only one panel. Non-zero when the mesh has
        an open boundary (missing lid, isolated sheet, dangling
        strip). See :func:`validate_panel_normals` for the
        false-negative implication and the emitted warning.
    """

    n_panels: int
    inward_indices: NDArray[np.int64]
    indeterminate_indices: NDArray[np.int64]
    n_degenerate_panels: int
    n_open_edges: int


@dataclass(frozen=True)
class VolumeReport:
    """Report from :func:`check_hydrostatic_volume`.

    Attributes
    ----------
    signed_volume
        Divergence-theorem sum over all panels as-stored,
        divided by 6. This is the **total enclosed volume of
        the closed mesh** (the displacement if the body were
        fully submerged), NOT the displaced volume at a
        waterline. For displaced volume at a free surface,
        clip the mesh at the waterline plane first. Positive
        for a mesh whose panels are collectively
        outward-oriented; sign-flipped or anomalously small
        for a mesh with reversed panels.
    displaced_mass
        ``rho * signed_volume``. Physical if the mesh is
        correctly oriented and represents a solid displacement
        body.
    mass
        Body mass provided to the check, or None.
    residual_fraction
        ``(displaced_mass - mass) / mass`` when mass is
        provided; None otherwise.
    """

    signed_volume: float
    displaced_mass: float
    mass: float | None
    residual_fraction: float | None


# ---------------------------------------------------------------------------
# GDF I/O
# ---------------------------------------------------------------------------


def load_gdf_panels(path: _PathLike) -> GdfMesh:
    """Parse a WAMIT GDF file into a :class:`GdfMesh`.

    Format expected (WAMIT v6/v7 quad-panel GDF):

      - Line 1: title comment (arbitrary string).
      - Line 2: ULEN GRAV (two floats, ignored by this module).
      - Line 3: ISX ISY (two integers, symmetry flags; ignored).
      - Line 4: NPAN (integer, number of quad panels).
      - Lines 5..4+4*NPAN: vertex coordinates (three floats each).
        Each block of 4 consecutive lines is one quad panel in
        vertex-winding order.

    Additional lines after the required vertex block are
    tolerated but ignored (some WAMIT-adjacent tools append
    trailer metadata).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"GDF file not found: {p}")
    text = p.read_text().splitlines()
    if len(text) < 4:
        raise ValueError(
            f"GDF file too short (need >= 4 header lines); got {len(text)}."
        )
    header = tuple(text[:4])
    parts_line4 = text[3].split()
    if not parts_line4:
        raise ValueError(f"GDF header line 4 (NPAN) is empty: {text[3]!r}.")
    n_panels = int(parts_line4[0])
    if n_panels <= 0:
        raise ValueError(f"GDF NPAN must be positive; got {n_panels}.")
    n_expected = 4 * n_panels
    vertex_lines = text[4:]
    if len(vertex_lines) < n_expected:
        raise ValueError(
            f"GDF file inconsistent: header claims {n_panels} panels "
            f"(needs {n_expected} vertex lines); got {len(vertex_lines)}."
        )
    verts_flat: list[list[float]] = []
    for i, ln in enumerate(vertex_lines[:n_expected]):
        parts = ln.split()
        if len(parts) < 3:
            raise ValueError(
                f"GDF vertex line {i + 5} malformed (need >= 3 fields); "
                f"got {parts!r}."
            )
        verts_flat.append([float(parts[0]), float(parts[1]), float(parts[2])])
    panels = np.asarray(verts_flat, dtype=np.float64).reshape(n_panels, 4, 3)
    return GdfMesh(header_lines=header, panels=panels)


def write_gdf_panels(mesh: GdfMesh, path: _PathLike) -> None:
    """Write a :class:`GdfMesh` to disk in WAMIT GDF format.

    Formatting matches the spar-fin study's
    ``fix_mesh_normals.py`` output convention:

      - 4 header lines verbatim from ``mesh.header_lines``.
      - Each vertex line: 3 leading spaces + three
        ``%12.6f`` values separated by 2 spaces.

    File ends with a single trailing newline. This format
    parity is required for the byte-compatible reproduction
    of the study's corrected fixture via the ``panel_mask``
    escape hatch of :func:`fix_panel_normals`.
    """
    p = Path(path)
    lines = list(mesh.header_lines)
    for panel in mesh.panels:
        for v in panel:
            lines.append(f"   {v[0]:12.6f}  {v[1]:12.6f}  {v[2]:12.6f}")
    p.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Vertex hashing + edge adjacency (retained for T-junction detection only)
# ---------------------------------------------------------------------------


def _vertex_hash_tol(panels: NDArray[np.float64]) -> float:
    """Compute the coordinate-hashing tolerance from bounding box."""
    all_verts = panels.reshape(-1, 3)
    bbox_min = all_verts.min(axis=0)
    bbox_max = all_verts.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    if diag <= 0.0:
        raise ValueError(
            "Degenerate mesh: all vertices coincide (bounding-box diagonal "
            "is zero); cannot check T-junction adjacency."
        )
    return _VERTEX_HASH_TOL_RATIO * diag


def _vertex_keys(
    panels: NDArray[np.float64], tol: float
) -> NDArray[np.int64]:
    """Map each vertex to an integer key by rounding coordinates to ``tol``.

    Returns a ``(n_panels, 4)`` array of vertex keys where equal
    integer values denote shared vertices in the mesh.
    """
    all_verts = panels.reshape(-1, 3)
    coord_keys = np.round(all_verts / tol).astype(np.int64)
    _, unique_idx = np.unique(coord_keys, axis=0, return_inverse=True)
    return unique_idx.reshape(panels.shape[0], 4)


def _build_edge_adjacency(
    vertex_keys: NDArray[np.int64],
) -> dict[frozenset[int], list[int]]:
    """Build an unordered edge -> list of panel indices sharing it.

    Retained ONLY for T-junction detection (the sixth-amendment
    ray-parity core does not require edge adjacency for
    orientation). WAMIT's degenerate-quad convention (v3 == v0
    for triangular panels) produces zero-length edges that
    would appear as single-panel entries; those are silently
    skipped by testing ``v_a == v_b`` before adding.

    Coordinate-hash boundary limitation. The adjacency uses
    the coordinate-hash keys from ``_vertex_keys`` at the
    tolerance ``_VERTEX_HASH_TOL_RATIO * bounding_box_diagonal``
    (~1e-9 relative). If the source file wrote inconsistent
    last-digit text for the same geometric vertex, the hash
    can bucket them into different keys and produce a
    spurious T-junction miss or false positive; the current
    tolerance is ~3 orders tighter than typical WAMIT GDF
    text quantisation (~1e-6) so this is unlikely on
    well-formed exports.
    """
    edges: dict[frozenset[int], list[int]] = {}
    n_panels = vertex_keys.shape[0]
    for pi in range(n_panels):
        for local_e in range(4):
            v_a = int(vertex_keys[pi, local_e])
            v_b = int(vertex_keys[pi, (local_e + 1) % 4])
            if v_a == v_b:
                # Degenerate quad edge (WAMIT triangular convention).
                continue
            key = frozenset((v_a, v_b))
            edges.setdefault(key, []).append(pi)
    return edges


# ---------------------------------------------------------------------------
# Ray-parity core (Q2 final amendment: assumption-free orientation)
# ---------------------------------------------------------------------------


def _panel_normal(panel: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Return a unit normal from the as-stored panel winding.

    Tries ``(v1 - v0) × (v3 - v0)`` first (the standard quad normal);
    falls back to ``(v1 - v0) × (v2 - v0)`` when v3 == v0 makes the
    first cross zero (WAMIT triangular-panel convention). Returns
    ``None`` if both attempts collapse to a zero vector — the
    panel is degenerate/tiny and cannot serve as a ray origin.
    """
    v0, v1, v2, v3 = panel
    n = np.cross(v1 - v0, v3 - v0)
    if np.linalg.norm(n) > 1.0e-15:
        return n / np.linalg.norm(n)
    n = np.cross(v1 - v0, v2 - v0)
    if np.linalg.norm(n) > 1.0e-15:
        return n / np.linalg.norm(n)
    return None


def _build_triangle_fan(
    panels: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Precompute the triangle fan for all panels.

    Each quad panel ``(v0, v1, v2, v3)`` contributes up to two
    triangles ``(v0, v1, v2)`` and ``(v0, v2, v3)``. Degenerate
    triangles (near-zero ``edge1 × edge2``) are skipped at build
    time so ray-triangle intersection can be applied without
    per-triangle validity checks.

    The ``1.0e-15`` degeneracy threshold is intentionally absolute
    (not bbox-scaled): a triangle is degenerate here iff its two
    fan edges are bit-identical repeated vertices (the WAMIT
    ``v3 == v0`` triangular-panel encoding); genuine small
    triangles from a fine mesh must be kept in the fan.

    Returns ``(v0_arr, edge1_arr, edge2_arr, panel_idx)`` — the
    per-triangle vertex-0 positions, edge vectors, and origin
    panel indices. Arrays have shape ``(n_tris, 3)`` except
    ``panel_idx`` which is ``(n_tris,)``.
    """
    n_panels = panels.shape[0]
    v0_list: list[NDArray[np.float64]] = []
    edge1_list: list[NDArray[np.float64]] = []
    edge2_list: list[NDArray[np.float64]] = []
    panel_idx_list: list[int] = []
    for pi in range(n_panels):
        p = panels[pi]
        for tri in ((p[0], p[1], p[2]), (p[0], p[2], p[3])):
            a, b, c = tri
            e1 = b - a
            e2 = c - a
            if np.linalg.norm(np.cross(e1, e2)) < 1.0e-15:
                continue
            v0_list.append(a)
            edge1_list.append(e1)
            edge2_list.append(e2)
            panel_idx_list.append(pi)
    return (
        np.asarray(v0_list, dtype=np.float64),
        np.asarray(edge1_list, dtype=np.float64),
        np.asarray(edge2_list, dtype=np.float64),
        np.asarray(panel_idx_list, dtype=np.int64),
    )


def _vec_moller_trumbore(
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    v0: NDArray[np.float64],
    edge1: NDArray[np.float64],
    edge2: NDArray[np.float64],
    tri_panel_idx: NDArray[np.int64],
    exclude_panel: int,
    parallel_tol: float,
    edge_tol: float,
    epsilon: float,
) -> tuple[int, bool]:
    """Vectorized Möller-Trumbore: one ray against all triangles.

    Returns ``(crossing_count, grazes_detected)`` where
    ``grazes_detected`` is True if any hit is near-degenerate:

    - barycentric coordinates within ``edge_tol`` of a triangle
      boundary (edge or vertex hit), OR
    - ray-parameter ``t < epsilon`` (hit within the origin-offset
      distance from the origin — the ray is entering the mesh
      through a face immediately adjacent to the excluded origin
      panel; count is unreliable regardless of the barycentric
      position).

    In either case the crossing count cannot be trusted by the
    caller and a retry with a perturbed direction is required.
    """
    h = np.cross(direction, edge2)
    a = np.einsum("ij,ij->i", edge1, h)
    valid_a = np.abs(a) >= parallel_tol
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(valid_a, 1.0 / a, 0.0)
    s = origin - v0
    u = f * np.einsum("ij,ij->i", s, h)
    valid_u = (u >= -edge_tol) & (u <= 1.0 + edge_tol)
    q = np.cross(s, edge1)
    v = f * np.einsum("j,ij->i", direction, q)
    valid_v = (v >= -edge_tol) & (u + v <= 1.0 + edge_tol)
    t_vals = f * np.einsum("ij,ij->i", edge2, q)
    valid_t = t_vals > parallel_tol
    hit = valid_a & valid_u & valid_v & valid_t
    hit = hit & (tri_panel_idx != exclude_panel)
    count = int(hit.sum())
    if count == 0:
        return 0, False
    hit_u = u[hit]
    hit_v = v[hit]
    hit_t = t_vals[hit]
    grazes = bool(
        np.any(
            (np.abs(hit_u) < edge_tol)
            | (np.abs(hit_v) < edge_tol)
            | (np.abs(1.0 - hit_u - hit_v) < edge_tol)
        )
    )
    grazes = grazes or bool(np.any(hit_t < epsilon))
    return count, grazes


def _perturb_direction(
    direction: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Deterministic 1e-3 rad rotation about the coordinate axis of
    smallest ``|direction|`` component.

    Pure function of the input — no randomness. Used as the
    retry direction after a grazing hit; small enough that the
    parity answer is not perturbed by more than ``edge_tol``
    for well-conditioned rays.
    """
    smallest_axis = int(np.argmin(np.abs(direction)))
    axis = np.zeros(3, dtype=np.float64)
    axis[smallest_axis] = 1.0
    theta = 1.0e-3
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    dot_da = float(np.dot(direction, axis))
    rotated = (
        direction * cos_t
        + np.cross(axis, direction) * sin_t
        + axis * dot_da * (1.0 - cos_t)
    )
    return rotated / np.linalg.norm(rotated)


def _ray_parity_inward_flags(
    panels: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Per-panel ray-parity: return ``(inward_flags, indeterminate_flags)``.

    Algorithm (Q2 final amendment 2026-07-03):

    1. Precompute the triangle fan for the whole mesh.
    2. For each panel with a non-degenerate normal:
         - Cast from ``centroid + epsilon * n`` along ``n``.
         - Vectorized Möller-Trumbore against all triangles;
           exclude triangles belonging to the origin panel.
         - Even count = ray exits body cleanly = outward.
         - Odd count = ray re-enters body = inward.
       If the first cast has any grazing hit (barycentric near
       a triangle boundary, or ray-parameter ``t < epsilon`` —
       a near-origin adjacent-face hit), retry once with a
       deterministic perturbed direction (1e-3 rad rotation
       about the smallest-component axis). If still grazing,
       flag indeterminate. Never guess.
    3. Panels with ``_panel_normal`` returning None (degenerate
       geometry) are also flagged indeterminate.

    Tolerances are scaled by the bounding-box diagonal:

      - epsilon = 1e-6 · diag (ray offset to avoid self-hit)
      - parallel_tol = 1e-12 · diag (ray parallel to triangle)
      - edge_tol = 1e-9 · diag (barycentric graze detection)
    """
    n_panels = int(panels.shape[0])
    all_verts = panels.reshape(-1, 3)
    bbox_diag = float(
        np.linalg.norm(all_verts.max(axis=0) - all_verts.min(axis=0))
    )
    epsilon = 1.0e-6 * bbox_diag
    parallel_tol = 1.0e-12 * bbox_diag
    edge_tol = 1.0e-9 * bbox_diag

    v0_arr, edge1_arr, edge2_arr, tri_panel_idx = _build_triangle_fan(panels)

    inward_flags = np.zeros(n_panels, dtype=bool)
    indeterminate_flags = np.zeros(n_panels, dtype=bool)

    for pi in range(n_panels):
        n = _panel_normal(panels[pi])
        if n is None:
            indeterminate_flags[pi] = True
            continue
        centroid = panels[pi].mean(axis=0)
        origin = centroid + epsilon * n
        count, grazes = _vec_moller_trumbore(
            origin, n, v0_arr, edge1_arr, edge2_arr, tri_panel_idx,
            exclude_panel=pi,
            parallel_tol=parallel_tol, edge_tol=edge_tol,
            epsilon=epsilon,
        )
        if grazes:
            perturbed = _perturb_direction(n)
            perturbed_origin = centroid + epsilon * perturbed
            count, grazes = _vec_moller_trumbore(
                perturbed_origin, perturbed, v0_arr, edge1_arr, edge2_arr,
                tri_panel_idx, exclude_panel=pi,
                parallel_tol=parallel_tol, edge_tol=edge_tol,
                epsilon=epsilon,
            )
            if grazes:
                indeterminate_flags[pi] = True
                continue
        if count % 2 == 1:
            inward_flags[pi] = True
    return inward_flags, indeterminate_flags


# ---------------------------------------------------------------------------
# Public API — Tier 1: orientation validate/fix
# ---------------------------------------------------------------------------


def _check_topology(mesh: GdfMesh) -> int:
    """T-junction hard raise + open-boundary UserWarning.

    Returns ``n_open_edges`` — the number of unique edges
    (frozenset of two vertex keys) adjacent to only one panel.
    T-junctions (edges shared by more than two panels) raise
    ``ValueError`` and are never returned as open-edge counts.

    Emits a single ``UserWarning`` when ``n_open_edges > 0``,
    pinning the ray-parity false-negative mode (reversed panel
    whose ray exits through an opening → reported as outward).
    Warning is attributed to the caller of the public
    ``validate_panel_normals`` / ``fix_panel_normals`` function
    (``stacklevel=3``: helper → public API → user code).
    """
    tol = _vertex_hash_tol(mesh.panels)
    vkeys = _vertex_keys(mesh.panels, tol)
    edges = _build_edge_adjacency(vkeys)
    n_junction = sum(1 for adj in edges.values() if len(adj) > 2)
    if n_junction > 0:
        raise ValueError(
            f"Mesh has {n_junction} edges shared by more than two panels "
            f"(T-junctions or self-intersecting topology). Panel-normal "
            f"validation requires unambiguous per-edge orientation. See "
            f"tracker entry PANEL-NORMAL-NONCONVEX-BODIES in "
            f"docs/phase2-followups.md for T-junction handling scope."
        )
    n_open = sum(1 for adj in edges.values() if len(adj) == 1)
    if n_open > 0:
        warnings.warn(
            f"Mesh has {n_open} open boundary edges: ray-parity "
            f"orientation is unreliable for panels near openings or on "
            f"isolated sheets -- a reversed panel whose ray exits through "
            f"an opening is reported as outward. See "
            f"PANEL-NORMAL-NONCONVEX-BODIES / BEM-MESH-THIN-SURFACE-"
            f"ORIENTATION.",
            UserWarning,
            stacklevel=3,
        )
    return n_open


def validate_panel_normals(
    mesh: GdfMesh, *, return_report: bool = False
) -> OrientationReport | None:
    """Verify all panel normals point outward via per-panel ray-parity.

    Behavior:

    - Runs a T-junction hard check first (edges shared by more
      than two panels → ``ValueError``; see tracker
      ``PANEL-NORMAL-NONCONVEX-BODIES``).
    - Emits a ``UserWarning`` when the mesh has open boundary
      edges (documented false-negative mode: a reversed panel
      whose ray exits through an opening is reported as
      outward — see tracker
      ``BEM-MESH-THIN-SURFACE-ORIENTATION``).
    - Runs per-panel ray-parity via ``_ray_parity_inward_flags``.
    - If ``return_report=True`` (programmatic use): returns an
      :class:`OrientationReport` regardless of whether inward
      panels were found. No raise.
    - If ``return_report=False`` (default): raises
      ``ValueError`` per plan §I2 wording when inward panels
      exist, otherwise returns ``None``. Indeterminate panels
      are appended to the message when non-zero but do not by
      themselves trigger a raise.

    The exact §I2 wording is preserved for backward
    compatibility with existing external documentation.
    """
    n_open_edges = _check_topology(mesh)

    inward_flags, indeterminate_flags = _ray_parity_inward_flags(mesh.panels)
    inward_indices = np.where(inward_flags)[0].astype(np.int64)
    indeterminate_indices = (
        np.where(indeterminate_flags)[0].astype(np.int64)
    )
    n_degenerate = int(
        sum(
            1
            for pi in range(int(mesh.panels.shape[0]))
            if _panel_normal(mesh.panels[pi]) is None
        )
    )

    report = OrientationReport(
        n_panels=int(mesh.panels.shape[0]),
        inward_indices=inward_indices,
        indeterminate_indices=indeterminate_indices,
        n_degenerate_panels=n_degenerate,
        n_open_edges=n_open_edges,
    )

    if return_report:
        return report

    if inward_indices.size == 0:
        return None

    n_inward = int(inward_indices.size)
    n_total = int(mesh.panels.shape[0])
    msg = (
        f"Mesh has {n_inward} panels with inward-facing normals "
        f"(out of {n_total} total; "
        f"{n_inward / n_total * 100:.1f}%).\n"
        f"BEM integration on this mesh will silently produce wrong "
        f"added-mass and radiation-damping values -- see conventions "
        f"doc Item 5 in docs/multibody-conventions.md and the spar-fin "
        f"study post-mortem in studies/spar-fin-decay/STEP-A-FINDING.md "
        f"for the pathology.\n"
        f"\n"
        f"To correct, either:\n"
        f"  (a) Re-export the mesh with outward-facing normals from the "
        f"upstream tool (OrcaWave, WAMIT, etc.), OR\n"
        f"  (b) Apply the study-local correction via "
        f"floatsim.hydro.mesh_hygiene.fix_panel_normals(mesh), then "
        f"pass the corrected mesh to the BEM solver.\n"
        f"\n"
        f"Reversed-panel indices (first 10): "
        f"{inward_indices[:10].tolist()}"
    )
    if indeterminate_indices.size > 0:
        msg += (
            f"\nAdditionally, {int(indeterminate_indices.size)} panels "
            f"were indeterminate under ray-parity (grazing on both "
            f"original and perturbed casts, or degenerate normal). "
            f"Indices (first 10): "
            f"{indeterminate_indices[:10].tolist()}"
        )
    raise ValueError(msg)


def fix_panel_normals(
    mesh: GdfMesh,
    *,
    panel_mask: NDArray[np.bool_] | NDArray[np.int64] | list[int] | None = None,
) -> GdfMesh:
    """Return a new :class:`GdfMesh` with reversed panels flipped.

    Default path (``panel_mask=None``): runs per-panel ray-parity
    to identify inward panels and flips their vertex order via
    ``[::-1]``. Panels flagged indeterminate are left untouched
    with a UserWarning citing their indices.

    Escape hatch (``panel_mask`` supplied): flips exactly the
    panels indicated by the mask. Accepts a boolean array of
    shape ``(n_panels,)`` or an index array/list. No ray-parity
    computation, no orientation judgment, no topology check.
    This is the byte-compatible reproduction path for the
    spar-fin study's ``fix_mesh_normals.py`` output.

    The default (ray-parity) path runs the T-junction hard
    check + open-boundary UserWarning via ``_check_topology``
    once per call; the escape hatch stays silent.

    The input mesh is never mutated; a new mesh is returned
    with a copy of the panels array.
    """
    if panel_mask is not None:
        if isinstance(panel_mask, np.ndarray) and panel_mask.dtype == bool:
            mask = panel_mask.astype(bool)
            if mask.shape != (mesh.panels.shape[0],):
                raise ValueError(
                    f"panel_mask (bool array) must have shape "
                    f"({mesh.panels.shape[0]},); got {mask.shape}."
                )
        else:
            idx = np.asarray(panel_mask, dtype=np.int64)
            mask = np.zeros(mesh.panels.shape[0], dtype=bool)
            mask[idx] = True
        new_panels = mesh.panels.copy()
        if mask.any():
            new_panels[mask] = new_panels[mask, ::-1, :]
        return GdfMesh(header_lines=mesh.header_lines, panels=new_panels)

    _check_topology(mesh)
    inward_flags, indeterminate_flags = _ray_parity_inward_flags(mesh.panels)
    if indeterminate_flags.any():
        indeterminate_idx = np.where(indeterminate_flags)[0]
        warnings.warn(
            f"fix_panel_normals: {int(indeterminate_flags.sum())} panels "
            f"were indeterminate under ray-parity (grazing on both original "
            f"and perturbed casts, or degenerate normal); left untouched. "
            f"Indices (first 10): {indeterminate_idx[:10].tolist()}",
            UserWarning,
            stacklevel=2,
        )
    new_panels = mesh.panels.copy()
    if inward_flags.any():
        new_panels[inward_flags] = new_panels[inward_flags, ::-1, :]
    return GdfMesh(header_lines=mesh.header_lines, panels=new_panels)


# ---------------------------------------------------------------------------
# Public API — Tier 2: hydrostatic-volume physics screen
# ---------------------------------------------------------------------------


def _panel_signed_volume_contribution(
    panel: NDArray[np.float64],
) -> float:
    """Signed tetrahedron-sum contribution of one quad panel.

    Split the quad into two triangles ``(v0, v1, v2)`` and
    ``(v0, v2, v3)``. Each triangle contributes
    ``dot(v0, cross(v1, v2))`` to the total signed volume
    before the final division by 6.

    Degenerate triangles (from ``v3 == v0`` etc.) contribute
    exactly zero via the triple-product identity
    ``a · (b × a) = 0``, so no special-case handling is
    needed here.
    """
    v0, v1, v2, v3 = panel
    contrib_t1 = float(np.dot(v0, np.cross(v1, v2)))
    contrib_t2 = float(np.dot(v0, np.cross(v2, v3)))
    return contrib_t1 + contrib_t2


def check_hydrostatic_volume(
    mesh: GdfMesh,
    rho: float,
    mass: float | None = None,
) -> VolumeReport:
    """Compute displaced volume via divergence theorem + optional
    buoyancy-vs-weight residual.

    No parity signing, no component logic, no topology
    assumption. Panels contribute according to their as-stored
    winding: reversed panels contribute with the wrong sign and
    corrupt the total. That's exactly the property the check
    exploits — the ``1.30-vs-21.11 kg`` A_inf failure class
    that motivated M7.5 leaves an obvious signature in the
    displaced volume.

    Two-tier blind-spot division. Tier 2 (this check) is strong
    against reversed faces on voluminous features (a wholly-inverted
    body flips the sign of V; a reversed hemispherical cap moves V
    by O(cap volume / body volume)). Tier 2 is weak against
    thin-plate reversals: on the spar-fin study fixture, flipping
    all 192 horizontal plate faces moves V by only 0.82%
    (ORIGINAL-vs-CORRECTED); auto-fixing the 24 strip panels
    moves V a further 1.92% (full_fix-vs-CORRECTED). Both are
    well below any residual threshold that would catch a
    16x A_inf error. Tier 1 (:func:`validate_panel_normals` +
    per-panel ray-parity) covers exactly this thin-plate gap by
    judging each panel individually against the mesh topology,
    independent of its volumetric weight. Callers should run
    tier 1 first; tier 2 is the physics-first backstop against
    a wholly-inverted mesh that would otherwise slip past a
    partial-scope tier-1 fix (as the study's 192-panel
    z-band-heuristic fix would have, if not for ray-parity).

    Parameters
    ----------
    mesh
        The mesh to screen. Panels are used as-stored.
    rho
        Water density (kg/m^3). Typical seawater value 1025.
    mass
        Optional body mass (kg). When provided, the report
        includes ``residual_fraction = (rho*V - m) / m`` — the
        buoyancy-vs-weight mismatch as a fraction of body mass.

    Returns
    -------
    VolumeReport
        Pure report; no raise. Callers decide the threshold at
        which a residual is considered acceptable.
    """
    total = 0.0
    for panel in mesh.panels:
        total += _panel_signed_volume_contribution(panel)
    signed_volume = total / 6.0
    displaced_mass = rho * signed_volume
    residual_fraction: float | None = None
    if mass is not None:
        residual_fraction = (displaced_mass - mass) / mass
    return VolumeReport(
        signed_volume=signed_volume,
        displaced_mass=displaced_mass,
        mass=mass,
        residual_fraction=residual_fraction,
    )
