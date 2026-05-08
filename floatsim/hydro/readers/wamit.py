"""WAMIT plain-text reader — ARCHITECTURE.md §8 M5.

Parses the WAMIT v6/v7 ASCII outputs into a :class:`HydroDatabase`:

    ``.1``   added mass ``A(omega)`` and radiation damping ``B(omega)``,
             plus ``A_inf`` from the ``PER == -1`` row and ``A_zero`` from
             the ``PER == 0`` row (latter discarded).
    ``.3``   wave excitation force coefficients ``X(omega, beta)`` (complex).
    ``.hst`` hydrostatic stiffness matrix ``C`` (6×6).
    ``.4``   motion RAOs — parsed by :func:`read_motion_rao` for cross-check
             use; not stored in the returned :class:`HydroDatabase`.

Conventions
-----------
- **Dimensional output required.** WAMIT writes nondimensional output by
  default; users must set the appropriate dimensional flags (typically
  ``IPLTDAT=15`` with the run-control ``IPER`` flag set so ``PER`` is in
  seconds and ``IFORCE`` so excitation is in N or N·m). See HydroDyn user
  manual §6 for the canonical recipe. The reader emits a heuristic warning
  if the magnitude of the values looks nondimensional (max |A| < 10).
- DOF order is WAMIT's: 1=surge, 2=sway, 3=heave, 4=roll, 5=pitch, 6=yaw.
  This matches :data:`floatsim.hydro.database.DOF_ORDER`.
- Phase convention: WAMIT default ("leads"). The complex coefficient is
  stored such that ``F_exc(t) = Re[X * A_wave * exp(i * omega * t)]``,
  matching the OrcaFlex VesselType YAML reader. Phase angles in ``.3`` and
  ``.4`` are degrees.
- IMPORTANT — phase **reporting** discipline for downstream consumers:
  under the +i convention, ``arg(x_hat)`` is the LEAD (negative-of-lag).
  ``cos+sin`` lstsq fits (e.g., ``tests/support/rao_extraction.py``)
  return ``atan2(B, A)`` which is the LAG. To compare an impedance-
  computed RAO phase against a time-domain lstsq fit, negate one of
  them. Surfaced at M6 PR4 Pre-3 — see
  ``docs/diagnostics/m6-pr4-pre3-phase-residual-diagnosis.md``.
- Periods are seconds, ``omega = 2*pi/PER``.

  * ``PER == -1``  →  infinite frequency. The corresponding row carries
    ``A_inf``; no damping column is expected (some WAMIT variants write a
    zero column, which is also tolerated).
  * ``PER ==  0``  →  zero frequency. Discarded — FloatSim does not
    consume the zero-frequency added mass.

Hydrostatic stiffness convention
--------------------------------
The WAMIT ``.hst`` file contains only the **buoyancy / waterplane**
contribution to hydrostatic restoring — it is *not* the full
restoring matrix. WAMIT does not know the body's mass distribution, so
gravity terms (``m * g * z_G`` for roll/pitch) are absent. Downstream
:class:`floatsim.bodies.Body` assembly is expected to add the gravity
contribution from the body's mass and CoG. By contrast, the OrcaFlex
VesselType YAML reader returns the *full* restoring (OrcaFlex bundles
mass into the VesselType). Cross-checking the two in M5 PR3 will
require the test to apply this offset explicitly.

Scope
-----
M5 PR1: single body, single draught. Multi-body output blocks, mean-drift
``.2``, QTFs (``.8``, ``.9``, ``.12``), and pressure ``.pat`` files are out
of scope.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from floatsim.hydro.database import HydroDatabase

# Path or stem types accepted at the public boundary.
_PathLike = str | Path

# Period sentinels in WAMIT .1.
_PER_INFINITE_FREQ: Final[float] = -1.0
_PER_ZERO_FREQ: Final[float] = 0.0
# Periods within this absolute tolerance of the sentinels are treated as such.
_PER_SENTINEL_ATOL: Final[float] = 1.0e-9

# Symmetrization tolerance: WAMIT writes both (i,j) and (j,i) entries for
# off-diagonal coefficients. They must agree to numerical precision; if not,
# we warn and use the average.
_SYM_RTOL: Final[float] = 1.0e-6

# M6 PR4 Pre-3 finding (locked 2026-05-06): the WAMIT public-format
# convention is **non-dimensional output** by default (WAMIT v7
# manual §4.2 / HydroDyn user guide §6 reference the same scheme).
# Readers must apply ``rho * ULEN^k`` / ``rho * g * ULEN^k``
# rescaling per DOF to recover SI units. The pre-fix reader returned
# raw non-dim values; the bug was latent through M5-M6 PR3 because
# free-decay periods are dominated by dimensional ``M`` and ``C``.
# See conventions doc Items 22, 23 and
# ``docs/post-mortems/m6-pr4-wamit-dim-bug.md``.
#
# WAMIT default sea-water properties (overridable per call):
_DEFAULT_RHO_KG_M3: Final[float] = 1025.0  # OpenFAST WtrDens
_DEFAULT_G_M_S2: Final[float] = 9.80665  # OpenFAST Gravity
_DEFAULT_ULEN_M: Final[float] = 1.0  # OC4 marin_semi (HydroDyn WAMITULEN)

# Dimensionality detection: when ``assume_dimensional=True`` is passed
# but the values look non-dim, fire a warning. We use a per-DOF
# expectation: dimensional rotational A on a typical floating platform
# is O(1e9) kg·m²; if it's O(1e7) the file is almost certainly the
# non-dim version with rho·ULEN^5 missing. The previous threshold
# (max|A| < 10) failed on marin_semi where surge A = 8527 non-dim.
_NONDIM_PITCH_THRESHOLD_KG_M2: Final[float] = 1.0e8
_DIMENSIONAL_THRESHOLD: Final[float] = 10.0  # kept as a backstop sanity

# Re/Im vs Mod/Pha consistency tolerance in .3 / .4. Either representation
# can be used — they should round-trip to within this tolerance.
_COMPLEX_REPR_RTOL: Final[float] = 1.0e-3
_COMPLEX_REPR_ATOL: Final[float] = 1.0e-3


# ---------------------------------------------------------------------------
# Dimensional rescaling per WAMIT v7 manual §4.2 (and HydroDyn UG §6)
# ---------------------------------------------------------------------------
#
# WAMIT's default output is non-dimensional, with the dimensionalisation
# factor depending on the DOF coupling:
#
#   .1 added mass / damping ``A_ij``, ``B_ij``:
#       k_ij = 3 + (1 if i in {4,5,6}) + (1 if j in {4,5,6})
#       A_ij_dim = rho * ULEN^k_ij * A_ij_nondim
#       B_ij_dim = rho * omega * ULEN^k_ij * B_ij_nondim
#
#   .3 excitation force / moment ``X_i`` (per unit wave amplitude):
#       k_i = 2 + (1 if i in {4,5,6})
#       X_i_dim = rho * g * ULEN^k_i * X_i_nondim
#
#   .hst hydrostatic stiffness ``C_ij``:
#       k_ij = 2 + (1 if i in {4,5,6}) + (1 if j in {4,5,6})
#       C_ij_dim = rho * g * ULEN^k_ij * C_ij_nondim
#
# Translation modes are 1, 2, 3 (surge, sway, heave); rotation modes are
# 4, 5, 6 (roll, pitch, yaw). Internally we use 0-indexed: i in {0,1,2}
# is translation; i in {3,4,5} is rotation.


def _is_rotation_dof(i_zero: int) -> bool:
    """Internal 0-indexed DOF: True for roll/pitch/yaw."""
    return 3 <= i_zero <= 5


def _added_mass_factor(i: int, j: int, *, rho: float, ulen: float) -> float:
    """rho * ULEN^k for added-mass index pair (i, j) (0-indexed)."""
    k = 3 + (1 if _is_rotation_dof(i) else 0) + (1 if _is_rotation_dof(j) else 0)
    return rho * ulen**k


def _damping_factor(i: int, j: int, *, rho: float, ulen: float, omega: float) -> float:
    """rho * omega * ULEN^k for damping index pair (i, j) (0-indexed)."""
    return _added_mass_factor(i, j, rho=rho, ulen=ulen) * omega


def _excitation_factor(i: int, *, rho: float, g: float, ulen: float) -> float:
    """rho * g * ULEN^k for excitation force on mode i (0-indexed)."""
    k = 2 + (1 if _is_rotation_dof(i) else 0)
    return rho * g * ulen**k


def _hydrostatic_factor(i: int, j: int, *, rho: float, g: float, ulen: float) -> float:
    """rho * g * ULEN^k for hydrostatic stiffness pair (i, j) (0-indexed)."""
    k = 2 + (1 if _is_rotation_dof(i) else 0) + (1 if _is_rotation_dof(j) else 0)
    return rho * g * ulen**k


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


def read_wamit(
    stem: _PathLike,
    *,
    reference_point: tuple[float, float, float] | None = None,
    assume_dimensional: bool = False,
    rho_water_kg_m3: float = _DEFAULT_RHO_KG_M3,
    g_m_s2: float = _DEFAULT_G_M_S2,
    ulen_m: float = _DEFAULT_ULEN_M,
) -> HydroDatabase:
    """Read a WAMIT case (``.1`` + ``.3`` + ``.hst``) into a HydroDatabase.

    Parameters
    ----------
    stem
        Path stem of the WAMIT case — i.e. the path without an extension.
        For a case named ``marin_semi``, pass ``.../marin_semi`` and the
        reader appends ``.1``, ``.3``, ``.hst`` (and optionally ``.4``)
        as needed.
    reference_point
        Inertial-frame point about which the BEM coefficients are defined.
        WAMIT does not encode this in the text outputs — it must be supplied
        out-of-band (typically from the WAMIT input ``XBODY`` block). Defaults
        to the origin ``(0, 0, 0)`` if ``None``.

    Returns
    -------
    HydroDatabase
        Validated single-body BEM database. ``A``, ``B``, ``A_inf``, ``C``
        are real 6×6 (×n_w) arrays; ``RAO`` is the complex first-order
        wave excitation force per unit wave amplitude (shape
        ``(6, n_w, n_h)``).

    Notes
    -----
    The ``.4`` motion RAO file, if present, is **not** loaded into the
    returned database — it is redundant with the assembled
    ``(M, A_inf, B, C, F_exc)`` system and is reserved for cross-check
    via :func:`read_motion_rao`.
    """
    s = Path(stem)
    if s.suffix in {".1", ".3", ".hst", ".4"}:
        s = s.with_suffix("")

    rescale_kwargs = dict(
        assume_dimensional=assume_dimensional,
        rho_water_kg_m3=rho_water_kg_m3,
        ulen_m=ulen_m,
    )
    excit_kwargs = {**rescale_kwargs, "g_m_s2": g_m_s2}
    omega, A, B, A_inf = read_added_mass_and_damping(s.with_suffix(".1"), **rescale_kwargs)
    heading_deg, F_exc = read_excitation_force(s.with_suffix(".3"), omega=omega, **excit_kwargs)
    C = read_hydrostatic_stiffness(s.with_suffix(".hst"), **excit_kwargs)

    ref = (
        np.zeros(3, dtype=np.float64)
        if reference_point is None
        else np.asarray(reference_point, dtype=np.float64)
    )

    metadata: dict[str, str] = {
        "source": "wamit",
        "stem": s.name,
    }

    return HydroDatabase(
        omega=omega,
        heading_deg=heading_deg,
        A=A,
        B=B,
        A_inf=A_inf,
        C=C,
        RAO=F_exc,
        reference_point=ref,
        # WAMIT .hst stores buoyancy/waterplane only; gravity m*g*z_G must
        # be added downstream via assemble_cummins_lhs (see HydroDatabase
        # docstring and floatsim/hydro/hydrostatics.py).
        C_source="buoyancy_only",
        metadata=metadata,
    )


def read_added_mass_and_damping(
    path: _PathLike,
    *,
    assume_dimensional: bool = False,
    rho_water_kg_m3: float = _DEFAULT_RHO_KG_M3,
    ulen_m: float = _DEFAULT_ULEN_M,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Parse a WAMIT ``.1`` file and return SI-dimensional coefficients.

    Parameters
    ----------
    path
        Path to the WAMIT ``.1`` file.
    assume_dimensional
        If ``True``, the file is assumed to already carry SI units (kg,
        kg·m, kg·m²) and the reader returns values verbatim. The
        dimensionality heuristic still fires (informationally) if the
        magnitudes look non-dim. Default ``False``: per the WAMIT v7
        manual §4.2 convention, the file is non-dimensional and the
        reader applies ``rho * ULEN^k`` rescaling per DOF (k = 3 / 4 / 5
        for trans-trans / trans-rot / rot-rot couplings). For OC4
        marin_semi: ``rho = 1025 kg/m^3``, ``ULEN = 1.0 m``, ``g = 9.81 m/s^2``.
    rho_water_kg_m3
        Sea-water density used in the dimensionalisation (ignored if
        ``assume_dimensional=True``). Defaults to OpenFAST's WtrDens.
    ulen_m
        WAMIT characteristic body length (the ``ULEN`` parameter from
        the WAMIT input or HydroDyn ``WAMITULEN``). Defaults to 1.0,
        the OC4 marin_semi value.

    Returns
    -------
    omega : (n_w,) float64
        Angular frequencies (rad/s), strictly increasing.
    A : (6, 6, n_w) float64
        Added mass at each finite frequency in SI units (kg, kg·m, kg·m²).
        Symmetric per slice.
    B : (6, 6, n_w) float64
        Radiation damping at each finite frequency in SI units
        (kg/s, kg·m/s, kg·m²/s). Symmetric per slice.
    A_inf : (6, 6) float64
        Infinite-frequency added mass (from the ``PER == -1`` row),
        same units as ``A``.

    Raises
    ------
    ValueError
        If no ``PER == -1`` row is present, if (i, j) duplicates disagree
        beyond :data:`_SYM_RTOL`, or if any matrix slice is not symmetric.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"WAMIT .1 file not found: {p}")

    # Bucket entries by period: a dict period -> (A_dict, B_dict) where each
    # inner dict maps (i_zero, j_zero) -> list of values.
    A_per: dict[float, dict[tuple[int, int], list[float]]] = {}
    B_per: dict[float, dict[tuple[int, int], list[float]]] = {}

    A_inf_dict: dict[tuple[int, int], list[float]] = {}

    for tokens in _iter_data_rows(p):
        if len(tokens) not in (4, 5):
            raise ValueError(
                f"WAMIT .1 row in {p.name} must have 4 or 5 fields "
                f"(PER I J A [B]); got {len(tokens)}: {tokens!r}"
            )
        per = float(tokens[0])
        i = int(tokens[1])
        j = int(tokens[2])
        a_val = float(tokens[3])
        b_val = float(tokens[4]) if len(tokens) == 5 else 0.0
        _check_dof_index(i, p.name)
        _check_dof_index(j, p.name)
        i0, j0 = i - 1, j - 1

        if abs(per - _PER_INFINITE_FREQ) <= _PER_SENTINEL_ATOL:
            A_inf_dict.setdefault((i0, j0), []).append(a_val)
            continue
        if abs(per - _PER_ZERO_FREQ) <= _PER_SENTINEL_ATOL:
            # Zero-frequency added mass is not consumed by FloatSim.
            continue

        A_per.setdefault(per, {}).setdefault((i0, j0), []).append(a_val)
        B_per.setdefault(per, {}).setdefault((i0, j0), []).append(b_val)

    if not A_inf_dict:
        raise ValueError(
            f"WAMIT .1 file {p.name} has no PER == -1 row — A_inf cannot be "
            f"populated. Re-run WAMIT with the infinite-frequency case enabled."
        )

    # --- Sort finite periods into ascending omega ---------------------------
    finite_periods = sorted(A_per.keys())
    omega = np.asarray([2.0 * np.pi / per for per in finite_periods], dtype=np.float64)
    omega = omega[::-1]  # ascending omega <=> descending period
    finite_periods = list(reversed(finite_periods))

    n_w = len(finite_periods)
    A = np.zeros((6, 6, n_w), dtype=np.float64)
    B = np.zeros((6, 6, n_w), dtype=np.float64)
    for k, per in enumerate(finite_periods):
        A[:, :, k] = _resolve_6x6_from_dict(A_per[per], label=f"A(PER={per})")
        B[:, :, k] = _resolve_6x6_from_dict(B_per[per], label=f"B(PER={per})")

    A_inf = _resolve_6x6_from_dict(A_inf_dict, label="A_inf")

    if not assume_dimensional:
        # Apply rho * ULEN^k rescaling (WAMIT v7 manual §4.2).
        for i in range(6):
            for j in range(6):
                fac = _added_mass_factor(i, j, rho=rho_water_kg_m3, ulen=ulen_m)
                A[i, j, :] *= fac
                A_inf[i, j] *= fac
                # Damping factor includes ω, applied per-frequency.
                for k, w in enumerate(omega):
                    B[i, j, k] *= _damping_factor(
                        i, j, rho=rho_water_kg_m3, ulen=ulen_m, omega=float(w)
                    )

    _maybe_warn_nondimensional(A_inf, A, label=p.name, assume_dimensional=assume_dimensional)
    return omega, A, B, A_inf


def read_excitation_force(
    path: _PathLike,
    *,
    omega: NDArray[np.float64],
    assume_dimensional: bool = False,
    rho_water_kg_m3: float = _DEFAULT_RHO_KG_M3,
    g_m_s2: float = _DEFAULT_G_M_S2,
    ulen_m: float = _DEFAULT_ULEN_M,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Parse a WAMIT ``.3`` excitation-force file (SI-dimensional output).

    Parameters
    ----------
    path
        Path to the ``.3`` file.
    omega
        Frequency grid this excitation is to be aligned with — usually the
        result of :func:`read_added_mass_and_damping`. Each row in the file
        must match a frequency in this grid to better than 1e-9 rad/s.
    assume_dimensional
        If ``True``, the file is assumed to already carry SI units and the
        reader returns values verbatim. Default ``False``: apply
        ``rho * g * ULEN^k`` rescaling per WAMIT v7 manual §4.2 (k = 2 for
        translational modes 1-3, N/m wave amplitude; k = 3 for rotational
        modes 4-6, N·m/m wave amplitude). For OC4 marin_semi: ``rho = 1025``,
        ``g = 9.80665``, ``ULEN = 1.0``.
    rho_water_kg_m3, g_m_s2, ulen_m
        Sea-water density, gravity, and characteristic body length for
        the dimensionalisation. Defaults match OpenFAST WtrDens / Gravity
        and the OC4 marin_semi WAMITULEN.

    Returns
    -------
    heading_deg : (n_h,) float64
        Wave headings (degrees), sorted ascending.
    F_exc : (6, n_w, n_h) complex128
        Complex first-order excitation force per unit wave amplitude in
        SI units (N/m for translational modes, N·m/m for rotational). Phase
        convention: ``F_exc(t) = Re[F_exc * A_wave * exp(i * omega * t)]``.
    """
    heading_deg, F_exc = _read_complex_per_dof(
        path,
        omega=omega,
        label="excitation",
    )
    if not assume_dimensional:
        for i in range(6):
            fac = _excitation_factor(i, rho=rho_water_kg_m3, g=g_m_s2, ulen=ulen_m)
            F_exc[i, :, :] *= fac
    return heading_deg, F_exc


def read_motion_rao(
    path: _PathLike,
    *,
    omega: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Parse a WAMIT ``.4`` motion-RAO file.

    Same row layout as ``.3``; provided as a separate parser so a test can
    cross-check assembled RAOs (from ``M, A_inf, B, C, F_exc``) against
    WAMIT's own RAOs without polluting :class:`HydroDatabase`.

    Parameters
    ----------
    path
        Path to the ``.4`` file.
    omega
        Frequency grid (matched as in :func:`read_excitation_force`).

    Returns
    -------
    heading_deg : (n_h,) float64
    rao : (6, n_w, n_h) complex128
    """
    return _read_complex_per_dof(
        path,
        omega=omega,
        label="motion_rao",
    )


def read_hydrostatic_stiffness(
    path: _PathLike,
    *,
    assume_dimensional: bool = False,
    rho_water_kg_m3: float = _DEFAULT_RHO_KG_M3,
    g_m_s2: float = _DEFAULT_G_M_S2,
    ulen_m: float = _DEFAULT_ULEN_M,
) -> NDArray[np.float64]:
    """Parse a WAMIT ``.hst`` hydrostatic-stiffness file (SI-dimensional output).

    Parameters
    ----------
    path
        Path to the ``.hst`` file.
    assume_dimensional
        If ``True``, return values verbatim. Default ``False``: apply
        ``rho * g * ULEN^k`` rescaling per WAMIT v7 manual §4.2
        (k = 2 / 3 / 4 for trans-trans / trans-rot / rot-rot pairs).

    Returns
    -------
    C : (6, 6) float64
        Hydrostatic restoring matrix in SI units (N/m for trans-trans,
        N·m/m for trans-rot, N·m/rad for rot-rot). Symmetric.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"WAMIT .hst file not found: {p}")

    by_pair: dict[tuple[int, int], list[float]] = {}
    for tokens in _iter_data_rows(p):
        if len(tokens) != 3:
            raise ValueError(
                f"WAMIT .hst row in {p.name} must have 3 fields (I J C); "
                f"got {len(tokens)}: {tokens!r}"
            )
        i = int(tokens[0])
        j = int(tokens[1])
        c_val = float(tokens[2])
        _check_dof_index(i, p.name)
        _check_dof_index(j, p.name)
        by_pair.setdefault((i - 1, j - 1), []).append(c_val)

    C = _resolve_6x6_from_dict(by_pair, label="C")

    if not assume_dimensional:
        for i in range(6):
            for j in range(6):
                C[i, j] *= _hydrostatic_factor(i, j, rho=rho_water_kg_m3, g=g_m_s2, ulen=ulen_m)
    return C


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _iter_data_rows(path: Path) -> Iterator[list[str]]:
    """Yield tokenized non-comment, non-blank lines from a WAMIT text file."""
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            # Skip a leading header line that does not parse as a number in
            # column 1 (some WAMIT post-processors prepend a column-name row).
            try:
                float(tokens[0])
            except ValueError:
                continue
            yield tokens


def _check_dof_index(i: int, file_label: str) -> None:
    if not 1 <= i <= 6:
        raise ValueError(f"WAMIT mode index must be 1..6 in {file_label}; got {i}")


def _resolve_6x6_from_dict(
    by_pair: dict[tuple[int, int], list[float]],
    *,
    label: str,
) -> NDArray[np.float64]:
    """Build a symmetric 6×6 matrix from a dict of (i,j)-keyed value lists.

    The output matrix is symmetric by construction:

    - If both ``M[i, j]`` and ``M[j, i]`` are populated in the file, the
      stored value is their arithmetic mean. This is the *correct* handling
      for WAMIT output: the solver writes both halves of a physically
      symmetric coupling, and panel-method noise produces small asymmetries
      (typically far below the diagonal scale). For example, marin_semi at
      ``T=12.57 s`` gives ``M[4,6]≈92`` and ``M[6,4]≈48`` (port-starboard
      symmetry of the OC4 semi makes both physically zero); the diagonal
      scale is ``~8e6``, so this is solver noise, not a file bug.
    - If only one of ``M[i, j]`` / ``M[j, i]`` is populated, the missing
      transpose is filled by mirroring.
    - Duplicate rows for the same ``(i, j)`` must agree to within
      ``rtol=_SYM_RTOL`` and a scale-relative absolute tolerance. Disagreement
      at this level indicates a corrupt file, not solver noise.

    Symmetry of the *averaged* matrix is enforced exactly; the validation
    in :class:`HydroDatabase.__post_init__` then sees a perfectly symmetric
    input and passes its own ``rtol=1e-6`` check trivially.
    """
    M = np.zeros((6, 6), dtype=np.float64)
    populated = np.zeros((6, 6), dtype=bool)
    duplicate_lists: dict[tuple[int, int], list[float]] = {}

    for (i, j), values in by_pair.items():
        if not values:
            continue
        M[i, j] = values[0]
        populated[i, j] = True
        if len(values) > 1:
            duplicate_lists[(i, j)] = values

    scale = float(np.max(np.abs(M))) if np.any(populated) else 0.0
    atol = max(_SYM_RTOL * scale, 1.0e-12)

    for (i, j), values in duplicate_lists.items():
        v0 = values[0]
        for v in values[1:]:
            if not _close(v, v0, atol=atol):
                raise ValueError(
                    f"{label}: duplicate ({i + 1},{j + 1}) entries disagree "
                    f"beyond rtol={_SYM_RTOL:.0e}, atol={atol:.3e}: {values!r}"
                )

    for i in range(6):
        for j in range(i + 1, 6):
            if populated[i, j] and populated[j, i]:
                avg = 0.5 * (M[i, j] + M[j, i])
                M[i, j] = avg
                M[j, i] = avg
            elif populated[i, j]:
                M[j, i] = M[i, j]
            elif populated[j, i]:
                M[i, j] = M[j, i]

    return M


def _close(a: float, b: float, *, atol: float = 1.0e-12) -> bool:
    return bool(np.isclose(a, b, rtol=_SYM_RTOL, atol=atol))


def _maybe_warn_nondimensional(
    A_inf: NDArray[np.float64],
    A: NDArray[np.float64],
    *,
    label: str,
    assume_dimensional: bool,
) -> None:
    """Strengthened post-fix heuristic (M6 PR4 Pre-3).

    The pre-fix check used a single ``max|A| < 10`` magnitude threshold,
    which missed the marin_semi case where surge ``A_inf = 8527`` non-dim
    (above the threshold) but rotational ``A_inf_pitch = 7.4e6`` non-dim
    (a typical OC4 dimensional pitch added mass is O(1e9), so this is
    100x too small).

    Post-fix logic:

    - Path 1 (``assume_dimensional=True``): the caller asserted
      dimensional. We sanity-check the *rotational* added masses --
      a real floating platform's pitch / yaw added mass is ≥ 1e8
      kg·m². If we see < 1e8 with ``assume_dimensional=True``, warn:
      the file is almost certainly the WAMIT non-dim default and the
      caller should either set ``assume_dimensional=False`` or
      pre-rescale the file.
    - Path 2 (``assume_dimensional=False`` -- the default): the reader
      already applied ``rho * ULEN^k``. We check that the post-rescale
      values look reasonable; if not, warn that the rescaling factors
      may be wrong (e.g. a non-OC4 deck with a different ULEN).

    Either way the warning is informational; behaviour is the same
    (rescale or pass-through) per the caller's request. The warning
    is the surface-level alarm that prevents the M6 PR4 Pre-3 class
    of latent bug from recurring.
    """
    a_rot_max = float(np.max(np.abs(A_inf[3:, 3:])))

    if assume_dimensional:
        # The reader returned values verbatim; the caller asserted
        # dimensional. Check rotational scale.
        if a_rot_max < _NONDIM_PITCH_THRESHOLD_KG_M2:
            warnings.warn(
                f"WAMIT .1 file {label} was read with assume_dimensional=True "
                f"but its rotational A_inf max = {a_rot_max:.3e} kg*m^2 is below "
                f"{_NONDIM_PITCH_THRESHOLD_KG_M2:.0e} -- this is the "
                "magnitude expected of a non-dimensional WAMIT file. The "
                "WAMIT default output convention is non-dimensional (manual "
                "v7 §4.2). Pass assume_dimensional=False (the default) so "
                "the reader applies rho * ULEN^k rescaling, OR confirm the "
                "file is genuinely dimensional and update this heuristic. "
                "See conventions doc Item 22.",
                stacklevel=3,
            )
        return

    # assume_dimensional=False: post-rescale sanity. Real platforms
    # have rotational A_inf in the 1e9-1e11 range; below 1e8 means
    # ULEN is probably wrong, or the file was already dimensional and
    # got over-rescaled.
    if a_rot_max < _NONDIM_PITCH_THRESHOLD_KG_M2:
        warnings.warn(
            f"WAMIT .1 file {label}: post-rescale rotational A_inf max = "
            f"{a_rot_max:.3e} kg*m^2 is below the {_NONDIM_PITCH_THRESHOLD_KG_M2:.0e} "
            "expected for a real floating platform. Possible causes: "
            "(a) the file was already dimensional and you should pass "
            "assume_dimensional=True; (b) the ULEN passed to the reader "
            "is wrong (default 1.0 m matches OC4 marin_semi; other decks "
            "may differ -- check WAMIT's ULEN input or HydroDyn's WAMITULEN). "
            "See conventions doc Item 22.",
            stacklevel=3,
        )


def _read_complex_per_dof(
    path: _PathLike,
    *,
    omega: NDArray[np.float64],
    label: str,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Shared parser for ``.3`` (excitation) and ``.4`` (motion RAO) files.

    Both share the row layout ``PER  BETA  I  Mod  Pha  Re  Im``. We use
    ``Re + 1j * Im`` as the canonical representation and verify it agrees
    with ``Mod * exp(1j * Pha_rad)`` to within
    :data:`_COMPLEX_REPR_RTOL` / :data:`_COMPLEX_REPR_ATOL`.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"WAMIT {label} file not found: {p}")

    omega_set = np.asarray(omega, dtype=np.float64)

    # First pass: discover headings, build (period -> omega_index) map.
    rows: list[tuple[float, float, int, float, float, float, float]] = []
    headings: set[float] = set()
    for tokens in _iter_data_rows(p):
        if len(tokens) != 7:
            raise ValueError(
                f"WAMIT {label} row in {p.name} must have 7 fields "
                f"(PER BETA I Mod Pha Re Im); got {len(tokens)}: {tokens!r}"
            )
        per = float(tokens[0])
        beta = float(tokens[1])
        i = int(tokens[2])
        _check_dof_index(i, p.name)
        rows.append(
            (
                per,
                beta,
                i,
                float(tokens[3]),
                float(tokens[4]),
                float(tokens[5]),
                float(tokens[6]),
            )
        )
        headings.add(beta)

    if not rows:
        raise ValueError(f"WAMIT {label} file {p.name} contains no data rows")

    heading_deg = np.asarray(sorted(headings), dtype=np.float64)
    n_h = heading_deg.size
    n_w = omega_set.size

    out = np.zeros((6, n_w, n_h), dtype=np.complex128)
    filled = np.zeros((6, n_w, n_h), dtype=bool)

    for per, beta, i, mod_x, pha_deg, re_x, im_x in rows:
        w = 2.0 * np.pi / per
        w_idx = _match_index(w, omega_set, atol=1.0e-9, label=f"omega in {p.name}")
        h_idx = _match_index(beta, heading_deg, atol=1.0e-9, label="heading")
        z_re_im = complex(re_x, im_x)
        z_mod_pha = mod_x * np.exp(1j * np.deg2rad(pha_deg))
        if not _complex_close(z_re_im, z_mod_pha):
            raise ValueError(
                f"{label} row in {p.name}: Re/Im={z_re_im} disagrees with "
                f"Mod*exp(i*Pha)={z_mod_pha} beyond rtol={_COMPLEX_REPR_RTOL}"
            )
        if filled[i - 1, w_idx, h_idx]:
            existing = out[i - 1, w_idx, h_idx]
            if not _complex_close(existing, z_re_im):
                raise ValueError(
                    f"{label}: duplicate row at (mode={i}, omega={w}, beta={beta}) "
                    f"with disagreeing values {existing!r} vs {z_re_im!r}"
                )
        out[i - 1, w_idx, h_idx] = z_re_im
        filled[i - 1, w_idx, h_idx] = True

    if not np.all(filled):
        missing = np.argwhere(~filled).tolist()
        raise ValueError(
            f"WAMIT {label} grid in {p.name} is incompletely populated; "
            f"missing (mode_idx, omega_idx, heading_idx): {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )

    return heading_deg, out


def _match_index(
    target: float,
    grid: NDArray[np.float64],
    *,
    atol: float,
    label: str,
) -> int:
    idx = int(np.argmin(np.abs(grid - target)))
    if abs(float(grid[idx]) - target) > atol:
        raise ValueError(
            f"{label} value {target} does not match any grid entry; "
            f"nearest is {float(grid[idx])} (atol={atol:.0e})"
        )
    return idx


def _complex_close(a: complex, b: complex) -> bool:
    return bool(
        np.isclose(a.real, b.real, rtol=_COMPLEX_REPR_RTOL, atol=_COMPLEX_REPR_ATOL)
        and np.isclose(a.imag, b.imag, rtol=_COMPLEX_REPR_RTOL, atol=_COMPLEX_REPR_ATOL)
    )
