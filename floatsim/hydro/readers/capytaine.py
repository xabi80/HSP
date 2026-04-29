"""Capytaine NetCDF reader -- ARCHITECTURE.md §8 M5, locked decision Q3.

Capytaine writes diffraction/radiation results as an xarray-backed NetCDF
file. The on-disk schema (see ``capytaine.io.xarray.assemble_dataset``
and ``separate_complex_values`` in the upstream source) is::

    dims:
        omega                (rad/s, float64; may include +inf for A_inf)
        wave_direction       (rad, float64)
        radiating_dof        (string DOF name, e.g. 'Surge', 'Heave')
        influenced_dof       (string DOF name)
        complex              (size 2, ['re', 'im']) -- added by
                             ``separate_complex_values`` whenever a
                             complex variable is written.

    data_vars:
        added_mass(omega, radiating_dof, influenced_dof)            float64
        radiation_damping(omega, radiating_dof, influenced_dof)     float64
        excitation_force(complex, omega, wave_direction, influenced_dof)
                                                                    float64
        Froude_Krylov_force, diffraction_force                      float64
            (split-complex; reader sums them as fallback when
             ``excitation_force`` is absent.)
        hydrostatic_stiffness(radiating_dof, influenced_dof)        float64
            (optional; defaults to zero if absent.)

Conventions translated by this reader
-------------------------------------
- Time:  Capytaine uses ``x(t) = Re[X * exp(-i*omega*t)]`` (LAGS).
  FloatSim's ``HydroDatabase.RAO`` uses
  ``F(t) = Re[X * A_wave * exp(+i*omega*t)]`` (LEADS, matching the
  OrcaFlex VesselType reader). Translation:
  ``RAO_floatsim = conj(excitation_force_capytaine)``.
- Wave direction: Capytaine stores radians, FloatSim stores degrees.
- DOF order: Capytaine accepts arbitrary DOF labels (string-typed).
  This reader requires the standard six labels (case-insensitive):
  ``Surge, Sway, Heave, Roll, Pitch, Yaw``. The 6x6 matrices and
  6-vectors are reordered into FloatSim's canonical
  (surge, sway, heave, roll, pitch, yaw) order.

A_inf handling
--------------
Capytaine does not separately store an "infinite-frequency added mass"
matrix. Two valid sources:

1. The dataset includes a sample at ``omega = +inf`` (Capytaine accepts
   ``omega=infinity`` as a problem condition; the resulting added mass
   is the high-frequency limit). The reader extracts that sample as
   ``A_inf`` and strips the ``+inf`` row from the finite-frequency
   ``A`` and ``B`` arrays.
2. The caller passes an explicit ``a_inf`` keyword.

If neither is available, ``read_capytaine`` raises ``ValueError`` --
guessing ``A_inf`` from finite samples gives a subtly wrong retardation
kernel (the M2 PR4 regression is the cautionary tale).

Hydrostatic stiffness
---------------------
Capytaine's ``hydrostatic_stiffness`` covers the **buoyancy/waterplane**
contribution exclusively (gravity ``m*g*z_G`` on roll/pitch is the
body's responsibility, since Capytaine does not know the mass
distribution -- same convention as WAMIT's ``.hst``). Downstream
``floatsim.bodies.Body`` assembly is expected to add the gravity term.

Reference point
---------------
Capytaine's NetCDF does not carry a single canonical reference point.
The default is ``(0, 0, 0)`` (the origin in the inertial frame the BEM
problem was solved in). Override via the ``reference_point`` kwarg if
the BEM run was not at the origin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from floatsim.hydro.database import HydroDatabase

_PathLike = str | Path

# Capytaine's canonical 6-DOF labels. Case-insensitive comparison;
# the order in which they appear on disk is arbitrary.
_REQUIRED_DOF_NAMES: Final[tuple[str, ...]] = (
    "surge",
    "sway",
    "heave",
    "roll",
    "pitch",
    "yaw",
)


def read_capytaine(
    path: _PathLike,
    *,
    a_inf: NDArray[np.float64] | None = None,
    reference_point: tuple[float, float, float] | None = None,
) -> HydroDatabase:
    """Read a Capytaine NetCDF file into a :class:`HydroDatabase`.

    Parameters
    ----------
    path
        Path to the ``.nc`` file written by ``capytaine.io.xarray``.
    a_inf
        Optional 6x6 infinite-frequency added mass matrix, in FloatSim
        DOF order. Required only when the file does not already include
        an ``omega = +inf`` sample. Symmetric within
        :data:`floatsim.hydro.database._SYMMETRY_RTOL`.
    reference_point
        Optional ``(x, y, z)`` point in the inertial frame about which
        the BEM coefficients were computed. Defaults to the origin.

    Returns
    -------
    HydroDatabase
        Single-body BEM database, with all conventions translated to
        FloatSim's internal forms (leads phase, degrees heading, SI
        units, canonical DOF order).

    Raises
    ------
    ValueError
        If required dimensions or variables are missing, if DOF labels
        do not match the canonical six, or if neither an ``omega=inf``
        sample nor an ``a_inf`` argument supplies the
        infinite-frequency added mass.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Capytaine NetCDF file not found: {p}")

    with xr.open_dataset(p) as ds:
        ds = _decode_string_coords(ds.load())

    ds = _merge_split_complex(ds)
    _check_required_dims(ds)

    dof_perm = _resolve_dof_permutation(ds)

    omega_all = np.asarray(ds["omega"].values, dtype=np.float64)
    finite_mask = np.isfinite(omega_all)
    inf_mask = np.isposinf(omega_all)
    if np.any(omega_all[finite_mask] < 0.0):
        raise ValueError("omega must be non-negative")
    if np.any(~(finite_mask | inf_mask)):
        raise ValueError("omega contains entries that are neither finite nor +inf")

    omega_finite = omega_all[finite_mask]
    sort_idx = np.argsort(omega_finite, kind="stable")
    omega_finite = omega_finite[sort_idx]
    if omega_finite.size < 2:
        raise ValueError(
            f"Capytaine dataset must contain at least 2 finite omega samples; "
            f"found {omega_finite.size}."
        )
    if np.any(np.diff(omega_finite) <= 0.0):
        raise ValueError("Capytaine omega grid contains duplicate finite samples")

    A_full = _extract_radiation(ds, "added_mass", dof_perm, finite_mask, sort_idx)
    B_full = _extract_radiation(ds, "radiation_damping", dof_perm, finite_mask, sort_idx)

    A_inf_resolved = _resolve_a_inf(ds, dof_perm, inf_mask, a_inf)

    heading_rad = np.asarray(ds["wave_direction"].values, dtype=np.float64)
    heading_deg = np.rad2deg(heading_rad)
    heading_sort = np.argsort(heading_deg, kind="stable")
    heading_deg = heading_deg[heading_sort]

    RAO = _extract_excitation(ds, dof_perm, finite_mask, sort_idx, heading_sort)

    C = _extract_hydrostatic(ds, dof_perm)

    ref = (
        np.asarray(reference_point, dtype=np.float64)
        if reference_point is not None
        else np.zeros(3, dtype=np.float64)
    )

    return HydroDatabase(
        omega=omega_finite,
        heading_deg=heading_deg,
        A=A_full,
        B=B_full,
        A_inf=A_inf_resolved,
        C=C,
        RAO=RAO,
        reference_point=ref,
        # Capytaine's hydrostatic_stiffness is buoyancy/waterplane only;
        # gravity m*g*z_G is added downstream by assemble_cummins_lhs.
        # See HydroDatabase docstring and floatsim/hydro/hydrostatics.py.
        C_source="buoyancy_only",
        metadata=_collect_metadata(ds, p),
    )


# ---------------------------------------------------------------------------
# Dataset-level helpers
# ---------------------------------------------------------------------------


def _decode_string_coords(ds: xr.Dataset) -> xr.Dataset:
    """Convert byte-string coordinates (NetCDF default) to UTF-8 Python str.

    NetCDF stores variable-length strings as bytes (``|S<n>``). xarray
    does not transparently decode them on open, so DOF and ``complex``
    coordinates come back as ``b'Surge'``, ``b're'`` and so on. This
    helper returns a dataset whose string-typed coords are plain ``str``
    so subsequent name-based selection works.
    """
    out = ds.copy()
    for coord_name in ("radiating_dof", "influenced_dof", "complex", "body_name"):
        if coord_name not in out.coords:
            continue
        values = out[coord_name].values
        if values.dtype.kind == "S":
            decoded = np.array([v.decode("utf-8") for v in values], dtype="U32")
            out = out.assign_coords({coord_name: decoded})
    return out


def _merge_split_complex(ds: xr.Dataset) -> xr.Dataset:
    """Re-pack ``complex=['re','im']``-split variables into native complex.

    Mirrors ``capytaine.io.xarray.merge_complex_values``. After this
    call, complex variables (e.g. ``excitation_force``,
    ``Froude_Krylov_force``) have ``dtype == complex128`` and the
    ``complex`` dimension is removed.
    """
    if "complex" not in ds.coords:
        return ds
    out = ds.copy()
    re_label, im_label = "re", "im"
    for var_name in list(out.data_vars):
        var = out[var_name]
        if "complex" not in var.dims:
            continue
        re = var.sel(complex=re_label).values
        im = var.sel(complex=im_label).values
        merged = re + 1j * im
        new_dims = tuple(d for d in var.dims if d != "complex")
        out = out.drop_vars(str(var_name))
        out[str(var_name)] = (new_dims, merged.astype(np.complex128))
    out = out.drop_vars("complex")
    return out


def _check_required_dims(ds: xr.Dataset) -> None:
    """Enforce presence of the four schema dimensions FloatSim consumes."""
    required = ("omega", "wave_direction", "radiating_dof", "influenced_dof")
    missing = [d for d in required if d not in ds.dims]
    if missing:
        raise ValueError(
            f"Capytaine dataset missing required dimensions {missing}; "
            f"found dims {tuple(ds.dims)}."
        )
    if "added_mass" not in ds.data_vars:
        raise ValueError("Capytaine dataset missing required variable 'added_mass'.")
    if "radiation_damping" not in ds.data_vars:
        raise ValueError("Capytaine dataset missing required variable 'radiation_damping'.")


def _resolve_dof_permutation(ds: xr.Dataset) -> NDArray[np.int_]:
    """Return a 6-vector mapping disk DOF order -> FloatSim DOF order.

    Capytaine accepts arbitrary user-defined DOF names; FloatSim's
    canonical order is ``(surge, sway, heave, roll, pitch, yaw)``.
    Comparison is case-insensitive. ``radiating_dof`` and
    ``influenced_dof`` must agree on the same six labels (Capytaine
    enforces this by construction for rigid-body problems).
    """
    rad_disk = [str(s).lower() for s in ds["radiating_dof"].values]
    inf_disk = [str(s).lower() for s in ds["influenced_dof"].values]

    if sorted(rad_disk) != sorted(inf_disk):
        raise ValueError(
            "radiating_dof and influenced_dof label sets disagree: "
            f"{ds['radiating_dof'].values!r} vs {ds['influenced_dof'].values!r}."
        )
    if sorted(rad_disk) != sorted(_REQUIRED_DOF_NAMES):
        raise ValueError(
            f"Capytaine DOF labels must be the six rigid-body names "
            f"(case-insensitive): {_REQUIRED_DOF_NAMES}. Found "
            f"{ds['radiating_dof'].values!r}."
        )

    perm = np.asarray([rad_disk.index(name) for name in _REQUIRED_DOF_NAMES], dtype=np.intp)
    return perm


# ---------------------------------------------------------------------------
# Per-variable extraction
# ---------------------------------------------------------------------------


def _extract_radiation(
    ds: xr.Dataset,
    var_name: str,
    dof_perm: NDArray[np.int_],
    finite_mask: NDArray[np.bool_],
    sort_idx: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Pull a (omega, radiating, influenced) variable into FloatSim layout (6,6,n_w)."""
    raw = ds[var_name]
    arr = np.asarray(raw.values, dtype=np.float64)
    arr = arr[finite_mask, :, :]
    arr = arr[sort_idx, :, :]
    arr = arr[:, dof_perm, :]
    arr = arr[:, :, dof_perm]
    # FloatSim expects (6, 6, n_w); Capytaine stored (n_w, 6, 6).
    arr = np.moveaxis(arr, 0, -1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"Capytaine '{var_name}' contains non-finite values at finite-omega samples."
        )
    return np.ascontiguousarray(arr)


def _extract_excitation(
    ds: xr.Dataset,
    dof_perm: NDArray[np.int_],
    finite_mask: NDArray[np.bool_],
    sort_idx: NDArray[np.int_],
    heading_sort: NDArray[np.int_],
) -> NDArray[np.complex128]:
    """Build (6, n_w, n_h) complex excitation in FloatSim's leads convention.

    Prefers a ``excitation_force`` variable; falls back to the sum of
    ``Froude_Krylov_force + diffraction_force`` when the precomputed
    sum was not written. Conjugates to translate Capytaine's
    ``exp(-i*omega*t)`` into FloatSim's ``exp(+i*omega*t)``.
    """
    if "excitation_force" in ds.data_vars:
        F = ds["excitation_force"]
    elif "Froude_Krylov_force" in ds.data_vars and "diffraction_force" in ds.data_vars:
        F = ds["Froude_Krylov_force"] + ds["diffraction_force"]
    else:
        raise ValueError(
            "Capytaine dataset has neither 'excitation_force' nor "
            "('Froude_Krylov_force' + 'diffraction_force'); cannot assemble RAO."
        )

    arr = np.asarray(F.values, dtype=np.complex128)
    # Capytaine layout after merge_complex: (omega, wave_direction, influenced_dof).
    arr = arr[finite_mask, :, :]
    arr = arr[sort_idx, :, :]
    arr = arr[:, heading_sort, :]
    arr = arr[:, :, dof_perm]
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "Capytaine excitation force contains non-finite values at " "finite-omega samples."
        )
    # Lags -> leads: conjugate.
    arr = np.conj(arr)
    # FloatSim expects (6, n_w, n_h); Capytaine layout is (n_w, n_h, 6).
    arr = np.moveaxis(arr, -1, 0)
    return np.ascontiguousarray(arr)


def _extract_hydrostatic(
    ds: xr.Dataset,
    dof_perm: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Pull hydrostatic stiffness into a 6x6, default zeros if absent.

    Capytaine's ``hydrostatic_stiffness`` covers buoyancy/waterplane
    only -- gravity ``m*g*z_G`` on roll/pitch is the body's job
    (same convention as WAMIT's ``.hst``).
    """
    if "hydrostatic_stiffness" not in ds.data_vars:
        return np.zeros((6, 6), dtype=np.float64)
    raw = np.asarray(ds["hydrostatic_stiffness"].values, dtype=np.float64)
    if raw.shape != (6, 6):
        raise ValueError(f"hydrostatic_stiffness must be 6x6; got shape {raw.shape}.")
    perm_matrix = raw[dof_perm, :][:, dof_perm]
    # Symmetrize against panel-method noise (same policy as the WAMIT reader).
    sym = 0.5 * (perm_matrix + perm_matrix.T)
    return sym


def _resolve_a_inf(
    ds: xr.Dataset,
    dof_perm: NDArray[np.int_],
    inf_mask: NDArray[np.bool_],
    a_inf_arg: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Source A_inf either from the dataset's omega=inf row or from the kwarg."""
    has_inf_row = bool(np.any(inf_mask))
    if has_inf_row and a_inf_arg is not None:
        raise ValueError(
            "Capytaine dataset already contains an omega=inf sample; "
            "do not also pass an explicit 'a_inf' kwarg."
        )

    if has_inf_row:
        n_inf = int(np.sum(inf_mask))
        if n_inf > 1:
            raise ValueError(
                f"Capytaine dataset contains {n_inf} omega=inf samples; expected 0 or 1."
            )
        raw = np.asarray(ds["added_mass"].values, dtype=np.float64)
        # raw layout: (omega, radiating, influenced)
        a_inf_disk = raw[inf_mask, :, :][0]
        a_inf = a_inf_disk[dof_perm, :][:, dof_perm]
    elif a_inf_arg is not None:
        a_inf = np.asarray(a_inf_arg, dtype=np.float64)
        if a_inf.shape != (6, 6):
            raise ValueError(f"a_inf must have shape (6, 6); got {a_inf.shape}.")
    else:
        raise ValueError(
            "Capytaine dataset has no omega=inf sample, and no 'a_inf' argument was "
            "supplied. Either add an infinite-frequency case to the Capytaine problem "
            "set, or pass a_inf=<6x6 matrix> to read_capytaine()."
        )

    if not np.all(np.isfinite(a_inf)):
        raise ValueError("Resolved A_inf contains non-finite entries.")
    # Symmetrize against panel-method noise (mirrors the WAMIT reader policy).
    a_inf_sym: NDArray[np.float64] = 0.5 * (a_inf + a_inf.T)
    return a_inf_sym


def _collect_metadata(ds: xr.Dataset, path: Path) -> dict[str, str]:
    """Surface a small subset of Capytaine's global attrs into HydroDatabase.metadata."""
    out: dict[str, str] = {"source_format": "capytaine", "source_path": str(path)}
    for key in ("rho", "g", "water_depth", "body_name", "capytaine_version"):
        if key in ds.attrs:
            out[key] = str(ds.attrs[key])
    return out
