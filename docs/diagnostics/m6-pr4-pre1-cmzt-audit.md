# M6 PR4 Pre-1 — `PtfmCMzt` convention audit

**Status**: ✅ **PR2 inconsistent but benign**. PR2's `cog_total_z`
diagnostic mixes conventions (uses OpenFAST's `PtfmCMzt = -8.66 m`
steel-only with Robertson's `1.347 × 10⁷ kg` with-ballast). However,
the inconsistency does NOT affect any element of `F_residual` —
PR2's heave / roll / pitch assertions are numerically unchanged
under either convention.

Recommendation: **document as conventions doc Item 17 and apply the
small parser fix on a `fix-pr2-cmzt` branch before PR4** (same
precedent as the M5 hydrostatic-gravity fix). The fix is one line
in `tests/support/openfast_deck.py` and prevents the convention
drift from leaking into future scenario PRs.

## Background

PR3 Setup B (combined-deck FloatSim mass aggregation) flagged a
convention mismatch: PR2's `compute_openfast_deck_residual`
parses OpenFAST's `PtfmCMzt = -8.66 m` (the steel-only platform
CoG; `PtfmMass = 3.85 × 10⁶ kg`) but assigns it to the
`platform_with_ballast` component whose mass is Robertson's
`1.347 × 10⁷ kg` (with ballast). Robertson 2014 Table 3-1's
matching CoG for the with-ballast mass is `-13.46 m`, ~5 m deeper.

The audit question: does this inconsistency change any number PR2
asserts on?

## Where `cog_total_z` enters the residual

`compute_openfast_deck_residual` returns a 6-vector
`F_residual` plus a `cog_total_z_m` diagnostic. The 6-vector
elements at the BEM reference (z = 0 = SWL) are:

```
F[2] = ρ · V₀ · g  −  m_total · g                 (vertical)
F[3] = -cog_total_y · weight_n  +  hd_cobyt · buoyancy_n   (roll moment)
F[4] = +cog_total_x · weight_n  -  hd_cobxt · buoyancy_n   (pitch moment)
F[0] = F[1] = F[5] = 0  (axisymmetric mass + on-axis CoB)
```

**`F[3]` and `F[4]` use only `cog_total_x` and `cog_total_y`**, not
`cog_total_z`. This is correct: at the linearisation point (`xi=0`)
the body's CoG is directly below the BEM reference, so the gravity
vector `(0, 0, -m·g)` produces moments only from horizontal lever
arms (`r × F` where `F` is purely vertical → moment depends on the
horizontal components of `r` only).

`cog_total_z` is reported in the `DeckResidual` dataclass for
diagnostic logging, and the M6 PR2 cross-check report quotes it
(0.488 m heave equilibrium, etc.), but **no PR2 assertion reads
it**. Verified by grep on `tests/validation/test_m6_openfast_static_eq.py`:
the only `DeckResidual` fields consumed are `F_residual`,
`m_total_kg`, `buoyancy_n`, `weight_n`, `iterations`,
`residual_norm`, `converged` — `cog_total_z_m` does not appear.

## Numerical verification

PR2 residual on S1 with the existing convention:

| Component | Value |
|-----------|------:|
| F[2] heave | 1.876 130 376 998 × 10⁶ N |
| F[3] roll | 0.000 N |
| F[4] pitch | 6.538 097 357 × 10⁶ N·m |
| `cog_total_z` (mixed) | -5.308 m |

Recomputed with Robertson `-13.46 m` substituted for the
`platform_with_ballast` z (all other inputs identical):

| Component | Value | Δ vs existing |
|-----------|------:|--------------:|
| F[2] heave | 1.876 130 376 998 × 10⁶ N | +4.66 × 10⁻¹⁰ N |
| F[3] roll | 0.000 N | 0 |
| F[4] pitch | 6.538 097 358 × 10⁶ N·m | +9.31 × 10⁻¹⁰ N·m |
| `cog_total_z` (all-Robertson) | -10.701 m | -5.39 m |

The F-vector deltas are float-precision noise (≤ 10⁻⁹). The only
substantive change is `cog_total_z`: -5.31 m → -10.70 m. Since no
assertion consumes this value, **PR2 results are unchanged**.

## Why the inconsistency is benign

The Cummins formulation used by the M6 cross-check linearises the
6-DOF dynamics about the BEM reference. The hydrostatic stiffness
matrix `C` already encodes the full restoring including the gravity
contribution `-m · g · z_G` per Robertson Table 3-3 (the
`OC4_C55_PITCH_NM_PER_RAD = 1.078 × 10⁹` value carries this
embedded). When the residual force `F_residual` is applied as
external load to the linearised system, the equilibrium offset is

```
xi_eq[i] = F_residual[i] / C[i, i]   (for the diagonal DOFs)
```

and `F_residual` is purely the load (gravity + buoyancy at xi=0)
without re-doing the gravity-stiffness decomposition. The z_G
inconsistency only enters the diagnostic `cog_total_z` field and
not the load itself.

The inconsistency would matter if the parser EVER computed a moment
that uses `cog_total_z` (e.g., a body with off-axis vertical CoG
contributing to torsional moments — yaw-axis-perpendicular
quantities, which are `0` for OC4 by symmetry). PR4 will inherit
the same scoping: S3 RAOs evaluate the linearised response per
DOF, with C-matrix gravity decomposition consistent with Robertson
throughout.

## Should the parser be fixed?

The fix is one line. Pros: prevents a future scenario PR (e.g., a
non-axisymmetric deck, or a moored case where vertical lever arms
matter for mooring-line moment computation) from inheriting a
silent error. Cons: a separate commit + branch for what is
provably zero numerical impact on the existing test suite.

**Recommendation**: small fix branch before PR4 starts. Pattern-
matches the M5 hydrostatic-gravity fix in spirit (a parser-level
caveat made explicit in code rather than only in prose). The
diagnostic doc (this file) plus the conventions doc Item 17 is the
permanent record.

## Item 17 (to be added to conventions doc)

> **Item 17 — `z_G` must be consistent with mass and `C` across all
> uses.** When mixing literature values for mass and OpenFAST input
> values for CoG (or vice versa), the conventions must match.
> Robertson 2014 Table 3-1's `1.347 × 10⁷ kg` is the
> platform-with-ballast mass and pairs with `z_G = -13.46 m`.
> OpenFAST's `PtfmMass = 3.85 × 10⁶ kg` is steel-only and pairs
> with `PtfmCMzt = -8.66 m`. The two pairs are NOT interchangeable;
> mixing one mass with the other CoG is a bookkeeping error that
> may not surface in a specific assertion (see PR4 Pre-1 audit
> finding for OC4: F-vector elements are independent of z_G under
> axisymmetric assumptions) but will eventually leak through a
> moment formula that does reference vertical lever arms.

## Files produced

- This document.
- (Pending) Small parser fix on `fix-pr2-cmzt` branch off main —
  one-line change in `tests/support/openfast_deck.py` line 386 to
  use Robertson's `-13.46 m`, plus regression run and PR2
  retrospective update.
