# M6 PR3 — pitch period gap diagnostic (Pre-step before Option A)

**Status**: ✅ classified. **F1 mostly explains** the gap.
PR3 proceeds with period xfail-strict under **F1-residual**.

Per the locked Option A workflow: classify the period gap between
FloatSim and OpenFAST on S2 (drag-off, post-Mod-1) before any test
code lands. Two FloatSim setups against the regenerated drag-off
OpenFAST reference.

## Reference

OpenFAST S2 (Morison drag disabled, `PtfmSurge=0` IC, `PtfmPitch=5°` IC):

| Quantity | Value |
|----------|------:|
| Pitch period (mean of first 10 inter-zero-crossings, last-60s mean removed) | **26.83 s** |
| Pitch peak at t = 26.8 s | 5.00° (essentially undecayed) |
| ζ over peaks 1–5 | 1.5 × 10⁻⁴ (numerical noise) |

## FloatSim setups

**Common parameters:**
- BEM: `marin_semi.1` via WAMIT reader (post-fix kernel; t_max = 200 s, dt = 0.05 s).
- Solver: generalised-α (`rho_inf = 1.0`); 600 s, IC `pitch = 5°`, no F_external.
- Hydrostatic decomposition (used by both setups for the buoyancy term):

  ```
  C_55_buoyancy_only = OC4_C55_full − m_platform · g · |z_G_platform|
                     = 1.0780e9 − 1.347e7 · 9.80665 · 13.46
                     = -7.004e8 N·m/rad   (.hst-style)
  ```

**Setup A — platform-only Robertson (the original PR3 plan):**

- Mass via `_oc4_rigid_body_mass_matrix()` (Robertson Table 3-1 / 3-3,
  `m = 1.3473e7 kg`, `z_G = −13.46 m`, `I_yy_cog = 6.827e9`,
  block-diagonal at SWL via internal parallel-axis).
- `C_55 = 1.078e9` (Robertson Table 3-3, full restoring at platform-only).
- I₅₅ at SWL = 9.27 × 10⁹ kg·m².

| | |
|--|--:|
| FloatSim period | **18.43 s** |
| rel-err vs OpenFAST | **−31.3 %** |
| Naive uncoupled `2π·√((M+A_inf)/C)` | 18.43 s — matches integration exactly |

**Setup B — combined deck (Robertson z_G_platform_with_ballast convention):**

Mass aggregation parses the OpenFAST S2 deck (post-Mod-1, drag-off)
for tower, hub, nacelle, yaw-bearing, and blade masses + positions.
Platform-with-ballast uses Robertson Table 3-1's CoG at −13.46 m
(see "Convention note" below — NOT OpenFAST's `PtfmCMzt = −8.66 m`,
which is steel-only).

| Component | Mass [kg] | z_G [m] | x_G [m] |
|-----------|----------:|--------:|--------:|
| platform_with_ballast | 1.347 × 10⁷ | −13.460 | 0.000 |
| tower                 | 2.497 × 10⁵ | +43.239 | 0.000 |
| hub                   | 5.678 × 10⁴ | +87.600 | +1.900 |
| yaw_bearing           | 0           | +87.600 | 0.000 |
| nacelle               | 2.400 × 10⁵ | +89.350 | +1.900 |
| blades_total (3 blades) | 5.411 × 10⁴ | +87.600 | +1.900 |
| **combined**          | **1.4074 × 10⁷** | **−9.904** | **+0.047** |

I₅₅ at SWL via point-mass parallel-axis from each component's CoG
(platform's intrinsic `I_yy_cog = 6.827e9` is added on top of its
parallel-axis term):

```
I_55_combined = 1.250 × 10¹⁰ kg·m²
```

`C_55` recomputed with combined CoG:

```
C_55_combined = C_55_buoyancy_only + (-m_total · g · z_G_combined)
              = -7.004e8 + (-1.4074e7 · 9.80665 · (-9.904))
              = -7.004e8 + 1.367e9
              = +6.665e8 N·m/rad
```

Mass matrix uses `cog_offset_body = (x_G, 0, z_G)` so the
surge–pitch coupling `M[0,4] = −m·z_G` is captured (FloatSim's
`rigid_body_mass_matrix` builds this automatically).

| | |
|--|--:|
| FloatSim period | **25.67 s** |
| rel-err vs OpenFAST | **−4.29 %** |

## Convention note — platform-with-ballast CoG

The OpenFAST S2 `*_ElastoDyn.dat` declares:

```
PtfmMass  =  3.85e6  kg     (steel-only platform)
PtfmCMzt  = -8.66    m      (steel-only CoG)
```

The water ballast is treated separately, via HydroDyn `FillGroups`
(member-fill volumes with `FillDens = 1025 kg/m³`). The combined
"platform with ballast" CoG is at z = −13.46 m per Robertson 2014
Table 3-1 (the published OC4 reference value, used by the M6 PR2
deck-residual parser as well — see
`tests/support/openfast_deck.py` line 287).

The Pre-step uses Robertson's −13.46 m for the combined platform
CoG, NOT OpenFAST's `PtfmCMzt`, to keep the mass aggregation
internally consistent with `OC4_PLATFORM_TOTAL_MASS_KG = 1.3473e7`
(which already includes ballast). Mixing OpenFAST's steel-only
`PtfmCMzt` with Robertson's with-ballast mass would underestimate
the combined CoG depth by ~5 m and skew the pitch period
calculation by a further ~7 % (verified during the diagnostic
debug pass — see commit history of this branch).

## Decision tree

Per the locked Option A workflow:

| Setup B rel-err vs OpenFAST | Decision | Result |
|-----------------------------|----------|--------|
| < 2 × 10⁻² | F1 fully explains; period asserts GREEN | not triggered |
| < 5 × 10⁻² | F1 mostly explains; xfail-strict under F1-residual | **TRIGGERED** (4.29 %) |
| ≥ 5 × 10⁻² | second deck-identity effect; pause and name F2 | not triggered |

**Locked**: PR3 proceeds. Period assertion will fire as
`pytest.mark.xfail(strict=True, reason="F1-residual: combined-deck
period mismatch with platform-only point-mass approximation; full
F1 closure requires distributed inertia integration of platform
ballast and tower/RNA components")`.

## What's in the residual 4.29 %?

Hypotheses for the remaining gap (not investigated further in PR3 —
flagged as "F1-residual" follow-up):

1. **Platform inertia approximation**. Setup B uses Robertson's
   `I_yy_cog = 6.827e9` for the platform-with-ballast as a single
   point mass at z = −13.46 m. The actual ballast water is
   distributed in the column-fill members (heave plates, columns)
   with their own volume distribution. A proper integration of the
   filled volumes' inertia would shift `I_yy_at_SWL` by O(5 %) —
   in either direction, depending on whether the ballast is more
   concentrated near the heave plates (lower z, more inertia) or
   along the columns (higher z, less inertia near SWL).

2. **Above-water components as point masses**. Tower, hub, nacelle,
   blades all assume point-mass at component CoG. Tower especially
   has distributed inertia; integrating its `TMassDen` profile would
   add a small contribution.

3. **Buoyancy decomposition**. `C_55_buoyancy_only = -7.004e8` was
   computed as `Robertson_C_55 - gravity_at_platform_only`. If
   Robertson's `C_55 = 1.078e9` is for a reference configuration
   different from "platform-only at -13.46 m" (e.g., it's the
   tower-locked configuration with combined CoG built in), this
   decomposition is internally inconsistent.

4. **A_inf coupling effects**. The pitch–surge coupling
   `A_inf[0,4] = -1.05e5` from marin_semi adds a small correction
   to the effective pitch inertia in the coupled mode. Setup B
   includes this via the BEM `A_inf` matrix — possibly OpenFAST
   treats it differently.

Hypothesis 1 is the largest of these by typical magnitude. A proper
F1 closure (in a future PR) would parse the HydroDyn `FillGroups`
and member geometry to compute the actual ballast inertia
distribution.

## Files produced

- `scripts/m6_pr3_prestep_period_gap.py` — Pre-step runner
  (parses the S2 deck, builds Setup A / Setup B, runs FloatSim,
  classifies the gap).
- This document.
