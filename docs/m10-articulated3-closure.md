# M10 — Articulated-3 cluster (coupled BEM + KKT joints): closure

**Status: CLOSED, 2026-07-29.** Branch `milestone-10-articulated3`
(`8be5c81..6c7ab50`, 8 commits ahead of `main` at `8aa95a9`). Plan:
[`docs/m10-articulated3-plan.md`](m10-articulated3-plan.md) (Q1-Q8 +
Amendments A1-A5). Program context:
[`docs/tier3-program-plan.md`](tier3-program-plan.md).

M10 assembled the first **articulated** FloatSim model — 3 hydrodynamic
buoys on the committed coupled 18-DOF BEM fixture + 1 dry structural hub,
joined by 3 `yaw_locked` joints — validated it against a pre-derived
heave cross-check, and **measured** the joint rotation under regular
waves. The rotation measurement surfaced the milestone's headline result:
an in-band drag-free rotational resonance that makes the LEVEL2 decision
**undeterminable without a drag term**.

---

## S1 — Scope, deliverables, and the capability arc

**Delivered (branch `milestone-10-articulated3`):**

| PR | Commit | Deliverable |
|----|--------|-------------|
| PR0 | `41e4537` | thread the Item-25 `asymptote_check_override` through the coupled `build_system` path (small cluster hulls need it) |
| PR0.5 | `84053dc` | structural (hydro-free) body support in the coupled assembly (the dry hub) |
| PR0.75 | `465979a` | reference-aware `JointSet` — the joint/coupled state-convention bridge |
| PR0.85 | `baff44a` | per-body label-resolved hydrostatic `C` in the coupled path |
| PR1 | `04d4e9c` | first articulated run — heave cross-check + zero-pitch gates |
| PR2 | `6c7ab50` | regular-wave rotation measurement + the in-band-resonance finding |

**The capability arc, honestly (3a).** The plan sized M10 as "assembly +
two gates" (Q8). It took **four capability PRs before PR1's gates could
run**, because the articulated coupled path had never been exercised
end-to-end and each cross-module seam hid a defect that was **invisible
to existing tests and loud on first real dynamics** — the CLAUDE.md §13
shape, three more instances:

- **PR0.75** — the M9 joint layer read `xi[6k:6k+3]` as *absolute* world
  position; the coupled Cummins system reads `xi` as *displacement from
  reference*. At rest the two disagreed by the arm separation
  (`max|phi| = 1.689`); the fix is a reference-aware `JointSet`.
- **PR0.5** — the coupled assembly rejected a dry body; a structural hub
  needed rigid-mass-only support with zero hydrodynamic blocks.
- **PR0.85** — the coupled BEM fixture carries `C = 0` (hydrostatics are
  per-body block-diagonal and cannot live in a cross-coupled database),
  so the assembled system had **no heave restoring** until per-body
  hydrostatic `C` was injected by label.

**Generalised lesson (A3(d), carried to M11): STRUCTURAL ASSERTIONS DO
NOT CATCH A MISSING RESTORING FORCE.** PR0.5's gate asserted the hub's
mass/rank/zeros and passed while the system had zero restoring;
PR0.85's gate is therefore a **decay**, not an assembly check.
Dynamics-bearing capability needs **dynamics-shaped gates**. Each of the
three findings was caught by tracing the cross-check path at PR-plan
time (CLAUDE.md §15 audit pattern), not by an assertion failing.

---

## S2 — Terminal gates (PASSED, measured)

All on the real 4-body topology (`tests/validation/test_m10_pr1_articulated.py`).

**Preconditions (Amendment A2 STEP-1 + a second-convention detector):**

| check | measured | gate |
|-------|----------|------|
| `phi(xi=0)` (reference-aware joints) | `0.000e+00` | `< 1e-9` |
| `rank(G)` on assembled model | `12` (shape 12x24) | `= 12` |
| equilibrium — no-force drift from `xi=0` over 20 s | `0.000e+00` | `< 1e-9` |

The no-drift check is the displacement-`xi` convention's static-solve
verification **and** a second-convention-mismatch detector; it is exactly
zero, so there is no residual reference-config force.

**GATE A — heave cross-check (correctness) — the milestone's validation
(3b).** The pure-heave free-decay period:

```
T_n = 3.105333 s   vs   pre-derived  2*pi*sqrt((M + A33_inf)/C33)
                        = 2*pi*sqrt((98.01 + 64.0738043)/663.2420101)
                        = 3.106087 s   (committed interaction.json
                        :T_n_with_interaction = 3.1060873560737936 s)
```

`rel = 2.4e-4`, **41.7x inside the rtol-1e-2 M8 cross-check band.** With
every joint translation locked the cluster's pure heave is
rigid-body-identical, so this is a **true cross-check, not a re-fit** —
and it is passable **only if joints, coupled hydrodynamics, the mass
split, and per-body hydrostatics were simultaneously correct.** GATE A is
the single assertion that certifies the whole assembled stack.

**GATE B — zero-pitch symmetry (correctness).** A symmetric heave IC
excites only symmetric modes (3-fold + y-mirror): `max|pitch| = 1.4e-5`,
`max|roll| = 1.3e-5 rad` (projection numerical floor), gate `1e-3` —
100x below the Item-2 physical threshold. No assembly/constraint
asymmetry. Joint constraints hold along the decay (`max|phi| = 2e-16`).

---

## S3 — Empirical findings

### The headline: an in-band drag-free rotational resonance (PR2)
Free rotational decay (buoy pitch IC, no wave) measured a
buoy-pitch-about-joint mode:

- `T_rot = 3.257 s`, `zeta = 0.373 %` (radiation-only), `Q ~ 134`,
  **in-band and adjacent to the 3.106 s heave resonance.**
- Post-Q2-confirmation (A5(b)) `T_rot` is **single-valued** — the
  alternative-split `3.431 s` was a sensitivity bound, now retired.
- The decay is **stable and bounded** (peak never exceeds its IC), so the
  integrator + KKT/projection handling are sound; the wave-case runaway
  (hundreds of rad, still building at 180 s) is **genuine resonant
  buildup, not numerical instability.**

**Amplitude arithmetic** (drag-free wave amplitude reaching the Item-2
`0.1 rad` threshold; the response is RAO-linear so it scales with `A`):

| regime | sensitivity | `A_crit(0.1 rad)` |
|--------|-------------|-------------------|
| off resonance (`T = 10 s`, worst joint) | `0.0185 rad/m` | `5.4 m` (`H ~ 10.8 m`) |
| near resonance (`T_rot`, via measured `Q`) | `~22 rad/m` | `~0.0045 m` (`H ~ 9 mm`) |

**~1200x contrast.** The near-resonance figure uses the **measured**
excitation-moment rise `F(omega_res)/F(omega_off) = 8.96` (tracking
`omega^2`, wave slope) x `Q = 134` — a correction to the flat-excitation
estimate that would have given `~0.04 m`. The record won; the conclusion
strengthened.

### Wave forcing delivered (PR2, Q6 = supported)
`make_regular_wave_force` (`floatsim/hydro/excitation.py:134`) is
dimension-agnostic (returns an 18-vector for the coupled RAO); the
`integrate_cummins(external_force=...)` hook is at
`floatsim/solver/newmark.py:218`; the buoy force scatters to global DOF
`[0:18]`, the hub `[18:24]` carries zero. The multibody RAO is
**origin-referenced** (`reference_point = (0,0,0)`), verified by
`arg(b0)-arg(b1) = -k*(x0-x1)` to 3-4 sig figs at three frequencies with
`b1 == b2` exactly (y-mirror). The driver declines turnkey wiring (Q1);
the caller composes.

### The (24, 12) KKT timing point (3d)
The assembled system is `n = 24` generalized DOF (`6 x 4` bodies) with
`m = 12` constraints (3 `yaw_locked` joints x 4 rows each). The bordered
KKT solve is `(n + m) = 36`. Honest scope note (per the plan risk
register): **this informs `m`-scaling only.** `n = 24` repeats M9's
`n = 24`; the `n`-scaling question (the 12-buoy `n = 72`) stays **open
for M11**.

---

## S4 — The LEVEL2 output: a dependency, not a number (3c)

M10's charter was to **measure** the rotation amplitudes the post-M10
LEVEL2 decision gate consumes (program plan Q1). M10's actual output is
that **those amplitudes are undeterminable without drag** near resonance:

- **NOT "LEVEL2 deferred"** — that claims rotations are safely small, but
  near resonance we cannot show that without drag.
- **NOT "LEVEL2 required"** — that claims rotations are large, but the
  drag-free number that says so is unphysical.
- **Validity domain:** off resonance (`T >= 10 s`, the energetic band)
  the linear model is valid and rotations are small (`A_crit = 5.4 m`);
  near the `3.1-3.3 s` band the measurement is **undetermined** pending
  drag.

**Dependency chain:** `drag capability (coupled assembly) ->
rotational-drag characterisation (tank) -> re-measure rotation -> decide
LEVEL2`. **LEVEL2 nonlinear-restoring is subordinate to drag** — drag
gates the very measurement LEVEL2's decision consumes. Drag moves from
**DEFERRED (Q8) to REQUIRED (M11)** — the *capability* (a coupled drag
`state_force`), distinct from `Cd` *calibration*, which stays a tank
input.

---

## S5 — Tracker dispositions (`docs/phase2-followups.md`)

- **`INBAND-ROTATIONAL-RESONANCE`** — opened at PR2 (first directional-wave
  activation of the coupled model); updated at close with the
  working-joints resolution, the Q2 confirmation (uncertainty removed,
  `T_rot` single-valued), and the campaign implications. Depends on M11
  drag capability.
- No M10 entry was *closed* by M10 (the milestone opened one and left it
  open by design — it is an M11 dependency). `LEVEL2-INTEGRATOR-UNWIRED`
  remains open and is now explicitly **subordinate to drag** (S4).

---

## S6 — Deviations from plan / process

- **Four capability PRs beyond the planned "assembly + two gates"**
  (PR0/PR0.5/PR0.75/PR0.85), each surfacing a §13 cross-module finding
  (S1). The Q8 estimate did not anticipate them; the M6-closure "3-4x
  per-PR" multiplier held again.
- **Q6 fork resolved SUPPORTED, then escalated.** The plan framed PR2 as
  "run a wave case or ship PROVISIONAL." Wave forcing *was* delivered,
  but the measurement changed the LEVEL2 recommendation and promoted drag
  to a required M11 capability (A4, program-plan amendment). The
  frequency-domain alternative was rejected and is now **permanently
  settled** by the amplitude-dependent (drag) physics (program-plan 2d).
- **Pre-existing F2 hypothesis red** (`test_connector_attachment_transform.py
  ::test_property_F_ref_equals_T_pullback_of_F_attach`) — the lone
  suite failure throughout M10, unrelated (M7 connector transform;
  hypothesis corner-case bound). Tracked `F2-HYPOTHESIS-TOLERANCE-EMPIRICAL`.
- **Carried fix- debt** (M9-closure S6): black conformance (3 files) and
  the F2 magnitude-scaled hypothesis-red bound — non-blocking, tracked
  for their own branches.

---

## S7 — What M10 hands forward to M11

1. **A restructured, staged validation strategy** (program-plan 2a): the
   tank campaign yields **cluster-scale** wave-response data before any
   12-buoy data — calibrate + validate at 3-buoy, then scale to 12-buoy.
   Stronger than the planned single jump.
2. **Drag escalated to an early M11 capability** (2b): required to
   *predict the cluster tests*, not only to resolve LEVEL2. The
   wave-height sweep **is** the drag experiment (2c: response-per-height
   falls near resonance — a falsifiable prediction the tool makes now).
3. **`T_rot = 3.257 s` as the programme's first external check** (A5(a)):
   a falsifiable tool-vs-tank prediction before the 12-buoy comparison.
4. **The rotational-`Cd` open question** (2f): the heave-plate `Cd = 5.0`
   (disc broadside to vertical flow) **cannot** be reused for the
   rotational mode (plate edge-on, moving horizontally) — different flow
   regime, same geometry. The tank's rotational decay is **essential,
   not confirmatory**; the drag-widened resonance bandwidth is unknown
   until measured.
5. **The campaign recommendation** (2e) for Xabier to forward: fine
   period sweep bracketing both resonances (core `[3.05, 3.30] s` at
   `<= 0.025 s` = the drag-free half-power width; shoulder `[2.8, 3.6] s`
   at `~0.1 s`; coarse elsewhere) x multiple wave heights.
6. **Architecture confirmed** (2d): time-domain regular-wave runs per
   `(H, T)` — the amplitude-dependent physics rules out a linear
   frequency-domain solver permanently.
7. **The `n`-scaling question** (S3): the 12-buoy `n = 72` KKT is
   unmeasured; M10 informs `m`-scaling only.
