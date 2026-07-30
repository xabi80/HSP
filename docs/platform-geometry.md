# 12-buoy platform — geometry & mass (SOURCE DOCUMENT)

**Purpose.** Single source of truth for the M11 12-buoy platform geometry
and mass. **§1 (supplied) and §2 (record) are inputs — do not alter.**
**§3 is derived from §1+§2** (arithmetic shown); **§4 lists the pending
confirmables.** The M11 plan ([`m11-platform-plan.md`](m11-platform-plan.md))
and future sessions **cite this document** rather than carrying the
numbers, so everything downstream is re-derivable.

---

## §1 — SUPPLIED BY XABIER (2026-07-29) — platform-level, unaltered

Supplied verbatim in the M11 Phase-1 session (image + statement),
2026-07-29:

| # | supplied fact | as given |
|---|---------------|----------|
| S1 | cluster count | **4 clusters of 3 buoys each** (12 buoys) |
| S2 | cluster placement | **each cluster centre is 1 m from the platform centre** |
| S3 | cluster↔platform connection | **each cluster is connected to the platform arm through an articulated joint** |
| S4 | platform arms | **the platform arms (central cross-truss) are RIGID** |
| S5 | structural mass budget | **the mass of the cluster arms and platform is < 60 kg in total** |
| S6 | overall configuration (image) | a rigid central cross-truss with 4 clusters of 3 vertical spar-buoys (heave plates at the base) suspended from the four arm-ends via articulated joints |
| S7 | cluster↔platform articulation type (2026-07-29) | **the cluster↔platform joint is `yaw_locked`** — the same articulation as buoy↔hub (R3). Resolves confirmable C4-b. |
| S8 | platform structure waterline (2026-07-29) | **the platform structure (cross-truss + arms) is fully ABOVE water at equilibrium** — contributes structural mass only, no buoyancy or hydrodynamics. Was inferred from S6 in the Phase-1 draft; now supplied. |

*Note on S6:* the image itself is not committed here (only this textual
transcription of what it shows). The numeric platform data is S1-S5, S7, S8.

---

## §2 — FROM THE COMMITTED RECORD (M10 3-buoy cluster) — intra-cluster, unaltered

Not supplied fresh this session; carried from the committed cluster study.
Cited to source so it is re-derivable:

| # | quantity | value | source (file:line) |
|---|----------|-------|--------------------|
| R1 | intra-cluster buoy radius | 0.5 m | `studies/cluster-3buoy-rigid/cluster_common.py:44` |
| R2 | intra-cluster buoy angles | [0, 120, 240]° | `cluster_common.py:47` |
| R3 | intra-cluster joint type | `yaw_locked` (3 translations + yaw locked; free roll/pitch) | `docs/m9-joints-plan.md:170`; M10 realization `m10-articulated3-plan.md:119` |
| R4 | per-buoy mass | 28.67 kg | `cluster_common.py:32` |
| R5 | per-buoy inertia | diag(24, 24, 0.114) kg·m² | `cluster_common.py:34-35` |
| R6 | cluster arm mass (per cluster) | 12 kg (3 rods × 4 kg) | `cluster_common.py:40,49` (`M_CLUSTER = 3·28.67 + 12 = 98.01`) |
| R7 | cluster arm-hub inertia | diag(0.5, 0.5, 1.0) kg·m² | `m10-articulated3-plan.md:127` (Q2) |
| R8 | measured rotational mode (M10) | T_rot = 3.257 s, ζ = 0.373 %, Q ≈ 134 | `m10-articulated3-plan.md` A4; `docs/phase2-followups.md` INBAND-ROTATIONAL-RESONANCE |

---

## §3 — DERIVED (from §1 + §2; arithmetic shown) — NOT supplied

Marked derived. Each line is reproducible from §1/§2 plus the confirmable
assumptions (§4).

### §3.1 Geometry
- **Cluster-centre positions** (S2 radius 1 m; **ASSUMES C4-a: 90° spacing**):
  `(±1, 0)`, `(0, ±1)` m.
- **Max buoy radius** from platform centre = R (S2) + r (R1) = `1 + 0.5 =
  **1.5 m**`.
- **Footprint** ≈ `2 × (1.5 + 0.215 plate) ≈ **3.43 m** diameter`.

### §3.2 Body / DOF / constraint count
- **Bodies = 17:** 12 buoys (hydro) + 4 cluster-hubs (dry, R6/R7) + 1
  platform cross (**dry — confirmed fully above water, S8**). ⇒ **n = 6 ×
  17 = 102 DOF.**
- **Joints = 16 IDENTICAL `yaw_locked`** (**per S7, resolved**): 12
  intra-cluster (4 × 3 buoy↔hub, R3) + 4 cluster↔platform (S3+S7 — same
  articulation as buoy↔hub, so the platform is **topologically uniform**).
  At 4 rows each: **m = 16 × 4 = 64 constraints.** *(Capability
  consequence: M9's `yaw_locked` constraint-Jacobian builder covers the
  ENTIRE platform — no new joint physics in M11.)*
- **Free DOF = n − m = 102 − 64 = 38** = platform 6 + 4 hubs × 2 rot +
  12 buoys × 2 rot.

### §3.3 Mass
- **Cluster arms** = 4 × 12 (R6) = **48 kg**.
- **Platform** = **≈ 10 kg (ASSUMES C4-c**: a light platform within S5;
  `48 + 10 = 58 < 60` ✓).
- **Total floating mass** = 12 × 28.67 (R4) + 48 + 10 = **402.04 kg**.
- **Per-buoy support** = 402.04 / 12 = **33.5 kg** (vs M10 cluster
  32.67 kg/buoy, **+2.6 %** → buoys sit slightly deeper → the 12-buoy BEM
  mesh must be built at the **platform** equilibrium draft, not the
  cluster draft).

### §3.4 Rotational-mode census (derived; drives the drag scope)
The cluster↔platform `yaw_locked` articulation (S7) yields **three**
rotational families in the wave band:
1. **buoy-vs-hub** roll/pitch — 12 buoys × 2 = 24 (the M10 mode, R8,
   T_rot = 3.257 s).
2. **cluster-vs-platform** roll/pitch — 4 clusters × 2 = **8** (NEW;
   **confirmed real by S7** — the `yaw_locked` joint frees each cluster's
   roll/pitch relative to the platform).
3. **platform rigid-body** — 6.
⇒ several lightly-damped rotational modes → **strengthens drag-REQUIRED**
(M11 plan Q3) and multiplies the rotational-`Cd` characterisation.

### §3.5 Compute (from M11 Phase-1 Measurements C, D)
- **BEM: 12 × 1488 = 17,856 panels** (matches the record). Measured peak
  ≈ 9.3 GB (2× single influence matrix) vs 34.1 GB free → **not
  memory-bound**; ~25-40 min (13-ω grid).
- **KKT** at (n + m) = 166 projects to **~84 µs = 0.4 % of a step** → B6
  sparse treatment stays deferred even at n = 102.

### §3.6 Inter-cluster spacing (derived under C4-a; verified 2026-07-29)
Cluster centres at (±1, 0), (0, ±1) m; buoys at 0/120/240° radius 0.5 m,
identical orientation. Buoy coordinates (adjacent clusters A@(1,0), B@(0,1)):
```
A -> (1.5, 0), (0.75, +0.433), (0.75, -0.433)
B -> (0.5, 1), (-0.25, 1.433), (-0.25, 0.567)
```
- **Closest cross-cluster buoy pair = 0.620 m** (A@(0.75,0.433) ↔
  B@(0.5,1)); **adjacent cluster-centre separation = 1.414 m**.
- **Intra-cluster spacing = 2 × 0.5 × sin60° = 0.866 m.**
- ⇒ the platform's **tightest hydrodynamic pairs are INTER-cluster,
  ~28 % closer (0.620 vs 0.866 m)** than anything the cluster study
  measured, and **inter-cluster coupling is UNMEASURED**.

**Consequence.** The program plan's intra-cluster **R = 1.011** (G2, added
mass at 0.866 m) is a **lower bound** on platform coupling, not
representative. The cluster study measured the **damping** coupling at
0.866 m as **B33 ×8.68** (G6) — so the stakes sit in **B33 (radiation
damping), not added mass**. First real inter-cluster number comes from
**M11a PR3** (the 2-cluster probe rebuilt at the actual 1.414 m centre
separation; 8928 panels already timed at 0.75 s/problem, Measurement C).

**Design observation (flag for Xabier — NOT a change).** The min
cross-cluster gap depends on cluster **orientation**: sweeping a common
rotation of all clusters, it ranges **0.557 m (40°) → 0.728 m (20°)**; the
assumed identical-0° orientation (C4-a) gives 0.620 m — **not** the
maximum. A ~20° rotation would widen the tightest pair ~17 %, reducing
inter-cluster coupling. Worth raising before the mesh is fixed.

---

## §4 — CONFIRMABLES

### Resolved (2026-07-29)
- **C4-b — RESOLVED → S7.** The cluster↔platform articulation is
  `yaw_locked` (m = 64, 38 free, the 8-mode cluster-vs-platform family is
  real). **Framing correction:** the alternative was **never "rigid"** —
  S3 already supplies *articulated*, and S4's rigidity refers to the
  **arms** (the cross-truss members), a different object from the joint.
  The genuine open question was **which articulation** (yaw_locked vs
  hinge vs ball); S7 closes it to `yaw_locked`. (The "rigid → m = 72,
  30 free" line in the Phase-1 draft was an illustrative what-if, not a
  live option.)
- **Platform-above-water — RESOLVED → S8.** The platform structure is
  fully above the waterline at equilibrium (was inferred from S6 in
  Phase-1; now supplied) — confirms the platform cross is a **dry**
  structural mass (§3.2/§3.3), no buoyancy or hydrodynamics.

### Still open (assumption in force + effect if wrong)
| id | assumption IN FORCE | if wrong → |
|----|---------------------|-----------|
| **C4-a** | 4 clusters at **90° spacing**, identical orientation (C4v point-group) | cluster/buoy positions and the BEM mesh change; the plan's symmetry-reduction option changes; also the min cross-cluster gap (§3.6) — orientation is a design lever, gap ranges 0.56–0.73 m |
| **C4-c** | arms/platform split = **48 kg arms (R6) + ~10 kg platform**; S5 fixes only the **sum < 60 kg** | a different split shifts total mass and the per-buoy draft (§3.3); the buoyancy balance / mesh draft re-derive |

---

## Provenance summary
- **Supplied (do not alter):** §1 (Xabier 2026-07-29, S1-S6), §2 (committed
  M10 record, R1-R8).
- **Derived (arithmetic shown, re-derivable):** §3.
- **Open:** §4 (C4-a/b/c). The M11 plan **cites this document**; when a
  confirmable resolves, update §4 here and the derivations in §3 follow.
