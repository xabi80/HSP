# Study plan — articulated (pin) vs whole-chain-rigid buoy platform

**Question.** Each buoy connects to the platform through the gimbal at the top of its
spar — modelled as a `yaw_locked` KKT joint (3 translations + yaw locked; **roll/pitch
free**). Does that articulation give the platform *better dynamics* than welding the whole
assembly rigid?

**"Better" = a still, level deck.** The target application is a platform steady enough to
**launch/land rockets and host datacenters or hotels**. So the objective is deck stillness:
minimal tilt, minimal heave, minimal acceleration, natural periods parked outside the sea's
energy band. Connection loads are a hard secondary (a rigid raft may be steadier but could
carry punishing joint moments — feasibility).

## The two configurations
16-buoy platform (21 bodies, 126 DOF). Everything held identical between runs — coupled BEM,
hydrostatics, masses/inertias, Morison drag, waves, `dt`, static equilibrium. The **only**
change is the joint type:

| Config | buoy→hub | hub→platform | meaning |
|---|---|---|---|
| **Articulated** (baseline) | `yaw_locked` | `yaw_locked` | buoys & hubs tilt freely; current model |
| **Rigid** (whole chain) | `rigid` | `rigid` | entire assembly is one rigid raft |

Physical essence: both joints lock translation, so both carry buoy wave *forces* into the
platform. The entire difference is whether each buoy's wave-induced **tilting moment** is
transmitted. Articulated = no; rigid = yes.

## Prerequisite — PR1: a `rigid` KKT joint (core, validation-first)
No rigid/weld joint exists (`floatsim/bodies/joints.py` has `hinge`, `yaw_locked` only; the
deck exposes just those). Add the 3-rotation completion of that family:

- **Core:** a `rigid` kind (6 rows = the exact translational block already there + a
  rotational-lock projection that is the full identity on the relative rotation vector, vs
  `hinge`'s 2 perpendicular rows and `yaw_locked`'s 1 axis row). `rigid_joint(...)` factory.
- **Deck:** a `RigidJoint(type="rigid")` class + wire into the discriminated `Joint` union
  and the deck→`JointSet` builder.
- **Spec first:** update `ARCHITECTURE.md` §7/§8 to record the rigid joint and its validation
  case **before** writing code (working-agreement §1, §8).
- **Validation (§7):**
  1. *Two-body weld ⇒ combined-mass equivalent body.* A rigid link between two bodies
     free-decays at the period of a single rigid body with summed mass + parallel-axis
     inertia (`rtol=1e-3`) — the spec's own M4-era reference case.
  2. Energy conservation in undamped free response (`rtol=1e-10`), constraint drift at
     machine noise (the KKT projection), same gates `hinge`/`yaw_locked` pass.
  3. Degenerate: a `rigid` joint between coincident, aligned bodies behaves as one body.

## PR2 — the comparison study (16-buoy)
Build both decks (swap all joints articulated↔rigid), run and compare:

1. **Deck tilt** — max & RMS platform pitch/roll (deg) vs wave period & heading. *The
   rocket-landing number.*
2. **Deck acceleration** — RMS & peak translational (m/s², g) and angular (reuse the
   accel-HT tooling). *The datacenter/hotel number.*
3. **Heave** — max & RMS platform vertical motion.
4. **Resonance placement** — platform heave/pitch/roll natural periods (free-decay) vs the
   design-sea band.
5. **Connection loads** — peak/RMS moment at the welds, from the constrained integrator's
   Lagrange multipliers (`lam`). *Feasibility gate for the rigid raft.*
6. **Articulation used** — buoy tilt relative to platform in the articulated case (how hard
   the gimbals work).

Runs: regular-wave RAO sweep + one irregular design sea, headings 0/45/90°.

## Prior (what the study should resolve)
The rigid raft has large pitch inertia **and** large distributed waterplane restoring (buoys
over the 1.5 m footprint) → potentially very stiff/level in long waves, but driven by every
buoy's wave moment → possible high accelerations near resonance and large joint loads. The
articulated platform decouples buoy tilt and adds low-frequency pendulum modes (each buoy
hangs from its top joint) → may *isolate* the deck from wave-frequency motion, or introduce
lightly-damped sway. Which wins for "still and level" is non-obvious — hence the study.

## Caveat
The joint rotational-lock rows are small-angle (first-order, valid to ~0.1 rad ≈ 6°). The
articulated buoys tilt more, so keep wave heights modest and flag large-angle response as
Phase-2, or note where fidelity degrades.

## Sequencing
- **PR1** — ARCHITECTURE spec update → `rigid` KKT joint + `RigidJoint` deck type → unit
  tests → 2-body combined-mass validation. (Gated: core change.)
- **PR2** — both 16-buoy decks → RAO + free-decay + accelerations + joint loads → figures +
  `PIN-VS-RIGID.md` write-up + a summary deck.
