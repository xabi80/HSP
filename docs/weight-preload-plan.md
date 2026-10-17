# Weight preload: static loads, the joint-constrained static solve and their geometric stiffness

**Status: done (PR1 `05128b7`, PR2 `dd889f2`, PR3 studies); the default stays opt-in.** Written 2026-10-17, branch
`fix-weight-preload`. Tracker entry `WEIGHT-PRELOAD-UNWIRED` (docs/phase2-followups.md).

## 1. The gap

ARCHITECTURE.md §9.4 specifies the static equilibrium as
`F_gravity(ξ_eq) + F_buoyancy(ξ_eq) + F_moor(ξ_eq) = 0`. The code
(`floatsim/solver/equilibrium.py`, `build_system`) solves `C ξ = F_state(0, ξ, 0)` instead:
- there is no gravity or buoyancy vector;
- the joints are ignored (the solve never sees `G`).

Every FloatSim model is therefore built as if each hydro body floated freely at its BEM draft, and
every structural body (hub, deck, payload) were weightless in statics. In dynamics those bodies
do have mass.

**What it loses.** The joints carry no static preload, so the KKT reactions `λ(t)` hold only the
dynamic part. The preload's geometric stiffness is then absent. That is the term M9 showed lives
in `G`'s configuration dependence (m9-joints-plan.md, finding ME); M9's hinge gate supplied
gravity as an external force, which is why the gate swings.

**Where it was measured** (studies/payload-stability, 2026-10-16). A buoy pushed down at its pin
by a load P, the pin L = 1.624 m above its CoG, loses P L of tilt stiffness about its CoG. The
same holds for the deck: its weight sits 0.183 m above the pins.

| Model | Buoys' in-phase rocking period: FloatSim today / with the preload (estimate) |
|---|---|
| Record flume platform, unloaded (hubs 4 × 2 kg, deck 6 kg) | 2.81 / 3.10 s |
| Record flume cluster, unloaded (hub 2 kg) | 2.79 / 2.95 s |
| Cluster with 12.5 kg on the hub | 2.86 / 4.22 s |
| Platform with 50 kg on the deck | 2.86 / 4.38 s |

The estimate added the geometric stiffness of the statics' pin loads to FloatSim's matrices
(`tilted_dynamics.geometric_stiffness`). At 50 kg the missing term is 50–75 % of the loaded
buoys' C55.

## 2. Governing equations

### 2.1 Static loads

For every body k, at its reference point (FloatSim's reference point is the CoG,
`_body_mass_matrix`):
- **weight** `W_k = (0, 0, −m_k g)`;
- for a hydro body, also **buoyancy** at the BEM reference draft, `B_k = (0, 0, ρ g V_k)`.
  - V_k is the displaced volume at that draft.
  - Default `V_k = m_k / ρ`: the body floats freely at its BEM draft, so its net static force is
    zero. This is today's implicit assumption, made explicit.

Both act at the reference point, so neither has a moment arm. Their moments under rotation are
already in `C`: the buoyancy-only hydrostatics plus `gravity_restoring_contribution`'s
`−m g z_G`. Nothing is counted twice. The buoyancy is assumed to act on the vertical through the
reference point at the reference configuration, true of every axisymmetric body in the records.
A general hull needs its centre of buoyancy, which is out of scope here (§5).

`F_static` stacks these as a constant generalized force: `(W_k + B_k)` in each body's
translational rows, zero in its rotational rows.

### 2.2 The joint-constrained static solve (replaces §9.4's implementation)

Unknowns ξ (6N) and the joint multipliers λ (m):

    C ξ + G(ξ)ᵀ λ = F_static + F_state(0, ξ, 0)
    φ(ξ) = 0

solved with `scipy.optimize.root`, in the existing solver's style, from ξ = the deck's initial
position and λ = 0. `G`, `φ` and the sign of λ are M9's (`floatsim/bodies/joints.py`); λ is the
physical joint force (m9 plan, λ-units derivation; Gate 3). Without joints this reduces to
today's solve plus `F_static`.

### 2.3 Dynamics

`F_static` joins the Cummins right-hand side as a constant (composed into `state_force`, so
`integrate_cummins` is unchanged):

    [M + A∞] ξ̈ + ∫K ξ̇ + C ξ = F_exc + F_state + F_static + G(ξ)ᵀ λ(t)

At `(ξ_eq, λ_eq)` the right-hand side balances, so a run at rest stays at rest. As a body
rotates, `G(ξ)` turns the preload `λ_eq` and produces the geometric stiffness: the M9 pendulum
mechanism, now driven by FloatSim's own loads.

### 2.4 Geometric stiffness for modal tools

Tools that work with matrices (mooring_verify-style eigen analyses, the restoring-PSD gate)
cannot see a `G(ξ)ᵀλ` that only appears during integration. `build_system` therefore also returns

    K_geo = − ∂(G(ξ)ᵀ λ_eq) / ∂ξ  at ξ_eq      (central differences, symmetrised)

with the size of its antisymmetric part reported. For loads that keep a fixed direction (gravity,
buoyancy) the symmetric part is the conservative geometric stiffness. The modal stiffness is
`C + K_geo`, and the PSD gate checks `C + K_geo` on the feasible subspace. A load that makes the
system statically unstable, such as a raised payload past its limit, then fails fast, as an
indefinite `C` does today.

References: the geometric stiffness of constraint forces in the linearisation of constrained
multibody systems about a loaded equilibrium (the λ-weighted constraint Hessian). See Géradin &
Cardona, *Flexible Multibody Dynamics: A Finite Element Approach* (Wiley, 2001), and Bauchau,
*Flexible Multibody Dynamics* (Springer, 2011); exact section citations are to be added in PR1.
For the floating body, the `−m g z_G` gravity term is already implemented and cited in
`floatsim/hydro/hydrostatics.py`.

## 3. Validation (derived before any code; never calibrated on FloatSim)

### Gate A — a floating body pinned at its top (new analytic case)

A hydro body (synthetic database: A = 0, constant small B, given C55 about the CoG) is pinned to
earth by a yaw-locked joint at L above its CoG. Heave is held by the pin.
- **Static:** with `V > m/ρ` the body pushes up on the pin with `P = ρ g V − m g`, so the
  recovered `λ` must be `(0, 0, −P)` on the body (the M9 Gate 3 pattern).
- **Small rotation θ about the pin.** The CoG swings horizontally by `L θ` and rises by
  `L θ²/2`. Raising the body against its net upward force P stores `−P L θ²/2`. So:

      (I_yy + m L²) θ̈ = −(C55 − P L) θ,     T = 2π √((I_yy + m L²) / (C55 − P L))

- **Numbers.** m = 21.52 kg, I_yy = 10.2 kg m², L = 1.624 m, C55 = 75.28 N m/rad (the OSU buoy,
  12-panel table):
  - P = +30 N: T = 9.98 s;
  - P = −30 N (hanging below the pin): T = 4.62 s;
  - FloatSim today: 5.93 s for both (no −P L).
- **Gate:** FloatSim's constrained free decay reproduces T at rtol 1e-3 (CLAUDE.md analytic
  tolerance), and the static λ at rtol 1e-10.

### Gate B — the default changes nothing for free hydro bodies

With every `V = m/ρ` and no structural mass, `F_static = 0`. Results must be byte-identical to
today: the M6 / M9 / M11 regression suite.

### Gate C — the articulated cluster against the large-angle statics (cross-check)

The record cluster with a centred payload on the hub, FloatSim against
`studies/payload-stability/cluster_statics.py` on the same 12-panel table, at small loads:
- **static sinkage:** equal (today FloatSim gives 0);
- **joint loads:** λ equals the statics' pin loads;
- **buoy-tilt modal period:** within 1 % of the study's estimate (2.79 → 2.95 s unloaded).

## 4. Delivery (PRs, each test-first)

The plan's statics and dynamics PRs were merged into one. A static solve that sinks the model
under `F_static`, with dynamics that do not carry `F_static`, would start every run away from
equilibrium.

1. **PR1, statics and dynamics.**
   - `F_static` assembly; the deck fields (§6); the joint-constrained static solve returning
     `(ξ_eq, λ_eq)`; `F_static` in the state force.
   - Tests:
     - `tests/unit/test_weight_preload_deck.py`;
     - `tests/integration/test_weight_preload_statics.py`: Gate A static λ and Gate B;
     - `tests/validation/test_weight_preload_pinned_float.py`: Gate A periods and the
       finite-amplitude shift;
     - `tests/validation/test_weight_preload_cluster.py`: Gate C statics.
   - ARCHITECTURE §2.1, §5 and §9.4 updated in the same PR.
   - **Measured (Gate A).** At θ0 = 0.005 rad: −9e-5 (+30 N), −6e-5 (−30 N), −1.2e-4 (off).
     - The fixture's first `B = 1` gave a 0.35 % short period. It was the Kramers–Kronig added
       mass implied by a non-zero `B` with `A = 0`, found and fixed in the fixture (`B = 1e-3`),
       not in the tolerance.
     - The residual scales as θ0², the body rising against C33: measured 3.53–3.63 θ0² against
       Duffing's derived 3.58 θ0², now pinned by its own test.
2. **PR2, modal stiffness.**
   - `K_geo` on `SimulationSetup`, by central differences of `G(ξ)ᵀλ_eq`.
   - A second restoring gate on `C + K_geo` at the preloaded equilibrium.
   - Tests:
     - `tests/integration/test_weight_preload_kgeo.py`: the pinned float's `K_geo` is
       `diag(−P L, −P L, 0)`; `C + K_geo` gives the closed-form ω² to 1e-6; P = 50 N
       (`P L > C55`) fails at build time;
     - `tests/validation/test_weight_preload_cluster.py`: on the flume cluster, the
       finite-difference `K_geo` equals the per-load closed form on the feasible subspace.
   - **Measured.** `K_geo` is symmetric to round-off on the whole matrix (max|K − Kᵀ| ≤ 1.4e-7
     against entries up to 55 N·m/rad). At a static equilibrium each body's net moment is zero,
     and the per-load antisymmetric parts sum to exactly that moment.
3. **PR3, studies: done.** `studies/payload-stability/tilted_dynamics.py --preload`, README
   "With FloatSim's weight preload".
   - FloatSim's constrained static solve keeps the statics geometry to ≤ 4e-15 m in all 8
     cases.
   - Its `C + K_geo` periods equal the study's closed-form estimate exactly.
   - With a payload, the buoys' rocking resonance is 4.2–4.4 s, at 11–12.6° in H = 0.1 m waves
     (FloatSim without the preload: 2.85 s).
   - The unloaded flume platform's rocking mode is 3.10 s (2.81 s without).

## 5. Out of scope

- **LEVEL2 nonlinear restoring** (`LEVEL2-INTEGRATOR-UNWIRED`). `C` stays the BEM-draft matrix.
  A large sinkage still needs the BEM re-run at that draft, as the payload study does.
- **The rotational lock rows' second-order terms.** They are first-order in M9 by design; moment
  preloads, such as yaw-lock reactions, carry no geometric stiffness.
- **Centre-of-buoyancy offsets and non-vertical buoyancy lines.** These matter for general hulls;
  the records are axisymmetric.
- **Reading V from BEM files.** Possible later (Capytaine computes it); see §6 Q2.

## 6. Decisions (Xabier, 2026-10-17)

- **Q1, default: LOCKED (a), opt-in.** A deck switch, `weight_preload: true`. Every existing
  result and test is unchanged until a deck switches it on; the studies opt in one by one.
  Flipping the default is a later decision, once validated.
- **Q2, buoyancy source: LOCKED (a), deck field.** An optional per-body `displaced_volume` (m³)
  for hydro bodies, default `m/ρ`. Set it when the BEM was run at a loaded draft. Reading V from
  BEM files is not done.

## 7. Risks

- **Sign convention of λ.** Pinned by Gate A's static λ before any dynamics.
- **The KKT static solve on structural bodies.** They have no `C`, so their position is fixed
  only by the joints. The root solve needs the joint rows; a free structural body is singular and
  must raise a clear error.
- **Large preloads near instability.** `C + K_geo` near singular; the PSD gate's margin must
  handle that, and Gate A at P → C55/L is the probe.
