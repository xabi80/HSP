# FloatSim — Architecture Specification

**Version:** 0.2 (Phase 1, decisions locked)
**Author:** Xabier
**Purpose:** Internal tool for time-domain simulation of floating platforms (multi-body, 6-DOF each), consuming BEM hydrodynamic databases from OrcaWave, WAMIT, or NEMOH/Capytaine.

---

## 1. Scope

### 1.1 In scope for Phase 1
- Rigid bodies, 1 to N, 6 DOF each
- Multi-body coupling (rigid constraints, linear springs, or mooring connections between bodies)
- Linear hydrostatic restoring
- First-order wave excitation (regular waves, single heading)
- Radiation forces via Cummins' equation (added mass + retardation convolution)
- Viscous drag via user-defined Morison drag elements
- Mooring: analytic catenary (Irvine) + linear springs
- Pluggable BEM database readers: OrcaWave, WAMIT, NEMOH/Capytaine
- Time integration: generalized-α (primary), RK4 (debug)
- Input: YAML deck
- Output: HDF5 time histories, basic post-processing

### 1.2 Out of scope for Phase 1 (deferred)
- Irregular seas (JONSWAP, PM) — Phase 2
- Second-order drift forces (QTFs) — Phase 2
- FE mooring lines (co-rotational beams) — Phase 2
- Wind/current loading — Phase 3
- Contact mechanics, seabed interaction — Phase 3
- GUI — Phase 4
- Parallelization beyond NumPy vectorization — Phase 4

---

## 2. Governing Equations

### 2.1 Cummins' equation (per body, in inertial frame)

$$
[M + A_\infty]\ddot{\xi}(t) + \int_0^t K(t-\tau)\dot{\xi}(\tau)\,d\tau + C\,\xi(t) = F_{exc}(t) + F_{visc}(\dot{\xi}, t) + F_{moor}(\xi) + F_{ext}(t)
$$

Where:
- `ξ` = 6-DOF displacement from equilibrium (surge, sway, heave, roll, pitch, yaw)
- `M` = 6×6 rigid-body mass/inertia matrix
- `A∞` = 6×6 infinite-frequency added mass
- `K(t)` = 6×6 retardation kernel (impulse response of radiation)
- `C` = 6×6 hydrostatic restoring matrix
- `F_exc` = first-order wave excitation (from RAO × wave elevation)
- `F_visc` = viscous drag (Morison elements)
- `F_moor` = mooring reaction forces
- `F_ext` = any external forces (user-defined)

### 2.2 Multi-body generalization

For N bodies, global state: `Ξ = [ξ₁ᵀ, ξ₂ᵀ, …, ξ_Nᵀ]ᵀ` of size 6N.

- `M_global`, `A∞_global`, `C_global`, `K_global(t)` are 6N×6N block matrices.
- Off-diagonal 6×6 blocks in `A∞` and `K` capture hydrodynamic interaction (only populated if BEM database was multi-body; otherwise block-diagonal).
- Mooring/spring connectors contribute coupling terms to the RHS, not to the mass/stiffness.

### 2.3 Retardation kernel

From radiation damping `B(ω)`:
$$
K(t) = \frac{2}{\pi} \int_0^\infty B(\omega) \cos(\omega t)\,d\omega
$$

Computed once at setup via discrete cosine transform on the BEM frequency grid. Stored as a truncated array (typically 20–60 s of lag).

### 2.4 Convolution evaluation

At each time step:
$$
\mu(t) = \int_0^t K(t-\tau)\dot{\xi}(\tau)\,d\tau \approx \sum_{k=0}^{N_K-1} K_k \cdot \dot{\xi}_{n-k} \cdot \Delta t
$$

Using a circular buffer of past velocities. Cost: O(N_K · 36 · N²) per step — acceptable for N ≤ ~10 bodies.

*Optional future optimization:* state-space approximation of K(t) via Prony or vector fitting (eliminates convolution entirely).

---

## 3. Coordinate Systems and State

### 3.1 Frames
- **Inertial (global):** Z up, origin at mean water level (MWL). Wave heading 0° = waves traveling in +X.
- **Body frame:** origin at body reference point (configurable; typically CoG or waterline centroid). x-forward, y-port, z-up when at rest.
- **Hydrodynamic reference:** BEM databases typically computed about their own origin; FloatSim applies rigid-body transformation to the body reference.

### 3.2 Orientation representation
- **Storage (internal):** quaternion `q = [q₀, q₁, q₂, q₃]` — unit norm, no gimbal lock.
- **Input/output:** Euler angles ZYX intrinsic (yaw-pitch-roll), in degrees or radians per deck setting. Matches OrcaFlex/OrcaWave convention.
- **Small-angle assumption:** the Cummins formulation above is linearized about equilibrium. For large rotations, forces must be transformed via rotation matrix at each step.

### 3.3 Per-body state vector (13 components)
```
s = [x, y, z,        # position of body origin in inertial frame
     q0, q1, q2, q3, # orientation quaternion
     u, v, w,        # linear velocity in body frame
     p, q, r]        # angular velocity in body frame
```

Global state: concatenation over N bodies → `13N`.

---

## 4. Module Layout

```
floatsim/
├── bodies/
│   ├── rigid_body.py           # 6-DOF kinematics, quaternion propagation, transforms
│   ├── mass_properties.py      # Mass matrix assembly, parallel-axis
│   └── connector.py            # Rigid links, springs, linear dampers between bodies
│
├── hydro/
│   ├── database.py             # Abstract BEM database (A(ω), B(ω), C, RAOs, QTFs-stub)
│   ├── readers/
│   │   ├── orcawave.py         # OrcaWave .yml + results
│   │   ├── wamit.py            # .1, .3, .hst, .4
│   │   └── capytaine.py        # NEMOH / Capytaine NetCDF
│   ├── hydrostatics.py         # Restoring matrix, gravity/buoyancy balance
│   ├── radiation.py            # Retardation kernel, convolution, Cummins assembly
│   ├── excitation.py           # First-order wave force from RAOs × wave elevation
│   └── drag.py                 # Morison drag on slender elements
│
├── waves/
│   ├── regular.py              # Airy wave kinematics at arbitrary point
│   ├── irregular.py            # [Phase 2] JONSWAP, PM, sum-of-cosines realization
│   └── kinematics.py           # Wheeler stretching, particle velocities
│
├── mooring/
│   ├── linear_spring.py        # 6-DOF spring between body and earth, or body-body
│   └── catenary_analytic.py    # Irvine closed-form elastic catenary
│
├── solver/
│   ├── state.py                # Global state vector assembly/disassembly
│   ├── rhs.py                  # Residual/force assembly (Cummins RHS)
│   ├── newmark.py              # Generalized-α integrator
│   └── rk4.py                  # Explicit RK4 (debug/reference)
│
├── io/
│   ├── deck.py                 # YAML deck schema + validation (pydantic)
│   └── results.py              # HDF5 writer/reader
│
├── post/
│   ├── timeseries.py           # Extraction, statistics
│   └── spectra.py              # PSDs, response spectra
│
└── validation/
    ├── cases/                  # Canonical benchmarks
    ├── analytical.py           # Closed-form references
    └── run_all.py              # Regression runner
```

---

## 5. Input Deck Schema (YAML)

Example — two-body semi-submersible + barge sharing a mooring:

```yaml
simulation:
  duration: 600.0              # s
  dt: 0.05                     # s
  integrator: generalized_alpha
  spectral_radius_inf: 0.8     # numerical damping
  retardation_memory: 60.0     # s (convolution truncation, §9.1)
  ramp_duration: 20.0          # s (excitation ramp-up, §9.3)
  skip_static_equilibrium: false  # debug only (§9.4)

environment:
  water_depth: 200.0           # m
  water_density: 1025.0        # kg/m³
  gravity: 9.80665

waves:
  type: regular
  height: 6.0                  # m
  period: 10.0                 # s
  heading: 0.0                 # deg

bodies:
  - name: semisub
    reference_point: [0, 0, -15.0]
    mass: 2.0e7
    inertia:                   # about reference_point, body frame
      Ixx: 3.0e10
      Iyy: 3.0e10
      Izz: 5.0e10
    hydro_database:
      format: orcawave
      path: ./hydro/semisub.yml
      body_index: 0
    drag_elements:
      - type: morison_member
        node_a: [ 20, 0, -10]
        node_b: [ 20, 0, -20]
        diameter: 8.0
        Cd: 0.8
        Ca: 1.0
    initial_conditions:
      position: [0, 0, 0, 0, 0, 0]
      velocity: [0, 0, 0, 0, 0, 0]

  - name: barge
    reference_point: [100, 0, -5]
    mass: 5.0e6
    inertia: { Ixx: 1.0e9, Iyy: 4.0e9, Izz: 4.0e9 }
    hydro_database:
      format: wamit
      path: ./hydro/barge
      body_index: 0

connections:
  - type: linear_spring
    body_a: semisub
    body_b: earth
    anchor_a_body: [30, 0, 0]
    anchor_b_global: [150, 0, -200]
    stiffness: 1.0e6
    rest_length: 180.0

  - type: catenary
    body_a: semisub
    body_b: barge
    attach_a_body: [ 30, 0, 0]
    attach_b_body: [-40, 0, 0]
    line:
      length: 120.0
      weight_per_length: 800.0    # N/m in water
      EA: 5.0e8

output:
  file: results.h5
  channels: [position, velocity, mooring_tensions, wave_elevation]
  sample_rate: 10.0            # Hz
```

---

## 6. Key Design Principles

1. **Pure-NumPy core, no global state.** Every solver function takes inputs, returns outputs. Pydantic for deck validation.
2. **BEM database is an interface, not an implementation.** All readers produce the same `HydroDatabase` object: `A(ω)`, `B(ω)`, `C`, `RAO(ω, heading)`, metadata. Core never checks origin.
3. **Vectorize across bodies, not across time.** Time-stepping is inherently sequential; inner-loop ops (convolution, force assembly) use dense NumPy.
4. **Test against analytical solutions, not just regression.** Every new module ships with a benchmark having a closed-form reference.
5. **HDF5 for everything persistent.** Human-readable summary files next to binary results.

---

## 7. Validation Plan (Phase 1)

| Case | What it tests | Reference |
|------|---------------|-----------|
| Free decay (heave, single body) | Restoring, radiation damping, Cummins convolution | Exponential decay + natural period from √(M+A∞)/C |
| Free decay (pitch) | Cross-coupling in C and A | Same, with coupled DOFs |
| Regular wave, fixed body | Excitation force magnitude | Direct RAO × wave amplitude |
| Regular wave, free body (single) | Full Cummins loop | RAO-based steady-state amplitude |
| Two bodies, rigid connector | Multi-body assembly, constraint handling | Combined-mass equivalent body |
| Two bodies, mooring line | Catenary restoring | Static catenary analytic |
| OrcaFlex cross-check | Full system | Output from licensed OrcaFlex for same deck |

---

## 8. Development Roadmap

### Milestone 0 — Skeleton (week 1)
Repo structure, pydantic deck schema, HDF5 writer stub, CI with pytest. No physics yet.

### Milestone 1 — Single-body frequency-domain sanity (week 2)
Implement `HydroDatabase` dataclass, reader dispatch interface, and an internal synthetic-YAML reader to drive tests. Rigid-body mass matrix assembly. Cummins LHS matrix assembly (M + A∞, C). Verify natural-period formula `T_i = 2π√((M+A∞)_ii / C_ii)` on a synthetic OC4 DeepCwind–shaped fixture reproduces published reference periods. No time stepping yet.

### Milestone 1.5 — OrcaFlex VesselType YAML reader (primary path)
Implement `floatsim/hydro/readers/orcaflex_vessel_yaml.py`. This is the human-readable text export produced by OrcaFlex when an OrcaWave `.owr` is imported and saved as YAML — it carries the full set of computed coefficients FloatSim needs: `FrequencyDependentAddedMassAndDamping` (A(ω), B(ω), and the `Infinity` row for A∞), `HydrostaticStiffness` (C), `DisplacementRAOs` and `LoadRAOs`, plus body mass/inertia. Fields honour the file's `UnitsSystem`, `WavesReferredToBy`, `RAOPhaseConvention`, and `RAOPhaseUnitsConvention` declarations; the reader validates these and converts to FloatSim's SI + rad/s + complex-RAO-with-`exp(-i·omega·t)` convention.

Direct `.owr` binary parsing via `OrcFxAPI` is deferred — demo-mode OrcFxAPI refuses API access and cannot be used. A future `floatsim/hydro/readers/orcawave_owr.py` can land behind a lazy import when a full FlexNet/HASP license is available.

Validation case: `PlatformOrcaflexSmall.yml` (OC4 DeepCwind–shaped semi-submersible demo), reproducing heave and pitch natural periods in the physical ranges established in Milestone 1.

### Milestone 2 — Single-body time domain (weeks 3–4)
Retardation kernel, convolution buffer, generalized-α integrator. Validate: heave free decay matches analytical period/damping.

### Milestone 3 — Waves + excitation (week 5)
Regular Airy waves, excitation from RAOs. Validate: single body in regular waves matches steady-state RAO response.

### Milestone 4 — Multi-body + mooring (weeks 6–7)
Global assembly, connectors, catenary. Validate: two-body rigid-link case, then moored case.

### Milestone 5 — WAMIT + Capytaine readers, drag (week 8)
Pluggable readers. Morison drag elements. Full Phase 1 validation suite.

### Milestone 6 — OrcaFlex cross-check (week 9)
Set up a representative deck in both tools. Document discrepancies.

---

## 9. Resolved Design Decisions

These four were open during architecture review and are now locked for Phase 1.

### 9.1 Convolution truncation
- **Strategy:** fixed truncation length, configurable per simulation via the deck.
- **Default:** `retardation_memory: 60.0` seconds.
- **Diagnostic:** at setup, warn if `|K(t_max)| > 0.01 · max(|K(t)|)` for any DOF — this indicates the buffer is too short for the case.
- **Deferred:** state-space (Prony / vector fitting) approximation of `K(t)` → Phase 2 performance optimization.

### 9.2 Fidelity level: rotated-frame hydro with nonlinear restoring (Level 2)
- Hydrostatic restoring is computed from **instantaneous position and orientation** (nonlinear, but analytically cheap for simple shapes; for general bodies use a precomputed restoring stiffness about equilibrium plus a gravity/buoyancy correction).
- Radiation and excitation forces use **linear BEM coefficients evaluated in the body frame**, then rotated to inertial via the current quaternion.
- Yaw is a free DOF with no artificial restoring — mooring provides the only yaw stiffness.
- This matches OrcaFlex's standard large-angle formulation and is the industry norm for floating platform time-domain simulation.

### 9.3 Simulation startup
- **Convolution initialization:** assume `ẋ(τ) = 0` for `τ < 0`. Retardation buffer starts empty.
- **Excitation ramp:** mandatory. Wave forces multiplied by smooth envelope `r(t)` where:
  - `r(0) = 0`, `r(t ≥ T_ramp) = 1`
  - Use half-cosine: `r(t) = 0.5 · (1 - cos(π · t / T_ramp))` for `t < T_ramp`
- **Default:** `ramp_duration: 20.0` seconds (override in deck; recommend ≥ 2× longest wave period).
- Post-processing tools must be aware of ramp duration for statistics windowing.

### 9.4 Static equilibrium
- **Mandatory pre-step** before any dynamic simulation. Solves:
  $$
  F_{gravity}(\xi_{eq}) + F_{buoyancy}(\xi_{eq}) + F_{moor}(\xi_{eq}) = 0
  $$
  via `scipy.optimize.root` (Levenberg-Marquardt or hybrid Powell).
- Time integration begins from `ξ_eq` with zero velocity.
- Outputs equilibrium offset and mooring pretensions to the log for sanity checking.
- Deck override `skip_static_equilibrium: true` available for debugging / specific test cases only.
- Mooring force functions must be written to expose analytical or finite-difference Jacobians for the solver.

---

## 10. Handoff to Claude Code

When moving to Claude Code:

1. Place this `ARCHITECTURE.md` at repo root.
2. Create `CLAUDE.md` with: coding standards (black, ruff, type hints, numpy-style docstrings), test-first policy, numerical tolerances.
3. Start with Milestone 0, one module at a time, each with tests before implementation.
4. Keep this document as the contract — update it when decisions change, don't let code and spec diverge.
