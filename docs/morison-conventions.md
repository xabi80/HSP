# Morison Drag Conventions

User-facing reference for `floatsim.hydro.morison.MorisonElement` and
`make_morison_state_force`. Covers the formulas, the body-frame node
convention, the member-normal projection, and the `include_inertia`
flag that prevents BEM double-counting. The module's docstring carries
the same content; this page is the "I want to add drag to my deck,
what do I need to know" reference.

## Drag formula — force per unit length

For a slender Morison cylinder of hydrodynamic diameter `D`, the
drag force per unit length is:

```
dF_drag/dl = 0.5 * rho * D * Cd * |u_n| * u_n
```

where `u_n` is the component of the relative flow `u_rel = u_fluid - u_body`
**projected onto the plane normal to the member axis**. For an axis
unit vector `e_axis = (node_b - node_a) / |node_b - node_a|`:

```
u_n = u_rel - (u_rel . e_axis) * e_axis
```

This is the standard textbook form (Faltinsen 1990, Ch. 4). The
projection is the **bug-catcher**: a member aligned with the flow
contributes negligible drag, and the full-vector formula
`dF/dl = 0.5*rho*D*Cd*|u_rel|*u_rel` is wrong for arbitrary
orientations.

The total force on a member is computed by midpoint quadrature:

```
F_drag = (dF_drag/dl)|_{midpoint} * L
```

with `L = |node_b - node_a|`. Midpoint quadrature is exact for
constant flow, accurate to `O(L^2)` for linearly-varying flow, and
appropriate in the slender-body limit `L << wavelength`. Members
longer than ~ a quarter-wavelength should be discretized into
sub-elements.

## Inertia formula (when `include_inertia=True`)

When the element flag is set, an additional Froude-Krylov +
added-mass term is applied:

```
dF_inertia/dl = rho * A_x * (1 + Ca) * a_fluid_n  -  rho * A_x * Ca * a_body_n
```

with `A_x = pi * D^2 / 4` the cross-sectional area. The sign and
factors match Sarpkaya (2010) Ch. 3.

The body acceleration `a_body_n` is **always taken as zero in Phase 1**
because the integrator's `state_force` callable only receives
`(t, xi, xi_dot)` — `xi_ddot` is not available. See the module
docstring "Body acceleration in the inertia term" for the full
rationale; the short version is that the dominant `include_inertia`
use case (a non-BEM Morison-only body) has the `Ca * a_body_n` term
already accounted for via the body's own mass on the LHS, so this
explicit-treatment limit is the correct Phase 1 default.

## The `include_inertia` rule — avoid BEM double-counting

**Default: `include_inertia=False`.** Most FloatSim bodies have a BEM
database (WAMIT, Capytaine, OrcaFlex VesselType). The BEM `A(omega)`
and `F_exc(omega)` already cover the Froude-Krylov + added-mass
inertia. Adding the Morison inertia on top **double-counts**.

When the user sets `include_inertia=True` on a member attached to a
BEM-equipped body, `floatsim.hydro.morison.startup_inertia_double_count_warnings`
emits a warning naming the offending element. The deck loader and CLI
surface these messages before integration starts; the integrator
itself is silent.

The warning resolves either way:
- **Set `include_inertia=False`** if the BEM data covers inertia
  already (most common).
- **Remove the BEM database from the body** if the body is genuinely
  Morison-only (a slender column or brace with no BEM solve).

## Body-frame node convention

Each `MorisonElement` is defined by two body-frame node coordinates:

```
node_a_body, node_b_body : Vec3
```

These are positions relative to the body's reference point, in
metres, in the body frame. They rotate with the body's orientation
each step. The integrator's generalized state
`xi = (surge, sway, heave, roll, pitch, yaw)` is decomposed into
translation `xi[0:3]` and ZYX-intrinsic Euler angles `xi[3:6]`. The
rotation `R(q)` is built via
`floatsim.bodies.rigid_body.quaternion_from_euler_zyx`.

The midpoint and axis vector in the inertial frame are:

```
mid_inertial  = r_ref + R @ (0.5 * (node_a + node_b))
axis_inertial = R @ (node_b - node_a)
```

The **order** of the two nodes is irrelevant for the drag formula
(it determines the sign of `e_axis`, but `e_axis` only ever appears
inside `(u . e_axis) * e_axis`, which is sign-invariant).

## Generalized-force assembly

The drag (and optional inertia) force at the midpoint is in the
inertial frame. It maps to a 6-DOF generalized force on the body's
reference point as:

```
F_translation = F_inertial                                   (3,)
F_rotation    = (mid_inertial - r_ref_inertial) x F_inertial (3,)
```

The 6-vector goes into the body's slot of the global `state_force`
output consumed by `floatsim.solver.newmark.integrate_cummins`.

For multiple elements on the same body, `make_morison_state_force`
caches the body's pose (translation + rotation matrix) per step so
the trig only happens once even when the body has many members.

## Wave kinematics

Phase 1 uses **linear Airy with no stretching, clipped at MWL**:

- Below the still water level (`z < 0`): the standard Airy
  exponentially-decaying field.
- Above the still water level (`z > 0`): the depth-decay factor
  `e^{kz}` is clamped to `e^0 = 1` (no stretching). This
  overestimates kinematics in the crest and underestimates in the
  trough — it is the correct linear-theory reference Phase 1
  calibrates against.

`floatsim.waves.kinematics.airy_velocity` and `airy_acceleration`
implement this. A `# TODO(phase-2): Wheeler stretching` comment in
the module marks the natural injection point for a Wheeler-style
position-mapping refinement (`z' = z * h / (h + eta)`).

For calm-sea tests pass `lambda p, t: np.zeros(3)` as the fluid
velocity callable; for a regular Airy wave wrap
`functools.partial(airy_velocity, wave)` (signature
`(point, t) -> vec3`).

## Validation

PR4 unit tests (`tests/unit/test_morison.py`, 33 cases) cover the
four red-tests from the M5 plan plus 45-degree members,
body-translation cancellation, body-velocity-doubles-relative-flow,
moment-about-reference, oscillating-flow inertia,
body-acceleration-subtracts, axis-aligned acceleration, multi-body
isolation, and yaw rotating a member into the flow plane.

PR5's free-decay validation gate
(`tests/validation/test_m5_drag_free_decay.py`) is the system-level
integration test: a heave-only single-body deck with a horizontal
drag plate matches the closed-form hyperbolic envelope
`xi_n = xi_0 / (1 + n * xi_0 * delta)` to `rtol=1e-4` over five cycles,
and a discrimination test asserts the envelope shape is genuinely
hyperbolic, not exponential.

## References

- Morison, J.R., O'Brien, M.P., Johnson, J.W., Schaaf, S.A., 1950.
  "The force exerted by surface waves on piles." Petroleum
  Transactions AIME 189, 149-154.
- Faltinsen, O.M., 1990. *Sea Loads on Ships and Offshore
  Structures*. Cambridge University Press. Ch. 4 (Morison's
  equation, KC number, drag-vs-inertia regime diagram).
- Sarpkaya, T., 2010. *Wave Forces on Offshore Structures*.
  Cambridge University Press. Chs. 3-5 (drag/inertia coefficients,
  separated flow).
