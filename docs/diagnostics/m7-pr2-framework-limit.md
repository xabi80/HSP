# M7-Foundation PR2 — Framework-limit diagnostic

**Date.** 2026-06-01.
**PR.** M7-Foundation PR2 — F2 attachment-offset transform.
**Commit.** `54703b7` on `milestone-7-foundation`.
**Tracker entry.** [`docs/phase2-followups.md`](../phase2-followups.md)
§ BB-OFFSET-CONNECTOR.

Discoverable record of a framework constraint surfaced during F2's
derivation. The PR2 commit message captures the punchline; this doc
captures the algebraic derivation for future investigators ("why
exactly can't body-body offsets be represented?") and the why-it-was-
latent observation ("why didn't M2-M6 hit this?").

---

## The constraint, in two lines

`LinearConnector` assumes symmetric Newton-III at reference points
(`F_on_b = -F_on_a` exactly). A non-zero attachment arm makes the
two reference-point generalised forces **asymmetric**, so the
framework's symmetric `K_LC` cannot express the physics.

## The constraint, derived

A connector couples bodies A and B via a 6-DOF stiffness `K_attach`
acting on the relative displacement of their **attachment points**.
Let `r_a`, `r_b` be the body-frame offsets from each body's reference
to its attachment; let `T_a`, `T_b` be the 6x6 small-angle linear
transforms (`T_a = [I, -r_a_tilde; 0, I]` and similarly for `T_b`,
per plan Q3).

The attachment-point relative displacement is::

    delta_attach = T_a @ xi_a - T_b @ xi_b - rest_offset

The force at the attachment (3+3 vector) is `F_attach = -K_attach @
delta_attach`. By virtual-work duality, the generalised forces at
the reference points are::

    F_a_ref = +T_a^T @ F_attach
    F_b_ref = -T_b^T @ F_attach

The sign convention follows from `∂delta_attach/∂xi_a = T_a` and
`∂delta_attach/∂xi_b = -T_b`.

**The asymmetry.** `F_a_ref` and `F_b_ref` are pulled back through
*different* transforms (`T_a^T` vs `T_b^T`). For non-zero arms,
`T_a^T ≠ T_b^T`, so `F_b_ref ≠ -F_a_ref` in general. In particular,
the moment block of `F_a_ref` carries `r_a x F_attach_trans` (the
A-side moment arm), and the moment block of `F_b_ref` carries
`r_b x F_attach_trans` (the B-side moment arm). These are not
negatives of each other unless `r_a = r_b` — which only happens
when both arms are zero (or symmetric by coincidence, which is a
measure-zero subset of configurations).

## What `LinearConnector` actually represents

The existing framework:

```python
delta_LC = xi_a - xi_b - rest_offset
F_on_a   = -K_LC @ delta_LC
F_on_b   = +K_LC @ delta_LC                # Newton III, exact
```

`F_on_b = -F_on_a` follows by construction from the single `K_LC`.
For this to match the physics with non-zero arms, we would need::

    +T_a^T @ K_attach @ T_a = K_LC                     # from F_a_ref's xi_a coeff
    +T_a^T @ K_attach @ T_b = K_LC                     # from F_a_ref's xi_b coeff (signs absorb)
    +T_b^T @ K_attach @ T_a = K_LC                     # from F_b_ref's xi_a coeff
    +T_b^T @ K_attach @ T_b = K_LC                     # from F_b_ref's xi_b coeff

All four expressions equal a **single** `K_LC` only if
`T_a^T = T_b^T`, i.e. `T_a = T_b`. Since `T_a` and `T_b` are uniquely
determined by `r_a` and `r_b` through `T = [I, -r_tilde; 0, I]`,
that requires `r_a = r_b`. The two special cases that satisfy this:

- **Both offsets zero** (`r_a = r_b = 0`, so `T_a = T_b = I`).
  The M4 PR3 reference-to-reference body-body case. Supported.
- **Single offset with the other endpoint at earth.** Earth has no
  reference-point generalised force (it absorbs whatever force the
  framework would apply to it), so the F_b_ref equation drops out
  entirely. The remaining constraint `K_LC = T_a^T @ K_attach @ T_a`
  determines the LinearConnector's `K`. The catenary-fairlead and
  earth-anchored-spring cases. Supported.

## Why the M2-M6 test suite did not surface this

Every existing connector / mooring fixture lives in one of the
supported subsets:

| Fixture | Configuration | Why supported |
|---|---|---|
| `tests/validation/test_m4_two_body_assembly.py` | Block-diagonal, no connectors | No connector at all |
| `tests/validation/test_m4_rigid_link_heave.py` | Body-body `heave_rigid_link` | Both ends at reference (zero offsets) |
| `tests/validation/test_m4_two_body_moored.py` | Body-body heave link + body-earth catenaries | Body-body link has zero offsets; catenaries are body-earth |
| `tests/validation/test_m6_openfast_moored_eq.py` (PR5) | OC4 3-line mooring | All three lines body-earth |
| `tests/validation/test_m7_n4_block_diagonal.py` (F4) | Block-diagonal, no connectors | No connector at all |

The M4 plan Q1's "rigid-link penalty" decision pre-empted the
need for attached-to-offset rigid links by choosing the
penalty-stiffness route; that route gave a 6x6 K at the reference
points (no attachment-point translation needed) and quietly avoided
the constraint. The catenary path went body-earth from M4 PR6
onward, so it never needed body-body offsets either.

The constraint was always there. It was simply never exercised
because no test asked for the un-supported subset.

## Disposition

F2 raises `NotImplementedError` for body-body with any non-zero
offset, with a message pointing at the tracker entry. F2's locked
scope (body-earth single-offset + both-zero-offset degenerate) is
sufficient for everything M7-Foundation needs:

- F3 catenary: body-earth (fairlead to anchor).
- F1 driver for `RigidLink` deck entries: both ends at reference
  (heave-only at M4 PR3).
- F1 driver for `Catenary` deck entries: body-earth (Q4-locked
  at M7 PR3).
- F1 driver for `LinearSpring` deck entries: the deck schema
  carries both `attach_a_body` and `attach_b_body`, so F1 must
  decide. **Locked at 2026-06-01**: F1 raises
  `NotImplementedError` on body-body with any non-zero offset,
  citing BB-OFFSET-CONNECTOR. See plan Q9.

## Resolution paths (for the future investigator)

Per the tracker entry:

1. **Direct.** Extend `LinearConnector` to per-endpoint K factors.
   ~1-2 weeks framework surgery. Ripples through
   `connector_drift`, `make_connector_state_force`, and any code
   that reads the existing 6x6 K shape.
2. **Free emergence from B2.** The Lagrange-multiplier DAE
   formulation handles the asymmetry naturally — different arms on
   each side simply contribute different rows to the constraint
   Jacobian, and the multipliers ensure Newton-III at the
   attachment point in the inertial frame (where it actually
   holds), not at the reference points.

The B2 path is cleaner if B2 is going to happen anyway.

---

*Diagnostic close. No code changes accompany this document.*
