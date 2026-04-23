# Milestone 4 Plan — Multi-body + Mooring

Working document. Status: **locked, not yet started.** Delete or archive after M4 merges.

Scope per ARCHITECTURE.md §8: global assembly, connectors, catenary.
Validation gates per §7:

1. Two bodies + rigid connector → combined-mass equivalent single-body free decay.
2. Two bodies + catenary mooring → Irvine closed-form static catenary.

Branch: `milestone-4-multibody-mooring` (not yet created).

---

## Decisions (from Xabier, locked)

**Q1 — Rigid link.** Penalty spring at **10³–10⁴ × max diag(C_global)**, deck-configurable. Default 10⁴, floor 10³. Diagnostic: **max rigid-link length drift** over run; warn if >0.1 % of rest length with a suggested stiffness bump. DAE path deferred to Phase 2. Accept (don't fight) the `dt < 2/ω_penalty` stability floor; document it, emit a startup diagnostic.

**Q2 — Catenary benchmark.** Use **elastic** Irvine (not inextensible).
  - L = 500 m, w = 1000 N/m (submerged), EA = 5×10⁸ N
  - water depth h = 200 m, horizontal span = 400 m
  - Targets: H (horizontal tension at fairlead), V (vertical tension at fairlead), touchdown point, top-angle
  - Tolerance **rtol = 1e-4** on H and V
  - Note: line length 500 vs straight-line diagonal √(400² + 200²) ≈ 447 → ~53 m slack, so case exercises seabed contact

**Q3 — rigid_body.py scope at M4.** Build it now. Minimal.
  - Hamilton-product quaternion update with per-step renormalization
  - R(q) and R(q)ᵀ (body ↔ inertial)
  - Newton–Euler 6-DOF assembly
  - NO angular-accel interpolation, NO higher-order quaternion integrators
  - Tests: (a) torque-free symmetric-top precession, (b) ‖q‖ preservation over 10⁴ steps, (c) R(q)·R(q)ᵀ = I

**Q4 — deck.py schema expansion.** Instantiate now.
  - `bodies:` list — `mass`, `inertia`, `reference_point`, `hydro_database`, `initial_conditions`
  - `connections:` list — **discriminated union** on `type: Literal["linear_spring", "catenary", "rigid_link"]`. No flat dict with optional fields.
  - Validation: Pydantic catches unknown fields, type mismatches, missing required keys
  - Round-trip test on the §5 two-body example (load → re-serialize → deep-equal)

Naming correction noted: the spec uses `connections:`, not `connectors:` or `mooring_lines:`.

---

## PR sequence

### PR1 — N-body global assembly
- `floatsim/solver/state.py`: pack/unpack Ξ ∈ ℝ^{6N} per §3.3
- Extend `floatsim/solver/newmark.py` to accept block-6N M, A∞, K, C, RHS
- **Red test:** two uncoupled bodies each reproduce their M2 free-decay period (M2 tolerance)
- ~300 lines

### PR2 — Rigid-body kinematics
- `floatsim/bodies/rigid_body.py` — scope per Q3
- **Red tests:**
  - Torque-free symmetric-top precession: ω_prec = (I₃ − I₁)/I₁ · Ω₃ (Goldstein §5)
  - ‖q‖ stays within 1e-12 of unity over 10⁴ steps
  - R(q)·R(q)ᵀ = I to 1e-14
- ~250 lines

### PR3 — Connector (linear spring + rigid-link penalty)
- `floatsim/bodies/connector.py`
- One model: 6-DOF linear spring-damper between two bodies (or body↔earth)
- Rigid-link = penalty spring; stiffness deck-configurable per Q1
- Drift diagnostic, dt-floor startup note
- **Red test (M4 gate 1):** two identical bodies rigidly linked in heave → combined-body period = √((2M + 2A∞_zz)/(2C_zz)) = single-body period, rtol = 5e-3; drift < 0.1 %
- ~400 lines

### PR4 — Irvine elastic catenary
- `floatsim/mooring/catenary_analytic.py`
- Elastic Irvine (1981) closed form with seabed contact
- Inputs: L, w, EA, fairlead + anchor positions, seabed depth h
- Outputs: H, V, touchdown point, top-angle
- Cite source in docstring (Irvine 1981 "Cable Structures"; Faltinsen Ch. 8 variant if that's what's accessible — flag in docstring)
- Keep derivation in `docs/catenary.md` so signs are reviewable separately
- **Red test (M4 gate 2):** benchmark from Q2, H and V to rtol=1e-4
- ~500 lines; biggest risk surface

### PR5 — Deck schema expansion
- `floatsim/io/deck.py` per Q4
- `bodies:` and `connections:` with discriminated union
- **Red tests:**
  - Load §5 two-body example → re-serialize → deep-equal
  - Negative: unknown `type`, missing `line.EA` on `catenary`, wrong-type `stiffness`
- ~300 lines

### PR6 — M4 integration deck
- Two-body moored case: one rigid-link pair + one catenary to earth
- Static equilibrium via `scipy.optimize.root` per §9.4
- Short dynamic run, spot-check natural-period shift from mooring stiffness
- No new gates — glue PR
- ~200 lines

---

## Ordering rationale

PR1 first — enabler. PR2 and PR3 are independent; PR2 → PR3 keeps each PR small (PR3 can use R(q) for attachment-point transforms). PR4 independent of PR2/PR3. PR5 before PR6 so the integration deck is schema-validated.

## Risks

- **PR4 catenary signs** — biggest error surface. Separate `docs/catenary.md` derivation note.
- **PR3 dt floor** — penalty stiffness × dt² stability; emit startup diagnostic, don't auto-shrink dt.
- **PR5 scope creep** — `drag_elements:` and `waves:` stay out of this PR (M3/M5 territory).

## Session-continuity notes

If a fresh session picks this up: M3 is merged at commit `f0934f9` on `main`. No M4 code exists yet. Start with PR1 on a new branch `milestone-4-multibody-mooring`.
