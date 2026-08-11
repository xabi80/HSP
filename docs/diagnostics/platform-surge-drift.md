# Platform surge drift — characterization

**Status:** characterized (mechanism measured, not inferred). **Scope:** the
12-buoy articulated platform time-domain runs (fin study + M11b PR8 pilot).
**Case anchor:** fin 0.215, T = 3.141 s, H = 0.08 m, ramp 20 s, unless noted.

Diagnostics that produced every number here live in
[`studies/platform-12buoy/drift/`](../../studies/platform-12buoy/drift/):
`platform_drift_check.py` (per-DOF, joints), `platform_drift_source.py`
(ramp scan + force decomposition), `platform_drift_components.py`
(component-resolved balance), `radiation_damping_check.py` (kernel vs `B(0)`).

---

## 1. What drifts, and what does not

| DOF | behaviour | why |
|-----|-----------|-----|
| **surge (x)** | **secular drift, −x, ≈ −1.15 mm/s (model)** | no hydrostatic restoring; unmoored |
| sway (y), yaw | machine zero (~1e-11) | heading 0 + x-symmetry ⇒ no net lateral/yaw force |
| heave, roll, pitch | bounded, oscillate about a fixed mean | buoyancy-restored |

Numerical (constraint) drift is **absent**: the velocity-level KKT joints hold to
`max|φ| = 1.8e-7 m` over a 220 s run (≈0.1 µm). The bodies do not separate; the
integrator is sound. All drift is **physical**, in the one unrestored translational
DOF.

## 2. The drift velocity is the primitive (not a total excursion)

`run_case` returns only the final six periods, so a drift figure taken from that
window (≈ 6 × per-period step) **understates the run total by ~16×** and must not
be quoted as the excursion. The correct primitive is the **steady drift
velocity** `v0 = −1.153 mm/s` (model). Total excursion = `v0 × duration`; over a
full ≈95-cycle integration that is ≈ 0.34 m model → **≈ 17–19 m full scale**
(Froude ×50), i.e. **~2.3 spar diameters** (D = 8.41 m), not a fraction of one.
Any downstream analysis that treats the end-of-run state as a small perturbation
about ξ = 0 is in the wrong regime — by the end of the run it is a **different
configuration**.

**Cross-harness validation.** FloatSim's `v0 = 1.153 mm/s` agrees with the
independent FloatFEA measurement (0.20 m/period ⇒ 1.228 mm/s) to **6.5 %**. Two
harnesses, two methods — the phenomenon is real and correctly characterised.

## 3. Source: steady rectification, not a startup transient

The startup ramp is exonerated by a **ramp scan** — the terminal drift velocity
is ramp-duration-independent (a one-time impulse would shrink adiabatically):

| ramp | 20 s | 60 s | 120 s |
|------|------|------|-------|
| v0 (mm/s) | −1.153 | −1.143 | −1.147 |

`v0(120)/v0(20) = 0.99`. The driving force is present every cycle → a **steady
second-order rectification** of the quadratic Morison drag. The wave excitation
is exonerated too: its mean surge force is **−0.003 N against a 95 N swing**.

> **DR2 (excitation sign convention) remains formally untested.** −0.003 N vs a
> 95 N swing is *suggestive, not decisive*; it is not a substitute for a direct
> sign-convention test. DR2 stays first, not downgraded.

## 4. Mechanism — resolve the drag by component, never net it

The single "Morison drag = −0.0136 N" row **hides two opposing ~0.42 N terms**.
Netting them is the same failure mode as averaging a per-case diagnostic — and it
is exactly what obscured the mechanism through two earlier (now-retracted)
readings. Component-resolved mean **system surge force** over the steady window
(+x downwave / −x upwave):

| component | mean force | sign | note |
|-----------|-----------|------|------|
| wave excitation | −0.003 N | −x | ≈ 0 (see DR2) |
| **drag — plate NORMAL** (Cd_n = 5.0, distributed) | **−0.432 N** | **−x** | the **driver**; scales with Cd_n |
| **drag — spar cylinder** (Cd = 1.2) | **+0.417 N** | **+x** | the **brake**; Cd_n-independent |
| drag — plate tangential (Cd_t = 1.5, centre-lumped) | +0.002 N | +x | negligible |
| radiation memory | ~5e-5 N | +x | negligible — see §5 |

The **plate-normal drag rectification drives −x** (the heave-plate broadside load
gains a surge component as the buoy pitches — a real coupling the *distributed*
normal patches capture), and the **spar cylinder drag brakes +x**. They nearly
cancel; the small residual −x net produces the drift. This directly explains
FloatFEA's **Cd_n/10 sign reversal**: cut Cd_n 10× and the plate-normal driver
falls to ≈ −0.043 N while the spar brake stays +0.417 N, so the net flips to +x.
**Cd_n is net-driving, not braking.**

The driver is the plate **normal** term, whose coefficient `Cd_n = 5.0` is the
*known* heave-plate value — so the driver magnitude is on firm ground. The plate
**tangential** term (the centre-lumped, non-distribution-resolved one, whose
`Cd_t = 1.5` is the midpoint of a tank-pending [1, 2] range) is **negligible
here** (+0.002 N), so the tangential-magnitude uncertainty (±33 %) does **not**
propagate into this drift. (This is a FloatSim finding; a code that lumps the
*normal* load differently could localise the drift elsewhere.)

## 5. Radiation is negligible — measured, not inferred

The residual that closes the terminal balance is **not** the radiation memory.
Zero-frequency radiation damping is the closure candidate, and it is ~0:

- kernel integral `κ = Σ_surge ∫₀^∞ K(s) ds = 0.046 N·s·m⁻¹` (Ogilvie: `κ = B(0)`),
- BEM `B_surge(ω→0) → 0` directly (while `B_surge(1.93) = 10.1` — the mapping is live).

So the radiation brake on the drift is ≈ +5e-5 N — three orders below the
±0.42 N drag terms. **The earlier inference that radiation supplies the missing
row is refuted by measurement.** At terminal velocity the balance closes because
the velocity-dependent part of the *drag* rises to meet the rectification driver
(confirmed by the ramp-scan-stable v0); a step-point post-hoc force sum still
leaves a ~0.014 N residual, which is a generalized-α evaluation offset (forces
summed at step points vs the integrator's α-weighted midpoint), **not** a missing
physical term — every Cummins term (excitation, drag-by-component, radiation,
inertia, restoring; internal constraints sum to zero) is accounted.

## 6. Scope of impact — read this before quoting "no impact"

- **FloatSim's own outputs are drift-immune.** Heave RAO and every acceleration
  channel are unaffected — heave is bounded, and a linear position drift has zero
  second derivative. Velocity picks up a constant ~1 mm/s offset (negligible vs
  the oscillation). Only **x-displacement** carries the trend. This is correct
  **for FloatSim outputs only.**
- **Position consumers are NOT covered.** Any downstream user of the platform
  *position* (e.g. FloatFEA's mean-wetted-surface / snapshot comparability) sees
  the ~2.3-spar-diameter excursion of §2 and is in a different regime. Do not
  read "drift-immune" as a global no-impact finding.

## 7. Provisional until wave-relative drag lands (M10 A4)

The drag here is **calm-water** (`driver.py` `_calm_fluid`): it uses body velocity
only, with no wave orbital velocity — the conventional wave-drift force (always
**downwave, +x**) is a *deliberately deferred* feature. The observed −x drift is
the pure calm-water rectification with that term omitted. **Enabling
wave-relative drag (M10 A4) is expected to add a larger +x drift that may
dominate or flip the sign.** Anything built downstream around the current drift
(direction or magnitude) is **provisional** until that term is included.

## 8. Open items

- **DR2** — direct excitation sign-convention test (still first).
- **Exact force-balance closure** — re-evaluate all rows at the integrator's
  α-weighted states to drive the §5 residual to numerical zero (a check, not a
  physics gap).
- **Wave-relative drag (M10 A4)** — the dominant real-world term; §7.
