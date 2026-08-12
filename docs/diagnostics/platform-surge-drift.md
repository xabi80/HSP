# Platform surge drift — characterization

**Status:** characterized (mechanism measured, not inferred); DR2 excitation
sign-convention **PASS** (§9). **Scope:** the
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

> **DR2 (excitation sign convention) — now validated (§9).** The −0.003 N vs a
> 95 N swing was suggestive, not decisive, on its own; the direct single-buoy
> phase test in §9 confirms the convention independently.

## 4. Mechanism — resolve the drag by component, never net it

The single "Morison drag = −0.0136 N" row **hides two opposing ~0.42 N terms**.
Netting them is the same failure mode as averaging a per-case diagnostic — and it
is exactly what obscured the mechanism through two earlier (now-retracted)
readings. Component-resolved mean **system surge force** over the steady window
(+x downwave / −x upwave):

| component | mean force | sign | note |
|-----------|-----------|------|------|
| wave excitation | −0.003 N | −x | ≈ 0 (see DR2) |
| **drag — plate NORMAL** (Cd_n = 5.0, distributed) | **−0.432 N** | **−x** | ∝ Cd_n (decomposed below) |
| **drag — spar cylinder** (Cd = 1.2) | **+0.417 N** | **+x** | Cd_n-independent (decomposed below) |
| drag — plate tangential (Cd_t = 1.5, centre-lumped) | +0.002 N | +x | negligible |
| radiation memory | ~5e-5 N | +x | negligible — see §5 |

Each drag row is **two things** — the oscillation-**rectified** force plus the
quadratic **resistance** to the existing drift — inseparable at one drift state.
Removing the mean drift velocity from the surge DOFs (= zero drift; exact since
v0 ≪ v_osc) separates them **by measurement**:

| component | total | = rectification (driver) | + drift-resistance (brake) |
|-----------|------:|-------------------------:|---------------------------:|
| plate-NORMAL | −0.432 | **−0.433** (−x, ∝ Cd_n) | +0.001 |
| spar cylinder | +0.417 | **+0.216** (+x, Cd_n-indep.) | +0.201 |
| plate-tangential | +0.002 | +0.001 | +0.001 |
| **net drag** | **−0.013** | **−0.216 (−x — the driver)** | **+0.203 (the brake)** |

(Drag rows only, so every column closes: each row's total = rect + resist, and
the net row's −0.216 + 0.203 = −0.013.) So the **driver is the net drag
rectification, −0.216 N** — the plate-normal term (the heave-plate load gaining a
surge component as the buoy pitches, captured by the *distributed* normal
patches) winning over the spar's own +x rectification — and the **brake is the
drift-resistance, +0.203 N**, almost entirely the spar's quadratic resistance to
v0. Adding the near-zero excitation mean −0.003 N (convention validated, §9)
gives the total driving force −0.219 N; it balances the +0.203 N resistance at v0 = −1.15 mm/s,
the ~0.016 N gap being the §5 evaluation offset, not a physical imbalance.

This makes FloatFEA's **Cd_n/10 sign reversal a measurement**: a pure brake
cannot reverse a drift, so the reversal *requires* a genuine Cd_n-independent +x
rectification — and the **spar rectification (+0.216 N) is exactly that**. Cut
Cd_n 10× and the plate-normal rectification falls to ≈ −0.043 N while the spar
rectification holds at +0.216 N, so the **net rectification flips to +x (+0.17 N)**
and the drift reverses. **Cd_n is net-driving.**

The driver is the plate **normal** term, whose coefficient `Cd_n = 5.0` is the
*known* heave-plate value — so the driver magnitude is on firm ground. The plate
**tangential** term (the centre-lumped, non-distribution-resolved one, whose
`Cd_t = 1.5` is the midpoint of a tank-pending [1, 2] range) is **negligible for
the surge mean here** (+0.002 N), so the tangential-magnitude uncertainty
(±33 %) does **not** propagate into this drift. (A separate caveat for structural
consumers: a load lumped at the disc centre has zero moment about that centre by
construction, so the centre-lumped tangential term contributes nothing to plate
*bending* — structurally absent, not merely small. Patch-resolving it is
required for plate-shell fidelity independent of its surge mean.)

**Q2 linkage.** The plate-normal rectification arises from the correlation
between **pitch angle and heave velocity**, so its magnitude depends on the
pitch–heave phase — the same quantity Q2's small-angle question addresses.
Exposure is small here (measured off-resonance, pitch ≈ 0.038 rad) but the
coupling is real, and it would grow near a pitch resonance.

## 5. Radiation is negligible — measured, not inferred

The residual that closes the terminal balance is **not** the radiation memory.
Zero-frequency radiation damping is the closure candidate, and it is ~0:

- kernel integral `κ = Σ_surge ∫₀^∞ K(s) ds = 0.046 N·s·m⁻¹` (Ogilvie: `κ = B(0)`),
- BEM `B_surge(ω→0) → 0` directly.

**Two frequencies, two answers** — state both so the next reader does not
re-derive it. `B_surge(ω_wave = 1.93) = 10.1 N·s·m⁻¹` is large: that is why the
oscillatory **response** is radiation- (not Cd-) controlled at the wave
frequency. `B_surge(0) → 0`: that is why the steady **drift** has no radiation
brake and is **drag-limited only**. Radiation damps the response; it cannot
brake the drift. So the radiation brake here is ≈ +5e-5 N — three orders below
the ±0.42 N drag terms — and the earlier inference that radiation supplies the
missing row is **refuted by measurement**.

At terminal velocity the balance closes because the velocity-dependent part of
the *drag* rises to meet the rectification driver (confirmed by the
ramp-scan-stable v0). A step-point post-hoc force sum still leaves a ~0.014 N
residual — a generalized-α **evaluation offset** (forces summed at step indices
vs the integrator's α-weighted states), **not** a missing physical term (every
Cummins term — excitation, drag-by-component, radiation, inertia, restoring — is
accounted; internal constraints sum to zero). This offset is a
**solve-state-export concern, not a drift item**: step-indexed force export
hands a consumer forces the integrator never *simultaneously* applied, at
`0.014 / 0.432 = 3.2 %`, which a downstream equilibrium gate (FloatFEA G4.1)
inherits. The fix — write each force at the α-state it was evaluated at —
belongs in the solve-state export design and closes this residual as a side
effect (§8).

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

**Do not enable it yet.** M10 A4 changes the *solve path* (it is not an additive
load), so it breaks additive-only composition, moves the model tag and
invalidates G1.5 — and FloatFEA needs a stable load source for the next ~8 weeks.
When it does land, the drift consequences must be **re-derived, not adjusted**:
FF1's 2.3-spar-diameter excursion, G4.6's magnitude and F5's snapshot
comparability all descend from a drift number whose **sign may flip**, so patching
them would be worse than recomputing from the new solve.

## 8. Open items

- **DR2 — PASSED (§9).** Excitation sign convention validated by a direct
  single-buoy phase test. No longer open.
- **α-state force export** (was "exact force-balance closure") — the §5 residual
  is a step-indexed vs α-weighted force mismatch, not a physics gap. Its
  resolution belongs in the **solve-state export design** (write each force at
  the α-state the integrator evaluated it at), where it also removes a ~3.2 %
  FloatFEA-G4.1 equilibrium floor. Tracked there, not as a drift item.
- **Wave-relative drag (M10 A4)** — the dominant real-world term; §7.

With DR2 passed, the drift investigation closes on the FloatSim side (pending
FloatFEA's ⟨θ·f_n⟩ reproduction); the α-residual lives in the solve-state export
design, not here.

## 9. DR2 — excitation sign convention: PASS

Validated on a **single isolated spar-fin buoy** — the convention is a per-body
property, so one 6-DOF body isolates it (no assembly, no joints). Conditions:
**drag off**, **off-resonance**, **surge and heave**. **Phase is the entire test —
a 180° sign error leaves |RAO| unchanged**, so an amplitude check would pass a
flipped convention.

**Primary result — the physical anchor.** Below the heave resonance (long waves)
the buoy rides the wave, so heave must be **in phase with the elevation η at the
body**. This is the test that matters, because it probes the *relative*
consistency between excitation and kinematics using **no BEM convention on either
side** — and a relative inconsistency is the only real failure: a *globally*
consistent sign flip is a coordinate choice (identical physics), not a defect.

| T (s) | ω (rad/s) | heave phase vs η | \|X\|_heave |
|-------|-----------|-----------------:|-----------|
| 6.0 | 1.047 | **−0.003°** (in phase) | 2.06e-2 |
| 4.5 | 1.396 | **+0.036°** (in phase) | 2.24e-2 |

Heave rides the wave to **within 0.04°**; a flipped sign would read 180°. Pass.

**Coverage — two clean points, stated honestly.** The pass rests on the two
long-wave rows above. A third frequency, T = 2.0 s (above resonance), sits near a
heave response node (|X| = 2.3e-4, ~90× smaller) where the phase is numerically
meaningless and contributes nothing. Two points suffice: a convention sign error
is **frequency-independent by construction**, so it needs no sweep.

**Corroboration — TD vs FD.** The full TD pipeline
(`make_regular_wave_force` → `integrate_cummins`) reproduces an independent FD
closed form `X = [−ω²(M+A(ω)) + iωB(ω) + C]⁻¹ (RAO·A)`:

| T (s) | surge Δ(TD−FD) | heave Δ(TD−FD) |
|-------|---------------:|---------------:|
| 6.0 | 0.758° | −0.003° |
| 4.5 | 0.436° | +0.036° |
| 2.0 | 1.837° | +0.006° |

This is *corroboration, not the primary test* — FD and TD read the excitation
phase from the same reader, so they could agree while both wrong; only the anchor
rules that out. **Heave Δ is clean (< 0.04°) while surge scatters 0.4–1.8°**:
heave has hydrostatic restoring and is well-conditioned; surge on a free body has
none, so its phase is intrinsically sensitive to the small free-decay residual in
the fit. That pattern *supports* the result — it is conditioning, not a convention
problem.

**Phase convention (so this is re-checkable).** Time convention `e^{+iωt}`:
`x(t) = Re{X·e^{+iωt}}`, `η(t) = Re{A·e^{+iωt}}` with A real (phase 0), so the
reported phase is `arg(X)` relative to η. Surge = **−90°** means `X = −i|X|`, i.e.
`x(t) = |X|·sin(ωt)`, which **lags** η = cos(ωt) by 90° — the body following the
horizontal orbital displacement, which lags the elevation by 90° for a +x
progressive wave (FD and TD both give −90°, confirming no axis-specific error).
Diagnostic: `studies/spar-fin-decay/dr2_excitation_sign_convention.py`.
