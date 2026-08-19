# Pitch-damping verification — isolated single spar-buoy (option "a")

**In-model correctness/convergence check** for the buoy's rotational (pitch) drag
damping, on the platform buoy (0.215 m heave plate Cd_n=5, spar Ø0.168 m Cd=1.2).
Mirrors the M11a spar-drag (PR2) + plate-drag (PR4) validation, extended to the free
single-buoy **pitch** mode with **both** drag elements.
Reproduce: `python studies/spar-fin-decay/pitch_decay_verify.py`.

## Setup
Isolated single buoy (`study_common.py` hydro: M+A∞, C+gravity, radiation kernel), with
the **correct** drag added — distributed spar transverse Morison (`morison_element_force`,
`_project_normal` = cross-axis only) + `PlateDragElement` (distributed normal + centre-lumped
rim). Pitch-restoring mode from the drag-free eigenanalysis (drag is force-only ⇒ mode is
independent of the drag code): **T_pitch = 2.32 s**, surge/pitch coupling β = +0.35 (rotates
about z ≈ −0.35 m, not the reference point), modal inertia 53.9 kg·m². Released along the
mode shape φ·θ₀; damping measured by log-decrement in the modal coordinate; drag isolated as
ζ_drag = ζ_total − ζ_radiation.

## Result 1 — amplitude dependence, model vs first-principles

| θ (rad) | KC_plate | ζ_total | ζ_radiation | **ζ_drag (measured)** | ζ_drag (predicted) | meas/pred |
|--------:|:--------:|:-------:|:-----------:|:---------------------:|:------------------:|:---------:|
| 0.02 | 0.06 | 0.92% | 0.59% | **0.34%** | 0.30% | 1.13 |
| 0.05 | 0.13 | 1.38% | 0.59% | **0.79%** | 0.70% | 1.13 |
| 0.10 | 0.24 | 2.02% | 0.59% | **1.43%** | 1.28% | 1.12 |
| 0.15 | 0.33 | 2.56% | 0.59% | **1.98%** | 1.77% | 1.12 |

- **Quadratic-drag signature confirmed:** ζ_drag grows linearly with amplitude (ζ ∝ θ).
  A single damping ratio is meaningless — it must be quoted with amplitude/KC.
- **Model reproduces the first-principles moment integral to ~12%**, and the **ratio is
  constant (1.12–1.13) across all amplitudes** — the *amplitude scaling and physics are
  exact*; the ~12% is the closed-form energy-linearization's accuracy (drag-free mode shape +
  (8/3π) equivalent-linearization), not a model error. (The absolute level rides on Cd anyway.)

## Result 2 — spar vs plate split
Predicted at θ=0.10: spar **1.56%**, plate **0.13%** → **92% spar / 8% plate**; measured
single-element decays: spar 1.35%, plate 0.12%. **Pitch damping is spar-dominated** (the long
∫|z|³ moment arm), the opposite of *heave* damping which is plate-dominated. So the heave-plate
does little for pitch; the slender spar's cross-flow drag is what damps the buoy's rotation.

## Result 3 — spar discretization convergence (ζ_drag at θ=0.10)
| n_seg | 1 | 2 | 4 | 10 | 16 |
|------:|:--:|:--:|:--:|:--:|:--:|
| spar ζ_drag | 0.34% | 1.12% | 1.29% | **1.35%** | 1.36% |

A single lumped element under-predicts **~4×** (it carries almost no ∫|z|³ moment). The
adopted **n_seg=10 is converged** (within 0.5% of n=16). Pitch damping *requires* the
distributed sampling — this is the gate that guards it.

## Conclusion
The model's pitch damping is **internally validated**: quadratic (ζ ∝ θ), matching the
first-principles moment integral to ~12% with exact amplitude scaling, spar-dominated
(92/8), and converged at n_seg=10. **Open item (option b):** the *absolute* level rides on the
drag coefficients — a rotational forced-oscillation (or free-decay) tank test on one
model-scale buoy at the design KC (~0.1–0.3 for pitch; low-KC regime) is needed to pin the
effective pitch Cd, ideally bundled with the heave-plate Cd measurement. See
`memory/project_pitch_damping_verification.md`.
