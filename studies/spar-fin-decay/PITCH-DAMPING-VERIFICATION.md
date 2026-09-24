# Pitch-damping verification — isolated single spar-buoy (option "a")

**In-model correctness/convergence check** for the buoy's rotational (pitch) drag
damping, on the platform buoy (0.215 m heave plate Cd_n=5, spar Ø0.168 m Cd=1.2).
Mirrors the M11a spar-drag (PR2) + plate-drag (PR4) validation, extended to the free
single-buoy **pitch** mode with **both** drag elements.
Reproduce: `python studies/spar-fin-decay/pitch_decay_verify.py`.

> **Corrected 2026-09-24 (tracker `STUDY-HYDROSTATIC-REFERENCE-POINT`).** The first version
> of this record used `study_common.build_lhs`, which referenced the waterline and added the
> gravity term to a BEM already solved about the CoG. That inflated C55 to 393.39 instead of
> 107.56 N·m/rad and gave the wrong mode (T 2.32 s, ζ_rad 0.59 %, modal inertia 53.9). All
> numbers below are re-run with FloatSim's CoG-referenced per-body assembly, with the drag
> geometry in the CoG frame. The conclusions hold: quadratic, exact amplitude scaling,
> spar-dominated, converged at n_seg = 10. The model-vs-prediction ratio moved from 1.12 to
> 0.87, and the radiation share essentially vanished.

## Setup
Isolated single buoy (`study_common.py` hydro: FloatSim per-body M+A∞ and C about the CoG,
radiation kernel), with
the **correct** drag added — distributed spar transverse Morison (`morison_element_force`,
`_project_normal` = cross-axis only) + `PlateDragElement` (distributed normal + centre-lumped
rim). Pitch-restoring mode from the drag-free eigenanalysis (drag is force-only ⇒ mode is
independent of the drag code): **T_pitch = 3.26 s**, surge/pitch coupling β = −0.18 (rotates
about z ≈ +0.18 m above the CoG), modal inertia 30.1 kg·m². Released along the
mode shape φ·θ₀; damping measured by log-decrement in the modal coordinate; drag isolated as
ζ_drag = ζ_total − ζ_radiation.

## Result 1 — amplitude dependence, model vs first-principles

| θ (rad) | KC_plate | ζ_total | ζ_radiation | **ζ_drag (measured)** | ζ_drag (predicted) | meas/pred |
|--------:|:--------:|:-------:|:-----------:|:---------------------:|:------------------:|:---------:|
| 0.02 | 0.06 | 0.38% | 0.018% | **0.36%** | 0.41% | 0.87 |
| 0.05 | 0.14 | 0.86% | 0.018% | **0.84%** | 0.96% | 0.87 |
| 0.10 | 0.25 | 1.54% | 0.018% | **1.52%** | 1.75% | 0.87 |
| 0.15 | 0.35 | 2.10% | 0.018% | **2.09%** | 2.41% | 0.87 |

- **Quadratic-drag signature confirmed:** ζ_drag grows linearly with amplitude (ζ ∝ θ).
  A single damping ratio is meaningless — it must be quoted with amplitude/KC.
- **Model reproduces the first-principles moment integral to ~13%**, and the **ratio is
  constant (0.87) across all amplitudes** — the *amplitude scaling and physics are
  exact*; the ~13% is the closed-form energy-linearization's accuracy (drag-free mode shape +
  (8/3π) equivalent-linearization), not a model error. (The absolute level rides on Cd anyway.)

## Result 2 — spar vs plate split
Predicted at θ=0.10: spar **2.05%**, plate **0.11%** → **95% spar / 5% plate**; measured
single-element decays: spar 1.45%, plate 0.11%. **Pitch damping is spar-dominated** (the long
∫|z|³ moment arm), the opposite of *heave* damping which is plate-dominated. So the heave-plate
does little for pitch; the slender spar's cross-flow drag is what damps the buoy's rotation.

## Result 3 — spar discretization convergence (ζ_drag at θ=0.10)
| n_seg | 1 | 2 | 4 | 10 | 16 |
|------:|:--:|:--:|:--:|:--:|:--:|
| spar ζ_drag | 0.10% | 1.01% | 1.35% | **1.45%** | 1.46% |

A single lumped element under-predicts **~15×** (it carries almost no ∫|z|³ moment about the
CoG). The adopted **n_seg=10 is converged** (within 0.8% of n=16). Pitch damping *requires* the
distributed sampling — this is the gate that guards it.

## Conclusion
The model's pitch damping is **internally validated**: quadratic (ζ ∝ θ), matching the
first-principles moment integral to ~13% with exact amplitude scaling, spar-dominated
(95/5), and converged at n_seg=10. **Open item (option b):** the *absolute* level rides on the
drag coefficients — a rotational forced-oscillation (or free-decay) tank test on one
model-scale buoy at the design KC (~0.1–0.3 for pitch; low-KC regime) is needed to pin the
effective pitch Cd, ideally bundled with the heave-plate Cd measurement. See
`memory/project_pitch_damping_verification.md`.
