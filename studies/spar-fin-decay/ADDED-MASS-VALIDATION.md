# Single-buoy heave added-mass validation (does Capytaine's A₃₃ make sense?)

Sanity-check of the spar-fin buoy heave added mass `A₃₃` from Capytaine
(`capytaine_bem.nc`), and whether it is all the fin. Reproduce with
`python studies/spar-fin-decay/added_mass_check.py` (writes `added_mass_single.png`).

## Geometry (measured from the BEM mesh)

`test2_spar_fin_fullfix_eqdraft.gdf`, waterline at z=0:

- spar column, radius **0.0841 m**, from z=+0.573 down to z=−1.279 m;
- a thin fin / heave plate, radius **0.215 m**, ~4 mm thick, at **z=−1.141 m**
  (the spar passes through it → hydrodynamically an **annular** plate),
  ~1.14 m below the waterline (depth / radius ≈ 5.3).

## Results

| quantity | value |
|----------|-------|
| A₃₃(ω) over 0.1–30 rad/s | **21.1 kg, essentially flat** (= A_inf) |
| A₃₃(ω_n=2.11 rad/s) | 21.11 kg |
| A_inf | 21.12 kg |
| B₃₃ (radiation damping) | **~0** (peak 0.09, ~0.02 kg/s at ω_n) |
| C₃₃ (BEM) | 221.1 N/m (analytic ρg·πR²_spar = 223.4, 1% mesh) |

**Is A₃₃ ≈ 21 kg reasonable? Yes.** An ideal thin *solid* disc has broadside
added mass `(8/3)ρa³` = **27.2 kg** at a=0.215; Capytaine gives **78%** of that.
The shortfall is expected: the fin is **annular** (the spar hole knocks the ideal
disc to ~25–26 kg), plus the spar-through-plate flow, finite plate thickness and
BEM mesh resolution take the rest. Dimensionally consistent (C₃₃ matches ρg·πR²,
so the added mass is genuine SI — no WAMIT non-dimensional trap).

**Is it all the fin? ~94% yes.** Added mass ∝ radius³, so the fin (0.215) vs the
spar's own bottom (0.084) is a 17:1 ratio — fin disc ≈ 27 kg, spar bottom ≈ 1.6 kg.
The vertical spar walls contribute nothing to heave. Remove the fin and A₃₃ falls
to ~1–2 kg.

## The important consequence: B₃₃ ≈ 0

The heave plate **stores kinetic energy (added mass) but radiates almost no
waves** — it is deep and quiet. This is why:

1. A₃₃ is frequency-flat (Kramers–Kronig: A disperses only where B is
   significant; B≈0 → A=A_inf). It is also why this BEM needs the small-body
   ITEM25 asymptote override (B·ω⁴ never reaches its asymptote).
2. The potential-flow BEM gives this buoy **almost no heave damping**. All real
   heave damping must come from **Morison drag (Cd)** — which is exactly why
   every RAO in the cross-model study is so Cd-sensitive and why the resonance
   is amplitude-gated. The fin's real job is *drag*, not radiation, and BEM
   cannot see drag.

## Minor aside (not a heave issue)

The drag model places the plate at z=−1.278 (the spar's bottom stub) but the
actual fin is at z=−1.141. Heave-irrelevant (drag resists vertical velocity
regardless of z), but it sets the pitch-drag lever arm — worth a look if
rotational response is ever assessed.
