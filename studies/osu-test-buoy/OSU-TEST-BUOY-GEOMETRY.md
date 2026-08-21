# OSU Test Buoy — geometry measurements (for FloatSim adaptation)

Measurements of the **physical test buoy** ("6-inch model") that will be tank-tested,
taken from the CAD model to re-point the FloatSim single spar-buoy to the real geometry.

- **Source:** `OSU Test Buoy.stp` (CATIA V5R19, STEP AP203, **units = mm**). Kept in the
  buoy OneDrive folder (`Documents/buoy/`), not in this repo (9.3 MB).
- **Tooling:** read with **gmsh** (OpenCASCADE kernel) — `pip install gmsh` on the
  contributor machine; it is NOT a FloatSim runtime dependency (like `openfast_io`,
  scripting-only). Reproduce with `step_measurements.py` / `mesh_from_step.py`.

## Overall
- Envelope **328 × 339 × 2274 mm**; total solid (steel/parts) volume **6.41 L**.
- 30 solids. Layout bottom→top: ballast/heave-plate frame · lower cap · 6″ pipe spar
  (with internal ballast carriage on a track) · upper cap + gimbal ring + gimbal mount +
  12 V motor + lanyard spool.

## Key parts (gmsh: bbox, volume, centroid z)
| part | size (mm) | volume | z_cog (mm) |
|---|---|---|---|
| **Pipe 6 sch 40 (spar)** | **Ø159 OD / ≈Ø149 ID** (wall ~5 mm), **L 1683** | 4.00 L (steel wall) | +841 |
| **Ballast base plate** | **328 × 198 × 50** (rectangular) | 0.310 L | −416 |
| Ballast webs (several) | 0.7–25 mm thick × ~250 mm plates | ~0.06–0.11 L each | −130…−400 |
| Lower cap 6 inch | 159 × 186 × 127 | 0.290 L | −29 |
| Ballast carriage | small, at z = 746 / 996 / 1246 / 1496 (track positions) | 0.007 L each | — |
| Upper cap assembly | 159 × 169 × 103 | 0.302 L | +1712 |
| 6″ gimbal ring | Ø≈220 × 45 | 0.211 L | +1696 |
| gimbal mount | 240 × 60 × 158 | 0.128 L | +1783 |

## ⚠ The heave plate is NOT a disc
The bottom ballast doubles as the heave plate and is an **open, webbed, perforated
frame** — a **rectangular ~328 × 198 mm base plate** with a **cage of thin webs (with
lightening holes)** flaring up to the spar. It is **non-axisymmetric** and thin in one
horizontal axis. (An earlier read as a "Ø396 flat disc" — from parsing circular *edges*
— was wrong; see `OSU_ballast_shape.png`, `OSU_Test_Buoy_mesh.png`.)

**Hydrodynamic implication:** the disc heave-plate model (Cd_n = 5, area πa²) does **not**
transfer. Also a potential-flow BEM of the CAD surface would **over-predict the added
mass** (it blocks the flow the real perforations pass through) and under-predict damping.
So:
- **Spar (pipe):** a clean cylinder → Capytaine BEM is reliable (added mass / radiation /
  excitation).
- **Heave-plate/ballast frame:** its added mass **and** drag are best obtained from the
  **tank test** (the "option b" rotational/heave forced-oscillation or free-decay); for an
  open frame that measurement is essential, not optional. Empirical porous-plate
  corrections are the fallback.

## FloatSim mapping (single spar-buoy)
| FloatSim param | current | → test buoy |
|---|---|---|
| spar radius `R_SPAR` | 0.0841 m | **0.0795 m** (Ø159) |
| spar length | ~1.85 m | **1.683 m** |
| spar waterplane stiffness ρg·A_wp | 221 N/m | **≈200 N/m** |
| heave plate | Ø0.43 disc, Cd_n=5 | **rectangular 0.328×0.198 webbed frame** (hydro from tank/BEM, not the disc model) |

The spar physics carries over almost unchanged (heave period still ~2.5 s); the plate is
the piece that needs the real hydro.

## Mass / floating condition (from `OSU Spar Buoy Platform Metric.xlsx` + measured waterline)
Anchors: **structure = 8.16 kg** (all parts minus the ballast box), **unloaded waterline =
967 mm** from the cylinder bottom (= 0.574·L). Reconciled (`hydro_from_measurements.py`):

| quantity | value |
|---|---|
| total floating mass | **≈21.5 kg (fresh) / 22.1 kg (salt)** — geometry check (cyl+frame+lead=21.8 kg) agrees ~1.5% |
| ballast (lead), to float at 967 mm | ~13.4 kg |
| CoG | **z = −0.907 m** (waterline frame; 0.059 m above cyl bottom) — deep, very stable |
| cylinder | z ∈ **[−0.967, +0.717]** (submerged 0.967 m, freeboard 0.717 m) |
| heave-plate/ballast frame | z ∈ [−1.10, −1.43] (base plate ~−1.38 m); draft 1.425 m |

(OSU Hinsdale lab is **fresh** water; period is density-independent for a free-floating body.)

## Spar BEM (capytaine)
Wetted spar cylinder (Ø0.1593, immersed 0.967 m), `hydro_from_measurements.py`:
- **C33 = 194 N/m** (matches the spreadsheet ~195 ✓); spar heave added mass only ~1.1 kg →
  **spar-only heave period 2.14 s**.
- **Heave-plate bracket:** a *solid* equal-area disc (Ø0.29) adds ~7.9 kg → 2.49 s (upper
  bound); the real perforated/webbed plate adds **less** → heave period likely **~2.2–2.4 s**.
  (Our current FloatSim buoy is ~3.0 s — its fin is larger; the OSU plate is smaller/
  perforated, so a **shorter** period. Re-point the model accordingly.)

## Status / next
- **DONE:** exact geometry + mesh; mass/CoG/draft reconciled; spar BEM (C33, heave added
  mass, period) — the spar hydro is a reliable drop-in.
- **NEXT (integration):** build the full FloatSim buoy database (all DOFs + RAO over the ω
  grid from the spar BEM), update the buoy constants (`R_SPAR 0.0841→0.0795`, spar length
  →1.683 m, mass 21.5 kg, CoG −0.907 m), re-run the decay/RAO.
- **OPEN (needs the tank):** the perforated heave-plate's **added mass AND damping** — a
  potential-flow BEM over-predicts them for an open frame; a heave/pitch free-decay or
  forced-oscillation at the design KC pins both (ties to the pitch-damping option-b test).
