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

## Status / next
- **DONE (this record):** exact geometry, mesh (`mesh_from_step.py` → `OSU_Test_Buoy.stl`,
  30 MB, not committed — regenerate), figures.
- **PENDING (option a):** mass / CoG / inertia need **material list + ballast masses +
  carriage position** (Xabier to provide, or CATIA mass-properties export). Then: compute
  draft → extract wetted hull, clip at waterline → Capytaine BEM for the spar → drop-in
  A(ω)/B(ω)/C/RAO; heave-plate hydro from the tank test.
