# CLAUDE.md — Working Agreement for Claude Code

This file tells Claude Code (the CLI coding agent) how to work in this repository. It is the operating manual. Read it fully before touching any code.

**Project:** FloatSim — time-domain simulator for floating platforms (multi-body, 6-DOF), consuming BEM hydrodynamic databases. Internal engineering tool.

**Architecture:** see [`ARCHITECTURE.md`](./ARCHITECTURE.md) — that document is the contract. Code follows it. If you (Claude) think the architecture needs to change, stop and ask; do not silently diverge.

---

## 1. Prime Directives

1. **The architecture document is the source of truth.** If a requested change contradicts `ARCHITECTURE.md`, stop and flag it. Update the spec first, then the code.
2. **One milestone at a time.** Do not sprint ahead. Milestones are defined in `ARCHITECTURE.md` §8. Finish and validate Milestone N before starting Milestone N+1.
3. **Tests before implementation.** Every new module starts with a failing test case tied to a validation reference from `ARCHITECTURE.md` §7.
4. **No invented physics.** If a formula, constant, or convention isn't in the architecture doc or a cited reference, stop and ask. Do not guess.
5. **Ask when unsure, don't improvise.** A paused agent is cheaper than a week of debugging wrong-but-plausible output.

---

## 2. Repository

- **GitHub:** `xabi80/HSP`
- **Default branch:** `main`
- **Branching:** feature branches named `milestone-N-short-description` or `fix-short-description`. Never commit directly to `main`.
- **Commits:** small, focused, imperative present tense. Example: `add retardation kernel cosine transform` not `added stuff`.
- **PRs:** one milestone sub-task per PR. Include a short description of what was validated and how.

---

## 3. Language, Style, and Tooling

- **Python 3.11+** (use modern typing: `list[int]`, `X | None`, no `Optional[X]` / `List[X]`).
- **Formatter:** `black`, line length 100.
- **Linter:** `ruff` with rules `E,F,I,N,UP,B,SIM,RUF`. No `# noqa` without justification in a comment.
- **Type checker:** `mypy --strict` on the `floatsim/` package. External libs excepted via `[[tool.mypy.overrides]]`.
- **Docstrings:** NumPy style. Every public function documents inputs, outputs, units, and the reference (equation number in `ARCHITECTURE.md` or external citation).
- **Units:** SI throughout. Angles in radians internally; degrees only at deck I/O boundary.
- **Units in names:** when ambiguity is possible, suffix variable names: `period_s`, `heading_deg`, `mass_kg`. This is verbose but prevents the one class of bug we cannot afford.

---

## 4. Project Structure

```
HSP/
├── ARCHITECTURE.md        # The spec. Source of truth.
├── CLAUDE.md              # This file.
├── README.md              # User-facing overview, install, quick start.
├── pyproject.toml         # Build, deps, ruff/black/mypy/pytest config.
├── floatsim/              # The package (layout per ARCHITECTURE.md §4).
├── tests/
│   ├── unit/              # Module-level, fast, isolated.
│   ├── integration/       # Multi-module, deck-driven.
│   └── validation/        # Benchmarks with analytical/tool references.
├── examples/              # Annotated YAML decks that users should read.
├── docs/                  # Derivations, convention notes, BEM format notes.
└── scripts/               # Dev helpers, never imported from the package.
```

Do not create directories outside this layout without updating the architecture doc.

---

## 5. Testing Discipline

- **`pytest`** is the runner. **`pytest -q`** must pass on every PR.
- **Three test tiers:**
  - `unit/` — runs in < 5 s total. Pure-function tests, no I/O, no solver calls.
  - `integration/` — runs in < 2 min. Small decks, short simulations.
  - `validation/` — may take longer, marked `@pytest.mark.slow`. Runs in CI nightly, not on every PR.
- **Numerical tolerances:** explicit and justified.
  - Conservation checks (energy, momentum in free-response): `rtol=1e-10, atol=1e-12`.
  - Analytical comparisons (free decay period): `rtol=1e-3` unless we can argue for tighter.
  - Tool cross-checks (OrcaFlex): `rtol=5e-2` is acceptable; investigate anything worse.
- **Property-based testing:** use `hypothesis` for coordinate transforms, quaternion math, mass-matrix assembly. These have invariants we can exploit (orthogonality, symmetry, positive-definiteness).

---

## 6. Numerical Conventions

From `ARCHITECTURE.md` §3 — repeated here as a quick reference:

- **Inertial frame:** Z up, origin at mean water level. Wave heading 0° = waves traveling in +X.
- **Rotations:** quaternions internally (`[q0, q1, q2, q3]`, scalar first, unit norm). Euler angles only at I/O (ZYX intrinsic = yaw-pitch-roll).
- **Fidelity level:** Level 2 — nonlinear restoring, linear hydro coefficients in body frame rotated to inertial per step.
- **Convolution:** fixed-length retardation buffer, default 60 s. Setup-time diagnostic warns if `|K(t_max)| > 0.01 · max|K(t)|`.
- **Startup:** zero velocity history + mandatory half-cosine excitation ramp (default 20 s).
- **Equilibrium:** mandatory static solve before dynamics via `scipy.optimize.root`.

---

## 7. BEM Database Readers

One of the architectural commitments is pluggable BEM input (OrcaWave, WAMIT, Capytaine). Rules:

- All readers produce the same `HydroDatabase` dataclass — defined in `floatsim/hydro/database.py`.
- Readers live in `floatsim/hydro/readers/`, one file per format.
- Readers are pure: file path in, `HydroDatabase` out. No side effects.
- Every reader ships with a minimal sample file in `tests/fixtures/bem/<format>/` and a round-trip test.

**Status of `orcawave.py` reader (as of M6 PR1):** unvalidated end-to-end. No working test fixture is currently available — Xabier no longer has OrcaWave or OrcaFlex licenses (license loss recorded in `docs/milestone-6-plan.md` v2 pivot). The module remains in the tree as a placeholder for future re-introduction; the verified paths for OrcaWave-derived BEM data are:

1. **`floatsim/hydro/readers/orcaflex_vessel_yaml.py`** (M1.5) — reads OrcaFlex's VesselType YAML export (the human-readable serialisation produced when OrcaWave `.owr` is imported into OrcaFlex and saved). Exercised by the `platform_small.yml` fixture and the M2/M4 OC4 validation tests.
2. **`floatsim/hydro/readers/wamit.py`** (M5 PR1) — reads WAMIT plain-text outputs directly. The OpenFAST/r-test OC4 case (M6 reference) ships its BEM data in this format.
3. **`floatsim/hydro/readers/capytaine.py`** (M5 PR2) — reads Capytaine NetCDF.

The original two implementation paths for `orcawave.py` (preferred: companion YAML/text exports; fallback: `OrcFxAPI` for direct `.owr` parsing) are still the right design when the module is re-validated. Until then, do not extend it; redirect users to the three verified paths above.

---

## 8. Validation-First Workflow per Milestone

For every milestone in `ARCHITECTURE.md` §8, follow this sequence:

1. **Read the relevant section** of `ARCHITECTURE.md`. Re-read it. Quote the governing equation in the PR description.
2. **Identify the validation case** from §7. If none fits, propose one and update the spec first.
3. **Write the test** that encodes the validation reference. It should fail.
4. **Implement the minimum code** to make it pass.
5. **Clean up** — docstrings, types, lint. Run `black`, `ruff`, `mypy`, `pytest`.
6. **Open a PR** with: the governing equation quoted, the validation result plotted or tabulated, and a one-paragraph summary of what was implemented and what was deferred.

---

## 9. What Claude Code Should NOT Do

- **Do not add dependencies** without asking. Every dep is weight. Current baseline: `numpy`, `scipy`, `pydantic`, `h5py`, `pyyaml`, `xarray`, `netCDF4`, `pytest`, `hypothesis`. Add only with a one-line justification and Xabier's approval. (`xarray` + `netCDF4` added in M5 to read Capytaine output natively — see `docs/milestone-5-plan.md` Q3.)
- **Do not write UI or visualization** in Phase 1 beyond basic `matplotlib` plots in validation scripts. No GUI, no web dashboards, no interactive tooling.
- **Do not parallelize** in Phase 1. NumPy vectorization is enough. Save `numba`/`multiprocessing`/`joblib` for Phase 4.
- **Do not refactor across milestones.** If Module A needs a cleanup pass, do it in a dedicated `refactor-<module>` PR, not mixed with new physics.
- **Do not skip the static equilibrium solve** when running dynamics, except when a test explicitly sets `skip_static_equilibrium: true`.
- **Do not silently widen tolerances** when a test fails. Investigate the physics. If the tolerance really is wrong, justify the change in the commit message.
- **Do not invent constants.** Water density, gravity, etc. come from the deck, not from code.

---

## 10. When to Stop and Ask

Stop coding and ask Xabier when:

- The architecture doc is silent or ambiguous on a required decision.
- A validation case fails and you can't explain why within one debugging session.
- A dependency, external tool, or licensed API (`OrcFxAPI`) is needed.
- You find yourself wanting to "temporarily" skip a principle from §9.
- You're about to spend more than ~200 lines of code without a test passing.

A short clarifying question is always cheaper than the wrong implementation.

---

## 11. First Session Checklist

When Claude Code opens this repo for the first time:

1. Read `ARCHITECTURE.md` end to end.
2. Read `CLAUDE.md` (this file) end to end.
3. Check `git status` and `git log --oneline -20` to understand current state.
4. Run `pytest -q` to confirm baseline passes (or fails expectedly).
5. Ask Xabier which milestone to work on. Do not assume.

---

## 12. Contact / Escalation

- Project owner: Xabier
- **Reference validation deck:** OC4 DeepCwind semi-submersible, configured per OpenFAST/r-test `5MW_OC4Semi_Linear/`. The case ships full BEM data as `marin_semi.{1,3,hst,4}` (a trimmed subset is committed at `tests/fixtures/bem/wamit/marin_semi_trimmed.*`); the geometry citation is Robertson et al. 2014, NREL/TP-5000-60601. Used by M5 PR1 reader unit tests and the M6 OpenFAST/HydroDyn cross-check (`docs/milestone-6-plan.md`).
- The earlier reference case (`model3small_nomor.owr` HSFP platform) is no longer the primary — the OrcaWave/OrcaFlex licenses needed to regenerate it are not currently available. M6 was re-planned around OpenFAST + OC4 DeepCwind in `docs/milestone-6-plan.md` v2.
- If BEM file format questions arise (especially for the unvalidated `orcawave.py` reader), ask before guessing.

---

## 13. Lessons Learned from Phase 1 Latent Bugs

Two latent bugs surfaced during the M6 OpenFAST cross-check that none
of the M1-M5 tests caught. Both fit the same anti-pattern:
**a downstream module silently consumed an upstream output without
gating the assumptions it depended on.** Future code reviews should
look hard for this shape.

### Example 1 — missing `m·g·z_G` gravity contribution to ``C``

The BEM readers (WAMIT, Capytaine) produced a buoyancy-only
hydrostatic restoring matrix and explicitly documented "downstream
``floatsim.bodies.Body`` assembly is expected to add the gravity
contribution from the body's mass and CoG." That downstream
assembly was never written. ``assemble_cummins_lhs`` consumed
``hdb.C`` verbatim and produced a negative pitch restoring on OC4.
See `docs/post-mortems/hydrostatic-gravity-bug.md`.

**Generalisation:** when a producer ships data with caveats in its
docstring, the consumer should either *honour* the caveat
explicitly (call the missing module) or *gate* it (raise on
inputs that violate the caveat). Silently passing the data through
is the bug.

### Example 2 — radiation kernel: truncation discontinuity + Nyquist aliasing

`compute_retardation_kernel` evaluated
``∫_0^∞ B(ω) cos(ωt) dω`` as a discrete trapezoidal sum on the BEM
grid, with no gate on whether the grid had reached the asymptotic
regime. On the OrcaFlex `platform_small.yml` fixture (10-point grid,
`B(ω_max) = 50 %` of peak), the missing tail was huge — but the
kernel was used through M2-M5 anyway. On well-resolved grids, the
*discrete cosine sum itself* failed Nyquist beyond `t = π/dω`,
producing sustained `ω_max`-frequency oscillations at amplitude
`~ K_max`. Period was forgiving (`M+A_inf+C` dominated); damping
was unforgiving and showed `t_max`-dependent values, including
sign flips. See `docs/post-mortems/m6-pr3-radiation-kernel-bug.md`.

**Generalisation:** for any oscillatory integral
``∫ f(ω) cos(ωt) dω`` evaluated at variable `t`, never sample the
cosine onto a grid — use a Filon-type rule that integrates the
oscillation analytically per segment. And gate the grid: refuse to
run when `B(ω_max)` is not at the noise floor, or when the
high-frequency asymptote is not clean.

### Pattern lock — the recurring shape

Both bugs lasted because:

1. **Caveats lived only in docstrings, not in code gates.** The
   reader said "downstream must add gravity"; the kernel's docstring
   said "B must reach asymptotic regime by ω_max." Neither was
   enforced by a runtime check. Consequence: every downstream
   consumer was free to violate the caveat silently.
2. **No test exercised the cross-module combination.** Each module's
   unit tests covered its piece in isolation, with controlled
   synthetic inputs that happened not to trigger the latent
   condition. The bug surfaced only when realistic inputs flowed
   end-to-end.
3. **Period was the M2-M5 validation lever; damping wasn't.**
   Period is dominated by `M+A_inf+C` and barely sees the
   convolution. Damping is the strict test of the kernel — and was
   asserted only on a constant-`B` synthetic (which happens to mask
   one of the two pathologies) until M6 PR3.

**For all future Phase 1 work:** when a module's docstring
documents a precondition on its input, that precondition belongs
in code as a `ValueError` gate, not as prose.

---

## 14. Pre-Cross-Check Audit Pattern

Before running any tool-cross-check milestone (M6 OpenFAST, future M7+), audit the FloatSim modules whose conventions sit on the path between the reference tool's output and the assertion under test. This is a workflow discipline: **conventions that look settled often aren't**, and disagreements between tools can hide as off-by-one errors in any module along the chain.

The motivating example is the hydrostatic-gravity bug (`docs/post-mortems/hydrostatic-gravity-bug.md`). M6 PR1's planning surfaced it: the BEM readers (WAMIT, Capytaine) explicitly documented that gravity is added by downstream `Body` assembly, but the assembly step did not exist. `floatsim/hydro/hydrostatics.py` was listed in ARCHITECTURE.md §4 but never written. `assemble_cummins_lhs` consumed the buoyancy-only `C` verbatim. No M5 test exercised the buggy combination, so the gap was invisible until the M6 audit traced it. A direct M6 PR2 assertion against OC4 pitch period would have failed with NaN, surfacing the bug as a *cross-check* failure rather than an *internal* one.

**Audit sequence (run before the first cross-check assertion fires):**

1. **List the modules on the cross-check path.** For M6 PR2 (S1 OC4 static equilibrium): the BEM reader → `assemble_cummins_lhs` → `floatsim/hydro/hydrostatics.py` → `static_equilibrium_solver` → comparison against the OpenFAST CSV.
2. **For each module, check that its docstring contract matches its implementation.** Reader docstrings claim "downstream assembly will add gravity" — does that downstream assembly exist? Does the integrator's `state_force` lag policy match what the connector docstring asserts? Does the conventions doc's pre-flight checklist cover every assumption the cross-check depends on?
3. **For every "TODO" or "expected to be added later" annotation found, treat it as a Phase-1 bug, not a future task.** If a downstream invariant is asserted in module A but not implemented, that's the bug to fix today.
4. **For convention-call entries, verify with both a written-source citation AND a runnable sanity-check protocol.** The HydroDyn conventions doc (`docs/openfast-cross-check-conventions.md` for M6) makes this explicit: every item carries both columns. Single-column items default to "not yet verified."
5. **For settling parameterisation choices, write a discriminator test BEFORE the cross-check fires.** The asymmetric-CoG test (`tests/validation/test_gravity_restoring_asymmetric_cog.py`) is the template — first-principles derivation alongside the implementation, with the test failing loudly when the convention is wrong.

The audit is not a one-time exercise; it runs at the start of every cross-check milestone. A pinned audit checklist lives in the milestone plan's Q8 (or equivalent) section, with each line marked verified / pending / N/A. Items that lack both a source citation and a sanity-check protocol are not allowed past PR1.

**This pattern is the institutional memory.** The hydrostatic-gravity bug stayed dormant for five milestones because no test exercised the buggy combination; the audit caught it the moment it became reachable. Future cross-checks will surface their own latent issues — the audit pattern surfaces them at PR1 plan time, not at the assertion-under-test failure.
