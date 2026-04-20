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

**Known issue for `orcawave.py` reader:** the `.owr` file is OrcaWave's binary results format and typically requires the `OrcFxAPI` Python module (licensed, ships with Orcina installs) to parse. Two implementation paths:
1. **Preferred:** read OrcaWave's companion YAML / text exports (load RAOs, QTFs, added mass, damping from human-readable output).
2. **Fallback:** detect `OrcFxAPI` at import time; if available, use it to read `.owr` directly. If not, raise a helpful error pointing at option 1.

The first validation file will be `C:\Users\xlama\OneDrive\Documents\buoy\Orca\orcawave\model3small_nomor.owr`. Confirm with Xabier which companion exports are available before writing the reader — this informs the approach.

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

- **Do not add dependencies** without asking. Every dep is weight. Current baseline: `numpy`, `scipy`, `pydantic`, `h5py`, `pyyaml`, `pytest`, `hypothesis`. Add only with a one-line justification and Xabier's approval.
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
- Reference OrcaFlex/OrcaWave case: `C:\Users\xlama\OneDrive\Documents\buoy\Orca\orcawave\model3small_nomor.owr` — to be used in Milestones 2, 3, and 6 validation.
- If OrcFxAPI licensing or BEM file format questions arise, ask before guessing.
