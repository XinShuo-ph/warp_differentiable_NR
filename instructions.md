# Numerical Relativity with NVIDIA Warp

## Goal
Implement differentiable numerical relativity algorithms using NVIDIA `warp` package, enabling backpropagation for ML integration.

---

## State Tracking Protocol

### On Session Start
1. Read `NR/STATE.md` (create if missing)
2. Read the current milestone's task file (e.g., `NR/m1_tasks.md`)
3. Resume from the first incomplete task

### On Session End (or when ~10k tokens remain)
1. Update `NR/STATE.md` with:
   - Current milestone number
   - Current task number
   - One-line status
   - Any blockers
2. Commit working code (no broken states)
3. Do NOT write summaries, reports, or READMEs

### File Structure
```
NR/
├── STATE.md              # Current progress state
├── m1_tasks.md           # Milestone 1 task breakdown
├── m2_tasks.md           # Milestone 2 task breakdown
├── ...
├── src/                  # Implementation code
├── tests/                # Test files
└── refs/                 # Extracted reference snippets (< 100 lines each)
```

---

## Milestones

### M1: Warp Fundamentals
**Goal:** Understand warp kernels, autodiff, and FEM basics.
**Entry:** Clone https://github.com/NVIDIA/warp.git locally.
**Exit criteria:** Successfully run and modify 3+ FEM examples.

Tasks to extract into `m1_tasks.md`:
- [ ] Install warp, run `warp.examples`
- [ ] Run `example_diffusion.py`, trace autodiff mechanism
- [ ] Run `example_navier_stokes.py`, document mesh/field APIs
- [ ] Run `example_adaptive_grid.py`, document refinement APIs
- [ ] Implement Poisson equation solver from scratch
- [ ] Verify Poisson solver against analytical solution

### M2: Einstein Toolkit Familiarization
**Goal:** Understand BBH simulation structure.
**Entry:** `docker pull rynge/einsteintoolkit:latest`
**Exit criteria:** Run BBH example, extract BSSN equation structure.

Tasks to extract into `m2_tasks.md`:
- [ ] Run Docker container, locate BBH example
- [ ] Execute BBH simulation, identify output files
- [ ] Extract McLachlan/BSSN evolution equations to `refs/bssn_equations.md`
- [ ] Extract grid structure and boundary conditions
- [ ] Document time integration scheme used

### M3: BSSN in Warp (Core)
**Goal:** Implement BSSN evolution equations in warp.
**Entry:** Completed M1 + M2.
**Exit criteria:** Evolve flat spacetime stably for 100+ timesteps.

Tasks to extract into `m3_tasks.md`:
- [ ] Define BSSN variables as warp fields
- [ ] Implement spatial derivative kernels (4th order FD)
- [ ] Implement RHS of BSSN equations (start with flat spacetime)
- [ ] Implement RK4 time integration
- [ ] Add Kreiss-Oliger dissipation
- [ ] Test constraint preservation on flat spacetime
- [ ] Verify autodiff works through one timestep

### M4: BSSN in Warp (BBH)
**Goal:** Reproduce BBH-like initial data evolution.
**Entry:** Completed M3.
**Exit criteria:** Match Einstein Toolkit output qualitatively.

### M5: Full Toolkit Port
**Goal:** Port remaining Einstein Toolkit core features.
**Entry:** Completed M4.

---

## Task Execution Rules

1. **One task at a time.** Complete fully before moving on.

2. **Atomic commits.** Each task should end with runnable code.

3. **Test-first when possible.** Write test, then implement.

4. **Validation = 2 consistent runs.** Run twice, same output = done.

5. **Minimal code comments.** Only explain non-obvious physics/math.

6. **Extract, don't summarize.** Save useful code snippets to `refs/`, not prose.

7. **Budget awareness:**
   - First 30k tokens of a milestone: explore, create task list
   - Remaining tokens: execute tasks
   - Last 10k tokens: wrap up, update STATE.md

---

## STATE.md Template

```markdown
# Current State

Milestone: M1
Task: 3 of 6
Status: Running example_navier_stokes.py, studying FemDomain API
Blockers: None

## Quick Resume Notes
- Warp installed at: C:\path\to\warp
- Working in: NR/src/
- Last successful test: tests/test_diffusion.py
```

---

## Context Efficiency Tips

- Don't re-read large files. Extract relevant snippets to `refs/`.
- Don't explore broadly. Follow task list linearly.
- Don't refactor working code. Move forward.
- Don't write documentation. Code + minimal comments only.
- If stuck > 5 attempts, note blocker in STATE.md and move to next task.
