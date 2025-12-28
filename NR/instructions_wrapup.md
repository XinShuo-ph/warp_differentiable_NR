# Numerical Relativity - Branch Wrapup

## Objective
Wrap up YOUR branch's work on differentiable BSSN numerical relativity using CPU backend. Validate, reproduce, and document what was built. Prepare for GPU stage.

---

## File Structure (create as needed in your branch)

```
NR/
├── instructions_wrapup.md   # This file (read-only)
├── WRAPUP_STATE.md          # Your progress tracker
├── README.md                # Documentation for your branch
├── src/                     # Implementation code
├── tests/                   # Test files
└── refs/                    # Reference equations/snippets
```

---

## State Management Protocol

### On Session Start
1. Read `WRAPUP_STATE.md` (create if missing)
2. Resume from documented next action

### On Session End (or ~20k tokens remaining)
1. Update `WRAPUP_STATE.md` with exact next action
2. Commit working changes
3. Push to remote
4. Stop—do not start new tasks

### WRAPUP_STATE.md Template
```markdown
# Wrapup State
- **Phase**: P1/P2/P3
- **Task**: [current task]
- **Status**: in_progress | blocked | completed

## Next Action
[Specific next step]

## Session Log
- [session]: [what was done]
```

---

## Phases

### P1: Validate & Reproduce
**Goal**: Verify your branch's code works from a clean state

**Tasks**:
1. Check what milestone your branch reached (read STATE.md, git log, file structure)
2. Install dependencies: `pip install warp-lang numpy`
3. Run tests to verify they work:
   - If you have `tests/`: `python -m pytest tests/ -v` or run individual test files
   - If you have `src/test_*.py`: run those directly
4. Test key functionality:
   - Poisson solver (if implemented)
   - Flat spacetime evolution (if implemented)
   - BSSN evolution (if implemented)
   - Autodiff through time step (if implemented)
5. Document what works and what doesn't in `WRAPUP_STATE.md`
6. Fix any minor issues (missing imports, path issues, etc.)

**Done when**: Core tests pass without errors

### P2: Document
**Goal**: Write clear README for your branch

**README.md must include**:
```markdown
# Differentiable Numerical Relativity with Warp - [Branch Name]

## Progress Summary
- Milestone reached: M1/M2/M3/M4/M5
- Key deliverables: [list what was built]

## What Works
- [x] [feature]: [brief description]
- [x] [feature]: [brief description]
- [ ] [incomplete feature]: [status]

## Requirements
```bash
pip install warp-lang numpy
```

## Quick Start
```bash
# Run tests
python -m pytest tests/ -v

# Or run specific components
python src/poisson_solver.py     # Poisson equation test
python src/bssn_evolution.py     # BSSN flat spacetime evolution
```

## File Structure
```
NR/
├── src/
│   ├── bssn_state.py       # [description]
│   ├── bssn_derivatives.py # [description]
│   ├── bssn_rhs.py         # [description]
│   └── ...
├── tests/
│   └── ...
└── refs/
    └── bssn_equations.md   # [description]
```

## Implementation Details

### BSSN Variables
[Brief list of implemented BSSN fields]

### Numerical Methods
- Spatial derivatives: [order, stencil]
- Time integration: [method]
- Dissipation: [if implemented]

### Test Results
| Test | Status | Notes |
|------|--------|-------|
| Flat spacetime stability | ✓/✗ | [notes] |
| Constraint preservation | ✓/✗ | [notes] |
| Autodiff | ✓/✗ | [notes] |

## Known Issues / TODOs
- [any unfinished work]
- [any bugs found]
```

**Done when**: README.md accurately describes your branch's state

### P3: GPU Analysis
**Goal**: Analyze what's needed to run on CUDA backend

**Tasks**:
1. Check if your code uses `device="cpu"` explicitly or defaults
2. Identify warp arrays that would need `device="cuda:0"`
3. Check if kernel launches specify device
4. Look for any CPU-only operations (numpy interop, etc.)
5. Estimate changes needed for GPU support
6. Document findings in `notes/gpu_analysis.md`

**notes/gpu_analysis.md template**:
```markdown
# GPU Analysis

## Current Device Usage
- Explicit device="cpu" in code: [Yes/No, where]
- Default device handling: [describe]

## Arrays Needing Device Change
| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| bssn_state.py | phi, chi, ... | cpu | device param |
| ... | | | |

## CPU-Only Operations
- [operation]: [file:line]
- [operation]: [file:line]

## Kernel Device Specification
- Kernels use explicit device: [Yes/No]
- wp.launch device param: [present/missing]

## Changes Needed for GPU
1. [change 1]
2. [change 2]

## Potential GPU Issues
- [ ] Memory transfers between CPU/GPU
- [ ] Array synchronization
- [ ] [other issues]

## Estimated Effort
- Low: [list simple changes]
- Medium: [list moderate changes]
- High: [list complex changes]
```

**Done when**: `notes/gpu_analysis.md` has concrete findings

---

## Key Commands Reference

```bash
# Check your branch status
git status
git log --oneline -5

# Install dependencies
pip install warp-lang numpy

# Run tests
python -m pytest tests/ -v
python -m pytest tests/test_bssn.py -v

# Run individual files
python src/poisson_solver.py
python src/bssn_evolution.py

# Commit and push
git add -A
git commit -m "wrapup: [brief description]"
git push origin HEAD
```

---

## NR-Specific Validation Checklist

For BSSN implementations, verify these if applicable:
- [ ] Flat spacetime remains flat (constraints ~0)
- [ ] Hamiltonian constraint preserved during evolution
- [ ] Momentum constraints preserved during evolution
- [ ] RK4 integration stable for 100+ timesteps
- [ ] Kreiss-Oliger dissipation working (if implemented)
- [ ] Autodiff gradient non-zero through time evolution

---

## Anti-Patterns (Avoid)

- ❌ Major refactoring or new features
- ❌ Writing lengthy physics documentation
- ❌ Trying to fix fundamental physics bugs
- ❌ Running long simulations
- ❌ Leaving code in broken state

---

## Token Budget

| Phase | Budget | Activities |
|-------|--------|------------|
| P1 | ~60k | Understand branch, run tests, fix minor issues |
| P2 | ~40k | Write README with implementation details |
| P3 | ~40k | GPU analysis |

Total estimate: 1-3 sessions per branch

