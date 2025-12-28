# NR Branch Merge

## Objective
Merge the best work from all 16 branches into a single production-ready differentiable BSSN numerical relativity codebase. You are working on your current branch (check with `git branch --show-current`).

---

## File Structure

```
NR/
├── instructions_merge.md    # This file (read-only)
├── branch_progresses.md     # Branch analysis (read-only reference)
├── MERGE_STATE.md           # Current progress tracker
├── merge_notes/             # Findings from each branch
│   ├── 0a7f_notes.md
│   ├── 0d97_notes.md
│   └── ...
├── src/                     # Merged production code
├── tests/                   # Merged test suite
├── refs/                    # Reference equations/snippets
└── README.md                # Final documentation
```

---

## State Management Protocol

### On Session Start
1. Identify and record your branch name:
   ```bash
   git branch --show-current
   ```
   Update `MERGE_STATE.md` with your branch name if not already recorded.
2. Read `MERGE_STATE.md` and `branch_progresses.md` for context
3. Resume from documented next action

### On Session End (or ~20k tokens remaining)
1. Update `MERGE_STATE.md` with:
   - Current phase and branch
   - Exact next action
   - Key findings from this session
2. Commit all changes with descriptive message
3. Push to remote
4. Stop—do not start new work

### MERGE_STATE.md Template
```markdown
# Merge State
- **Phase**: P1/P2
- **Current Branch**: [branch suffix, e.g., 0a7f]
- **Branches Completed**: [list]
- **Status**: in_progress | ready_for_next

## Next Action
[Specific next step with exact commands/files]

## Key Findings This Session
- [finding 1]
- [finding 2]

## Merge Decisions Made
- [decision 1]: [rationale]
- [decision 2]: [rationale]

## Session Log
- [session]: [what was done]
```

---

## Branch Processing Order

Process in this order (from `branch_progresses.md` ranking):

### Tier 1 - Most Advanced (MUST process)
1. **0a7f** - M5, 14 tests, full BSSN + BBH + ML ready
2. **0d97** - M5, ML pipeline, waveforms, optimization
3. **c633** - M4, BBH framework, 3300+ lines, comprehensive tests
4. **9052** - M5, puncture evolution, long-term tests

### Tier 2 - M3-M4 Complete (process for features)
5. **1183** - M5, full driver with RK4 and BCs
6. **bd28** - M4 started, **dissipation kernel** (unique)
7. **16a3** - M4 started, clean modular structure
8. **8b82** - M4 started, ETK structure docs
9. **3a28** - M3, README and completion report
10. **99cb** - M3, evolver with derivative tests

### Tier 3 - M2-M3 (scan for unique code)
11. **c374** - M3, basic BSSN + autodiff
12. **2b4b** - M3 started, extensive refs

### Tier 4 - M1-M2 Only (scan quickly)
13. **2eb4** - M1, Poisson Jacobi
14. **5800** - M2 started, ETK snippets
15. **7134** - M2 started, autodiff smoke
16. **95d7** - M2 started, diffusion trace

---

## Phase 1: Explore & Document

**Goal**: Understand each branch, extract reusable components

### Per-Branch Workflow

For each branch (in order above):

#### Step 1: Quick Assessment (~2k tokens)
```bash
# View key files from branch
git show origin/cursor/following-instructions-md-{SUFFIX}:NR/STATE.md 2>/dev/null
git show origin/cursor/following-instructions-md-{SUFFIX}:NR/src/bssn_rhs.py 2>/dev/null | head -50
git ls-tree --name-only -r origin/cursor/following-instructions-md-{SUFFIX} | grep -E '\.(py|md)$' | head -30
```

Check:
- What milestone reached?
- Does BSSN evolution work?
- Any unique features not in previous branches?

#### Step 2: Test Run (~5k tokens)
```bash
# Create temp workspace
mkdir -p /tmp/test_{SUFFIX}
cd /tmp/test_{SUFFIX}

# Copy test files from branch
git --git-dir=/path/to/NR/.git show origin/cursor/following-instructions-md-{SUFFIX}:NR/tests/test_flat_evolution.py > test_flat.py 2>/dev/null
git --git-dir=/path/to/NR/.git show origin/cursor/following-instructions-md-{SUFFIX}:NR/src/bssn_rhs.py > bssn_rhs.py 2>/dev/null

# Try running a test
python test_flat.py
```

#### Step 3: Document Findings (~1k tokens)
Create `merge_notes/{SUFFIX}_notes.md`:
```markdown
# Branch {SUFFIX} Analysis

## Quick Stats
- Milestone: M?
- Tests passing: N
- BSSN evolution works: Yes/No

## Unique Features
- [feature]: [file:function]

## BSSN Components Present
- [ ] Variables/State
- [ ] Derivatives (4th order FD)
- [ ] RHS equations
- [ ] RK4 integrator
- [ ] Constraints (Hamiltonian/Momentum)
- [ ] Dissipation (Kreiss-Oliger)
- [ ] Initial data (flat/BBH)
- [ ] Boundary conditions
- [ ] Autodiff verified

## Code Quality
- Clean: Yes/No
- Tests: Yes/No
- Docs: Yes/No

## Recommended for Merge
- [ ] bssn_rhs.py - [reason]
- [ ] bssn_derivatives.py - [reason]
- [ ] [other files]

## Skip
- [file]: [reason to skip]
```

#### Step 4: Commit & Push
```bash
git add merge_notes/{SUFFIX}_notes.md
git commit -m "P1: Analyze branch {SUFFIX}"
git push origin HEAD
```

### Phase 1 Exit Criteria
- All 16 branches have notes in `merge_notes/`
- Clear list of which files to take from which branch
- MERGE_STATE.md updated with merge plan

---

## Phase 2: Merge & Build

**Goal**: Create unified BSSN codebase from best components

### Step 1: Initialize from Best Base (~10k tokens)

```bash
# You are already on the working branch (check: git branch --show-current)
# Pull code from best base (0a7f or c633)
git checkout origin/cursor/following-instructions-md-0a7f -- NR/src/
git checkout origin/cursor/following-instructions-md-0a7f -- NR/tests/
git checkout origin/cursor/following-instructions-md-0a7f -- NR/refs/

git add -A
git commit -m "P2: Initialize from 0a7f base"
git push origin HEAD
```

### Step 2: Iterative Improvement

For each remaining branch (in order):

#### 2a. Baseline Test
```bash
# Run current tests
python -m pytest tests/ -v 2>/dev/null || python tests/test_flat_evolution.py
# Record: success/fail, which tests pass
```

#### 2b. Identify Improvements
Review `merge_notes/{SUFFIX}_notes.md`:
- What unique features does this branch have?
- What's better than current code?

#### 2c. Apply Improvements
Options:
- **Replace file**: `git show origin/cursor/following-instructions-md-{SUFFIX}:NR/src/file.py > src/file.py`
- **Merge function**: Copy specific function into existing file
- **Add new file**: For unique utilities (e.g., dissipation from bd28)

#### 2d. Verify Improvement
```bash
# Run tests again
python -m pytest tests/ -v 2>/dev/null || python tests/test_flat_evolution.py
# Compare: same or better?
```

#### 2e. Commit & Push with Rationale
```bash
git add -A
git commit -m "P2: Merge {SUFFIX} - [what improved]"
git push origin HEAD
```

If no improvement:
```bash
git commit --allow-empty -m "P2: Skip {SUFFIX} - [why no improvement]"
git push origin HEAD
```

---

## Component Merge Reference

Based on `branch_progresses.md`:

| Component | Primary Source | Alternatives |
|-----------|---------------|--------------|
| **BSSN Variables** | 0d97, c633 | 9052 |
| **BSSN Derivatives** | c633, bd28 | 99cb |
| **BSSN RHS** | c633, 0d97 | 9052 |
| **RK4 Integrator** | c633, 1183 | bd28 |
| **Dissipation** | **bd28** (unique) | - |
| **Constraints** | 9052, bd28 | 0d97 |
| **BBH Initial Data** | c633, 9052 | 0d97 |
| **Poisson Solver** | 0a7f, 3a28 | multiple |
| **Autodiff Tests** | c633, 99cb | 3a28 |
| **ML Pipeline** | **0d97** (unique) | - |
| **README/Docs** | c633, 3a28 | - |

### BSSN Components Checklist
Final merged code should include:

**Core Evolution:**
- [ ] `bssn_variables.py` - φ, χ, γ̄ᵢⱼ, Āᵢⱼ, K, Γ̄ⁱ
- [ ] `bssn_derivatives.py` - 4th order finite difference
- [ ] `bssn_rhs.py` - Evolution equation right-hand sides
- [ ] `bssn_integrator.py` - RK4 time stepping

**Stability:**
- [ ] `bssn_dissipation.py` - Kreiss-Oliger dissipation
- [ ] `bssn_constraints.py` - Hamiltonian & momentum constraints
- [ ] `bssn_boundary.py` - Boundary conditions

**Initial Data:**
- [ ] `initial_data.py` - Flat spacetime, BBH punctures
- [ ] `poisson_solver.py` - For constraint solving

**Validation:**
- [ ] `tests/test_flat_evolution.py` - Flat spacetime stability
- [ ] `tests/test_constraints.py` - Constraint preservation
- [ ] `tests/test_autodiff.py` - Gradient verification

**Optional (ML):**
- [ ] `bssn_ml_pipeline.py` - ML integration (from 0d97)
- [ ] `bssn_losses.py` - Physics-informed losses (from 0d97)

---

## Final Validation Checklist

Before marking Phase 2 complete:

```bash
# 1. Flat spacetime evolution stable
python tests/test_flat_evolution.py
# Should run 100+ steps without blowing up

# 2. Constraints preserved
python tests/test_constraints.py
# Hamiltonian/momentum should stay ~0

# 3. Autodiff works
python tests/test_autodiff.py
# Should compute non-zero gradients

# 4. All core modules import
python -c "from src.bssn_rhs import *; from src.bssn_derivatives import *; print('OK')"

# 5. Run full test suite
python -m pytest tests/ -v
```

---

## Git Commands Reference

```bash
# View file from branch without checkout
git show origin/cursor/following-instructions-md-{SUFFIX}:NR/src/file.py

# Copy file from branch
git show origin/cursor/following-instructions-md-{SUFFIX}:NR/src/file.py > src/file.py

# Checkout directory from branch
git checkout origin/cursor/following-instructions-md-{SUFFIX} -- NR/src/

# Compare files between branches
diff <(git show origin/cursor/following-instructions-md-0a7f:NR/src/bssn_rhs.py) \
     <(git show origin/cursor/following-instructions-md-c633:NR/src/bssn_rhs.py)

# List files in branch
git ls-tree -r --name-only origin/cursor/following-instructions-md-{SUFFIX}
```

---

## Token Budget

| Activity | Budget | Notes |
|----------|--------|-------|
| P1 per branch (Tier 1-2) | ~12k | Deep analysis, test runs |
| P1 per branch (Tier 3-4) | ~4k | Quick scan |
| P2 initialization | ~15k | Set up base |
| P2 per branch | ~10k | Merge + verify |
| Final validation | ~15k | Full test suite |

**Estimated total**: 250-350k tokens (6-10 sessions)

---

## Anti-Patterns (Avoid)

- ❌ Running long simulations during merge
- ❌ Rewriting physics code from scratch
- ❌ Merging without testing before/after
- ❌ Skipping Tier 1 branches
- ❌ Processing branches out of order
- ❌ Committing code that breaks tests
- ❌ Starting Phase 2 before Phase 1 complete
- ❌ Ignoring unique features (dissipation, ML pipeline)

---

## Success Criteria

Phase 2 is complete when:
1. Single unified codebase in `src/`
2. Flat spacetime evolution stable for 100+ steps
3. Constraints preserved during evolution
4. Autodiff verified through time step
5. Kreiss-Oliger dissipation included
6. README with quick start instructions
7. All tests passing
8. All merge decisions documented in MERGE_STATE.md

