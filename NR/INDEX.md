# Project Navigation Guide

## Quick Links

### Getting Started
1. **Read First:** [README.md](README.md) - Project overview and quick start
2. **Understanding Progress:** [STATE.md](STATE.md) - Current status and achievements
3. **Detailed Report:** [COMPLETION_REPORT.md](COMPLETION_REPORT.md) - Full task completion details
4. **Technical Details:** [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Comprehensive technical summary

### Running the Code

#### Test Everything
```bash
cd /workspace/NR
./run_all_tests.sh
```

#### Individual Tests
```bash
# M1: Poisson Solver
cd src
python3 poisson_solver.py

# M3: BSSN Flat Spacetime
python3 bssn_warp.py

# M3: Full Test Suite
cd ../tests
PYTHONPATH=../src:$PYTHONPATH python3 test_bssn.py

# M3: Autodiff Tests
PYTHONPATH=../src:$PYTHONPATH python3 test_autodiff_bssn.py
```

### Code Structure

#### Source Code (`src/`)
- `poisson_solver.py` - **M1**: Validated Poisson equation solver
- `bssn_warp.py` - **M3**: BSSN infrastructure (grid, RK4, variables)
- `bssn_rhs.py` - **M3**: BSSN evolution equations (RHS kernels)
- `finite_diff.py` - **M3**: 4th order finite difference operators

#### Tests (`tests/`)
- `test_bssn.py` - Comprehensive BSSN evolution tests
- `test_autodiff_bssn.py` - Autodiff infrastructure tests
- `test_poisson_autodiff.py` - Poisson solver autodiff tests

#### Documentation (`refs/`)
- `bssn_equations.md` - **M2**: Complete BSSN equations from McLachlan
- `grid_and_boundaries.md` - **M2**: Grid structure, AMR, boundary conditions
- `time_integration.md` - **M2**: RK4, MoL, time integration schemes
- `warp_fem_basics.py` - **M1**: Warp FEM API reference

#### Progress Tracking
- `m1_tasks.md` - Milestone 1 task list (6/6 complete)
- `m2_tasks.md` - Milestone 2 task list (5/5 complete)
- `m3_tasks.md` - Milestone 3 task list (7/7 complete)

### External Resources
- `warp/` - NVIDIA Warp source code (cloned from GitHub)
- `mclachlan/` - McLachlan BSSN source code (cloned from BitBucket)

## Milestones

### ✅ M1: Warp Fundamentals
**Goal:** Learn Warp, implement and validate Poisson solver

**Key Files:**
- Implementation: `src/poisson_solver.py`
- Reference: `refs/warp_fem_basics.py`
- Tasks: `m1_tasks.md`

**Validation:**
```bash
cd src && python3 poisson_solver.py
# Expected: L2 error ~ 10⁻⁵, consistency check passes
```

### ✅ M2: Einstein Toolkit Familiarization
**Goal:** Extract and document BSSN equations

**Key Files:**
- BSSN Equations: `refs/bssn_equations.md`
- Grid Structure: `refs/grid_and_boundaries.md`
- Time Integration: `refs/time_integration.md`
- Tasks: `m2_tasks.md`

**Source Reference:**
- McLachlan code: `mclachlan/m/McLachlan_BSSN.m`
- Parameter files: `mclachlan/par/`

### ✅ M3: BSSN in Warp (Core)
**Goal:** Implement BSSN evolution in Warp

**Key Files:**
- Infrastructure: `src/bssn_warp.py`
- Evolution Equations: `src/bssn_rhs.py`
- Finite Differences: `src/finite_diff.py`
- Tests: `tests/test_bssn.py`
- Tasks: `m3_tasks.md`

**Validation:**
```bash
cd tests
PYTHONPATH=../src:$PYTHONPATH python3 test_bssn.py
# Expected: All tests pass
```

## Key Results

### Numerical Accuracy
- **Spatial:** 4th order finite differences
- **Temporal:** 4th order Runge-Kutta
- **Constraint violation:** H < 10⁻⁶
- **Stability:** 200+ timesteps on flat spacetime

### Test Coverage
- ✅ Flat spacetime stability (zero drift)
- ✅ Gauge wave propagation
- ✅ Small perturbation boundedness
- ✅ Constraint preservation
- ✅ Autodiff infrastructure

## Next Steps (Optional)

### M4: BSSN in Warp (BBH)
Would require:
1. Full Ricci tensor implementation
2. BBH initial data (TwoPunctures equivalent)
3. Moving puncture gauge conditions
4. Validation against Einstein Toolkit

**Estimated effort:** Several weeks

### M5: Full Toolkit Port
Would require:
1. Complete boundary conditions (NewRad)
2. Adaptive mesh refinement
3. Wave extraction (Weyl scalars)
4. GPU optimization
5. ML framework integration

**Estimated effort:** Several months

## Troubleshooting

### Tests Failing?
```bash
# Ensure Warp is installed
pip install warp-lang

# Run individual tests to isolate issues
cd /workspace/NR/src
python3 bssn_warp.py
```

### Import Errors?
```bash
# Set Python path
export PYTHONPATH=/workspace/NR/src:$PYTHONPATH

# Or use full paths
cd /workspace/NR/tests
PYTHONPATH=../src:$PYTHONPATH python3 test_bssn.py
```

### Performance Issues?
- Current implementation runs in CPU mode
- GPU support requires CUDA-capable device
- For large grids (>64³), consider GPU or reduce resolution

## Contributing

### Code Style
- Minimal comments (physics/math only)
- Type hints in function signatures
- Warp kernels use `@wp.kernel` decorator
- Test functions start with `test_`

### Adding Tests
1. Create test file in `tests/`
2. Import from `src/` (use PYTHONPATH)
3. Add to `run_all_tests.sh`
4. Ensure all tests pass

### Documentation
- Technical details → `refs/`
- User guide → `README.md`
- Progress tracking → `STATE.md`
- Task lists → `m*_tasks.md`

## Citation

If you use this work, please cite:
```
Differentiable Numerical Relativity with NVIDIA Warp
BSSN formulation with automatic differentiation infrastructure
GitHub: [when available]
```

## Contact & Support

- **Documentation:** See `refs/` directory
- **Examples:** See `tests/` directory
- **Source Code:** See `src/` directory
- **Issues:** Review test output for diagnostics

---

**Last Updated:** 2025-12-25
**Status:** ✅ Complete (M1-M3)
**Test Suite:** 100% passing
