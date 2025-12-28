# NR (Numerical Relativity) Branch Progress Summary

## Overview
16 branches implementing differentiable BSSN numerical relativity using NVIDIA Warp.

## Milestone Reference
- **M1**: Warp fundamentals, run FEM examples, Poisson solver
- **M2**: Einstein Toolkit familiarization, extract BSSN equations
- **M3**: BSSN core in Warp (flat spacetime evolution)
- **M4**: BSSN BBH (binary black hole initial data)
- **M5**: Full toolkit port, ML integration

---

## Branch Details

### Tier 1: Most Advanced (M4-M5, BBH/ML Features)

| Branch | Milestone | Tests | Key Features |
|--------|-----------|-------|--------------|
| **0a7f** | M5 ✓ | 14 | Full BSSN evolution, BBH initial data, ML ready |
| **0d97** | M5 ✓ | - | ML pipeline, waveform extraction, optimization |
| **c633** | M4 ✓ | 5 | BBH framework, 3300+ lines, comprehensive tests |
| **9052** | M5 ✓ | 5 | Puncture evolution, long evolution tests |

**0a7f Files:**
- `src/bssn.py`, `bssn_evol.py`, `poisson.py`
- `tests/test_bssn.py`, `test_bssn_evol.py`, `test_poisson.py`
- `refs/bssn_equations.md`, `warp_autodiff.py`, `warp_fem_api.py`
- All 5 milestone task files

**0d97 Files:**
- `src/bssn_vars.py`, `bssn_rhs.py`, `bssn_rhs_full.py`, `bssn_integrator.py`
- `src/bssn_initial_data.py`, `bssn_boundary.py`, `bssn_constraints.py`
- `src/bssn_ml_pipeline.py`, `bssn_optimization.py`, `bssn_waveform.py`, `bssn_losses.py`
- `refs/ml_integration_api.py`, `schwarzschild_comparison.md`

**c633 Files:**
- `src/bssn_state.py`, `bssn_derivatives.py`, `bssn_rhs.py`, `bssn_rhs_full.py`
- `src/bssn_rk4.py`, `bbh_initial_data.py`, `poisson_solver.py`
- `tests/test_bbh_evolution.py`, `test_bssn_complete.py`, `test_bssn_autodiff.py`
- `README.md`, `FINAL_STATUS.md`, `SUMMARY.md`

**9052 Files:**
- `src/bssn_variables.py`, `bssn_derivatives.py`, `bssn_rhs.py`, `bssn_rhs_full.py`
- `src/bssn_integrator.py`, `bssn_initial_data.py`, `bssn_boundary.py`, `bssn_constraints.py`
- `tests/test_puncture_evolution.py`, `test_long_evolution.py`, `test_bssn_evolution.py`

---

### Tier 2: M3-M4 Complete (BSSN Core Working)

| Branch | Milestone | Tests | Key Features |
|--------|-----------|-------|--------------|
| **1183** | M5 ✓ | 3 | Full BSSN driver with RK4 and BCs |
| **bd28** | M4 started | - | M3 complete, dissipation kernel, constraints |
| **16a3** | M4 started | 3 | BSSN core, RK4 integrator, derivs module |
| **8b82** | M4 started | 2 | BSSN geometry, initial data, BBH test |
| **3a28** | M3 ✓ | 3 | BSSN warp, finite diff, README |
| **99cb** | M3 ✓ | 4 | BSSN evolver, full derivative tests |

**1183 Files:**
- `src/bssn.py`, `bssn_evolve.py`, `bssn_full.py`, `poisson.py`
- `tests/test_bssn.py`, `test_bssn_full.py`, `test_poisson.py`
- `refs/warp_fem_adaptive.py`, `warp_fem_mesh_field.py`

**bd28 Files:**
- `src/bssn_defs.py`, `bssn_rhs.py`, `bssn_solver.py`
- `src/derivatives.py`, `dissipation.py`, `dissipation_kernel.py`, `constraints.py`, `rk4.py`
- `src/test_bssn_rhs.py`, `test_constraints.py`, `test_derivatives.py`, `test_autodiff_bssn.py`

**16a3 Files:**
- `src/bssn.py`, `derivs.py`, `integrator.py`, `rhs.py`, `poisson.py`
- `tests/test_flat.py`, `test_autodiff.py`, `test_poisson.py`
- `refs/grid_structure.md`, `time_integration.md`

**8b82 Files:**
- `src/bssn_defs.py`, `bssn_geometry.py`, `bssn_rhs.py`, `bssn_solver.py`
- `src/derivatives.py`, `initial_data.py`, `poisson_solver.py`
- `tests/test_flat_spacetime.py`, `test_bbh_evolution.py`
- `refs/etk_bbh_structure.md`

**3a28 Files:**
- `src/bssn_warp.py`, `bssn_rhs.py`, `finite_diff.py`, `poisson_solver.py`
- `tests/test_bssn.py`, `test_autodiff_bssn.py`, `test_poisson_autodiff.py`
- `README.md`, `COMPLETION_REPORT.md`, `PROJECT_SUMMARY.md`

**99cb Files:**
- `src/bssn_variables.py`, `bssn_derivatives.py`, `bssn_rhs.py`, `bssn_evolver.py`
- `tests/test_flat_evolution.py`, `test_derivatives.py`, `test_autodiff_bssn.py`
- `PROGRESS.md`

---

### Tier 3: M2-M3 (BSSN Started)

| Branch | Milestone | Key Features |
|--------|-----------|--------------|
| **c374** | M3 ✓ | BSSN core, derivatives, autodiff test |
| **2b4b** | M3 started | BSSN vars/derivs, extensive refs |

**c374 Files:**
- `src/bssn.py`, `bssn_rhs.py`, `derivatives.py`, `poisson.py`
- `tests/test_bssn_autodiff.py`
- `refs/bssn_equations.md`, `warp_fem_api.md`

**2b4b Files:**
- `src/bssn_vars.py`, `bssn_derivatives.py`, `poisson_solver.py`
- `tests/test_bssn_vars.py`, `test_bssn_derivatives.py`, `test_poisson.py`
- `refs/bssn_equations.md`, `grid_boundary_conditions.md`, `autodiff_mechanism.md`

---

### Tier 4: M1-M2 (Poisson Solver, Basics)

| Branch | Milestone | Key Features |
|--------|-----------|--------------|
| **2eb4** | M1 ✓ | Poisson Jacobi, diffusion autodiff |
| **5800** | M2 started | FD Poisson, ETK snippets extracted |
| **7134** | M2 started | Poisson solver, autodiff smoke test |
| **95d7** | M2 started | Poisson solver, diffusion trace |

**2eb4 Files:**
- `src/poisson_jacobi.py`, `m1_diffusion_autodiff.py`
- `tests/test_poisson_jacobi.py`
- `refs/warp_tape_patterns.py`, `navier_stokes_mesh_field_api.py`

**5800 Files:**
- `src/poisson_fd.py`, `m1_run_example_*.py`
- `tests/test_poisson_fd.py`
- `refs/m2_time_integration_mol_rk4.md`, `m2_grid_and_boundary_conditions.md`

**7134 Files:**
- `src/poisson.py`, `autodiff_smoke.py`
- `tests/test_poisson.py`
- `refs/m1_autodiff_snippets.md`, `m1_navier_stokes_mesh_field_snippets.md`

**95d7 Files:**
- `src/poisson.py`, `m1_diffusion_autodiff_trace.py`
- `tests/test_poisson.py`
- `refs/warp_fem_autodiff_tape_record_func_snippet.py`

---

## Recommended Merge Strategy

### For Core BSSN:
1. **Primary base**: `c633` or `0a7f` (most complete, tested)
2. **ML features**: `0d97` (optimization, waveforms, losses)
3. **Dissipation**: `bd28` (dissipation_kernel.py)
4. **Constraints**: `9052`, `bd28` (constraint monitoring)

### Key Components by Category:

| Component | Best Sources |
|-----------|--------------|
| **BSSN Variables** | 0d97, c633, 9052 |
| **BSSN Derivatives** | c633, bd28, 99cb |
| **BSSN RHS** | c633, 0d97, 9052 |
| **Time Integration (RK4)** | c633, 1183, bd28 |
| **Dissipation** | bd28 |
| **Constraints** | 9052, bd28, 0d97 |
| **Initial Data (BBH)** | c633, 9052, 0d97 |
| **Poisson Solver** | 0a7f, 3a28, multiple |
| **Autodiff Tests** | c633, 99cb, 3a28 |
| **ML Pipeline** | 0d97 (unique) |
| **README/Docs** | c633, 3a28 |

### Implementation Patterns:

**BSSN State Variables:**
- `phi` (conformal factor)
- `chi` (χ = e^{-4φ})
- `gamma_bar_ij` (conformal metric)
- `A_bar_ij` (conformal traceless extrinsic curvature)
- `K` (trace of extrinsic curvature)
- `Gamma_bar^i` (conformal connection functions)

**Numerical Methods:**
- Spatial derivatives: 4th order finite difference
- Time integration: RK4
- Dissipation: Kreiss-Oliger (in bd28)
- Boundary conditions: Various (flat, periodic, outflow)

---

## Common File Structure

Most complete branches follow:
```
NR/
├── src/
│   ├── bssn_variables.py    # BSSN field definitions
│   ├── bssn_derivatives.py  # Spatial derivative kernels
│   ├── bssn_rhs.py          # Evolution equation RHS
│   ├── bssn_integrator.py   # RK4 time stepping
│   ├── bssn_constraints.py  # Hamiltonian/momentum constraints
│   ├── initial_data.py      # BBH initial data
│   └── poisson_solver.py    # Poisson equation solver
├── tests/
│   ├── test_flat_evolution.py
│   ├── test_bssn_autodiff.py
│   └── test_poisson.py
├── refs/
│   └── bssn_equations.md    # BSSN formulation reference
└── tasks/
    ├── m1_tasks.md through m5_tasks.md
```

---

## Unique Features by Branch

| Branch | Unique Feature |
|--------|----------------|
| **0d97** | ML pipeline, loss functions, waveform extraction |
| **c633** | Most comprehensive test suite, BBH framework |
| **bd28** | Kreiss-Oliger dissipation kernel |
| **9052** | Puncture evolution, long-term stability tests |
| **3a28** | Clean documentation, completion report |
| **8b82** | Einstein Toolkit structure documentation |

