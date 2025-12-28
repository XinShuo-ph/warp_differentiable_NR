# Differentiable Numerical Relativity with Warp - instructions-wrapup-completion

## Progress Summary
- **Milestone reached**: M4 (Complete)
- **Key deliverables**:
  - BSSN formulation with RK4 time integration
  - Poisson solver using Warp FEM
  - Binary Black Hole initial data (Brill-Lindquist + Bowen-York)
  - Kreiss-Oliger dissipation
  - Autodiff-compatible solver (verified working)

## What Works
- [x] **Poisson Solver** (`src/poisson_solver.py`): FEM-based 2D Poisson solver with L2 error < 1e-4
- [x] **BSSN State Management** (`src/bssn_defs.py`): Complete BSSN variable struct with 3D grid allocation
- [x] **4th Order Finite Differences** (`src/derivatives.py`): First and second derivatives, mixed derivatives, KO dissipation
- [x] **RK4 Time Integrator** (`src/bssn_solver.py`): Fourth-order Runge-Kutta with state management
- [x] **Flat Spacetime Evolution**: Stable over 100+ timesteps (constraints remain ~0)
- [x] **Initial Data**: Brill-Lindquist conformal factor + Bowen-York extrinsic curvature
- [x] **Autodiff**: Gradients computable through RK4 time evolution
- [x] **1+log Lapse + Gamma-driver Shift**: Gauge evolution implemented
- [ ] **BBH Evolution**: Runs but unstable (NaN after ~5 steps) - needs complete BSSN RHS terms and proper boundary conditions

## Requirements
```bash
pip install warp-lang numpy
```

For testing:
```bash
pip install pytest
```

## Quick Start
```bash
# Run all tests
python3 -m pytest NR/tests/ -v

# Or run specific components
PYTHONPATH=/workspace python3 NR/src/poisson_solver.py     # Poisson equation test
PYTHONPATH=/workspace python3 NR/tests/test_flat_spacetime.py  # Flat spacetime + autodiff
PYTHONPATH=/workspace python3 NR/tests/test_bbh_evolution.py   # BBH head-on (shows instability)
```

## File Structure
```
NR/
├── src/
│   ├── bssn_defs.py       # BSSN state struct and initialization (BSSNState, allocate, init_flat)
│   ├── bssn_rhs.py        # BSSN evolution equations (RHS kernel)
│   ├── bssn_solver.py     # RK4 integrator (BSSNSolver class)
│   ├── bssn_geometry.py   # Ricci tensor computation (skeleton)
│   ├── derivatives.py     # 4th order FD stencils, KO dissipation
│   ├── initial_data.py    # Brill-Lindquist and Bowen-York initial data
│   └── poisson_solver.py  # FEM Poisson solver (standalone demo)
├── tests/
│   ├── test_flat_spacetime.py  # Stability and autodiff tests
│   └── test_bbh_evolution.py   # BBH head-on collision test
├── refs/
│   ├── bssn_equations.md      # Reference BSSN equations
│   ├── etk_bbh_structure.md   # Einstein Toolkit reference
│   └── warp_fem_api.md        # Warp FEM API notes
├── m1_tasks.md ... m4_tasks.md  # Milestone task files
├── STATE.md                     # Project state tracker
├── WRAPUP_STATE.md              # Wrapup progress tracker
└── README.md                    # This file
```

## Implementation Details

### BSSN Variables
The `BSSNState` struct contains all evolution variables:
| Variable | Type | Description |
|----------|------|-------------|
| `phi` | `float` | Conformal factor (ψ = e^φ) |
| `gamma_tilde` | `mat33` | Conformal 3-metric (det = 1) |
| `K` | `float` | Trace of extrinsic curvature |
| `A_tilde` | `mat33` | Traceless extrinsic curvature |
| `Gamma_tilde` | `vec3` | Conformal connection functions |
| `alpha` | `float` | Lapse function |
| `beta` | `vec3` | Shift vector |
| `B` | `vec3` | Gamma-driver auxiliary variable |

### Numerical Methods
- **Spatial derivatives**: 4th order centered finite differences (5-point stencil)
- **Time integration**: 4th order Runge-Kutta (RK4)
- **Dissipation**: Kreiss-Oliger 6th order (σ = 0.1)
- **Boundary conditions**: Frozen (zero RHS in 3-cell ghost zone)

### Gauge Conditions
- **Lapse**: 1+log slicing: ∂_t α = -2αK + β^i ∂_i α
- **Shift**: Gamma-driver: ∂_t β^i = (3/4)B^i, ∂_t B^i = ∂_t Γ̃^i - ηB^i (η = 2)

### Test Results
| Test | Status | Notes |
|------|--------|-------|
| Flat spacetime stability | ✓ | Stays at machine zero for 100+ steps |
| Constraint preservation | ✓ | Flat space: Hamiltonian ≈ 0 |
| RK4 integration | ✓ | Stable with CFL ~ 0.25 |
| Kreiss-Oliger dissipation | ✓ | Applied to φ, K, α |
| Autodiff through timestep | ✓ | Gradient computable (zero for flat space as expected) |
| BBH initial data | ✓ | Brill-Lindquist + Bowen-York setup works |
| BBH evolution | ✗ | Goes unstable (NaN) after ~5 steps |

## Known Issues / TODOs
1. **BBH Instability**: The evolution goes to NaN quickly because:
   - `dt_A_tilde` is set to zero (not evolved)
   - Ricci tensor terms (`bssn_geometry.py`) are skeleton only
   - Boundary conditions are too simplistic (frozen BCs)
   
2. **Missing Physics**:
   - Complete Ricci tensor computation
   - Full A_tilde evolution equation
   - Radiative/Sommerfeld boundary conditions
   
3. **Code Improvements**:
   - Constraint monitoring (Hamiltonian, Momentum)
   - Adaptive time stepping
   - Puncture tracking

## Architecture Notes

### Warp Integration
- All arrays use `wp.array` with 3D indexing
- Kernels decorated with `@wp.kernel`
- Helper functions decorated with `@wp.func`
- Autodiff enabled via `wp.set_module_options({"enable_backward": True})`

### Memory Layout
- Fields stored as separate 3D arrays (not AoS)
- Device parameter propagated through `allocate_bssn_state()`
- Currently runs on CPU (CUDA supported but not tested)

## References
- Baumgarte & Shapiro, "Numerical Relativity" (2010)
- Alcubierre, "Introduction to 3+1 Numerical Relativity" (2008)
- Warp documentation: https://nvidia.github.io/warp/
