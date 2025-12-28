# Differentiable Numerical Relativity with Warp

## Progress Summary
- **Milestone reached**: M3 (BSSN in Warp - Core)
- **Key deliverables**:
  - Complete BSSN variable definitions (25 evolved fields)
  - 4th-order finite difference operators with verified convergence
  - BSSN RHS evolution equations (simplified for flat spacetime)
  - RK4 time integrator with Kreiss-Oliger dissipation
  - Autodiff verification through evolution timesteps
  - Poisson solver (from M1)

## What Works
- [x] **Flat spacetime evolution**: Stable for 100+ timesteps, deviations remain at machine precision
- [x] **4th-order spatial derivatives**: Verified convergence rate of 4.0 on test functions
- [x] **RK4 time integration**: 4-stage Runge-Kutta with CFL stability
- [x] **Kreiss-Oliger dissipation**: 5th-order dissipation for high-frequency damping
- [x] **Autodiff through timesteps**: Gradient flow verified through evolution kernels
- [x] **Poisson solver**: 2nd-order convergence verified for degree-1 elements
- [ ] **Full Ricci tensor**: Simplified for flat spacetime testing
- [ ] **Non-trivial initial data**: Not yet implemented (black holes, punctures)

## Requirements
```bash
pip install warp-lang numpy pytest
```

## Quick Start
```bash
cd NR

# Run all tests
python3 -m pytest tests/ -v

# Run specific components
python3 tests/test_flat_evolution.py      # BSSN flat spacetime evolution
python3 tests/test_derivatives.py         # FD convergence verification  
python3 tests/test_autodiff_bssn.py       # Autodiff verification
python3 src/poisson_solver.py             # Poisson equation test
```

## File Structure
```
NR/
├── src/
│   ├── bssn_variables.py       # BSSN variable containers (25 evolved fields)
│   ├── bssn_derivatives.py     # 4th-order FD operators
│   ├── bssn_rhs.py             # Evolution equation RHS kernels
│   ├── bssn_evolver.py         # RK4 time integrator
│   └── poisson_solver.py       # Warp FEM Poisson solver (M1)
├── tests/
│   ├── test_derivatives.py         # FD convergence tests
│   ├── test_flat_evolution.py      # Flat spacetime stability
│   ├── test_autodiff_bssn.py       # Autodiff verification
│   └── test_poisson_verification.py # Poisson convergence study
└── refs/
    ├── autodiff_mechanism.md        # Warp autodiff documentation
    ├── mesh_field_apis.md           # FEM mesh/field APIs
    ├── adaptive_grid_apis.md        # AMR APIs
    ├── bssn_equations.md            # Complete BSSN formulation
    ├── grid_and_boundaries.md       # Grid structure, BCs
    └── time_integration.md          # RK4, MoL, CFL conditions
```

## Implementation Details

### BSSN Variables
All 25 evolved BSSN variables implemented in `bssn_variables.py`:
- `phi`: Conformal factor (log form)
- `gt_xx, gt_xy, gt_xz, gt_yy, gt_yz, gt_zz`: Conformal metric (6 components)
- `At_xx, At_xy, At_xz, At_yy, At_yz, At_zz`: Traceless extrinsic curvature (6)
- `Gamma_x, Gamma_y, Gamma_z`: Contracted Christoffel symbols (3)
- `K`: Trace of extrinsic curvature
- `alpha`: Lapse function
- `beta_x, beta_y, beta_z`: Shift vector (3)

### Numerical Methods
- **Spatial derivatives**: 4th-order centered finite differences
  - First derivatives: `(-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*dx)`
  - Second derivatives: 4th-order accurate
  - Mixed derivatives: Nested 4th-order stencils
- **Time integration**: Classical RK4 (4-stage Runge-Kutta)
- **Dissipation**: 5th-order Kreiss-Oliger for high-frequency damping
- **Advection**: Upwind differencing based on shift direction

### Gauge Conditions
- **Lapse**: 1+log slicing: `∂_t α = -2αK + β^i∂_iα`
- **Shift**: Gamma driver: `∂_t β^i = (3/4)Γ^i + β^j∂_jβ^i`

### Test Results
| Test | Status | Notes |
|------|--------|-------|
| Flat spacetime stability | ✓ | 100+ steps, deviations < 1e-6 |
| Constraint preservation | ✓ | Zero initial data preserved |
| 4th-order FD convergence | ✓ | Rate 4.0 on sin(πx)cos(πy) |
| Autodiff gradient flow | ✓ | Non-zero gradients through timestep |
| Poisson solver (deg 1) | ✓ | Convergence rate 2.0 |

## Known Issues / TODOs
- Full Ricci tensor computation not implemented (simplified for flat spacetime)
- No non-trivial initial data (punctures, Brill-Lindquist)
- Constraint monitoring not implemented
- Matter sources not implemented
- Gravitational wave extraction not implemented

## Next Steps (M4: BSSN with Black Holes)
1. Implement full conformal Ricci tensor
2. Add Brill-Lindquist puncture initial data
3. Implement constraint monitoring (Hamiltonian, momentum)
4. Test with single Schwarzschild black hole
5. Extend to binary black hole system
