# Current State

Milestone: M3
Task: 7 of 7 (COMPLETE)
Status: M3 completed - Core BSSN implementation in Warp complete
Blockers: None

## Quick Resume Notes
- Warp 1.10.1 installed at: /home/ubuntu/.local/lib/python3.12/site-packages/warp
- Warp repo cloned at: NR/warp/
- McLachlan repo cloned at: NR/mclachlan/
- Working in: NR/src/
- References extracted to: NR/refs/
- Tests in: NR/tests/

## M1 Accomplishments
- Ran 3 FEM examples: diffusion, navier_stokes, adaptive_grid
- Documented autodiff mechanism in refs/autodiff_mechanism.md
- Documented mesh/field APIs in refs/mesh_field_apis.md
- Documented adaptive grid APIs in refs/adaptive_grid_apis.md
- Implemented Poisson solver from scratch in src/poisson_solver.py
- Verified solver with analytical solution, confirmed 2nd-order convergence

## M2 Accomplishments
- Cloned McLachlan (BSSN implementation) repository
- Extracted BSSN evolution equations from Mathematica and C++ source
- Documented complete BSSN formulation in refs/bssn_equations.md
- Documented grid structure (AMR, Carpet) and boundary conditions in refs/grid_and_boundaries.md
- Documented time integration (RK4, MoL) in refs/time_integration.md
- Analyzed parameter files to understand simulation setup

## M3 Accomplishments
- Defined all 25 BSSN variables as warp arrays in src/bssn_variables.py
- Implemented 4th-order finite difference derivatives in src/bssn_derivatives.py
  - Verified 4th-order convergence on test functions
  - Includes upwind derivatives and Kreiss-Oliger dissipation
- Implemented simplified BSSN RHS in src/bssn_rhs.py
  - Evolution equations for conformal factor, metric, gauge variables
  - Starting with flat spacetime for testing
- Implemented RK4 time integrator in src/bssn_evolver.py
  - 4-stage Runge-Kutta scheme
  - Modular design for easy extension
- Successfully evolved flat spacetime for 100+ timesteps
  - Verified stability: deviations remain at machine precision
- Verified autodiff works through evolution step
  - Gradient flow confirmed through kernels

## Code Structure
```
NR/
├── src/
│   ├── bssn_variables.py      # Variable definitions and containers
│   ├── bssn_derivatives.py    # 4th-order FD kernels
│   ├── bssn_rhs.py            # RHS computation kernels
│   ├── bssn_evolver.py        # RK4 time integrator
│   └── poisson_solver.py      # From M1
├── tests/
│   ├── test_derivatives.py         # FD convergence tests
│   ├── test_flat_evolution.py      # Flat spacetime evolution
│   ├── test_autodiff_bssn.py       # Autodiff verification
│   └── test_poisson_verification.py # From M1
└── refs/
    ├── autodiff_mechanism.md        # From M1
    ├── mesh_field_apis.md           # From M1
    ├── adaptive_grid_apis.md        # From M1
    ├── bssn_equations.md            # From M2
    ├── grid_and_boundaries.md       # From M2
    └── time_integration.md          # From M2
```

## Next Steps
- M4: BSSN in Warp (BBH) - Implement binary black hole evolution
  - Add full Ricci tensor computation
  - Implement proper initial data (e.g., Brill-Lindquist)
  - Test with single puncture black hole
  - Extend to binary black hole system
