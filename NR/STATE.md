# Current State

Milestone: M5 (COMPLETED)
Task: 6 of 6
Status: Full BSSN evolution driver with boundary conditions and constraint monitoring
Blockers: None

## Quick Resume Notes
- Warp installed: version 1.10.1 (CPU-only mode, no CUDA)
- Working in: NR/src/
- All tests passing

## Completed Milestones

### M1: Warp Fundamentals (DONE)
- refs/warp_autodiff.py - autodiff mechanism documentation
- refs/warp_fem_mesh_field.py - mesh/field API documentation  
- refs/warp_fem_adaptive.py - adaptive grid API documentation
- src/poisson.py - Poisson equation solver
- tests/test_poisson.py - verification tests

### M2: Einstein Toolkit Familiarization (PARTIAL - Docker blocked)
- refs/bssn_equations.md - BSSN evolution equations, boundary conditions, grid structure
- refs/mclachlan/ - McLachlan source code reference

### M3: BSSN in Warp Core (DONE)
- src/bssn.py - BSSN state variables, 4th order FD, RHS, RK4, KO dissipation
- tests/test_bssn.py - stability and constraint tests
- Features:
  - BSSNState struct with all 21 evolved variables
  - 4th order centered finite differences
  - 6th order Kreiss-Oliger dissipation
  - Flat spacetime evolution (100+ steps stable)
  - Autodiff support via wp.Tape

### M4: BSSN in Warp BBH (DONE)
- src/bssn_full.py - Full BSSN with Ricci tensor, Gamma-driver, initial data
- tests/test_bssn_full.py - gauge wave and puncture tests
- Features:
  - Christoffel symbol computation
  - Conformal Ricci tensor  
  - Full BSSN RHS with curvature terms
  - Gamma-driver shift condition (3/4 Γ̃^i - η β^i)
  - Gauge wave initial data (analytic test)
  - Brill-Lindquist (single puncture) initial data
  - Pre-collapsed lapse for puncture stability

### M5: Full Toolkit Port (DONE)
- src/bssn_evolve.py - Complete evolution driver
- Features:
  - Radiative (Sommerfeld) boundary conditions
  - RK4 time integration with all intermediate stages
  - Hamiltonian constraint monitoring (L2 and L∞ norms)
  - Checkpoint save/load capability
  - Flat spacetime evolution: 100+ steps stable
  - Gauge wave evolution: 100+ steps with bounded lapse
  - Puncture initial data: correctly initialized (full evolution requires moving-puncture gauge)

## Summary

Implemented differentiable numerical relativity in NVIDIA Warp:
- BSSN formulation with all 21 evolved variables
- 4th order spatial finite differences
- 6th order Kreiss-Oliger dissipation  
- RK4 time integration
- Gamma-driver shift condition and 1+log lapse
- Radiative boundary conditions
- Constraint monitoring
- Support for Warp autodiff via wp.Tape

### Limitations / Future Work
1. **No CUDA**: Running in CPU-only mode (Warp CUDA driver not found)
2. **Puncture evolution**: Requires moving-puncture gauge conditions for stability
3. **No AMR**: Uniform grids only (AMR would need CUDA for Warp's adaptive features)
4. **No Einstein Toolkit comparison**: Docker unavailable for running ET
