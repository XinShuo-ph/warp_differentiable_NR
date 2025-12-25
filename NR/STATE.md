# Current State

Milestone: M3 (COMPLETED)
Task: 7 of 7
Status: BSSN evolution on flat spacetime working with autodiff support
Blockers: None

## Quick Resume Notes
- Warp installed: version 1.10.1 (CPU-only mode, no CUDA)
- Working in: NR/src/
- Last successful test: tests/test_bssn.py (all passing)

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

## Next: M4 - BSSN in Warp (BBH)
Goal: Reproduce BBH-like initial data evolution
- Add puncture initial data
- Implement full Ricci tensor computation
- Add Gamma-driver shift condition
- Compare with Einstein Toolkit output
