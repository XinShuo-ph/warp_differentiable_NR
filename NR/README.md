# Differentiable Numerical Relativity with Warp - cursor/instructions-wrapup-completion-f134

## Progress Summary
- Milestone reached: M1 (Warp Fundamentals)
- Key deliverables:
  - Poisson equation solver using Warp FEM with CG linear solver
  - Diffusion example integration demonstrating Warp FEM workflow
  - Autodiff tracing through FEM integration operations
  - Reference code snippets for mesh/field APIs and tape recording

## What Works
- [x] **Poisson Solver**: 2D Poisson equation with sinusoidal source and Dirichlet BCs, validated against analytical solution
- [x] **FEM Integration**: Using `warp.fem` for weak-form assembly on Grid2D geometry
- [x] **Autodiff Trace**: Gradient computation through `fem.integrate()` using `wp.Tape()`
- [x] **Diffusion Example**: Running Warp's built-in diffusion example on CPU
- [ ] **Adaptive Grid**: Blocked - requires CUDA (wp.Volume.allocate_by_tiles / Nanogrid)
- [ ] **M2 Einstein Toolkit**: Blocked - docker not available in environment

## Requirements
```bash
pip install warp-lang numpy pytest
```

## Quick Start
```bash
# Run tests
python3 -m pytest NR/tests/ -v

# Or run specific components
python3 NR/src/poisson.py                    # (import only, use test)
python3 NR/src/m1_run_example_diffusion.py   # Diffusion example
python3 NR/src/m1_diffusion_autodiff_trace.py # Autodiff demonstration
```

## File Structure
```
NR/
├── src/
│   ├── poisson.py                    # Poisson equation solver with FEM
│   ├── m1_run_example_diffusion.py   # Warp diffusion example runner
│   └── m1_diffusion_autodiff_trace.py # Autodiff through FEM integration
├── tests/
│   └── test_poisson.py               # Pytest tests for Poisson solver
├── refs/
│   ├── navier_stokes_mesh_field_api_snippet.py    # Mesh/field API reference
│   ├── warp_fem_autodiff_tape_record_func_snippet.py # Tape recording reference
│   ├── navier_stokes_timestep_saddle_solve_snippet.py # Time-stepping reference
│   └── adaptive_grid_refinement_api_snippet.py    # Adaptive grid API (CUDA-only)
├── notes/
│   └── gpu_analysis.md               # GPU migration analysis
├── m1_tasks.md                       # M1 task checklist
├── m2_tasks.md                       # M2 task checklist (blocked)
├── STATE.md                          # Original state tracker
├── WRAPUP_STATE.md                   # Wrapup progress tracker
└── README.md                         # This file
```

## Implementation Details

### Poisson Solver
Solves the 2D Poisson equation: -∇²u = f with:
- **Source term**: f(x,y) = 2π² sin(πx) sin(πy)
- **Analytical solution**: u(x,y) = sin(πx) sin(πy)
- **Boundary conditions**: Homogeneous Dirichlet (u=0 on boundary)
- **Geometry**: Grid2D (structured quadrilateral mesh)
- **Function space**: Polynomial degree 2 (Q2 elements)
- **Linear solver**: Conjugate gradient with tolerance 1e-10

### Numerical Methods
- **Spatial discretization**: Warp FEM with polynomial spaces on Grid2D
- **Weak form assembly**: `fem.integrate()` with test/trial fields
- **Boundary conditions**: Nodal projection via `fem.project_linear_system()`
- **Linear solver**: CG via `fem_example_utils.bsr_cg()`

### Autodiff Approach
- Uses `wp.Tape()` context manager to record operations
- FEM integration is differentiable when `enable_backward=True`
- Custom reduction kernels work with tape backward pass
- Gradient verified: d/dval[sum(boundary_rhs)] matches baseline

### Test Results
| Test | Status | Notes |
|------|--------|-------|
| Poisson L2 error (res=8) | ✓ | Error < 2e-3 |
| Poisson L2 error (res=16) | ✓ | Error < 2e-4 |
| Diffusion example | ✓ | Deterministic checksum |
| Autodiff gradient | ✓ | grad ≈ baseline (2.0) |
| Flat spacetime stability | N/A | Not implemented (M2+) |
| BSSN evolution | N/A | Not implemented (M2+) |

## Known Issues / TODOs
- **Adaptive grid example blocked**: `example_adaptive_grid.py` requires CUDA backend for `wp.Volume.allocate_by_tiles()` and Nanogrid APIs
- **M2 blocked**: Einstein Toolkit docker container not available in environment
- **CPU-only mode**: All current code runs on CPU; see `notes/gpu_analysis.md` for GPU migration requirements
- **No BSSN implementation yet**: M1 focuses on Warp fundamentals; BSSN evolution is M3+
