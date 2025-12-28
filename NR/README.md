# Differentiable Numerical Relativity with Warp - cursor/instructions-wrapup-completion-5d34

## Progress Summary
- Milestone reached: M1 (Foundation)
- Key deliverables:
  - Poisson equation solver with Jacobi iteration (Warp kernels)
  - Diffusion energy integrand with automatic differentiation (Warp FEM + Tape)
  - Reference API snippets for mesh/field handling, tape patterns, and adaptive grids

## What Works
- [x] **Poisson Solver** (`src/poisson_jacobi.py`): 2D Jacobi iteration with Dirichlet BCs, verified against sin(πx)sin(πy) analytic solution
- [x] **Diffusion Autodiff** (`src/m1_diffusion_autodiff.py`): Gradient energy density computation with `wp.Tape` backward pass
- [x] **Tests** (`tests/test_poisson_jacobi.py`): Convergence and determinism tests (2/2 passing)
- [x] **Reference snippets**: Tape patterns, Navier-Stokes mesh/field API, adaptive grid refinement API
- [ ] **Adaptive Grid Example**: Blocked - requires CUDA (`wp.Volume.allocate_by_voxels` needs GPU tiles)

## Requirements
```bash
pip install warp-lang numpy pytest
```

## Quick Start
```bash
# Run tests
python3 -m pytest NR/tests/ -v

# Run Poisson solver demo
python3 NR/src/poisson_jacobi.py --n 32 --iters 800
# Output: rel_l2_error: ~0.0155

# Run diffusion autodiff demo
python3 NR/src/m1_diffusion_autodiff.py
# Output: energy, grad_u norms (non-zero gradients confirm autodiff works)
```

## File Structure
```
NR/
├── src/
│   ├── poisson_jacobi.py         # Jacobi Poisson solver with Warp kernels
│   └── m1_diffusion_autodiff.py  # FEM diffusion with Tape autodiff
├── tests/
│   └── test_poisson_jacobi.py    # Unit tests for Poisson solver
├── refs/
│   ├── warp_tape_patterns.py         # Tape context manager & record_func examples
│   ├── navier_stokes_mesh_field_api.py  # FEM spaces, fields, boundary projectors
│   └── adaptive_grid_refinement_api.py  # Nanogrid adaptive refinement
├── notes/
│   └── gpu_analysis.md           # GPU migration analysis
├── README.md                     # This file
├── WRAPUP_STATE.md               # Session state tracking
├── STATE.md                      # Milestone progress
└── m1_tasks.md                   # M1 task checklist
```

## Implementation Details

### Poisson Solver (`poisson_jacobi.py`)
Solves `-Δu = f` on [0,1]² with homogeneous Dirichlet BCs using Jacobi iteration.

**Warp Kernels:**
- `init_sin_sin_rhs`: Initializes RHS `f = 2π²sin(πx)sin(πy)` and exact solution
- `jacobi_step_dirichlet0`: Single Jacobi update with boundary enforcement

**Numerical Method:**
- Spatial discretization: 2nd-order finite differences (5-point stencil)
- Iteration: Standard Jacobi (no relaxation)
- Convergence: ~800 iterations for 32×32 grid → rel_l2_error ≈ 0.0155

### Diffusion Autodiff (`m1_diffusion_autodiff.py`)
Demonstrates differentiable PDE simulation using Warp FEM.

**Key Components:**
- `fem.Grid2D`: 2D structured geometry
- `fem.make_polynomial_space`: Polynomial function space (configurable degree)
- `fem.integrate`: Domain integration with automatic kernel generation
- `wp.Tape`: Records operations for reverse-mode autodiff

**Energy Functional:**
```python
@fem.integrand
def grad_energy_density(s: fem.Sample, u: fem.Field, nu: float):
    g = fem.grad(u, s)
    return nu * wp.dot(g, g)
```

### Reference API Snippets (`refs/`)
Extracted from Warp examples for future BSSN implementation:
- **Tape patterns**: Context manager usage, custom backward hooks
- **Mesh/Field API**: Mixed function spaces, boundary condition projectors
- **Adaptive grids**: Nanogrid refinement based on SDF fields (requires CUDA)

## Test Results
| Test | Status | Notes |
|------|--------|-------|
| Poisson convergence | ✓ | rel_l2 < 0.03 threshold met |
| Poisson determinism | ✓ | Two runs produce identical results |
| Diffusion autodiff | ✓ | Non-zero gradients (l2=310, max=55) |

## Known Issues / TODOs
- Adaptive grid example (`example_adaptive_grid.py`) requires CUDA - blocked on CPU-only environment
- No BSSN implementation yet (M1 is foundation milestone)
- Kreiss-Oliger dissipation not implemented
- No RK4 time integration (only steady-state Jacobi iteration)

## GPU Readiness
See `notes/gpu_analysis.md` for detailed GPU migration analysis. Summary:
- Code uses `wp.ScopedDevice(args.device)` for device flexibility
- No hardcoded `device="cpu"` - arrays inherit from scoped device
- GPU migration is straightforward: pass `--device cuda:0` at runtime
