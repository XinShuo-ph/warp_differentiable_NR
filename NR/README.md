# Differentiable Numerical Relativity with Warp - Branch cursor/instructions-wrapup-completion-41b2

## Progress Summary
- **Milestone reached**: M1 (complete) + M2 (partially complete)
- **Key deliverables**:
  - Red-Black Gauss-Seidel Poisson solver with SOR relaxation
  - Analytical verification against manufactured solutions
  - BSSN evolution equations extracted from Einstein Toolkit McLachlan
  - Time integration scheme (RK4 via Method of Lines) documented
  - Grid structure and boundary conditions documented

## What Works
- [x] **Poisson solver**: 2D FD Laplacian solver using Red-Black Gauss-Seidel with Successive Over-Relaxation
- [x] **Analytical verification**: L2 error < 5e-3 against sin(πx)sin(πy) manufactured solution
- [x] **Warp FEM examples**: Diffusion, graph capture, Navier-Stokes all validated on CPU
- [x] **BSSN equation extraction**: McLachlan RHS equations documented from Einstein Toolkit
- [x] **RK4 time integration**: Method of Lines with 4 intermediate steps documented
- [ ] **Adaptive grid refinement**: Blocked - requires CUDA (wp.Volume.allocate_by_voxels)
- [ ] **Einstein Toolkit Docker runs**: Blocked - Docker daemon not available in sandbox

## Requirements

```bash
pip install warp-lang numpy pytest
```

**Python**: 3.12+
**Warp**: 1.10.1+ (CPU mode works, CUDA optional)

## Quick Start

```bash
# Run tests
python3 -m pytest NR/tests/ -v

# Run Poisson solver verification
python3 -c "
import warp as wp
from NR.src.poisson_fd import make_sin_sin_problem, solve_poisson_dirichlet_rbgs
import numpy as np

wp.init()
prob = make_sin_sin_problem(n=65)
f = wp.array(prob.f, dtype=wp.float32)
u = solve_poisson_dirichlet_rbgs(f, num_iters=400, omega=1.9)
err = np.sqrt(np.mean((u.numpy()[1:-1, 1:-1] - prob.u_exact[1:-1, 1:-1])**2))
print(f'L2 error: {err:.2e}')
"

# Run Warp example validation scripts
python3 NR/src/m1_run_example_diffusion_checksum.py
python3 NR/src/m1_run_example_graph_capture_checksum.py
python3 NR/src/m1_run_example_navier_stokes_checksum.py
```

## File Structure

```
NR/
├── README.md                # This file
├── WRAPUP_STATE.md          # Wrapup progress tracker
├── STATE.md                 # Original session state tracker
├── m1_tasks.md              # Milestone 1 task checklist
├── m2_tasks.md              # Milestone 2 task checklist
├── src/
│   ├── poisson_fd.py        # Red-Black Gauss-Seidel Poisson solver
│   ├── m1_run_example_diffusion_checksum.py      # Diffusion example validation
│   ├── m1_run_example_graph_capture_checksum.py  # Graph capture validation
│   ├── m1_run_example_navier_stokes_checksum.py  # Navier-Stokes validation
│   └── m1_run_example_adaptive_grid_checksum.py  # Adaptive grid (CUDA-only)
├── tests/
│   └── test_poisson_fd.py   # Pytest for Poisson solver
└── refs/
    ├── bssn_equations.md                # McLachlan BSSN RHS equations
    ├── m2_time_integration_mol_rk4.md   # RK4 Method of Lines config
    ├── m2_grid_and_boundary_conditions.md # BBH grid setup
    ├── m2_output_files_snippet.md       # Output file configuration
    ├── m1_autodiff_tape_snippet.md      # Warp autodiff tape API
    ├── m1_diffusion_backward_disabled_snippet.md  # Diffusion backward note
    ├── m1_navier_stokes_mesh_field_apis.md        # Mesh/field APIs
    └── m1_adaptive_grid_refinement_apis.md        # Adaptive grid APIs
```

## Implementation Details

### Poisson Solver (`src/poisson_fd.py`)

**Problem**: Solve ∇²u = f with Dirichlet boundary conditions on [0,1]² grid

**Method**: Red-Black Gauss-Seidel with Successive Over-Relaxation (SOR)
- Alternating updates on red (even i+j) and black (odd i+j) cells
- SOR factor ω = 1.9 for accelerated convergence
- 400 iterations achieve L2 error < 5e-3 on 65×65 grid

**Warp Kernel**:
```python
@wp.kernel
def _rbgs_update(u, f, h2, parity, omega):
    i, j = wp.tid()
    # Skip boundary and wrong-parity cells
    if i == 0 or j == 0 or i == n-1 or j == m-1:
        return
    if ((i + j) & 1) != parity:
        return
    # Gauss-Seidel update with SOR
    nb = u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]
    u_gs = 0.25 * (nb - h2 * f[i,j])
    u[i,j] = (1-omega)*u[i,j] + omega*u_gs
```

### BSSN Variables (from Einstein Toolkit McLachlan)

Extracted evolution variables from `ML_BSSN_EvolutionInterior.cc`:
- **φ (phi)**: Conformal factor
- **γ̃ᵢⱼ (gt11, gt12, ...)**: Conformal metric (6 components)
- **K (trK)**: Trace of extrinsic curvature
- **Ãᵢⱼ (At11, At12, ...)**: Traceless extrinsic curvature (6 components)
- **Γ̃ⁱ (Xt1, Xt2, Xt3)**: Conformal connection functions
- **α (alpha)**: Lapse function
- **βⁱ (beta1, beta2, beta3)**: Shift vector

### Numerical Methods (Einstein Toolkit Reference)

**Time Integration**:
- Method of Lines (MoL) with RK4
- 4 intermediate steps, 1 scratch level
- CFL factor: dt = 0.25 × dx

**Spatial Derivatives**:
- Standard centered differencing for advection terms
- Upwind differencing for shift advection (symmetric + antisymmetric split)
- Kreiss-Oliger dissipation: ε × ∇⁶ terms for stability

**Boundary Conditions**:
- Ghost zones: 3 cells per boundary
- Initial: Extrapolate gammas
- RHS: NewRad (radiative) with radpower=2

### Test Results

| Test | Status | Notes |
|------|--------|-------|
| Poisson RBGS verification | ✓ | L2 < 5e-3 vs sin·sin exact |
| Diffusion example checksum | ✓ | 18757.091437863 |
| Graph capture checksum | ✓ | 1212.342285156 |
| Navier-Stokes checksum | ✓ | u=24.569, p=0.000 |
| Adaptive grid (CUDA) | ✗ | wp.Volume requires CUDA |

## Known Issues / TODOs

- **M1 Task 4 Blocked**: Adaptive grid refinement requires CUDA (wp.Volume.allocate_by_voxels not supported on CPU)
- **M2 Docker Blocked**: Einstein Toolkit Docker container cannot run (no Docker daemon in sandbox)
- **Next Milestone**: M3 - Implement flat spacetime BSSN evolution in Warp
- **GPU Support**: See `notes/gpu_analysis.md` for migration analysis
