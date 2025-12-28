# Differentiable Numerical Relativity with Warp - cursor/instructions-wrapup-completion-6a86

## Progress Summary
- Milestone reached: M1 (Warp Fundamentals) - nearly complete (5/6 tasks)
- M2 blocked: Docker not available for Einstein Toolkit
- Key deliverables:
  - Poisson equation solver using Red-Black Gauss-Seidel SOR
  - Autodiff smoke test demonstrating gradient computation through warp kernels
  - Reference snippets from warp FEM examples (autodiff, mesh/field APIs)

## What Works
- [x] Poisson solver: Solves -Δu = f on [0,1]² with Dirichlet boundary conditions
- [x] Autodiff: Forward/backward pass through warp kernels (wp.Tape)
- [x] FEM API exploration: Extracted snippets for mesh, field, and integration APIs
- [ ] Adaptive grid refinement: Blocked (requires CUDA Volumes, not available on CPU)

## Requirements
```bash
pip install warp-lang numpy pytest
```

## Quick Start
```bash
# Run tests
python3 -m pytest NR/tests/ -v

# Run Poisson solver test directly
python3 -m NR.tests.test_poisson

# Run autodiff smoke test
python3 NR/src/autodiff_smoke.py --device cpu --x 3.0
```

## File Structure
```
NR/
├── src/
│   ├── poisson.py          # Poisson solver with Red-Black Gauss-Seidel SOR
│   └── autodiff_smoke.py   # Simple autodiff demonstration (y = x²)
├── tests/
│   └── test_poisson.py     # Validates Poisson solver against sin(πx)sin(πy)
└── refs/
    ├── m1_autodiff_snippets.md              # wp.Tape API from warp examples
    ├── m1_navier_stokes_mesh_field_snippets.md  # FEM mesh/field APIs
    └── m1_adaptive_grid_refinement_snippets.md  # Adaptive grid (CUDA-only)
```

## Implementation Details

### Poisson Solver
Solves the 2D Poisson equation -Δu = f with zero Dirichlet boundary conditions using:
- 5-point stencil for Laplacian discretization
- Red-Black Gauss-Seidel iteration with Successive Over-Relaxation (SOR)
- GPU-portable warp kernel with explicit device selection

Key parameters:
- Grid: n × n including boundary points
- Relaxation factor ω: configurable (default 1.8)
- Iterations: configurable (default 800)

### Autodiff Mechanism
Warp's autodiff uses `wp.Tape` to record kernel launches and replay them in reverse for gradient computation:
1. Create arrays with `requires_grad=True`
2. Wrap kernel launches in `with tape:` context
3. Call `tape.backward(loss=output)` to compute gradients
4. Access gradients via `array.grad`

### Numerical Methods
- Spatial derivatives: 5-point stencil (2nd order finite difference)
- Time integration: N/A (iterative solver, not time-stepping)
- Dissipation: N/A

### Test Results
| Test | Status | Notes |
|------|--------|-------|
| Poisson vs analytical | ✓ | L2 error < 5e-3, L∞ error < 2e-2 for n=33 |
| Autodiff gradient | ✓ | y=x² gives dy/dx=2x correctly |
| Flat spacetime stability | N/A | Not implemented (M3) |
| BSSN evolution | N/A | Not implemented (M3+) |

## Known Issues / TODOs
- M1 Task 4 blocked: `example_adaptive_grid.py` requires CUDA Volumes (`wp.Volume`), unavailable on CPU
- M2 blocked: Docker not available in this environment for Einstein Toolkit
- No BSSN implementation yet (starts at M3)
