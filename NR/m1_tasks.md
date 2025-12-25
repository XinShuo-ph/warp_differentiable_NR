# Milestone 1: Warp Fundamentals

**Goal:** Understand warp kernels, autodiff, and FEM basics.

## Tasks

- [x] 1. Install warp, run `warp.examples`
- [x] 2. Run `example_diffusion.py`, trace autodiff mechanism
- [x] 3. Run `example_navier_stokes.py`, document mesh/field APIs
- [x] 4. Run `example_adaptive_grid.py`, document refinement APIs (source read only, requires CUDA)
- [x] 5. Implement Poisson equation solver from scratch
- [x] 6. Verify Poisson solver against analytical solution

## Summary
- Warp 1.10.1 installed (CPU mode, no CUDA)
- Autodiff mechanism: wp.Tape() for recording, tape.backward(loss) for gradients
- FEM APIs: fem.Grid2D, fem.Cells, fem.make_polynomial_space, fem.integrate
- Poisson solver shows O(h^3) convergence at coarse grids
