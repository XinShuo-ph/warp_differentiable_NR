# Milestone 1: Warp Fundamentals

**Goal:** Understand warp kernels, autodiff, and FEM basics.

## Tasks

- [x] 1. Install warp, run `warp.examples`
- [x] 2. Run `example_diffusion.py`, trace autodiff mechanism
- [x] 3. Run `example_navier_stokes.py`, document mesh/field APIs
- [~] 4. Run `example_adaptive_grid.py`, document refinement APIs (CUDA required)
- [x] 5. Implement Poisson equation solver from scratch
- [x] 6. Verify Poisson solver against analytical solution

## Notes

- Warp 1.10.1 installed, running in CPU-only mode (no CUDA)
- Autodiff mechanism: wp.Tape() records ops, tape.backward(loss) computes gradients
- Key FEM APIs documented in refs/warp_fem_api.py
- Poisson solver implemented in src/poisson.py, tests in tests/test_poisson.py
- Task 4 skipped: example_adaptive_grid.py requires CUDA for wp.Volume.allocate_by_tiles
