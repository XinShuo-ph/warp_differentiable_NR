# Current State

Milestone: M1
Task: 6 of 6
Status: Poisson solver verified against analytic sin(pi x)sin(pi y) (2 consistent runs); adaptive grid example remains blocked on missing CUDA.
Blockers: `warp.examples.fem.example_adaptive_grid` requires CUDA (`wp.Volume.allocate_by_voxels` -> tiles).

## Quick Resume Notes
- Working in: NR/
- Warp is installed (pip user site).
- `python3 /workspace/NR/src/m1_diffusion_autodiff.py` prints energy + grad norms (CPU ok).
- Poisson demo: `python3 /workspace/NR/src/poisson_jacobi.py --n 32 --iters 800` (rel_l2_error ~ 0.0155).
- Tests: `python3 -m unittest discover -s /workspace/NR/tests`.
