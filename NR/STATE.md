# Current State

Milestone: M1
Task: 4 of 6
Status: Poisson solver implemented + tested; adaptive grid example still blocked (CUDA-only)
Blockers: example_adaptive_grid.py requires CUDA (wp.Volume.allocate_by_tiles / Nanogrid)

## Quick Resume Notes
- Working in: NR/
- Warp installed via pip: warp-lang 1.10.1
- Last run: `python3 NR/src/m1_run_example_diffusion.py` (prints checksum/l2)
- Autodiff trace: `python3 NR/src/m1_diffusion_autodiff_trace.py`
- Poisson test: `python3 -m pytest -q NR/tests`
- Next action: Revisit Task 4 on a CUDA-capable machine, or replace with a CPU-compatible third FEM example
