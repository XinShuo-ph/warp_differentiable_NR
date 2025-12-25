# Current State

Milestone: M1
Task: 4 of 6
Status: M1 complete except adaptive grid (CUDA-only); M2 blocked (docker not available)
Blockers: example_adaptive_grid.py requires CUDA (wp.Volume.allocate_by_tiles / Nanogrid); docker not installed in this environment

## Quick Resume Notes
- Working in: NR/
- Warp installed via pip: warp-lang 1.10.1
- Last run: `python3 NR/src/m1_run_example_diffusion.py` (prints checksum/l2)
- Autodiff trace: `python3 NR/src/m1_diffusion_autodiff_trace.py`
- Poisson test: `python3 -m pytest -q NR/tests`
- Warp repo cloned to: `NR/warp` (depth=1)
- Next action: If docker can be installed/enabled, run `docker pull rynge/einsteintoolkit:latest`
