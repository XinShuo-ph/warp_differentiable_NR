# Current State

Milestone: M1
Task: 6 of 6 (Task 4 blocked)
Status: Implemented FD Poisson solver (RBGS) in Warp + passing analytical verification test
Blockers: M1 Task 4 requires CUDA (wp.Volume.allocate_by_voxels not supported on CPU)

## Quick Resume Notes
- Working in: NR/
- Warp installed (pip): warp-lang 1.10.1
- Local warp repo clone: NR/warp
- Validation script: NR/src/m1_run_example_graph_capture_checksum.py
- Last checksum (2 runs): 1212.342285156
- Diffusion validation script: NR/src/m1_run_example_diffusion_checksum.py
- Last diffusion checksum (2 runs): 18757.091437863
- Navier-Stokes validation script: NR/src/m1_run_example_navier_stokes_checksum.py
- Last Navier-Stokes checksum (2 runs):
  - sum(u)=24.569379807
  - sum(p)=0.000000000
- Autodiff refs:
  - NR/refs/m1_autodiff_tape_snippet.md
  - NR/refs/m1_diffusion_backward_disabled_snippet.md
- Navier-Stokes refs:
  - NR/refs/m1_navier_stokes_mesh_field_apis.md
- Adaptive-grid refs:
  - NR/refs/m1_adaptive_grid_refinement_apis.md
- Poisson solver:
  - NR/src/poisson_fd.py
  - NR/tests/test_poisson_fd.py
- Next: Start M2 (Docker-based Einstein Toolkit familiarization)

---

Milestone: M2
Task: 1 of 5
Status: Extracted BBH parfile snippets + McLachlan BSSN RHS snippets; Docker-based run remains blocked
Blockers: Docker daemon cannot run in this sandbox (no systemd, overlayfs/fuse-overlayfs missing, iptables nat not permitted)

## Quick Resume Notes (M2)
- BBH example parfiles: `NR/einsteintoolkit/repos/einsteinexamples/par/arXiv-1111.3344/bbh/`
- Extracts:
  - `NR/refs/bssn_equations.md`
  - `NR/refs/m2_time_integration_mol_rk4.md`
  - `NR/refs/m2_grid_and_boundary_conditions.md`
  - `NR/refs/m2_output_files_snippet.md`
