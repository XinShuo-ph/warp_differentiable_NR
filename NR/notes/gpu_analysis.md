# GPU Analysis

## Current Device Usage
- Explicit `device="cpu"` in code: **No**
- Default device handling: Uses `wp.ScopedDevice(args.device)` context manager, which allows runtime device selection via `--device` argument

## Arrays Needing Device Change
| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| `src/poisson_jacobi.py` | `f`, `u_exact`, `u`, `u_new` | Default (inherits from ScopedDevice) | No change needed - pass `--device cuda:0` |
| `src/m1_diffusion_autodiff.py` | `u.dof_values`, `energy` | Default (inherits from ScopedDevice) | No change needed - pass `--device cuda:0` |
| `tests/test_poisson_jacobi.py` | `f`, `u_exact`, `u`, `u_new` | `wp.ScopedDevice(None)` (default device) | Change to `wp.ScopedDevice("cuda:0")` for GPU testing |

## CPU-Only Operations
- `src/poisson_jacobi.py:85-88`: `u.numpy()`, `u_exact.numpy()` - NumPy interop for error computation
  - **Impact**: Triggers GPU→CPU transfer if running on GPU
  - **Mitigation**: Keep as-is (only used post-solve for validation)
  
- `src/m1_diffusion_autodiff.py:30-31`: `rng.standard_normal()` → `wp.array()`
  - **Impact**: Initial DOF values generated on CPU, transferred to device
  - **Mitigation**: Keep as-is (one-time initialization cost)

- `src/m1_diffusion_autodiff.py:50-52`: `.numpy()` for energy and gradient printing
  - **Impact**: Small GPU→CPU transfer for scalar output
  - **Mitigation**: Keep as-is (only for output display)

## Kernel Device Specification
- Kernels use explicit device: **No** (not required)
- `wp.launch` device param: **Not specified** (defaults to current device context)
- **Note**: Warp kernels automatically dispatch to the device of their input arrays

## Changes Needed for GPU

### No Code Changes Required
The current implementation uses `wp.ScopedDevice(args.device)` which supports GPU out-of-the-box:

```bash
# Run Poisson solver on GPU
python3 NR/src/poisson_jacobi.py --device cuda:0 --n 64 --iters 1000

# Run diffusion autodiff on GPU
python3 NR/src/m1_diffusion_autodiff.py --device cuda:0
```

### Test File Updates (Optional)
For GPU-specific tests, change:
```python
# Current
with wp.ScopedDevice(None):  # Uses default device

# GPU explicit
with wp.ScopedDevice("cuda:0"):  # Force GPU
```

## Potential GPU Issues
- [x] Memory transfers between CPU/GPU - Handled cleanly (numpy interop only for output)
- [x] Array synchronization - Warp handles automatically
- [ ] CUDA Volume operations - `refs/adaptive_grid_refinement_api.py` uses `wp.Volume` which requires CUDA

### Volume/Nanogrid Blocker
The adaptive grid refinement example (`refs/adaptive_grid_refinement_api.py`) uses:
```python
wp.volume_world_to_index()
wp.volume_sample_f()
fem.adaptivity.adaptive_nanogrid_from_field()
```
These require CUDA for `wp.Volume.allocate_by_voxels` (uses GPU tiles). This is documented as blocked in M1 tasks.

## Estimated Effort

### Low (No changes, works immediately)
- `src/poisson_jacobi.py` - Just pass `--device cuda:0`
- `src/m1_diffusion_autodiff.py` - Just pass `--device cuda:0`

### Medium (Minor test updates)
- `tests/test_poisson_jacobi.py` - Parameterize for both CPU and GPU testing

### High (Requires CUDA hardware)
- `refs/adaptive_grid_refinement_api.py` - Blocked until CUDA environment available

## GPU Performance Expectations
- **Poisson Jacobi**: Expect ~10-50x speedup on GPU for large grids (n≥128)
  - Each Jacobi step is embarrassingly parallel
  - GPU kernel launch overhead dominates for small grids
  
- **FEM Integration**: Expect ~5-20x speedup
  - `fem.integrate` generates GPU-optimized kernels
  - Tape backward pass benefits from GPU parallelism

## Summary
The codebase is **GPU-ready** with no code modifications required. All device selection is handled via `wp.ScopedDevice()` context manager with command-line `--device` argument. The only blocker is the adaptive grid example which requires actual CUDA hardware for `wp.Volume` operations.
