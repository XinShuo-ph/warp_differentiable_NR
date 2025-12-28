# GPU Analysis

## Current Device Usage
- Explicit `device="cpu"` in code: **Yes**
  - `poisson.py`: `device="cpu"` parameter (default)
  - `m1_run_example_diffusion.py`: `wp.ScopedDevice("cpu")`
  - `m1_diffusion_autodiff_trace.py`: `wp.ScopedDevice("cpu")`
- Default device handling: Uses `wp.ScopedDevice()` context manager consistently

## Arrays Needing Device Change
| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| poisson.py | `rhs`, `x`, `bd_rhs`, `err`, all FEM fields | cpu | Pass `device="cuda:0"` param |
| m1_diffusion_autodiff_trace.py | `val`, `loss`, `bd_rhs` | cpu | Change `ScopedDevice("cuda:0")` |
| m1_run_example_diffusion.py | Example internal arrays | cpu | Change `ScopedDevice("cuda:0")` |

## CPU-Only Operations
- `numpy` interop for final results:
  - `poisson.py:89`: `err.numpy()[0]` - read-back for L2 error
  - `m1_run_example_diffusion.py:15-17`: `dof_values.numpy()` - read-back for checksum
  - `m1_diffusion_autodiff_trace.py:40`: `bd_rhs_unit.numpy()` - baseline computation
  - `m1_diffusion_autodiff_trace.py:61-62`: `val.grad.numpy()`, `loss.numpy()` - gradient read-back
- These are end-of-computation reads and don't affect GPU execution

## Kernel Device Specification
- Kernels use explicit device: **Implicit via ScopedDevice**
- `wp.launch` device param: **Not explicitly specified** (inherits from ScopedDevice)
- `fem.integrate` device: **Implicit** (uses geometry's device)

## Changes Needed for GPU

### Low Effort (simple parameter changes)
1. Change `device="cpu"` to `device="cuda:0"` in `solve_poisson_sin_dirichlet()`
2. Change `wp.ScopedDevice("cpu")` to `wp.ScopedDevice("cuda:0")` in example scripts
3. No kernel code changes required - Warp handles CPU/GPU compilation automatically

### Medium Effort (minor code changes)
1. Add device parameter to any helper functions for consistency
2. Ensure all `wp.array()` allocations use same device as geometry
3. Add `wp.synchronize()` before numpy read-backs if timing is important

### No Changes Required
- FEM integration (`fem.integrate`) handles device automatically
- BSR matrices and CG solver work on both CPU and GPU
- Tape-based autodiff works identically on GPU

## Potential GPU Issues
- [x] Memory transfers between CPU/GPU - Only at end for result extraction (acceptable)
- [x] Array synchronization - Warp handles implicitly; add explicit sync if timing needed
- [ ] CUDA driver availability - Code already handles gracefully with CPU fallback
- [ ] Multi-GPU - Current code assumes single device; would need device mapping

## GPU Migration Strategy

### Step 1: Add device parameter (5 min)
```python
def solve_poisson_sin_dirichlet(
    *,
    resolution: int = 32,
    degree: int = 2,
    device: str = "cuda:0",  # Change default
    ...
)
```

### Step 2: Update example scripts (2 min)
```python
with wp.ScopedDevice("cuda:0"):  # Change from "cpu"
    ...
```

### Step 3: Test (5 min)
```bash
# Verify GPU is detected
python3 -c "import warp as wp; wp.init(); print(wp.get_devices())"

# Run tests with GPU
python3 -m pytest NR/tests/ -v
```

## Estimated Effort
- **Low (< 10 min)**: Device parameter changes for existing code
- **Medium (30 min)**: Add runtime device detection and fallback
- **High (N/A)**: No complex GPU-specific changes needed for current M1 code

## Notes
- Warp's design makes CPU/GPU transition trivial for this codebase
- Main blocker is CUDA driver availability, not code changes
- Adaptive grid example (`wp.Volume`) is the only truly CUDA-only feature
- For BSSN (M3+), GPU will be more important for performance
