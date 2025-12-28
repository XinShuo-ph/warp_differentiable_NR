# GPU Analysis

## Current Device Usage
- Explicit device="cpu" in code: Yes, as default parameter in `solve_poisson_dirichlet_zero`
- Default device handling: Uses `wp.ScopedDevice(device)` context manager - device is configurable

## Arrays Needing Device Change
| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| poisson.py | u, ff | cpu (default) | Pass `device="cuda:0"` to function |
| autodiff_smoke.py | x, y | cpu (default via args) | Pass `--device cuda:0` CLI arg |

## CPU-Only Operations
- `poisson.py:74`: `u.numpy()` - transfers array from device to CPU (works but involves GPU→CPU copy)
- `poisson.py:66-67`: NumPy array reshape before `wp.array()` creation - runs on CPU before GPU allocation
- `test_poisson.py:14`: Hardcoded `device="cpu"` - needs update for GPU testing

## Kernel Device Specification
- Kernels use explicit device: No (kernels are device-agnostic in Warp)
- wp.launch device param: Not specified (uses ScopedDevice context)

## Changes Needed for GPU

1. **Parameter Changes** (trivial):
   - Change `device="cpu"` default to `device=None` (use Warp default) or accept from caller
   - Update test to parameterize device

2. **Array Transfers** (minimal):
   - `.numpy()` call already handles GPU→CPU transfer automatically
   - Input arrays created from NumPy are already handled correctly

3. **Test Updates**:
   - Add pytest parameterization for `device` in `test_poisson.py`
   - Add GPU availability check: `wp.is_cuda_available()`

## Potential GPU Issues
- [x] Memory transfers between CPU/GPU: Already handled by `.numpy()` method
- [ ] Array synchronization: Not an issue (Warp handles synchronization)
- [ ] Large grid memory: May need to reduce grid size for GPU memory constraints

## Estimated Effort
- **Low** (< 1 hour):
  - Change default device parameter from "cpu" to None
  - Add `@pytest.mark.skipif(not wp.is_cuda_available())` for GPU tests
  - Pass device parameter through function calls

- **Medium** (N/A for current code):
  - None identified

- **High** (N/A for current code):
  - None identified

## Code Changes Required

### poisson.py
```python
# Change line 44
def solve_poisson_dirichlet_zero(
    f: np.ndarray,
    *,
    n: int,
    iters: int = 800,
    omega: float = 1.8,
    device: str | None = None,  # Changed from "cpu"
) -> PoissonSolveResult:
```

### test_poisson.py
```python
import pytest
import warp as wp

@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []))
def test_sine_solution_matches(self, device):
    # ... use device parameter
```

### autodiff_smoke.py
No changes needed - already accepts `--device` argument.

## Summary
The codebase is **GPU-ready** with minimal changes. Warp's design makes device portability straightforward:
- Kernels are automatically compiled for the target device
- `wp.ScopedDevice` context manager handles device selection
- Array transfers between CPU and GPU are handled transparently
