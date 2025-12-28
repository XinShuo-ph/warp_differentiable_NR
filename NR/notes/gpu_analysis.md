# GPU Analysis

## Current Device Usage

- **Explicit device="cpu" in code**: Yes, via `wp.ScopedDevice("cpu")` context managers
  - `src/poisson_solver.py:99`
  - `src/verify_poisson.py:120`
  - `src/test_autodiff_diffusion.py:45`
  - `tests/test_bssn_vars.py:15`
  - `tests/test_bssn_derivatives.py:30`
  - `tests/test_poisson.py:24`

- **Default device handling**: Warp defaults to first available CUDA device if present, otherwise CPU

## Arrays Needing Device Change

| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| bssn_vars.py | W, gamt_xx, gamt_xy, gamt_xz, gamt_yy, gamt_yz, gamt_zz | `wp.zeros(shape, dtype)` | Add `device` param to `BSSNVars.__init__` |
| bssn_vars.py | exKh, exAt_xx/xy/xz/yy/yz/zz | `wp.zeros(shape, dtype)` | Same - single device param |
| bssn_vars.py | trGt_x/y/z, alpha, beta_x/y/z | `wp.zeros(shape, dtype)` | Same - single device param |
| poisson_solver.py | x, bd_rhs | `wp.zeros_like(rhs)` | Inherits from FEM arrays (automatic) |
| test_autodiff_diffusion.py | x | `wp.zeros_like(rhs)` | Inherits from FEM arrays (automatic) |

**Total BSSN arrays**: 21 arrays × grid_size³ × 4 bytes (float32)
- For 64³ grid: 21 × 262144 × 4 = ~22 MB
- For 128³ grid: 21 × 2097152 × 4 = ~176 MB
- For 256³ grid: 21 × 16777216 × 4 = ~1.4 GB

## CPU-Only Operations

| Operation | File:Line | Issue | Solution |
|-----------|-----------|-------|----------|
| `.numpy()` | tests/test_bssn_derivatives.py:55 | Requires CPU copy | Use only for validation/testing |
| `.numpy()` | tests/test_bssn_vars.py:23,27-29,36,40 | Requires CPU copy | Use only for validation/testing |
| `.numpy()` | src/poisson_solver.py:81,87 | Requires CPU copy | Use only for validation/testing |
| `.numpy()` | src/test_autodiff_diffusion.py:39 | Requires CPU copy | Use only for validation/testing |

**Note**: The `.numpy()` calls are only in test/verification code, not in the core computation path.

## Kernel Device Specification

- **Kernels use explicit device**: No, device is inferred from array locations
- **wp.launch device param**: Not explicitly specified; inherits from input arrays
- **Example** (from test_bssn_derivatives.py):
  ```python
  wp.launch(
      kernel=compute_deriv_x_kernel,
      dim=(res, res, res),
      inputs=[f, df, grid.idx, res, res, res]
  )
  ```
  Device will match where `f` and `df` arrays are allocated.

## FEM Module GPU Support

The `warp.fem` module (used in poisson_solver.py) has built-in GPU support:
- `fem.Grid2D` can be created on any device
- Integration and assembly use device from geometry
- `fem.integrate()` returns arrays on same device as geometry
- BiCGSTAB solver in `warp.optim.linear` supports GPU

## Changes Needed for GPU

### 1. Low Effort (Immediate)
```python
# In bssn_vars.py, modify BSSNVars.__init__:
def __init__(self, resolution: int, device: str = None):
    self.device = device
    self.W = wp.zeros(self.shape, dtype=wp.float32, device=device)
    # ... same for all arrays
```

### 2. Low Effort (Test/Verification Files)
```python
# Replace wp.ScopedDevice("cpu") with:
device = "cuda:0" if wp.is_cuda_available() else "cpu"
with wp.ScopedDevice(device):
    # ... test code
```

### 3. Medium Effort (Poisson Solver)
```python
# In solve_poisson(), add device parameter:
def solve_poisson(resolution=32, degree=2, device=None):
    geo = fem.Grid2D(res=wp.vec2i(resolution), device=device)
    # Rest should work automatically
```

### 4. Future (BSSN Evolution)
When implementing BSSN RHS computation:
- Ensure all kernel launches use arrays on same device
- Consider device synchronization for multi-step time integration
- Profile memory bandwidth for stencil operations

## Potential GPU Issues

- [ ] **Memory transfers between CPU/GPU**: Only in test code, not in hot path
- [ ] **Array synchronization**: May need `wp.synchronize()` between kernel launches in time evolution
- [ ] **Mixed precision**: Currently all float32, may want float64 for numerical stability
- [ ] **Boundary conditions**: One-sided stencils at boundaries may have divergent control flow
- [ ] **Large grid memory**: 256³ with 21 fields requires ~1.4 GB VRAM

## Estimated Effort

### Low (< 1 hour)
- Add `device` parameter to `BSSNVars` class
- Update test files to auto-detect GPU
- Run existing tests on GPU

### Medium (1-4 hours)
- Add device parameter to `solve_poisson`
- Ensure FEM components work on GPU
- Add GPU-specific tests

### High (> 4 hours)
- Profile GPU performance vs CPU
- Optimize memory access patterns for stencils
- Implement GPU-specific boundary handling
- Add multi-GPU support for large domains

## Recommendations

1. **Start simple**: Add `device` parameter to `BSSNVars` and test on GPU
2. **Keep CPU fallback**: Auto-detect GPU availability with CPU fallback
3. **Profile early**: Compare CPU vs GPU performance for derivative kernels
4. **Batch operations**: When implementing time integration, batch all RHS computations before synchronizing
5. **Memory planning**: For large grids (>128³), consider memory layout optimization
