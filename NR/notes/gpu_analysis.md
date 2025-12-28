# GPU Analysis

## Current Device Usage

- **Explicit device="cpu" in code**: No - code is device-agnostic
- **Default device handling**: Uses `wp.ScopedDevice(device)` pattern which accepts device parameter

The Poisson solver already supports GPU via the optional `device` parameter:

```python
def solve_poisson_dirichlet_rbgs(
    f: wp.array,
    num_iters: int,
    omega: float = 1.9,
    device: str | None = None,  # <-- GPU-ready
) -> wp.array:
```

## Arrays Needing Device Change

| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| `src/poisson_fd.py` | u (output) | Uses ScopedDevice | ✓ Already GPU-ready |
| `tests/test_poisson_fd.py` | f (input) | Default device | Pass `device="cuda:0"` to `wp.array()` |

**Key Insight**: The solver uses `wp.ScopedDevice(device)` context manager, so:
- All `wp.zeros()` calls inside respect the scoped device
- All `wp.launch()` calls inside run on the scoped device
- Input array `f` must be on the same device as the solver

## CPU-Only Operations

| Operation | File:Line | Migration Strategy |
|-----------|-----------|-------------------|
| `u.numpy()` | `tests/test_poisson_fd.py:14` | Keep for validation (auto CPU→GPU transfer) |
| `wp.array(prob.f)` | `tests/test_poisson_fd.py:11` | Add `device="cuda:0"` parameter |
| `example.*numpy()` | `src/m1_run_example_*.py` | For checksums only, can stay |

**Note**: `.numpy()` automatically handles GPU→CPU transfer in Warp, but incurs synchronization overhead. For production use, minimize these calls.

## Kernel Device Specification

- **Kernels use explicit device**: No (uses ScopedDevice pattern instead)
- **wp.launch device param**: Missing - relies on ScopedDevice context

Current pattern (implicit device from context):
```python
with wp.ScopedDevice(device):
    wp.launch(_rbgs_update, dim=(n, n), inputs=[...])
```

Explicit device pattern (alternative):
```python
wp.launch(_rbgs_update, dim=(n, n), inputs=[...], device=device)
```

Both work; ScopedDevice is cleaner for multiple launches.

## Changes Needed for GPU

### Minimal Changes (Poisson Solver)

1. **Test file** (`tests/test_poisson_fd.py`):
   ```python
   # Before
   f = wp.array(prob.f, dtype=wp.float32)
   u = solve_poisson_dirichlet_rbgs(f, num_iters=400, omega=1.9)
   
   # After (GPU)
   device = "cuda:0"
   f = wp.array(prob.f, dtype=wp.float32, device=device)
   u = solve_poisson_dirichlet_rbgs(f, num_iters=400, omega=1.9, device=device)
   ```

2. **That's it!** The solver itself is already GPU-ready.

### For Future BSSN Implementation

When implementing BSSN evolution:

1. **State arrays**: Create with explicit device
   ```python
   phi = wp.zeros((nx, ny, nz), dtype=wp.float32, device=device)
   chi = wp.zeros((nx, ny, nz), dtype=wp.float32, device=device)
   # ... etc for all BSSN fields
   ```

2. **Kernel launches**: Use ScopedDevice or explicit device param
   ```python
   with wp.ScopedDevice(device):
       wp.launch(compute_rhs, dim=(nx, ny, nz), inputs=[...])
       wp.launch(rk4_update, dim=(nx, ny, nz), inputs=[...])
   ```

3. **Boundary conditions**: May need separate kernel for ghost zones

## Potential GPU Issues

- [x] Memory transfers between CPU/GPU: **Minimal** - only for initial/output data
- [x] Array synchronization: **Handled by Warp** - wp.synchronize() if needed
- [ ] Memory usage: **Monitor** - 3D BSSN grids with 17+ fields can be large
- [ ] Kernel occupancy: **Profile** - stencil operations have limited parallelism per cell

### Memory Estimation for BSSN (Future)

For a 128³ grid with 17 BSSN fields + scratch for RK4:
- Base: 17 fields × 128³ × 4 bytes = ~145 MB
- RK4 scratch: 4 stages × 17 fields × 128³ × 4 bytes = ~580 MB
- Total: ~725 MB (fits easily on modern GPUs)

## Estimated Effort

### Low (< 1 hour)
- [x] Poisson solver GPU support: **Already done** (just pass device param)
- [ ] Test file GPU path: Add device parameter to array creation

### Medium (1-2 hours)
- [ ] Add GPU/CPU test matrix: pytest parametrize over devices
- [ ] Add benchmark script: Compare CPU vs GPU performance

### High (Part of M3+ milestones)
- [ ] Implement BSSN state management with device support
- [ ] Implement BSSN RHS kernels for GPU
- [ ] Add constraint monitoring on GPU
- [ ] Implement IO with GPU→CPU transfer

## Recommendations

1. **Use `wp.ScopedDevice`**: Current pattern is correct and clean
2. **Avoid frequent `.numpy()` calls**: Batch them or keep on GPU
3. **Consider `wp.Tape` for autodiff**: GPU-compatible automatic differentiation
4. **Profile before optimizing**: Use Warp's built-in profiling
