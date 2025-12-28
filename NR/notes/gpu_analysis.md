# GPU Analysis

Analysis of code changes required to run on CUDA backend.

## Current Device Usage

- Explicit `device="cpu"` in code: **No**
- Default device handling: Code uses Warp's default device (CPU if CUDA unavailable)
- Warp auto-detection: Warp 1.10.1 automatically uses CPU when CUDA is not available

## Arrays Needing Device Change

| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| `bssn_warp.py` | `vars`, `rhs`, `k1-k4`, `temp` | Default (no device param) | Add `device=device` param |
| `bssn_warp.py` | `wp.from_numpy()` calls | Default | Add `device=device` param |
| `poisson_solver.py` | `x`, CG arrays | Default/numpy | Add device param, use Warp arrays |

### Detailed Array List (bssn_warp.py)

```python
# BSSNGrid.__init__ - lines 84-92
self.vars = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32)  # needs device=
self.rhs = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32)   # needs device=
self.k1 = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32)    # needs device=
self.k2 = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32)    # needs device=
self.k3 = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32)    # needs device=
self.k4 = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32)    # needs device=
self.temp = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32)  # needs device=

# initialize_flat_spacetime - line 138
self.vars = wp.from_numpy(vars_np, dtype=wp.float32)  # needs device=
```

## CPU-Only Operations

| Operation | File:Line | Description | GPU Solution |
|-----------|-----------|-------------|--------------|
| `np.zeros()` | bssn_warp.py:98 | Initialize flat spacetime | Use `wp.zeros()` with kernel init |
| `vars.numpy()` | bssn_warp.py:290,306 | Extract values for checking | Keep for validation, or use `wp.sum()` |
| `np.copy()` | bssn_warp.py:290 | Copy initial state | Use `wp.copy()` |
| Numpy CG | poisson_solver.py:78-122 | CG solver in numpy | Use `warp.sparse` CG or keep on CPU |

## Kernel Device Specification

- Kernels use explicit device: **No**
- `wp.launch` device param: **Missing**

### Current kernel launches (bssn_warp.py):
```python
wp.launch(compute_rhs_kernel, dim=(...), inputs=[...])  # missing device=
wp.launch(rk4_stage_kernel, dim=(...), inputs=[...])    # missing device=
wp.launch(rk4_final_kernel, dim=(...), inputs=[...])    # missing device=
```

## Changes Needed for GPU

### 1. Add Device Parameter to BSSNGrid (Low Effort)

```python
class BSSNGrid:
    def __init__(self, nx, ny, nz, ..., device="cuda:0"):
        self.device = device
        self.vars = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32, device=device)
        # ... same for all arrays
```

### 2. Update wp.launch Calls (Low Effort)

```python
wp.launch(
    compute_rhs_kernel,
    dim=(grid.nx, grid.ny, grid.nz),
    inputs=[...],
    device=grid.device  # Add this
)
```

### 3. Update wp.from_numpy Calls (Low Effort)

```python
self.vars = wp.from_numpy(vars_np, dtype=wp.float32, device=self.device)
```

### 4. Use Kernel-based Initialization (Medium Effort)

Replace numpy-based initialization with a Warp kernel:

```python
@wp.kernel
def initialize_flat_kernel(vars: wp.array4d(dtype=wp.float32)):
    i, j, k = wp.tid()
    # Set flat spacetime values
    vars[i, j, k, PHI] = 0.0
    vars[i, j, k, GT11] = 1.0
    # ...
```

### 5. Keep Validation on CPU (No Change)

For validation/testing, keep `.numpy()` calls - they will automatically transfer from GPU to CPU.

## Potential GPU Issues

- [ ] Memory transfers between CPU/GPU - only for initialization and validation
- [ ] Array synchronization - Warp handles automatically with `wp.synchronize()`
- [ ] Grid size limitations - current 16³-64³ grids are small, GPU shines at larger sizes
- [ ] Atomic operations - not currently used
- [ ] Shared memory - not currently used (could optimize stencil operations)

## Estimated Effort

### Low (1-2 hours):
- Add `device` parameter to `BSSNGrid` class
- Add `device` parameter to all `wp.zeros()` calls
- Add `device` parameter to all `wp.launch()` calls
- Add `device` parameter to `wp.from_numpy()` calls

### Medium (4-8 hours):
- Replace numpy-based initialization with Warp kernel
- Optimize Poisson solver for GPU (use Warp sparse CG)
- Add device selection CLI flag

### High (Not Required):
- Shared memory optimization for stencil operations
- Multi-GPU support
- Stream-based overlap of computation and communication

## Recommended GPU Migration Steps

1. **Add device parameter** to `BSSNGrid.__init__` with default `"cpu"`
2. **Propagate device** to all array allocations and kernel launches
3. **Test with `device="cuda:0"`** when CUDA is available
4. **Optimize initialization** with a Warp kernel (optional)
5. **Profile and optimize** for larger grid sizes

## Code Example: GPU-Ready BSSNGrid

```python
class BSSNGrid:
    def __init__(self, nx, ny, nz, xmin=-1.0, xmax=1.0, ..., device=None):
        if device is None:
            device = wp.get_preferred_device()
        self.device = device
        
        self.vars = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32, device=device)
        self.rhs = wp.zeros((nx, ny, nz, NUM_VARS), dtype=wp.float32, device=device)
        # ... etc
    
    def initialize_flat_spacetime(self):
        wp.launch(
            initialize_flat_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[self.vars],
            device=self.device
        )
```

## Summary

The codebase is **well-structured for GPU migration**. The main changes are:

1. Adding `device` parameters (straightforward)
2. Updating kernel launches with device (straightforward)
3. Optionally converting numpy initialization to Warp kernels

**Estimated total effort: 2-4 hours for basic GPU support**

The Warp framework handles most GPU complexities automatically. No fundamental architectural changes are needed.
