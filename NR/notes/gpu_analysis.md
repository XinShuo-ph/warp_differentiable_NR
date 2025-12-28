# GPU Analysis

## Current Device Usage

- **Explicit device="cpu" in code**: No
- **Default device handling**: All arrays use default device (currently CPU since no CUDA available). Warp automatically falls back to CPU when CUDA is not found.

## Arrays Needing Device Change

All warp arrays in the codebase use `wp.zeros()` or `wp.array()` without explicit device specification. For GPU execution, a device parameter would need to be added.

| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| bssn_vars.py | phi, gt*, At*, Xt*, alpha, beta*, *_rhs (42 arrays) | default | Add device param to BSSNGrid |
| bssn_integrator.py | *_0, *_acc storage arrays (42 arrays) | default | Add device param to RK4Integrator |
| bssn_constraints.py | H, M1, M2, M3 | default | Add device param |
| bssn_losses.py | loss arrays | default | Add device param |
| bssn_waveform.py | Psi4, theta, phi grids | default | Add device param |

**Recommended approach**: Add a `device` parameter to `BSSNGrid.__init__()` and propagate it through all array allocations. Example:

```python
class BSSNGrid:
    def __init__(self, nx, ny, nz, dx, requires_grad=False, device="cpu"):
        self.device = device
        self.phi = wp.zeros(n_points, dtype=wp.float32, 
                           requires_grad=requires_grad, device=device)
        # ... etc
```

## CPU-Only Operations

The following operations use `.numpy()` which requires CPU data:

| Operation | File:Lines | Purpose | GPU-compatible alternative |
|-----------|------------|---------|---------------------------|
| print statistics | bssn_vars.py:195-209 | Test output | Keep as-is (test only) |
| reshape for debug | bssn_initial_data.py:287-289 | Test output | Keep as-is (test only) |
| det verification | bssn_vars.py:204-209 | Test | Keep as-is (test only) |
| constraint stats | bssn_evolution_test.py:211,227,242 | Monitoring | Use wp.utils reduction |
| gradient checks | bssn_autodiff_*.py | Testing | Keep as-is (test only) |
| scipy CG solver | poisson_solver.py, test_autodiff_diffusion.py | Linear solve | Not needed for BSSN core |

**Key finding**: All `.numpy()` calls are in test/verification code, not in the core evolution kernels. The main evolution loop (RK4 stepping + RHS computation) is fully GPU-compatible.

## Kernel Device Specification

- **Kernels use explicit device**: No. All `wp.launch()` calls use default device.
- **wp.launch device param**: Not specified (uses array device automatically)

Warp kernels automatically run on the device where their input arrays reside. No changes needed to kernel launch calls if arrays are created on GPU.

## Changes Needed for GPU

### Low Effort (< 10 lines each)

1. **Add device parameter to BSSNGrid**
   ```python
   def __init__(self, nx, ny, nz, dx, requires_grad=False, device="cuda:0"):
   ```

2. **Add device parameter to RK4Integrator**
   ```python
   def __init__(self, grid, device=None):
       device = device or grid.device  # Inherit from grid
   ```

3. **Add device parameter to constraint/loss functions**
   ```python
   def compute_constraints(grid, device=None):
       device = device or grid.device
       H = wp.zeros(grid.n_points, dtype=wp.float32, device=device)
   ```

### Medium Effort

4. **Update test files to optionally use GPU**
   - Add command-line argument for device selection
   - Keep `.numpy()` calls for final result verification (will trigger sync)

### No Changes Needed

- **Warp kernels**: Already device-agnostic
- **RK4 integration logic**: Pure kernel launches
- **BSSN RHS computation**: Pure kernel computation
- **Boundary conditions**: Pure kernel computation
- **Initial data**: Already implemented as kernels

## Potential GPU Issues

- [x] Memory transfers between CPU/GPU
  - **Status**: Only occurs in test code for verification
  - **Impact**: Minimal, test-time only
  
- [x] Array synchronization
  - **Status**: Warp handles this automatically
  - **Impact**: None expected
  
- [ ] Memory capacity
  - **Status**: 48³ grid = ~110K points × 42 arrays × 4 bytes = ~18 MB
  - **Impact**: Even 48³×48³×48³ = 110M points would be ~18 GB, fits on modern GPUs
  
- [ ] Kernel occupancy
  - **Status**: Simple element-wise operations, should scale well
  - **Impact**: May need tuning for very large grids

## Estimated Effort

### Low (< 1 hour)
- Add `device` parameter to BSSNGrid class
- Add `device` parameter to RK4Integrator class
- Add `device` parameter to constraint/loss helpers

### Medium (1-2 hours)
- Update all test files to accept device argument
- Add GPU smoke tests
- Verify results match CPU

### High (requires investigation)
- None identified. Core code is GPU-ready.

## Summary

The codebase is **well-designed for GPU portability**. Key observations:

1. **No hardcoded devices**: All arrays use default device
2. **Core logic in kernels**: All heavy computation is in `@wp.kernel` functions
3. **CPU operations are test-only**: Production code path is pure kernels
4. **Estimated GPU migration**: < 2 hours of work

To enable GPU:
```python
# Just change this in BSSNGrid:
device = "cuda:0"  # Instead of default (cpu)
# Everything else follows automatically
```
