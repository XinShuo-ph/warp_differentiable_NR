# GPU Analysis

## Current Device Usage
- Explicit device="cpu" in code: **No** - code uses Warp defaults
- Default device handling: Warp uses CPU by default when no CUDA device is available

## Arrays Needing Device Change
| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| bssn_variables.py | phi, gt_xx, gt_xy, gt_xz, gt_yy, gt_yz, gt_zz | wp.zeros (default) | Add device param |
| bssn_variables.py | At_xx, At_xy, At_xz, At_yy, At_yz, At_zz | wp.zeros (default) | Add device param |
| bssn_variables.py | Gamma_x, Gamma_y, Gamma_z, K | wp.zeros (default) | Add device param |
| bssn_variables.py | alpha, beta_x, beta_y, beta_z | wp.zeros (default) | Add device param |
| bssn_variables.py | All RHS variables (21 arrays) | wp.zeros (default) | Add device param |

**Total: 42 arrays in BSSNVariables + BSSNRHSVariables classes**

## CPU-Only Operations
- `.numpy()` calls: tests/test_derivatives.py:73,74,116
- `.numpy()` calls: tests/test_flat_evolution.py:51-53,64-67
- `.numpy()` calls: tests/test_autodiff_bssn.py:85,107,126,127
- `.numpy()` calls: tests/test_poisson_verification.py:160,165
- `.numpy()` calls: src/poisson_solver.py:149

**Note**: All `.numpy()` calls are in tests for validation, not in core evolution code.

## Kernel Device Specification
- Kernels use explicit device: **No**
- `wp.launch` device param: **Missing** - uses array device automatically
- Warp automatically infers device from input arrays

## Changes Needed for GPU

### 1. Add device parameter to BSSNVariables/BSSNRHSVariables (Low effort)
```python
class BSSNVariables:
    def __init__(self, nx, ny, nz, device="cpu"):
        self.device = device
        self.phi = wp.zeros(self.shape, dtype=wp.float32, device=device)
        # ... repeat for all 21 arrays
```

### 2. Add device parameter to GridParameters (Low effort)
Pass device through to any array allocations.

### 3. Add device parameter to BSSNEvolver (Low effort)
```python
class BSSNEvolver:
    def __init__(self, nx, ny, nz, dx, dy, dz, dt, eps_diss=0.1, device="cpu"):
        self.device = device
        self.vars = BSSNVariables(nx, ny, nz, device=device)
        # ... etc
```

### 4. Update tests for GPU validation (Medium effort)
- Add `wp.synchronize()` before `.numpy()` calls
- Or use `array.numpy()` which handles sync automatically in newer Warp versions

## Potential GPU Issues
- [ ] Memory transfers between CPU/GPU for test validation
- [ ] Array synchronization before numpy() calls
- [ ] Ensure all arrays are on same device before kernel launch
- [ ] FEM (Poisson solver) may need separate GPU adaptation

## Estimated Effort

### Low (1-2 hours):
- Add `device` parameter to all class constructors
- Pass device through initialization chain
- Update wp.zeros() calls with device parameter

### Medium (2-4 hours):
- Update tests to handle GPU arrays
- Add wp.synchronize() where needed
- Test on CUDA device

### High (may not apply):
- No complex changes identified
- Code is already structured for easy GPU migration
- Warp handles device placement automatically once arrays are on GPU

## Summary

The codebase is **well-prepared for GPU migration**:
1. No hardcoded CPU device specifications
2. All arrays created through consistent pattern (wp.zeros)
3. Kernel launches infer device from arrays
4. CPU-only operations limited to test validation code

**Migration strategy**: Add single `device` parameter that flows through class hierarchy. Estimated effort: 1-2 hours for core code, additional 1-2 hours for test updates.
