# GPU Analysis

Analysis of code changes needed to run on CUDA backend.

## Current Device Usage

- **Explicit device="cpu" in code**: No
- **Default device handling**: Code uses `device=None` parameter which defaults to Warp's default device. This is GPU-friendly design.

### Device Parameter Locations

| File | Line | Usage |
|------|------|-------|
| `bssn_defs.py` | 53, 71 | `allocate_bssn_state(shape, device=None)` - arrays use passed device |
| `bssn_defs.py` | 102, 108 | `initialize(state, device=None)` - kernel launch uses device |

**Good**: The allocation functions already accept a `device` parameter, making GPU transition straightforward.

## Arrays Needing Device Change

| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| `bssn_defs.py` | All 24 BSSN fields | `device=None` (default) | Pass `device="cuda:0"` |
| `bssn_solver.py` | state, k_state, tmp_state, accum_state | Uses `allocate_bssn_state()` | Already supports device param |
| `test_derivatives.py` | f, df_dx, d2f_dx2 | `device=None` | Add `device="cuda:0"` |
| `test_autodiff_bssn.py` | state, rhs | Uses `allocate_bssn_state()` | Already supports device param |

**Summary**: Only need to pass `device="cuda:0"` to allocation functions. No structural changes needed.

## CPU-Only Operations

These `.numpy()` calls require GPU→CPU data transfer:

| File | Line | Operation | Purpose |
|------|------|-----------|---------|
| `bssn_defs.py` | 120, 122 | `state.phi.numpy()`, `state.alpha.numpy()` | Verification in main |
| `bssn_solver.py` | 85 | `solver.state.phi.numpy()` | Max deviation check |
| `constraints.py` | 25, 31-33 | `state.K.numpy()`, `state.A_*.numpy()` | Constraint checking |
| `test_bssn_rhs.py` | 42, 48 | `rhs.phi.numpy()`, `rhs.gamma_xx.numpy()` | Test verification |
| `test_derivatives.py` | 29, 53-54 | `f.numpy()`, `df_dx.numpy()` | Test setup and verification |
| `test_autodiff_bssn.py` | 69, 92, 112 | `.grad.numpy()`, `.numpy()` | Gradient verification |
| `poisson_test.py` | 87 | `dof_values.numpy()` | Solution verification |
| `trace_diffusion_autodiff.py` | 57, 62, 66 | `.numpy()` | Loss and gradient readout |

**Impact**: These are all in test/verification code, not in the hot path. `.numpy()` will trigger implicit synchronization and data transfer, which is fine for testing but should be minimized in production.

## Kernel Device Specification

- **Kernels use explicit device**: No (relies on default)
- **wp.launch device param**: Not explicitly specified

### Kernel Launch Analysis

```python
# Current pattern (all launches):
wp.launch(kernel, dim=shape, inputs=[...])

# GPU-compatible pattern (optional, Warp infers from arrays):
wp.launch(kernel, dim=shape, inputs=[...], device="cuda:0")
```

**Note**: Warp automatically infers the device from input arrays, so explicit `device` parameter in `wp.launch` is optional if arrays are already on the correct device.

## Changes Needed for GPU

### Low Effort (Minutes)
1. **BSSNSolver constructor**: Add `device` parameter
   ```python
   def __init__(self, res=32, domain_size=1.0, sigma=0.01, device="cuda:0"):
       ...
       self.state = allocate_bssn_state(self.shape, device=device)
   ```

2. **Test scripts**: Pass device to allocation
   ```python
   state = allocate_bssn_state((res, res, res), device="cuda:0")
   ```

### Medium Effort (Tens of Minutes)
1. **Constraint checking**: Keep `.numpy()` calls (acceptable overhead for diagnostics)
2. **Gradient seeding in autodiff test**: Current pattern works but could use wp.copy
   ```python
   # Current (works):
   rhs_alpha_grad_np = rhs.alpha.grad.numpy()
   rhs_alpha_grad_np[8, 8, 8] = 1.0
   rhs.alpha.grad = wp.array(rhs_alpha_grad_np, dtype=float, device=rhs.alpha.device)
   ```

### No Changes Needed
- All `@wp.kernel` and `@wp.func` decorators work on GPU
- `wp.Tape()` autodiff works on GPU
- Warp FEM (`poisson_test.py`) automatically handles device

## Potential GPU Issues

- [ ] **Memory transfers between CPU/GPU**: `.numpy()` calls will be slow on GPU (but only used for testing/diagnostics)
- [ ] **Array synchronization**: Not an issue - Warp handles this automatically
- [ ] **Kernel compilation**: First run on GPU will compile CUDA kernels (cached after)
- [ ] **Memory limits**: 3D grids with 24 fields × 4 states (RK4) may need monitoring for large resolutions

## Estimated Effort

### Low (< 30 min)
- Add `device` parameter to `BSSNSolver.__init__`
- Update test scripts to accept device parameter
- Test on CUDA-enabled machine

### Medium (1-2 hours)
- Add device auto-detection (prefer CUDA if available)
- Performance profiling and optimization
- Add CUDA stream management for concurrent kernel execution

### High (Future work, not needed for basic GPU support)
- Shared memory optimization for stencil operations
- Multi-GPU support with domain decomposition
- Overlap computation and communication

## Recommended GPU Migration Steps

1. **Minimal change**: Add `device` parameter to `BSSNSolver`:
   ```python
   class BSSNSolver:
       def __init__(self, res=32, domain_size=1.0, sigma=0.01, device=None):
           self.device = device
           self.state = allocate_bssn_state(self.shape, device=device)
           self.k_state = allocate_bssn_state(self.shape, device=device)
           self.tmp_state = allocate_bssn_state(self.shape, device=device)
           self.accum_state = allocate_bssn_state(self.shape, device=device)
   ```

2. **Test with**:
   ```python
   solver = BSSNSolver(res=32, device="cuda:0")
   ```

3. **Verify**: Run existing tests with CUDA device

## Conclusion

The codebase is **well-designed for GPU migration**:
- Device parameterization already exists in allocation functions
- All kernels use standard Warp patterns that work on GPU
- No CPU-specific operations in the hot path
- Estimated effort: **Low** (< 30 minutes for basic GPU support)
