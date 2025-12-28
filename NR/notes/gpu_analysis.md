# GPU Analysis

## Current Device Usage
- **Explicit device="cpu" in code**: No
- **Default device handling**: All arrays created with `wp.zeros()` without device parameter, defaulting to CPU. All `wp.launch()` calls use default device (CPU).

## Arrays Needing Device Change
| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| bssn.py | phi, gamma_tilde, K, A_tilde, Gam_tilde, alpha, beta, B | default (cpu) | Add `device` param to `wp.zeros()` |
| integrator.py | (uses BSSNState allocations) | default (cpu) | via BSSNState |
| poisson.py | x (CG solution) | default (cpu) | Add `device` param |

### Detailed Array Count
- **bssn.py**: 8 arrays allocated via `wp.zeros()`
  - phi: 3D scalar
  - gamma_tilde: 4D (3D + 6 components)
  - K: 3D scalar  
  - A_tilde: 4D (3D + 6 components)
  - Gam_tilde: 4D (3D + 3 components)
  - alpha: 3D scalar
  - beta: 4D (3D + 3 components)
  - B: 4D (3D + 3 components)

## CPU-Only Operations
- **test_flat.py:34-35**: `state.K.numpy()`, `state.phi.numpy()` - automatic GPU→CPU transfer, no change needed
- **test_autodiff.py:49**: `state.phi.grad.numpy()` - automatic transfer, no change needed
- **test_poisson.py:21-22**: `solver.space.node_positions().numpy()`, `solver.field.dof_values.numpy()` - automatic transfer, no change needed

Note: `.numpy()` on GPU arrays automatically copies data to CPU. This is fine for tests/visualization but should be minimized in hot paths.

## Kernel Device Specification
- **Kernels use explicit device**: No
- **wp.launch device param**: Not present (uses default)

### Kernel Inventory
| File | Kernel | Launches |
|------|--------|----------|
| bssn.py | init_flat_kernel | 1 (`init_flat_spacetime()`) |
| rhs.py | bssn_rhs_kernel | via integrator |
| integrator.py | update_kernel | 3 per RK4 step |
| integrator.py | final_update_kernel | 1 per RK4 step |
| test_autodiff.py | init_noise_kernel, loss_kernel | test only |

## Changes Needed for GPU

### Low Effort (Simple Parameter Changes)
1. **BSSNState.__init__**: Add `device` parameter to constructor
   ```python
   def __init__(self, res, bounds_lo, bounds_hi, device="cpu"):
       self.device = device
       self.phi = wp.zeros(self.shape, dtype=float, device=device)
       # ... repeat for all arrays
   ```

2. **BSSNState.init_flat_spacetime**: Add device to wp.launch
   ```python
   wp.launch(..., device=self.device)
   ```

3. **RK4Integrator.compute_rhs/update_state/final_update**: Pass device to wp.launch
   ```python
   wp.launch(..., device=self.state.device)
   ```

4. **PoissonSolver.solve**: Pass device to wp.zeros_like
   ```python
   x = wp.zeros_like(rhs, device=...)
   ```

### Medium Effort
5. **warp.fem GPU support**: Verify warp.fem geometry, spaces, and integration work on GPU. May need:
   - `fem.Grid2D(..., device="cuda:0")`
   - Check if FEM integrators support GPU

### No Changes Needed
- **Kernel code**: Warp kernels are device-agnostic; same code runs on CPU/GPU
- **Derivatives (derivs.py)**: `@wp.func` are inlined into kernels, device-agnostic
- **Test numpy() calls**: Automatic GPU→CPU transfer works transparently

## Potential GPU Issues
- [ ] **Memory transfers between CPU/GPU**: Only in tests, acceptable
- [ ] **Array synchronization**: Warp handles automatically, should be fine
- [ ] **warp.fem GPU compatibility**: Need to verify FEM module works on GPU
- [ ] **Autodiff GPU**: Need to verify tape.backward() works on GPU
- [ ] **Large grid memory**: GPU memory limits may constrain resolution

## Estimated Effort

### Low (< 1 hour)
- Add `device` parameter to BSSNState constructor
- Propagate device to all wp.zeros() calls
- Add device to wp.launch() calls in integrator

### Medium (1-2 hours)  
- Test and verify all kernels work on GPU
- Verify autodiff works on GPU
- Profile and optimize if needed

### High (if needed)
- warp.fem GPU adaptation (if not trivially supported)
- Memory optimization for large grids
- Multi-GPU support (future)

## Recommended Migration Steps
1. Add `device` parameter to `BSSNState` (default to "cpu" for backward compatibility)
2. Propagate device to `RK4Integrator` and all `wp.launch()` calls
3. Run existing tests with `device="cuda:0"` on GPU machine
4. Profile and compare CPU vs GPU performance
5. Address any GPU-specific issues that arise

## GPU Performance Expectations
- **BSSN evolution**: Should see significant speedup due to embarrassingly parallel nature
- **Expected bottleneck**: Memory bandwidth (stencil operations are memory-bound)
- **Autodiff**: May have overhead for tape recording; profile to verify
