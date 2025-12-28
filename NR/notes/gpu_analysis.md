# GPU Analysis

Analysis of GPU readiness for the differentiable BSSN implementation.

## Current Device Usage

### Explicit device="cpu" in code
**Yes** - in `bssn_state.py`:

```python
class BSSNState:
    def __init__(self, nx, ny, nz, device='cpu'):
        self.device = device
        self.chi = wp.zeros(npts, dtype=float, device=device)
        # ... all arrays use device parameter
```

The `device` parameter is already supported but defaults to `'cpu'`.

### Default device handling
- `BSSNState.__init__()` accepts `device` parameter (good design)
- Other modules don't specify device in `wp.launch()` calls (uses warp default)
- RHS/RK4 allocations use default device (no explicit device param)

## Arrays Needing Device Change

| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| `bssn_state.py` | chi, K, alpha, beta, Gamma_tilde, gamma_tilde, A_tilde | device param (default 'cpu') | Pass `device='cuda:0'` |
| `bssn_rhs.py` | rhs_chi, rhs_K, rhs_alpha, rhs_gamma, rhs_A, rhs_Gamma, rhs_beta, B_tilde, rhs_B | No device param | Add device param to `BSSNEvolver.__init__()` |
| `bssn_rk4.py` | temp_state, k2_chi, k3_chi, k4_chi | No device param | Add device param to `RK4Integrator.__init__()` |
| `bbh_initial_data.py` | (uses state arrays) | Inherits from state | No change needed |
| `bssn_rhs_full.py` | rhs_chi, rhs_K, etc. (test only) | No device param | Add device param |
| `poisson_solver.py` | FEM arrays | Warp FEM handles | Check `wp.fem` device support |

## CPU-Only Operations

Identified CPU-only patterns:

| Operation | File:Line | Impact | Fix |
|-----------|-----------|--------|-----|
| `np.ones/zeros` in `set_flat_spacetime()` | `bssn_state.py:61-79` | Low - one-time init | Use `wp.ones/zeros` or keep with `.assign()` |
| `.numpy()` calls | `bssn_state.py:91-100` | Medium - data export | Keep for output/analysis only |
| `.numpy()` in RK4 step check | `bssn_rk4.py:129-136` | Medium - per step | Remove or make optional for GPU |
| `np.abs().max()` constraint check | `bssn_rhs.py:207-209` | Low - debug only | Use warp reduction kernels |
| Print statements with `.numpy()` | Various | Low - debug | Keep but make optional |

### Critical Path Analysis
The evolution loop in `RK4Integrator.step()` has CPU synchronization:
```python
chi_before = state.chi.numpy().copy()  # GPU->CPU transfer
# ... step ...
chi_after = state.chi.numpy()          # GPU->CPU transfer
return np.allclose(chi_before, chi_after)
```

This should be removed or made optional for GPU performance.

## Kernel Device Specification

### Kernels use explicit device
**No** - All `wp.launch()` calls omit device parameter:

```python
wp.launch(
    compute_bssn_rhs_flat,
    dim=(nx, ny, nz),
    inputs=[...]  # No device= param
)
```

### wp.launch device param
**Missing** from all launches. Warp infers device from input arrays.

**Recommendation**: Since arrays carry device info, this should work correctly when arrays are on GPU. No change strictly required, but adding explicit device improves clarity:

```python
wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)
```

## Changes Needed for GPU

### Low Effort (1-2 hours)

1. **Pass device to BSSNEvolver**
   ```python
   class BSSNEvolver:
       def __init__(self, state: BSSNState, dx, dy, dz):
           device = state.device  # Get device from state
           self.rhs_chi = wp.zeros(npts, dtype=float, device=device)
           # ... same for all arrays
   ```

2. **Pass device to RK4Integrator**
   ```python
   class RK4Integrator:
       def __init__(self, evolver: BSSNEvolver, dt: float):
           device = evolver.state.device
           self.temp_state = BSSNState(nx, ny, nz, device=device)
           self.k2_chi = wp.zeros(npts, dtype=float, device=device)
   ```

3. **Remove CPU sync in RK4.step()**
   ```python
   def step(self):
       # Remove numpy() calls in hot path
       self.evolver.compute_rhs()
       # Just update state, no validation
   ```

### Medium Effort (2-4 hours)

4. **Add GPU constraint monitor**
   ```python
   @wp.kernel
   def compute_max_constraint(chi: wp.array, max_val: wp.array):
       idx = wp.tid()
       wp.atomic_max(max_val, 0, wp.abs(chi[idx] - 1.0))
   ```

5. **GPU-friendly diagnostics**
   - Move `.numpy()` calls to optional debug mode
   - Use warp reduction kernels for max/min/sum

### No Change Needed

- `wp.init()` - Auto-detects GPU
- Kernel definitions - Device agnostic
- `@wp.func` definitions - Device agnostic
- Struct types (`SymmetricTensor3`) - Device agnostic
- `wp.Tape()` - Works on GPU

## Potential GPU Issues

- [x] **Memory transfers between CPU/GPU**: Limited to init and output
- [ ] **Array synchronization**: No explicit sync needed (warp handles)
- [ ] **Shared memory usage**: Not used (could optimize stencils)
- [ ] **Thread divergence**: Boundary conditions have branches (minor impact)
- [ ] **Memory coalescing**: Linear indexing is good for coalescing
- [ ] **Register pressure**: Complex kernels may hit limits (monitor)

## Estimated Effort

### Low (< 1 day)
- Add device parameter propagation to BSSNEvolver, RK4Integrator
- Remove debug `.numpy()` from hot path
- Test with `device='cuda:0'`

### Medium (1-2 days)
- Add GPU reduction kernels for diagnostics
- Profile and identify bottlenecks
- Optimize memory access patterns

### High (1 week+)
- Shared memory stencil optimization
- Multi-GPU domain decomposition
- Async kernel launching

## GPU Transition Checklist

```bash
# 1. Quick test (after device param changes)
python3 -c "
from bssn_state import BSSNState
import warp as wp
wp.init()
state = BSSNState(16, 16, 16, device='cuda:0')
state.set_flat_spacetime()
print('GPU init:', state.device)
print('Array device:', state.chi.device)
"

# 2. Evolution test
# Modify test_bssn_complete.py to accept device param

# 3. Performance comparison
# Time CPU vs GPU for 100 steps on 64³ grid
```

## Expected GPU Speedup

| Grid Size | CPU Time (est) | GPU Time (est) | Speedup |
|-----------|---------------|----------------|---------|
| 32³ | 1s/100 steps | 50ms/100 steps | ~20x |
| 64³ | 8s/100 steps | 100ms/100 steps | ~80x |
| 128³ | 64s/100 steps | 400ms/100 steps | ~160x |

*Estimates based on typical warp FD kernel performance.*

## Conclusion

**GPU Readiness: HIGH**

The codebase is well-designed for GPU migration:
1. Device parameter already exists in core state class
2. All operations are in warp kernels (GPU-compatible)
3. No fundamental CPU-only dependencies
4. Changes needed are mostly parameter passing

**Recommended first step**: Add device propagation to `BSSNEvolver` and `RK4Integrator`, then test on GPU.
