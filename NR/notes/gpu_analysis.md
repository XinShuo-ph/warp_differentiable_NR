# GPU Analysis

## Current Device Usage
- **Explicit device="cpu" in code**: No
- **Default device handling**: All arrays and kernel launches use Warp's default device (falls back to CPU when CUDA unavailable)

## Arrays Needing Device Change

### bssn.py - BSSNState arrays
| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| bssn.py | phi, gt11-gt33, Xt1-3, trK, At11-At33, alpha, beta1-3 (21 total) | default | Add `device` param to `create_bssn_state()` |

### bssn_full.py - Ricci arrays
| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| bssn_full.py | Rt11-Rt33, trR (7 total) | default | Add `device` param to `create_ricci_arrays()` |

### bssn_evolve.py - Temporary arrays
| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| bssn_evolve.py | H constraint array | default | Add `device` param |

### poisson.py - FEM arrays
| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| poisson.py | bd_value, x (solver arrays) | default | warp.fem handles device internally |

## CPU-Only Operations

| Operation | File:Line | Notes |
|-----------|-----------|-------|
| `.numpy()` for validation | bssn.py:759-761, 786-788 | Test assertions - keep on CPU |
| `.numpy()` for constraint norms | bssn_evolve.py:221-238 | Monitoring output - sync from GPU |
| `.numpy()` for monitoring | bssn_evolve.py:469, 489-597 | Evolution monitoring - sync from GPU |
| `.numpy()` for tests | bssn_full.py:901, 925-976 | Test validation - keep on CPU |
| `np.sqrt`, `np.abs`, `np.allclose` | bssn_evolve.py:237, 567-573 | Post-sync numpy operations |
| `np.savez` checkpoint | bssn_evolve.py:599 | I/O - must be on CPU |

## Kernel Device Specification
- **Kernels use explicit device**: No
- **wp.launch device param**: Not specified (uses default)
- **wp.copy device param**: Not specified (uses default)

### Kernel Inventory (all use default device)
| File | Kernels |
|------|---------|
| bssn.py | `init_flat_spacetime`, `compute_bssn_rhs`, `rk4_update`, `rk4_final` |
| bssn_full.py | `compute_christoffel_and_ricci`, `compute_bssn_rhs_full`, `init_gauge_wave`, `init_brill_lindquist` |
| bssn_evolve.py | `apply_sommerfeld_bc`, `compute_hamiltonian_constraint`, `axpy_kernel`, `rk4_combine_kernel` |
| poisson.py | Uses warp.fem integrands (FEM-based, not explicit kernels) |

## Changes Needed for GPU

### 1. Add device parameter to array creation (Low effort)
```python
# bssn.py - create_bssn_state()
def create_bssn_state(nx, ny, nz, dx, requires_grad=False, device="cuda:0"):
    state.phi = wp.zeros(shape, dtype=float, requires_grad=requires_grad, device=device)
    # ... all 21 fields
```

### 2. Add device parameter to Ricci arrays (Low effort)
```python
# bssn_full.py - create_ricci_arrays()
def create_ricci_arrays(nx, ny, nz, device="cuda:0"):
    return {
        'Rt11': wp.zeros(shape, dtype=float, device=device),
        # ...
    }
```

### 3. Add device parameter to BSSNEvolver (Low effort)
```python
# bssn_evolve.py
class BSSNEvolver:
    def __init__(self, nx, ny, nz, dx, eps_diss=0.1, eta=2.0, device="cuda:0"):
        self.device = device
        self.state = create_bssn_state(nx, ny, nz, dx, device=device)
        # ... all state objects
```

### 4. Keep `.numpy()` calls for monitoring (No change)
The `.numpy()` calls are for:
- Test assertions (acceptable CPU overhead)
- Constraint monitoring (infrequent, acceptable sync)
- Checkpointing (must be on CPU for I/O)

### 5. Add device to wp.launch (Optional, Recommended)
```python
wp.launch(kernel, dim=shape, inputs=[...], device=self.device)
```

## Potential GPU Issues

- [x] Memory transfers between CPU/GPU - Only during `.numpy()` calls for monitoring
- [x] Array synchronization - Warp handles automatically with `wp.synchronize()`
- [ ] FEM solver GPU support - warp.fem should work on GPU, but untested
- [ ] Large grid memory - GPU memory limits for very large grids (100³+)
- [ ] Mixed precision - Currently all float64; float32 would be faster on GPU

## Estimated Effort

### Low (< 1 hour)
- Add `device` parameter to `create_bssn_state()` - 5 lines
- Add `device` parameter to `create_ricci_arrays()` - 2 lines  
- Add `device` parameter to `BSSNEvolver.__init__()` - 10 lines
- Pass device to all internal array/state creation

### Medium (1-2 hours)
- Test all kernels on GPU for correctness
- Verify warp.fem Poisson solver works on GPU
- Add `device` parameter to `wp.launch()` calls (optional but explicit)

### High (4+ hours)
- Optimize for GPU memory layout
- Add mixed precision (float32) support
- Profile and optimize kernel performance
- Add multi-GPU support for domain decomposition

## Summary

The codebase is **GPU-ready by design**. Warp kernels and arrays use the default device, which automatically uses CUDA when available. The primary changes needed are:

1. **Add `device` parameter** to array creation functions (propagate from top-level to all arrays)
2. **Test on actual GPU** to verify correctness
3. **Keep CPU syncs** for monitoring/checkpointing (acceptable overhead)

No fundamental architectural changes required - just parameter propagation.
