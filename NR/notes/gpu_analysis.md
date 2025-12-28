# GPU Analysis

## Current Device Usage

### Explicit device="cpu" in Code

| File | Lines | Usage |
|------|-------|-------|
| bssn_variables.py | 103-136 | `device=self.device` (parameter, defaults to "cpu") |
| bssn_derivatives.py | 198, 199, etc. | `device="cpu"` explicit in test code only |
| bssn_integrator.py | 118-131 | No device specified (defaults to CPU) |
| bssn_initial_data.py | 152-175 | No device specified |
| bssn_boundary.py | 169-175 | No device specified |
| bssn_constraints.py | 110-114 | No device specified |
| poisson_solver.py | N/A | Uses warp.fem (device handled internally) |

**Summary**: Most code uses Warp's default device (CPU). Only `BSSNFields` class accepts a `device` parameter but defaults to "cpu".

### Default Device Handling

Warp's `wp.zeros()`, `wp.array()`, etc. default to CPU when no device is specified. The codebase relies on this implicit default in most places.

## Arrays Needing Device Change

### Core BSSN Fields (bssn_variables.py)

| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| bssn_variables.py | chi, gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz | device param (cpu default) | Pass device="cuda:0" |
| bssn_variables.py | K, A_xx, A_xy, A_xz, A_yy, A_yz, A_zz | device param | Pass device="cuda:0" |
| bssn_variables.py | Gamma_x, Gamma_y, Gamma_z | device param | Pass device="cuda:0" |
| bssn_variables.py | alpha, beta_x, beta_y, beta_z, B_x, B_y, B_z | device param | Pass device="cuda:0" |

### Integrator Scratch Arrays (bssn_integrator.py)

| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| bssn_integrator.py | chi_tmp, K_tmp | cpu (implicit) | Add device param |
| bssn_integrator.py | k1_chi, k1_K, k2_chi, k2_K, k3_chi, k3_K, k4_chi, k4_K | cpu (implicit) | Add device param |

### Constraint Monitor (bssn_constraints.py)

| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| bssn_constraints.py | H, L2_norm, Linf_norm, count | cpu (implicit) | Add device param |

### Initial Data (bssn_initial_data.py)

| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| bssn_initial_data.py | All 24 BSSN field arrays | cpu (implicit) | Add device param |

### RHS Arrays (test files, evolver classes)

| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| test_long_evolution.py | All rhs_* arrays in FullBSSNEvolver | cpu (implicit) | Add device param |

## CPU-Only Operations

| Operation | File:Line | Description | GPU Impact |
|-----------|-----------|-------------|------------|
| `.numpy()` | bssn_variables.py:201-228 | Get field values for inspection | Need `wp.synchronize()` before `.numpy()` for correctness |
| `.numpy()` | bssn_constraints.py:145-147 | Read norm values | Same; also could use device-side reduction |
| `.numpy()` | test files | Check values in tests | Tests likely stay on CPU or need sync |
| `np.linspace()` | bssn_derivatives.py:183-186 | Create coordinate arrays | Happens once at init, not performance critical |
| `np.meshgrid()` | bssn_derivatives.py:186 | Create grid | Same; init-time only |

## Kernel Device Specification

### Current State
- Kernels use `wp.launch()` without explicit device parameter
- Warp infers device from input arrays
- If all input arrays are on same device, kernel runs there

### Recommendation
No changes needed to kernel launches if all arrays use same device. Warp's automatic device inference works correctly.

**Example of implicit device handling:**
```python
# If chi, K, alpha all on cuda:0, kernel runs on cuda:0
wp.launch(
    compute_simple_rhs,
    dim=(n, n, n),
    inputs=[chi, K, alpha, rhs_chi, rhs_K, ...]
)
```

## Changes Needed for GPU

### 1. Add Device Parameter Throughout (Low Effort)

```python
# bssn_variables.py - Already has device param, just change default
class BSSNFields:
    def __init__(self, ..., device: str = "cuda:0"):  # Change default
        ...

# bssn_integrator.py - Add device param
class SimpleRK4Integrator:
    def __init__(self, nx, ny, nz, h, sigma_ko=0.1, device="cuda:0"):
        ...
        self.chi_tmp = wp.zeros(shape, dtype=float, device=device)
        ...

# bssn_constraints.py - Add device param
class ConstraintMonitor:
    def __init__(self, nx, ny, nz, dx, ng=3, device="cuda:0"):
        ...
        self.H = wp.zeros(shape, dtype=float, device=device)
        ...

# bssn_initial_data.py - Add device param
class BrillLindquistData:
    def __init__(self, ..., device="cuda:0"):
        ...
```

### 2. Add Synchronization Before NumPy Conversion (Low Effort)

```python
# Before any .numpy() call for GPU arrays:
wp.synchronize_device("cuda:0")  # Or just wp.synchronize()
values = array.numpy()
```

### 3. Update Test Files (Low Effort)

Tests can remain CPU-only for simplicity, or add device parameter to test functions.

### 4. Handle FEM Solver (Medium Effort)

The `poisson_solver.py` uses `warp.fem` which has its own device handling. Check `warp.fem` docs for GPU support. May need:
```python
geo = fem.Grid2D(res=wp.vec2i(resolution), device="cuda:0")
```

## Potential GPU Issues

- [ ] **Memory transfers between CPU/GPU**: Minimize `.numpy()` calls during evolution; batch diagnostics
- [ ] **Array synchronization**: Add `wp.synchronize()` before reading GPU arrays on CPU
- [ ] **Atomic operations**: Used in constraint norms (`wp.atomic_add`, `wp.atomic_max`) - these work on GPU but may need care for performance
- [ ] **FEM solver compatibility**: Verify `warp.fem` GPU support for Poisson solver
- [ ] **Large memory footprint**: 24 fields × grid_size³ × 8 bytes; ensure GPU memory sufficient

## Estimated Effort

### Low (1-2 hours)
- Add `device` parameter to all classes: `BSSNFields`, `SimpleRK4Integrator`, `ConstraintMonitor`, `BrillLindquistData`, `BSSNBoundaryConditions`
- Add synchronization before `.numpy()` calls
- Update evolver classes in test files

### Medium (2-4 hours)
- Create unified device configuration (e.g., config dict or global setting)
- Update Poisson solver for GPU
- Add GPU memory management utilities
- Create GPU-specific test suite

### High (4+ hours)
- Optimize atomic operations for constraint monitoring (use warp's parallel reduction patterns)
- Profile and optimize kernel launch configurations
- Add multi-GPU support for domain decomposition
- Implement GPU-friendly output/visualization pipeline

## Recommended GPU Migration Steps

1. **Phase 1: Basic GPU Support**
   - Add `device` parameter to all array-allocating classes
   - Default to `"cuda:0"` or use environment variable
   - Add `wp.synchronize()` before `.numpy()` calls
   
2. **Phase 2: Test & Validate**
   - Run existing test suite on GPU
   - Compare results to CPU baseline
   - Profile for obvious bottlenecks

3. **Phase 3: Optimize**
   - Replace atomic operations with parallel reductions where beneficial
   - Tune grid dimensions for GPU occupancy
   - Minimize CPU-GPU memory transfers

## Code Snippet: Minimal GPU Migration

```python
# Add to top of each module or create config.py:
import os
DEFAULT_DEVICE = os.environ.get("WARP_DEVICE", "cuda:0")

# Then in array allocations:
self.chi = wp.zeros(shape, dtype=float, device=DEFAULT_DEVICE)

# Before numpy access:
wp.synchronize()
values = self.chi.numpy()
```

## References

- [Warp Device Management](https://nvidia.github.io/warp/basics.html#devices)
- [Warp Memory Model](https://nvidia.github.io/warp/basics.html#memory-model)
- [Warp FEM Module](https://nvidia.github.io/warp/modules/fem.html)
