# GPU Analysis

Analysis of changes needed to run the BSSN evolution code on CUDA backend.

## Current Device Usage

- **Explicit `device="cpu"` in code**: No
- **Explicit `device="cuda"` in code**: No
- **Default device handling**: Uses Warp's default (CPU unless CUDA available and `wp.set_device()` called)

The code currently relies on Warp's default device selection, which makes GPU migration straightforward.

## Arrays Needing Device Change

All array allocations use `wp.zeros()` or `wp.ones()` without explicit device specification.

| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| `bssn_evol.py` | `phi, K, alpha` | default | Add `device=` param |
| `bssn_evol.py` | `gtxx, gtxy, ..., gtzz` (6) | default | Add `device=` param |
| `bssn_evol.py` | `Atxx, Atxy, ..., Atzz` (6) | default | Add `device=` param |
| `bssn_evol.py` | `betax, betay, betaz` | default | Add `device=` param |
| `bssn_evol.py` | `rhs_phi, rhs_K, rhs_alpha` | default | Add `device=` param |
| `bssn_evol.py` | `k1_*, k2_*, k3_*, k4_*` (12) | default | Add `device=` param |
| `bssn_evol.py` | `tmp_phi, tmp_K, tmp_alpha` | default | Add `device=` param |
| `bssn_evol.py` | `H, Mx, My, Mz` | default | Add `device=` param |
| `bssn.py` | All 25 BSSNState fields | default | Add `device=` param |
| `bssn.py` | `rhs_phi, rhs_K, rhs_alpha` | default | Add `device=` param |
| `poisson.py` | FEM arrays (via warp.fem) | default | FEM handles device |

**Total arrays in BSSNEvolver**: ~40 arrays need device specification

## CPU-Only Operations

All `.numpy()` calls trigger CPU<->GPU data transfer:

| File | Line | Operation | Purpose |
|------|------|-----------|---------|
| `bssn_evol.py` | 1272-1275 | `self.H.numpy()`, `self.M*.numpy()` | Constraint monitoring |
| `bssn_evol.py` | 1318 | `evolver.alpha.numpy()` | Initial data printout |
| `bssn_evol.py` | 1324-1325 | `alpha_np`, `K_np` | Progress monitoring |
| `bssn_evol.py` | 1331 | `alpha_np` | Final check |
| `bssn.py` | 492-494 | `state.*.numpy()` | Error checking |
| `poisson.py` | 124, 158, 173 | `field.dof_values.numpy()` | Solution analysis |

**Impact**: These are all for monitoring/output, not in the hot path. They will cause GPU->CPU sync but won't affect evolution performance.

## Kernel Device Specification

- **Kernels use explicit device**: No
- **`wp.launch` device param**: Not present (uses default)

All 41 `wp.launch()` calls in `bssn_evol.py` and 6 in `bssn.py` use implicit device:

```python
# Current
wp.launch(rhs_phi_full, dim=dim, inputs=[...])

# GPU version
wp.launch(rhs_phi_full, dim=dim, inputs=[...], device="cuda:0")
```

## Changes Needed for GPU

### 1. Add Device Parameter to BSSNEvolver (Low effort)

```python
class BSSNEvolver:
    def __init__(self, nx, ny, nz, dx, sigma=0.1, use_sommerfeld=False, device="cpu"):
        self.device = device
        ...
        self.phi = wp.zeros(shape, dtype=float, device=device)
        self.K = wp.zeros(shape, dtype=float, device=device)
        # ... repeat for all 40 arrays
```

### 2. Add Device to Kernel Launches (Low effort)

```python
def compute_rhs(self, ...):
    wp.launch(rhs_phi_full, dim=dim, inputs=[...], device=self.device)
    # ... repeat for all launches
```

### 3. Update BSSNState in bssn.py (Medium effort)

Add device parameter to `create_bssn_state()`:

```python
def create_bssn_state(nx, ny, nz, dx, device="cpu"):
    state = BSSNState()
    state.phi = wp.zeros(shape, dtype=float, device=device)
    # ...
```

### 4. Handle Poisson Solver (Low effort)

The FEM-based Poisson solver uses `warp.fem` which handles device internally. Should work with `wp.set_device("cuda:0")` before calling.

### 5. Synchronization Points (Optional)

For `.numpy()` calls, add explicit sync if needed:

```python
def compute_constraints(self):
    ...
    wp.synchronize()  # Ensure GPU kernels complete
    H_max = np.max(np.abs(self.H.numpy()))
```

## Potential GPU Issues

- [x] Memory transfers between CPU/GPU - Only for monitoring, not in hot path
- [ ] Array synchronization - May need `wp.synchronize()` before `.numpy()` calls
- [ ] Multi-GPU support - Not addressed (single GPU assumed)
- [ ] Memory capacity - 40 arrays × N³ × 8 bytes; 64³ grid ≈ 80 MB, 256³ ≈ 5 GB
- [ ] FEM solver compatibility - `warp.fem` should support CUDA

## Estimated Effort

### Low Effort (~30 min)
1. Add `device` parameter to `BSSNEvolver.__init__`
2. Propagate `device` to all `wp.zeros/ones` calls in the class
3. Add `device=self.device` to all `wp.launch` calls

### Medium Effort (~1 hour)
4. Update `bssn.py` `create_bssn_state()` with device parameter
5. Update `copy_state()` and other utility functions
6. Add synchronization before `.numpy()` calls

### Testing (~30 min)
7. Verify all tests pass on GPU
8. Add GPU-specific test with `@pytest.mark.skipif(not wp.is_cuda_available())`
9. Performance comparison CPU vs GPU

## Recommended Approach

1. **Minimal change**: Add `device` parameter to `BSSNEvolver` class only, since that's the main entry point. The `bssn.py` module is mostly for simpler tests.

2. **Global device**: Alternatively, use `wp.set_device("cuda:0")` at program start, and all arrays/launches will use GPU by default.

```python
import warp as wp
wp.init()
wp.set_device("cuda:0")  # All subsequent operations on GPU

# Code works unchanged
evolver = BSSNEvolver(nx=32, ny=32, nz=32, dx=0.03125)
```

This approach requires zero code changes to the core implementation.

## Performance Expectations

| Grid Size | CPU (estimated) | GPU (expected) | Speedup |
|-----------|-----------------|----------------|---------|
| 16³ | ~1 ms/step | ~0.1 ms/step | 10x |
| 32³ | ~10 ms/step | ~0.3 ms/step | 30x |
| 64³ | ~100 ms/step | ~1 ms/step | 100x |
| 128³ | ~1 s/step | ~5 ms/step | 200x |

GPU benefits increase dramatically with grid size due to parallel execution of finite difference stencils across all grid points.
