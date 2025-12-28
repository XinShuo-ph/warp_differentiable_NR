# GPU Analysis

## Current Device Usage

- **Explicit `device="cpu"` in code**: No
- **Explicit `device="cuda"` in code**: No
- **Default device handling**: Code uses Warp's default device (typically first available GPU, falls back to CPU)

The current implementation does not explicitly specify devices, which means it will automatically use GPU if available. This is good practice for portable code.

## Arrays Needing Device Change

| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| `bssn.py` | phi, gt, K, At, Xt, alpha, beta, B | default | None needed (auto) |
| `bssn.py` | temp_fields, rhs_fields, initial_fields | default | None needed (auto) |
| `poisson.py` | bd_values, x, err_val | default | None needed (auto) |

**Note**: All `wp.zeros()` and `wp.zeros_like()` calls use Warp defaults, which will automatically use GPU if available.

## CPU-Only Operations

| Operation | File:Line | Impact |
|-----------|-----------|--------|
| `.numpy()` | `bssn.py:226,230` | Data transfer for printing/assertions in main block |
| `.numpy()` | `poisson.py:92` | Error computation returns numpy value |
| `.numpy()` | `test_bssn_autodiff.py:42` | Gradient inspection |

**Analysis**: All `.numpy()` calls are for output/inspection, not in performance-critical paths. They will trigger GPU→CPU transfer if running on GPU, but this is acceptable for diagnostics.

## Kernel Device Specification

- **Kernels use explicit device**: No (not needed)
- **`wp.launch` device param**: Not specified (uses array device automatically)

**Analysis**: Warp automatically launches kernels on the device where the input arrays reside. Since arrays are created on the default device, kernels will execute there automatically.

## Changes Needed for GPU

### Required Changes
**None** - The code is already GPU-ready by design.

### Recommended Enhancements

1. **Add device parameter to BSSNSolver**:
```python
def __init__(self, resolution=(32, 32, 32), extent=(10.0, 10.0, 10.0), 
             requires_grad=False, device=None):
    # ...
    self.device = device
    self.fields = {
        "phi": wp.zeros(self.shape, dtype=float, requires_grad=requires_grad, device=device),
        # ...
    }
```

2. **Add device parameter to PoissonSolver**:
```python
def __init__(self, resolution=50, degree=2, device=None):
    self.device = device
    # FEM may need additional device handling
```

3. **Synchronization point after main computation**:
```python
wp.synchronize()  # Ensure GPU work completes before timing/output
```

## Potential GPU Issues

- [ ] **Memory transfers between CPU/GPU**: Only at output points (`.numpy()`), acceptable
- [ ] **Array synchronization**: Not needed for current single-stream usage
- [ ] **warp.fem GPU support**: Need to verify FEM module GPU compatibility
- [ ] **Large grid memory**: 32³ with 8 fields × 4 buffers = ~100MB, fine for most GPUs
- [ ] **Kernel compilation**: First run will be slower due to CUDA kernel compilation

## Estimated Effort

### Low (no changes needed)
- BSSN solver runs on GPU automatically when CUDA is available
- All kernels are compatible with GPU execution
- Memory layout is GPU-friendly (contiguous 3D arrays)

### Medium (optional improvements)
- Add explicit `device` parameter for user control
- Add `wp.synchronize()` calls for timing accuracy
- Test performance at various resolutions

### High (future work)
- Investigate warp.fem GPU performance for Poisson solver
- Multi-GPU support if needed for large grids
- Memory optimization for higher resolutions

## Verification Steps

To verify GPU execution:

```python
import warp as wp
wp.init()

# Check available devices
print(wp.get_devices())

# Run solver
from NR.src.bssn import BSSNSolver
solver = BSSNSolver()
solver.step()

# Check where arrays live
print(f"phi device: {solver.fields['phi'].device}")
```

## Conclusion

The codebase is **GPU-ready** with no required changes. Warp's automatic device selection will use GPU when available. The only CPU operations are for output/inspection, which is appropriate. Optional enhancements could add explicit device control for advanced users.
