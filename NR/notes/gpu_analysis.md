# GPU Analysis

## Current Device Usage

### Explicit device="cpu" in code: No
The code does not explicitly specify `device="cpu"`. Instead, it uses `device=None` or passes the device as a parameter.

### Default device handling
- `allocate_bssn_state(res, device=None)` accepts optional device parameter
- `BSSNSolver(resolution=64, device=None)` propagates device to state allocation
- If no device specified, Warp uses its default device (usually first available GPU, or CPU if no GPU)

## Arrays Needing Device Change

All arrays are created in `bssn_defs.py:allocate_bssn_state()` with the `device` parameter:

| File | Arrays | Current | GPU Change |
|------|--------|---------|------------|
| `bssn_defs.py` | phi, gamma_tilde, K, A_tilde, Gamma_tilde, alpha, beta, B | `device=device` (param) | ✓ Already parameterized |
| `bssn_solver.py` | state, state_temp, rhs, state_out | via `allocate_bssn_state(..., device=device)` | ✓ Already parameterized |
| `initial_data.py` | Uses `state.phi.device` for launch | Reads from state | ✓ Device-aware |
| `poisson_solver.py` | FEM arrays | Warp FEM default | May need explicit device |

## CPU-Only Operations

### NumPy Interop (for analysis/debugging only)
These operations require CPU↔GPU transfers but are only used in tests, not in hot path:

| Operation | File:Line | Context |
|-----------|-----------|---------|
| `solver.state.phi.numpy()` | `test_flat_spacetime.py:26` | Printing norms |
| `solver.state.K.numpy()` | `test_flat_spacetime.py:27` | Printing norms |
| `np.linalg.norm(...)` | `test_flat_spacetime.py:26,27,30` | Computing norms |
| `np.max/min(...)` | `test_bbh_evolution.py:52-53` | Checking initial data |
| `self.space.node_positions().numpy()` | `poisson_solver.py:86` | Error computation |
| `self.u_field.dof_values.numpy()` | `poisson_solver.py:87` | Error computation |

### No CPU-only operations in evolution kernels
The core evolution loop (`rk4_step`, `compute_rhs`, `bssn_rhs_kernel`) uses only Warp operations and is fully GPU-compatible.

## Kernel Device Specification

### Current state
- **Kernels use explicit device**: Partially
- `wp.launch device param`: Present in `BSSNSolver` methods

### Details
| Method | Device Spec | Notes |
|--------|-------------|-------|
| `compute_rhs()` | `device=self.device` | ✓ Explicit |
| `update_step()` | `device=self.device` | ✓ Explicit |
| `accumulate_step()` | `device=self.device` | ✓ Explicit |
| `init_bssn_state()` | None | Uses array's default device |
| `setup_brill_lindquist()` | `device=state.phi.device` | ✓ Device-aware |
| `setup_bowen_york()` | `device=state.A_tilde.device` | ✓ Device-aware |

## Changes Needed for GPU

### Required Changes (Low Effort)

1. **Specify device in initialization**:
   ```python
   # Change from:
   solver = BSSNSolver(resolution=64)
   # To:
   solver = BSSNSolver(resolution=64, device="cuda:0")
   ```

2. **Add device to init_bssn_state launch** (`bssn_defs.py`):
   ```python
   def init_bssn_state(state, device=None):
       wp.launch(
           kernel=initialize_flat_spacetime,
           dim=dim,
           inputs=[...],
           device=device or state.phi.device  # Add this
       )
   ```

### Optional Changes (Medium Effort)

3. **GPU-side norm computation** (for monitoring):
   Replace `np.linalg.norm(solver.state.phi.numpy())` with a Warp reduction kernel to avoid CPU transfers during evolution.

4. **Poisson solver device**:
   The FEM-based Poisson solver uses Warp's FEM infrastructure which should work on GPU, but may need verification.

### No Changes Needed

5. **Kernels**: All `@wp.kernel` and `@wp.func` decorated functions are automatically compiled for the target device.

6. **Array operations**: Warp handles device placement automatically based on array creation.

## Potential GPU Issues

- [ ] **Memory transfers between CPU/GPU**: Only in test code, not in hot path ✓
- [ ] **Array synchronization**: Warp handles this automatically ✓
- [x] **init_bssn_state missing device param**: Minor fix needed
- [ ] **Large grid sizes**: May need to check GPU memory for high resolutions
- [ ] **Kernel launch overhead**: Should be negligible for 3D grids

## Estimated Effort

### Low (< 1 hour)
- Add `device` parameter to `init_bssn_state()` launch
- Test with `device="cuda:0"` on GPU machine
- Verify all tests pass on GPU

### Medium (1-2 hours)
- Add GPU-side reduction kernels for monitoring (norms, max/min)
- Verify Poisson FEM solver works on GPU
- Add device selection to Poisson solver

### High (if needed)
- None identified - code is already well-structured for GPU

## Summary

The codebase is **GPU-ready** with minimal changes. The main architectural decisions (parameterized device, explicit device in launches) are already in place. Only one kernel launch (`init_bssn_state`) is missing an explicit device specification, which is a simple fix.

To enable GPU execution:
```python
import warp as wp
wp.init()

solver = BSSNSolver(resolution=64, device="cuda:0")
# Evolution will run entirely on GPU
```
