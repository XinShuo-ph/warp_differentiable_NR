# Warp Autodiff Mechanism - Diffusion Example

## Key Observations

1. **Tape-based autodiff**: Warp uses `wp.Tape()` to record operations
   - Create tape with context manager
   - Operations inside context are recorded
   - Call `tape.backward()` to compute gradients

2. **FEM Integration is differentiable**:
   - `fem.integrate()` works with tape
   - Can differentiate through assembly process
   - Both bilinear and linear forms support autodiff

3. **Enable backward mode**: Must set module option
   ```python
   wp.set_module_options({"enable_backward": True})
   ```

4. **Integrand decorator**: `@fem.integrand` marks functions for compilation
   - Supports field operations: `u(s)`, `fem.grad(u, s)`
   - Compiled to efficient kernels with autodiff support

5. **Matrix assembly is differentiable**:
   - Can compute gradients w.r.t. parameters
   - Enables inverse problems and optimization
   - Works seamlessly with sparse matrices (BSR format)

## Code Pattern
```python
tape = wp.Tape()
with tape:
    matrix = fem.integrate(bilinear_form, fields={...}, values={...})
    rhs = fem.integrate(linear_form, fields={...})

# Solve system
x = solve(matrix, rhs)

# Backpropagate
tape.backward()
```
