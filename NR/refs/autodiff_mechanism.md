# Autodiff in Warp FEM

## Key Mechanisms

### 1. Integrand Compilation
- `@fem.integrand` decorator marks functions for autodiff
- Integrands compiled to warp kernels
- Warp auto-generates backward pass for each kernel

### 2. Spatial Derivatives
```python
@fem.integrand
def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field, nu: float):
    return nu * wp.dot(fem.grad(u, s), fem.grad(v, s))
```

**Forward pass**: `fem.grad(u, s)` returns gradient vector ∇u
**Backward pass**: Adjoint is spatial divergence operator

### 3. Gradient Flow
- Integration preserves gradient flow through assembly
- Full chain rule: integrand → assembly → solve
- All warp ops (dot, math functions) are differentiable

### 4. Differentiation w.r.t. Parameters
For diffusion_form: `L = nu * dot(grad(u), grad(v))`
- `dL/dnu = dot(grad(u), grad(v))`
- `dL/du = nu * div(grad(v))`

## Example Usage
```python
wp.set_module_options({"enable_backward": True})

# Define differentiable integrand
@fem.integrand
def my_form(s, u, v, param):
    return param * wp.dot(fem.grad(u, s), fem.grad(v, s))

# Assemble with tape recording
matrix = fem.integrate(my_form, fields={"u": trial, "v": test}, values={"param": p})

# Backward pass available through warp's tape mechanism
```
