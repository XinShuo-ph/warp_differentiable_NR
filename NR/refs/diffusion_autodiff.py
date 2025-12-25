# Key patterns from example_diffusion.py for autodiff

import warp as wp
import warp.fem as fem

# 1. Integrand decorator - creates differentiable kernels
@fem.integrand
def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field, nu: float):
    """Bilinear form: int nu * grad(u) · grad(v) dx"""
    return nu * wp.dot(fem.grad(u, s), fem.grad(v, s))

# 2. Integration produces warp arrays that support autodiff
# matrix = fem.integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": viscosity})
# Returns BSR sparse matrix with .values array that can be differentiated

# 3. Enable backward mode for autodiff
# wp.set_module_options({"enable_backward": True})

# 4. Fields are warp arrays: field.dof_values is wp.array that tracks gradients
# Can use wp.Tape() to record operations and compute gradients

# Key APIs:
# - fem.Sample: quadrature point
# - fem.Field: function in finite element space
# - fem.grad(field, sample): gradient operator
# - fem.integrate(): assembles forms into matrices/vectors
# - fem.make_test/make_trial: test/trial spaces for weak formulation
