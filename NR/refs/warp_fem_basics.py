# Warp FEM Key Concepts (extracted from examples)

# 1. GEOMETRY
# Grid2D, Trimesh2D, Quadmesh2D, Nanogrid (adaptive)
# geo = fem.Grid2D(res=wp.vec2i(50))

# 2. FUNCTION SPACES
# make_polynomial_space(geo, degree=2, dtype=float/wp.vec2/wp.vec3)
# space = fem.make_polynomial_space(geo, degree=2)

# 3. FIELDS
# field = space.make_field()
# field.dof_values = solution_array

# 4. DOMAINS
# fem.Cells(geometry=geo)
# fem.BoundarySides(geo)

# 5. INTEGRANDS (decorated with @fem.integrand)
# - Uses fem.Sample to evaluate at quadrature points
# - fem.grad(field, sample) - gradient
# - fem.div(field, sample) - divergence
# - fem.D(field, sample) - symmetric gradient
# - field(sample) - evaluate field at sample

# 6. INTEGRATION
# trial = fem.make_trial(space, domain)
# test = fem.make_test(space, domain)
# matrix = fem.integrate(bilinear_form, fields={"u": trial, "v": test}, values={...})
# rhs = fem.integrate(linear_form, fields={"v": test})

# 7. BOUNDARY CONDITIONS
# bd_matrix = fem.integrate(projector_form, fields={...}, assembly="nodal")
# fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)

# 8. AUTODIFF
# Enable with: wp.set_module_options({"enable_backward": True})
# Use tape: with wp.Tape() as tape: ...
# Backward pass: tape.backward(loss)

import warp as wp
import warp.fem as fem

# Example: Laplace operator integrand
@fem.integrand
def laplace_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))

# Example: Source term integrand
@fem.integrand
def source_form(s: fem.Sample, v: fem.Field, f: float):
    return f * v(s)
