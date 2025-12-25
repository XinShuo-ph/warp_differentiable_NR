# Warp FEM API Reference
# Key patterns from example_navier_stokes.py

import warp as wp
import warp.fem as fem

# ============ GEOMETRY ============
# Grid2D: Regular structured grid
# geo = fem.Grid2D(res=wp.vec2i(resolution))

# Trimesh2D: Unstructured triangle mesh
# geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions, build_bvh=True)

# Quadmesh2D: Unstructured quad mesh
# geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions, build_bvh=True)

# ============ DOMAINS ============
# Cells: Interior of the mesh (volume/area integration)
# domain = fem.Cells(geometry=geo)

# BoundarySides: Boundary faces/edges
# boundary = fem.BoundarySides(geo)

# ============ FUNCTION SPACES ============
# make_polynomial_space: Creates FEM function space
# scalar_space = fem.make_polynomial_space(geo, degree=2)              # scalar field
# vector_space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec2)  # vector field

# ============ FIELDS ============
# Discrete field from function space
# field = space.make_field()
# field.dof_values  # access degrees of freedom

# ImplicitField: Field defined by a function
# field = fem.ImplicitField(domain, func=my_func, values={"param": value})

# ============ TEST/TRIAL FUNCTIONS ============
# test = fem.make_test(space=space, domain=domain)
# trial = fem.make_trial(space=space, domain=domain)

# ============ INTEGRANDS ============
# @fem.integrand decorator for weak form functions
# Key parameters:
#   s: fem.Sample - quadrature point info
#   u, v: fem.Field - trial and test functions
#   domain: fem.Domain - for position/normal queries

# Key fem functions in integrands:
#   u(s)           - evaluate field at sample point
#   fem.grad(u, s) - gradient of scalar field
#   fem.D(u, s)    - symmetric gradient (deformation) of vector field
#   fem.div(u, s)  - divergence of vector field
#   fem.normal(domain, s) - outward normal
#   fem.lookup(domain, pos, s) - lookup sample at different position (for advection)

# ============ INTEGRATION ============
# Linear form (returns vector):
# rhs = fem.integrate(linear_form, fields={"v": test})

# Bilinear form (returns matrix):
# matrix = fem.integrate(bilinear_form, fields={"u": trial, "v": test}, values={"param": val})

# Assembly options:
#   assembly="nodal" - for boundary conditions, integrates at nodes

# ============ BOUNDARY CONDITIONS ============
# Dirichlet BCs:
# fem.project_linear_system(matrix, rhs, projector, values)
# fem.normalize_dirichlet_projector(projector, values)
