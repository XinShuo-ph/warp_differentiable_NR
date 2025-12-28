# Warp FEM Mesh and Field API Reference

import warp as wp
import warp.fem as fem

# =============================================================================
# GEOMETRY TYPES
# =============================================================================

# 2D Grid (structured)
geo = fem.Grid2D(res=wp.vec2i(resolution))

# 2D Triangle mesh
positions, tri_vidx = gen_trimesh(res=wp.vec2i(resolution))
geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions, build_bvh=True)

# 2D Quad mesh
positions, quad_vidx = gen_quadmesh(res=wp.vec2i(resolution))
geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions, build_bvh=True)

# 3D Grid
geo = fem.Grid3D(res=wp.vec3i(resolution))

# =============================================================================
# DOMAINS
# =============================================================================

# Cell domain (interior)
domain = fem.Cells(geometry=geo)

# Boundary sides domain
boundary = fem.BoundarySides(geo)

# Subdomain (subset of boundary based on mask)
subdomain = fem.Subdomain(boundary, element_mask=mask_array)

# =============================================================================
# FUNCTION SPACES
# =============================================================================

# Scalar polynomial space
scalar_space = fem.make_polynomial_space(geo, degree=2)

# Vector polynomial space
vector_space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec2)

# Serendipity elements (fewer DOFs)
scalar_space = fem.make_polynomial_space(geo, degree=2, element_basis=fem.ElementBasis.SERENDIPITY)

# =============================================================================
# FIELDS
# =============================================================================

# Discrete field (DOF values at nodes)
field = scalar_space.make_field()
field.dof_values  # access underlying array

# Implicit field (defined by a function)
@wp.func
def boundary_value(x: wp.vec2, param: float):
    return wp.vec2(param, 0.0)

implicit_field = fem.ImplicitField(
    domain=boundary,
    func=boundary_value,
    values={"param": 1.0}
)

# =============================================================================
# TEST AND TRIAL FUNCTIONS
# =============================================================================

# Test function (rows of matrix)
test = fem.make_test(space=scalar_space, domain=domain)

# Trial function (columns of matrix)  
trial = fem.make_trial(space=scalar_space, domain=domain)

# =============================================================================
# INTEGRANDS
# =============================================================================

@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))

@fem.integrand
def stiffness_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))

# =============================================================================
# INTEGRATION
# =============================================================================

# Matrix assembly (bilinear form)
matrix = fem.integrate(mass_form, fields={"u": trial, "v": test})

# Vector assembly (linear form)
rhs = fem.integrate(linear_form, fields={"v": test})

# Nodal assembly (for boundary conditions)
bd_matrix = fem.integrate(mass_form, fields={"u": trial, "v": test}, assembly="nodal")

# =============================================================================
# FEM OPERATORS (inside integrands)
# =============================================================================

# fem.grad(field, sample)  - gradient of scalar field
# fem.div(field, sample)   - divergence of vector field
# fem.D(field, sample)     - symmetric gradient (strain rate)
# fem.normal(domain, s)    - outward normal at boundary
# fem.lookup(domain, pos, s) - lookup sample at position (semi-Lagrangian)

# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

# Project linear system for Dirichlet BC
fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)

# Normalize Dirichlet projector
fem.normalize_dirichlet_projector(bd_matrix, bd_rhs)
