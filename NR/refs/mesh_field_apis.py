# Mesh and Field APIs from Navier-Stokes example

import warp as wp
import warp.fem as fem

# === GEOMETRY (Mesh) APIs ===

# 1. Built-in structured grids
geo = fem.Grid2D(res=wp.vec2i(resolution))  # Regular 2D grid

# 2. Unstructured meshes
positions, tri_vidx = gen_trimesh(res=wp.vec2i(resolution))
geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions, build_bvh=True)

positions, quad_vidx = gen_quadmesh(res=wp.vec2i(resolution))
geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions, build_bvh=True)
# build_bvh=True enables point lookup for semi-Lagrangian advection

# === DOMAIN APIs ===

domain = fem.Cells(geometry=geo)  # Integration over cells
boundary = fem.BoundarySides(geo)  # Integration over boundary sides

# === FUNCTION SPACE APIs ===

# Scalar space
scalar_space = fem.make_polynomial_space(geo, degree=2)

# Vector space (for velocity)
vector_space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec2)

# Mixed spaces for Stokes/Navier-Stokes (Q_d - Q_{d-1})
u_space = fem.make_polynomial_space(geo, degree=degree, dtype=wp.vec2)  # velocity
p_space = fem.make_polynomial_space(geo, degree=degree-1)  # pressure

# === FIELD APIs ===

# 1. Discrete fields (dof values stored)
velocity_field = u_space.make_field()
pressure_field = p_space.make_field()
# Access/modify DOF values:
velocity_field.dof_values = wp.array(...)  # wp.array of dof values

# 2. Implicit fields (computed from function)
@wp.func
def boundary_func(x: wp.vec2, param: float):
    return wp.vec2(param, 0.0)

u_bd_field = fem.ImplicitField(
    domain=boundary, 
    func=boundary_func, 
    values={"param": 1.0}
)

# === FIELD OPERATORS in integrands ===

@fem.integrand
def operators_demo(s: fem.Sample, u: fem.Field, domain: fem.Domain):
    # Value evaluation
    val = u(s)
    
    # Gradient
    grad_u = fem.grad(u, s)
    
    # Divergence
    div_u = fem.div(u, s)
    
    # Symmetric gradient (strain rate)
    D_u = fem.D(u, s)  # D_ij = 0.5*(du_i/dx_j + du_j/dx_i)
    
    # Domain position
    pos = domain(s)
    
    # Lookup (for advection, requires BVH)
    new_pos = pos - vel * dt
    new_s = fem.lookup(domain, new_pos, s)
    advected_val = u(new_s)
    
    return val  # dummy return

# === TEST/TRIAL SPACES ===

test = fem.make_test(space=u_space, domain=domain)
trial = fem.make_trial(space=u_space, domain=domain)

# === INTEGRATION ===

# Bilinear form -> sparse matrix
matrix = fem.integrate(bilinear_form, fields={"u": trial, "v": test})

# Linear form -> vector
rhs = fem.integrate(linear_form, fields={"v": test})

# Nodal assembly (for boundary conditions)
bd_matrix = fem.integrate(form, fields={...}, assembly="nodal")

# === BOUNDARY CONDITIONS ===

# Hard (strong) BCs via projection
fem.normalize_dirichlet_projector(bd_projector, bd_value)
fem.project_linear_system(matrix, rhs, bd_projector, bd_value, normalize_projector=False)

# Key matrix types:
# - BSRMatrix: Block sparse row format
# - Supports @ operator for matrix-vector products
# - Can combine: A + B, scalar*A
