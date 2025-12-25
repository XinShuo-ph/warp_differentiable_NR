# Warp FEM Mesh and Field APIs Reference

# 1. GEOMETRY TYPES
# Grid2D - regular 2D grid
geo = fem.Grid2D(res=wp.vec2i(resolution))

# Trimesh2D - triangular mesh
positions, tri_vidx = fem_example_utils.gen_trimesh(res=wp.vec2i(resolution))
geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions, build_bvh=True)

# Quadmesh2D - quadrilateral mesh
positions, quad_vidx = fem_example_utils.gen_quadmesh(res=wp.vec2i(resolution))
geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions, build_bvh=True)

# 2. DOMAINS
# Cells - interior of geometry (volume elements)
domain = fem.Cells(geometry=geo)

# BoundarySides - boundary faces
boundary = fem.BoundarySides(geo)

# 3. FUNCTION SPACES
# Scalar polynomial space
scalar_space = fem.make_polynomial_space(geo, degree=2)

# Vector polynomial space (for velocity fields)
u_space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec2)

# Pressure space (one degree lower for inf-sup stability)
p_space = fem.make_polynomial_space(geo, degree=1)

# 4. FIELDS
# Create discrete field from function space
u_field = u_space.make_field()
p_field = p_space.make_field()

# ImplicitField - field defined by a function
u_bd_field = fem.ImplicitField(
    domain=boundary, 
    func=u_boundary_value, 
    values={"top_velocity": 1.0, "box_height": 1.0}
)

# 5. TEST/TRIAL FUNCTIONS
test = fem.make_test(space=scalar_space, domain=domain)
trial = fem.make_trial(space=scalar_space, domain=domain)

# 6. INTEGRATION
# Bilinear form -> matrix
matrix = fem.integrate(bilinear_form, fields={"u": trial, "v": test}, values={"param": value})

# Linear form -> vector
rhs = fem.integrate(linear_form, fields={"v": test})

# Nodal assembly (for boundary conditions)
bd_projector = fem.integrate(form, fields={"u": bd_trial, "v": bd_test}, assembly="nodal")

# 7. FIELD OPERATIONS IN INTEGRANDS
@fem.integrand
def example_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    # Evaluate field at sample point
    u_val = u(s)
    
    # Gradient
    grad_u = fem.grad(u, s)
    
    # Divergence
    div_u = fem.div(u, s)
    
    # Symmetric gradient (strain rate tensor)
    D_u = fem.D(u, s)
    
    # Domain position
    pos = domain(s)
    
    # Normal vector (on boundaries)
    nor = fem.normal(domain, s)
    
    # Lookup operator (for semi-Lagrangian)
    conv_s = fem.lookup(domain, conv_pos, s)
    
    return result

# 8. BOUNDARY CONDITIONS
# Project linear system with Dirichlet BC
fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)

# Normalize Dirichlet projector
fem.normalize_dirichlet_projector(u_bd_projector, u_bd_value)

# 9. FIELD DATA ACCESS
# Get DOF values as warp array
dof_values = field.dof_values

# Set DOF values
field.dof_values = solution_array
