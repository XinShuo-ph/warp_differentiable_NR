# Warp FEM API Reference
# Extracted from example_diffusion.py and example_navier_stokes.py

# Key imports
# import warp as wp
# import warp.fem as fem

# --- GEOMETRY ---
# fem.Grid2D(res=wp.vec2i(N))                          # 2D structured grid
# fem.Trimesh2D(tri_vertex_indices, positions)         # 2D triangle mesh
# fem.Quadmesh2D(quad_vertex_indices, positions)       # 2D quad mesh

# --- DOMAINS ---
# fem.Cells(geometry=geo)                              # Interior cells
# fem.BoundarySides(geo)                               # Boundary faces

# --- FUNCTION SPACES ---
# fem.make_polynomial_space(geo, degree=2)             # Scalar space
# fem.make_polynomial_space(geo, degree=2, dtype=wp.vec2)  # Vector space
# space.make_field()                                   # Create field in space

# --- TEST/TRIAL FUNCTIONS ---
# fem.make_test(space=space, domain=domain)
# fem.make_trial(space=space, domain=domain)

# --- INTEGRANDS (weak forms) ---
# @fem.integrand
# def form(s: fem.Sample, u: fem.Field, v: fem.Field, param: float):
#     return wp.dot(fem.grad(u, s), fem.grad(v, s))

# --- INTEGRATION ---
# fem.integrate(form, fields={"u": trial, "v": test}, values={"param": 1.0})
# Returns BSR matrix for bilinear forms, array for linear forms

# --- FIELD OPERATIONS ---
# fem.grad(u, s)      # gradient
# fem.div(u, s)       # divergence  
# fem.D(u, s)         # strain tensor (symmetric gradient)
# u(s)                # field value at sample point

# --- BOUNDARY CONDITIONS ---
# fem.project_linear_system(matrix, rhs, bc_matrix, bc_rhs)  # Hard BC
# matrix += bc_matrix * strength  # Weak BC (penalty method)

# --- LINEAR SOLVE ---
# from warp.fem.linalg import bsr_cg
# bsr_cg(matrix, b=rhs, x=solution)  # Conjugate gradient
