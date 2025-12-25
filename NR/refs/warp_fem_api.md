# Warp FEM API Reference

## Mesh Creation
```python
# Grid
geo = fem.Grid2D(res=wp.vec2i(resolution))

# Triangle Mesh
positions, tri_vidx = fem_example_utils.gen_trimesh(res=wp.vec2i(resolution))
geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions, build_bvh=True)

# Quad Mesh
positions, quad_vidx = fem_example_utils.gen_quadmesh(res=wp.vec2i(resolution))
geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions, build_bvh=True)
```

## Adaptive Grid (Refinement)
```python
# Load volume (requires CUDA for some operations)
collider = wp.Volume.load_from_nvdb(file)

# Create adaptive grid
refinement = fem.ImplicitField(
    domain=fem.Cells(fem.Nanogrid(sim_vol)), 
    func=refinement_field, 
    values={"volume": collider.id}
)
geo = fem.adaptive_nanogrid_from_field(
    sim_vol, 
    level_count=4, 
    refinement_field=refinement, 
    grading="face"
)
```

## Function Spaces
```python
# Scalar space
scalar_space = fem.make_polynomial_space(geo, degree=degree)

# Vector space
vector_space = fem.make_polynomial_space(geo, degree=degree, dtype=wp.vec2)
```

## Fields
```python
# Discrete field (holds values)
u_field = vector_space.make_field()

# Implicit field (function-based)
u_bd_field = fem.ImplicitField(
    domain=boundary, 
    func=u_boundary_value, 
    values={"top_velocity": top_velocity}
)
```

## Weak Forms
```python
# Test and Trial functions
u_test = fem.make_test(space=u_space, domain=domain)
u_trial = fem.make_trial(space=u_space, domain=domain)

# Integrands
@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))

# Integration
matrix = fem.integrate(
    mass_form,
    fields={"u": u_trial, "v": u_test},
    values={}
)
```

## Boundary Conditions
```python
boundary = fem.BoundarySides(geo)

# Projector for Dirichlet BC
u_bd_test = fem.make_test(space=u_space, domain=boundary)
u_bd_trial = fem.make_trial(space=u_space, domain=boundary)
u_bd_projector = fem.integrate(mass_form, fields={"u": u_bd_trial, "v": u_bd_test}, assembly="nodal")

# Normalize
fem.normalize_dirichlet_projector(u_bd_projector, u_bd_value)

# Project linear system
fem.project_linear_system(matrix, rhs, u_bd_projector, u_bd_value, normalize_projector=False)
```
