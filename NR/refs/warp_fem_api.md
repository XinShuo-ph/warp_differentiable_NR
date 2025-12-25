# Warp FEM API Snippets

## Geometry
```python
# Grid
geo = fem.Grid2D(res=wp.vec2i(resolution))

# Trimesh
positions, tri_vidx = fem_example_utils.gen_trimesh(res=wp.vec2i(resolution))
geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions, build_bvh=True)

# Quadmesh
positions, quad_vidx = fem_example_utils.gen_quadmesh(res=wp.vec2i(resolution))
geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions, build_bvh=True)

# Adaptive Nanogrid
sim_vol = wp.Volume.allocate_by_voxels(..., device="cuda") # Requires CUDA
refinement = fem.ImplicitField(
    domain=fem.Cells(fem.Nanogrid(sim_vol)), 
    func=refinement_field
)
geo = fem.adaptive_nanogrid_from_field(
    sim_vol, 
    level_count, 
    refinement_field=refinement, 
    grading="face"
)
```

## Function Spaces & Fields
```python
# Polynomial space
u_space = fem.make_polynomial_space(geo, degree=degree, dtype=wp.vec2)
p_space = fem.make_polynomial_space(geo, degree=degree - 1)

# Fields
u_field = u_space.make_field()
p_field = p_space.make_field()

# Test/Trial functions
u_test = fem.make_test(space=u_space, domain=domain)
u_trial = fem.make_trial(space=u_space, domain=domain)
```

## Integration
```python
@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))

# Assemble matrix
matrix = fem.integrate(
    mass_form,
    fields={"u": u_trial, "v": u_test},
    values={"dt": dt}
)

# Assemble vector (RHS)
rhs = fem.integrate(
    linear_form,
    fields={"v": test}
)
```

## Boundary Conditions
```python
boundary = fem.BoundarySides(geo)

# Nodal integration for BCs
bd_test = fem.make_test(space=u_space, domain=boundary)
bd_projector = fem.integrate(mass_form, fields={"u": u_bd_trial, "v": u_bd_test}, assembly="nodal")

# Implicit field for BC values
u_bd_field = fem.ImplicitField(domain=boundary, func=u_boundary_value)
```

## Refinement
```python
@wp.func
def refinement_field(xyz: wp.vec3, volume: wp.uint64):
    # Returns negative value inside refinement region? 
    # Or some metric for refinement level.
    # In example: signed distance to collider.
    pass
```
