# Warp FEM Mesh and Field APIs

## Geometry Types
- `fem.Grid2D(res=wp.vec2i(n))` - Structured 2D grid
- `fem.Trimesh2D(tri_vertex_indices, positions, build_bvh)` - Triangle mesh
- `fem.Quadmesh2D(quad_vertex_indices, positions, build_bvh)` - Quad mesh
- BVH (build_bvh=True) enables fast spatial queries for semi-Lagrangian schemes

## Domain Types
- `fem.Cells(geometry=geo)` - Interior cell domain
- `fem.BoundarySides(geo)` - Boundary edges/faces

## Function Spaces
```python
# Scalar space
scalar_space = fem.make_polynomial_space(geo, degree=d)

# Vector space
vector_space = fem.make_polynomial_space(geo, degree=d, dtype=wp.vec2)

# Mixed spaces (Stokes/NS): Q(d)-Q(d-1)
u_space = fem.make_polynomial_space(geo, degree=d, dtype=wp.vec2)
p_space = fem.make_polynomial_space(geo, degree=d-1)
```

## Field Operations
- `u(s)` - Evaluate field at sample point
- `fem.grad(u, s)` - Gradient of scalar field
- `fem.div(u, s)` - Divergence of vector field
- `fem.D(u, s)` - Symmetric gradient (strain rate): 0.5*(grad u + grad^T u)
- `fem.normal(domain, s)` - Normal vector at boundary sample

## Spatial Queries
```python
# Semi-Lagrangian advection pattern
pos = domain(s)  # Physical position of sample
new_s = fem.lookup(domain, new_pos, s)  # Find sample at new position
value = u(new_s)  # Evaluate field at new sample
```

## Test/Trial Functions
```python
test = fem.make_test(space=space, domain=domain)
trial = fem.make_trial(space=space, domain=domain)
```

## Implicit Fields
For boundary conditions:
```python
bc_field = fem.ImplicitField(
    domain=boundary,
    func=bc_function,  # wp.func decorated
    values={"param1": val1, ...}
)
```

## Assembly Modes
- Default (elemental): Standard FEM assembly
- `assembly="nodal"`: Nodal (lumped) assembly for BCs
