# Mesh and Field APIs in Warp FEM

## Geometry Types

### Grid2D
```python
geo = fem.Grid2D(res=wp.vec2i(resolution))
```
- Uniform 2D Cartesian grid
- Simple, fast, no explicit connectivity

### Trimesh2D
```python
geo = fem.Trimesh2D(
    tri_vertex_indices=tri_vidx, 
    positions=positions,
    build_bvh=True  # For fast lookups in semi-Lagrangian advection
)
```
- Unstructured triangle mesh
- Requires vertex positions and connectivity

### Quadmesh2D
```python
geo = fem.Quadmesh2D(
    quad_vertex_indices=quad_vidx,
    positions=positions,
    build_bvh=True
)
```
- Unstructured quadrilateral mesh

## Domain Types

### Cells
```python
domain = fem.Cells(geometry=geo)
```
- Volume integration over mesh elements
- Used for PDE operators

### BoundarySides
```python
boundary = fem.BoundarySides(geo)
```
- Surface integration over mesh boundaries
- Used for boundary conditions

## Function Spaces

### Polynomial Spaces
```python
# Scalar space
scalar_space = fem.make_polynomial_space(geo, degree=degree)

# Vector space (for velocity)
vector_space = fem.make_polynomial_space(geo, degree=degree, dtype=wp.vec2)

# Mixed spaces (Q_d - Q_{d-1} for Stokes)
u_space = fem.make_polynomial_space(geo, degree=u_degree, dtype=wp.vec2)
p_space = fem.make_polynomial_space(geo, degree=u_degree - 1)
```

Options:
- `degree`: Polynomial degree (1=linear, 2=quadratic, etc.)
- `dtype`: Scalar or vector types (wp.float32, wp.vec2, wp.vec3, etc.)
- `element_basis`: Optional basis type (e.g., SERENDIPITY)

## Fields

### DiscreteField
```python
# Create field from space
field = space.make_field()

# Access DOF values
field.dof_values  # warp array of coefficients
```

### ImplicitField
```python
# Field defined by a function
u_bd_field = fem.ImplicitField(
    domain=boundary,
    func=u_boundary_value,  # wp.func returning value
    values={"top_velocity": 1.0, "box_height": 1.0}
)
```
- Used for analytical boundary conditions
- Evaluated on-the-fly during integration

## Test and Trial Functions

```python
test = fem.make_test(space=space, domain=domain)
trial = fem.make_trial(space=space, domain=domain)
```
- Test: Weighting functions for weak form
- Trial: Unknown functions to solve for

## Field Operations in Integrands

### Evaluation
```python
u(s)  # Evaluate field u at sample point s
```

### Gradient
```python
fem.grad(u, s)  # Spatial gradient ∇u
```

### Divergence
```python
fem.div(u, s)  # Divergence ∇·u (for vector fields)
```

### Symmetric Gradient (Deformation tensor)
```python
fem.D(u, s)  # (∇u + ∇u^T) / 2
```

### Field Lookup (for semi-Lagrangian advection)
```python
pos = domain(s)  # Get physical coordinates
new_s = fem.lookup(domain, target_pos, s)  # Find sample at target_pos
u(new_s)  # Evaluate field at new location
```

## Integration

```python
# Linear form (produces vector)
rhs = fem.integrate(linear_form, fields={"v": test})

# Bilinear form (produces matrix)
matrix = fem.integrate(bilinear_form, fields={"u": trial, "v": test})

# With parameters
matrix = fem.integrate(
    form,
    fields={"u": trial, "v": test},
    values={"nu": viscosity, "dt": dt}
)

# Nodal assembly (for boundary conditions)
bd_matrix = fem.integrate(form, fields={"u": bd_trial, "v": bd_test}, assembly="nodal")
```

Assembly modes:
- Default: Accumulate contributions from all quadrature points
- `"nodal"`: One-to-one mapping for boundary conditions

## Matrix Operations

```python
# Matrix-vector multiply
result = matrix @ vector

# Sparse operations
wp.sparse.bsr_mv(matrix, x=u, y=result, alpha=1.0, beta=0.0)

# Matrix addition
total_matrix = matrix1 + matrix2 * scale
```
