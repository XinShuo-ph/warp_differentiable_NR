# Warp FEM Mesh and Field APIs

## Geometry and Domain
- `fem.Grid2D(res=wp.vec2i(res))`: Creates a 2D grid geometry with specified resolution.
- `fem.Cells(geometry=geo)`: Creates a domain covering the geometry cells.
- `fem.BoundarySides(geo)`: Creates a domain covering the boundary sides of the geometry.
- `fem.Sides(geo)`: Creates a domain covering all sides (faces) of the geometry (useful for DG/adaptive).

## Adaptive Grids (Nanovdb/Sparse)
- `fem.Nanogrid(volume)`: Creates a grid from a volume (e.g. from VDB).
- `fem.adaptive_nanogrid_from_field(volume, level_count, refinement_field=..., grading=...)`: Creates an adaptive grid.
  - `refinement_field`: Scalar field defining refinement level/criteria.
  - `grading`: e.g. "face".

## Function Spaces
- `fem.make_polynomial_space(geo, degree=..., dtype=...)`: Creates a polynomial function space.
  - `dtype=wp.vec2` for vector fields.
  - Default `dtype=float` for scalar fields.
  - `element_basis=fem.ElementBasis.RAVIART_THOMAS`: For H(div) conforming spaces.

## Fields
- `space.make_field()`: Creates a discrete field (with DOF values) in the function space.
- `fem.make_test(space=..., domain=...)`: Creates a test field for integration.
- `fem.make_trial(space=..., domain=...)`: Creates a trial field for integration.
- `fem.ImplicitField(domain=..., func=..., values=...)`: Creates a field defined by a function `func(x, ...)`.

## Integration
- `fem.integrate(form, fields={...}, values={...}, output=..., assembly=...)`: Integrates a form.
  - `fields`: Dictionary mapping form arguments to Fields.
  - `values`: Dictionary mapping form arguments to global values (arrays or scalars).
  - `output`: Optional output array (Vector) or Matrix.
  - `assembly`: "nodal" for boundary conditions, "dispatch" (default) or "generic".

## Operators inside Integrands
- `u(s)`: Evaluate field `u` at sample `s`.
- `fem.grad(u, s)`: Gradient of field `u` at sample `s`.
- `fem.div(u, s)`: Divergence of field `u` at sample `s`.
- `fem.D(u, s)`: Symmetric gradient (Strain rate tensor) of `u` at `s`.
- `fem.curl(u, s)`: Curl.
- `fem.jump(u, s)`: Jump of field across element boundaries.
- `fem.average(u, s)`: Average of field across element boundaries.
- `domain(s)`: Get world position of sample `s`.
- `fem.lookup(domain, pos, s)`: Lookup sample at position `pos`.

## Linear Algebra
- `fem.project_linear_system(...)`: Enforce Dirichlet BCs.
- `fem.normalize_dirichlet_projector(...)`: Normalize BC projector.
- `wp.sparse.bsr_mv(...)`: Sparse matrix-vector multiplication.
- `fem_example_utils.bsr_cg(...)`: Conjugate Gradient solver.
- `fem_example_utils.bsr_solve_saddle(...)`: Solve saddle point systems (Stokes).
