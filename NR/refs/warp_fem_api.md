# Warp FEM API Reference

## Geometry
- `fem.Grid2D(res=wp.vec2i(nx, ny), bounds_lo=..., bounds_hi=...)`: Structured 2D grid.
- `fem.Grid3D(res=wp.vec3i(nx, ny, nz), ...)`: Structured 3D grid.
- `fem.Trimesh2D`: Unstructured triangle mesh.
- `fem.Quadmesh2D`: Unstructured quad mesh.

## Function Spaces
- `fem.make_polynomial_space(geo, degree=1, dtype=float)`: Creates a finite element function space on a geometry.
  - `degree`: Polynomial degree of basis functions.
  - `dtype`: value type (scalar `float` or vector `wp.vec2/3`).

## Fields
- `space.make_field()`: Creates a discrete field (holds DOFs).
- `fem.make_test(space, domain)`: Creates a test function for weak forms.
- `fem.make_trial(space, domain)`: Creates a trial function for bilinear forms.
- `fem.ImplicitField(domain, func, values)`: Field defined by a function (e.g. for analytic BCs).

## Integration
- `@fem.integrand`: Decorator for weak form kernels.
- `fem.integrate(form, fields={...}, values={...})`:
  - `form`: The integrand function.
  - `fields`: Dictionary mapping argument names to Test/Trial/Discrete fields.
  - `values`: Dictionary mapping argument names to constants or arrays.
  - Returns:
    - If Test & Trial provided: `BsrMatrix` (stiffness/mass matrix).
    - If only Test provided: `wp.array` (RHS vector).

## Boundary Conditions
- `fem.BoundarySides(geo)`: Domain representing the boundary of the geometry.
- `fem.dirichlet`: Utilities for imposing Dirichlet BCs.
- `fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)`: Enforces hard BCs by modifying the system.

## Linear Algebra
- `fem.linalg.array_axpy`: AXPY operation.
- `warp.sparse.bsr_mv`: Matrix-vector multiplication.
- `warp.sparse.bsr_cg`: Conjugate Gradient solver for BSR matrices.

## Adaptive Grid / NanoVDB
- `wp.Volume.load_from_nvdb(file)`: Loads a sparse volume structure (NanoVDB).
- `fem.Nanogrid(volume)`: Wraps a volume as a grid.
- `fem.adaptive_nanogrid_from_field(volume, level_count, refinement_field, grading="face")`:
  - Creates an adaptive octree-like grid based on a refinement field.
  - `refinement_field`: ImplicitField returning values to control refinement (e.g. SDF).
  - `grading`: Refinement strategy.
