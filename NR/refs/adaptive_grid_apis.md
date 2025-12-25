# Adaptive Grid Refinement APIs in Warp FEM

## Overview
Adaptive grids allow local mesh refinement based on a refinement field (e.g., distance to boundaries, error estimates).

**Note**: Requires CUDA device (uses NanoVDB volumes).

## Creating Adaptive Grids

### From Refinement Field
```python
# 1. Create base coarse grid
res = wp.vec3i(16, 8, 8)
bounds_lo = wp.vec3(-50.0, 0.0, -17.5)
bounds_hi = wp.vec3(50.0, 12.5, 17.5)
sim_vol = fem_example_utils.gen_volume(res=res, bounds_lo=bounds_lo, bounds_hi=bounds_hi)

# 2. Define refinement field (distance function, error estimate, etc.)
@wp.func
def refinement_field(xyz: wp.vec3, volume: wp.uint64):
    # Negative values indicate refinement needed
    # Positive values indicate coarsening allowed
    uvw = wp.volume_world_to_index(volume, xyz)
    sdf = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)
    return sdf

# 3. Create implicit field from refinement function
refinement = fem.ImplicitField(
    domain=fem.Cells(fem.Nanogrid(sim_vol)),
    func=refinement_field,
    values={"volume": collider.id}
)

# 4. Build adaptive grid
adaptive_geo = fem.adaptive_nanogrid_from_field(
    sim_vol,
    level_count,  # Number of refinement levels
    refinement_field=refinement,
    grading="face"  # Grading mode: limits resolution jumps
)
```

### Grading Modes
- `"face"`: Adjacent cells differ by at most 1 refinement level
- Ensures smooth transitions, avoids hanging nodes in some schemes

## Geometry Type

### Nanogrid
```python
geo = fem.Nanogrid(volume)
```
- Based on NVIDIA NanoVDB
- Sparse hierarchical grid structure
- Efficient for localized features

### Adaptive Nanogrid
- Returned by `adaptive_nanogrid_from_field()`
- Automatically refined based on field values
- Supports standard FEM operations (cells, sides, boundaries)

## Using Adaptive Grids

### Domain Operations
```python
# Cell integration (same as uniform grids)
domain = fem.Cells(geometry=adaptive_geo)
test = fem.make_test(space=space, domain=domain)

# Boundary conditions
boundary = fem.BoundarySides(adaptive_geo)

# Interior sides (for DG methods)
sides = fem.Sides(adaptive_geo)
```

### Handling Resolution Boundaries

At **t-junctions** (where refined and coarse cells meet):

```python
# Need to account for velocity jumps at resolution boundaries
divergence_matrix = fem.integrate(
    divergence_form,
    fields={"u": u_trial, "psi": p_test}
)

# Add side contribution for discontinuities
p_side_test = fem.make_test(p_space, domain=fem.Sides(geo))
u_side_trial = fem.make_trial(u_space, domain=fem.Sides(geo))
divergence_matrix += fem.integrate(
    side_divergence_form,
    fields={"u": u_side_trial, "psi": p_side_test}
)
```

### Side Integrand for Discontinuities
```python
@fem.integrand
def side_divergence_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, psi: fem.Field):
    # Normal velocity jump at resolution boundaries
    return -wp.dot(fem.jump(u, s), fem.normal(domain, s)) * fem.average(psi, s)
```

## Field Operations on Adaptive Grids

### Jump and Average
```python
fem.jump(u, s)     # [u] = u^+ - u^- across side
fem.average(u, s)  # {u} = (u^+ + u^-) / 2
```
Used in DG methods and at resolution boundaries.

## Function Spaces on Adaptive Grids

All standard function spaces work:
```python
# Lagrange polynomial spaces
u_space = fem.make_polynomial_space(adaptive_geo, degree=degree, dtype=wp.vec3)

# H(div)-conforming spaces (Raviart-Thomas)
u_space = fem.make_polynomial_space(
    geo=adaptive_geo,
    element_basis=fem.ElementBasis.RAVIART_THOMAS,
    degree=degree,
    dtype=wp.vec3
)
```

## Key Differences from Uniform Grids

1. **Construction**: Need refinement field
2. **T-junctions**: Require special handling in some formulations
3. **Performance**: More DOFs only where needed
4. **Storage**: Sparse structure via NanoVDB

## Typical Workflow

1. Start with coarse base grid
2. Define refinement criterion (distance, gradient, error)
3. Build adaptive grid with desired level count
4. Use same FEM workflow as uniform grids
5. Handle t-junctions if using discontinuous methods
