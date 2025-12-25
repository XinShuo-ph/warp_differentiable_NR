# Warp Adaptive Grid Refinement APIs

Note: Adaptive grids require CUDA device, not available in CPU-only mode.

## Adaptive Nanogrid Creation

```python
# Start with coarse base grid
base_vol = fem_example_utils.gen_volume(res=res, bounds_lo=lo, bounds_hi=hi)

# Define refinement criterion as ImplicitField
refinement = fem.ImplicitField(
    domain=fem.Cells(fem.Nanogrid(base_vol)),
    func=refinement_func,  # wp.func returning refinement indicator
    values={"param": value}
)

# Create adaptive grid with multiple levels
adaptive_geo = fem.adaptive_nanogrid_from_field(
    base_vol,
    level_count,
    refinement_field=refinement,
    grading="face"  # or "none" for no grading
)
```

## Refinement Function Pattern

```python
@wp.func
def refinement_field(xyz: wp.vec3, volume: wp.uint64):
    # Return negative values where refinement needed
    # Positive values indicate coarsening region
    # Magnitude controls refinement strength
    return distance_or_indicator
```

## Handling Non-Conforming Meshes

Adaptive grids create T-junctions (hanging nodes). Need special treatment:

```python
# For discontinuous quantities at resolution boundaries
sides_domain = fem.Sides(geo)
side_test = fem.make_test(space, domain=sides_domain)
side_trial = fem.make_trial(space, domain=sides_domain)

# Account for jumps at non-conforming interfaces
@fem.integrand
def side_divergence_form(s, domain, u, psi):
    return -wp.dot(fem.jump(u, s), fem.normal(domain, s)) * fem.average(psi, s)
```

## Key APIs for Non-Conforming Grids

- `fem.Sides(geo)` - Interior sides/faces domain
- `fem.jump(u, s)` - Jump discontinuity across sides
- `fem.average(u, s)` - Average value across sides
- Needed for correct divergence operator on adaptive meshes
