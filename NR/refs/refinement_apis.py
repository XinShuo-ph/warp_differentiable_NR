# Adaptive Refinement APIs from example_adaptive_grid.py

import warp as wp
import warp.fem as fem

# === NANOGRID (Sparse Volume Grid) ===

# 1. Create base volume
sim_vol = wp.Volume.allocate_by_voxels(
    min=bounds_lo, 
    max=bounds_hi, 
    voxel_size=(bounds_hi - bounds_lo) / res
)
# or use fem_example_utils.gen_volume()

# 2. Define refinement field (function that determines refinement level)
@wp.func
def refinement_field(xyz: wp.vec3, volume: wp.uint64):
    # Return negative for fine regions, positive for coarse
    # E.g., distance to obstacle/interface
    uvw = wp.volume_world_to_index(volume, xyz)
    sdf = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)
    return sdf

# 3. Create implicit field for refinement
refinement = fem.ImplicitField(
    domain=fem.Cells(fem.Nanogrid(sim_vol)),
    func=refinement_field,
    values={"volume": collider.id}
)

# 4. Build adaptive grid from refinement field
geo = fem.adaptive_nanogrid_from_field(
    sim_vol,                    # Base volume
    level_count,                # Number of refinement levels
    refinement_field=refinement,
    grading="face"              # "face" or "vertex" grading (limits level jumps)
)

# === USING ADAPTIVE GRIDS ===

# Use same FEM APIs as regular grids
u_space = fem.make_polynomial_space(geo=geo, degree=degree, dtype=wp.vec3)
domain = fem.Cells(geo)
boundary = fem.BoundarySides(geo)

# === SPECIAL CONSIDERATIONS ===

# 1. T-junctions (hanging nodes) at resolution boundaries
# Need to handle velocity jumps on interior sides:
@fem.integrand
def side_divergence_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, psi: fem.Field):
    # fem.jump() handles discontinuity across sides
    return -wp.dot(fem.jump(u, s), fem.normal(domain, s)) * fem.average(psi, s)

# Integrate over interior sides
divergence_matrix += fem.integrate(
    side_divergence_form,
    fields={"u": u_trial, "psi": p_test},
    domain=fem.Sides(geo)  # Interior sides
)

# 2. H(div)-conforming elements (Raviart-Thomas)
# Useful for adaptive grids to ensure divergence conformity
u_space = fem.make_polynomial_space(
    geo=geo,
    element_basis=fem.ElementBasis.RAVIART_THOMAS,
    degree=degree,
    dtype=wp.vec3
)

# === FIELD OPERATORS FOR DISCONTINUOUS ELEMENTS ===

# fem.jump(field, sample): jump across interface [u] = u+ - u-
# fem.average(field, sample): average across interface {{u}} = 0.5*(u+ + u-)

# === NANOGRID GEOMETRY ===

# Can also use directly without refinement:
base_grid = fem.Nanogrid(volume)
# Sparse voxel storage, efficient for mostly empty domains
