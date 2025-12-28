# Warp FEM Adaptive Grid API Reference
# NOTE: Requires CUDA (wp.Volume uses GPU-only sparse tiles)

import warp as wp
import warp.fem as fem

# =============================================================================
# NANOGRID (Sparse 3D Grid based on NanoVDB)
# =============================================================================

# Create a base volume (dense grid with given resolution)
sim_vol = fem_example_utils.gen_volume(res=res, bounds_lo=bounds_lo, bounds_hi=bounds_hi)

# Create Nanogrid geometry from volume
geo = fem.Nanogrid(sim_vol)

# =============================================================================
# ADAPTIVE GRID FROM REFINEMENT FIELD
# =============================================================================

# Define refinement field function
@wp.func
def refinement_field(xyz: wp.vec3, volume: wp.uint64):
    # Return negative values where refinement is needed
    uvw = wp.volume_world_to_index(volume, xyz)
    sdf = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)
    return sdf

# Create implicit field for refinement
refinement = fem.ImplicitField(
    domain=fem.Cells(fem.Nanogrid(sim_vol)),
    func=refinement_field,
    values={"volume": collider.id}
)

# Generate adaptive nanogrid with multiple refinement levels
adaptive_geo = fem.adaptive_nanogrid_from_field(
    sim_vol,                    # base volume
    level_count,                # number of refinement levels
    refinement_field=refinement, # field to drive refinement
    grading="face"              # grading type ("face" ensures face-connected cells)
)

# =============================================================================
# H(DIV) CONFORMING ELEMENTS (Raviart-Thomas)
# =============================================================================

# Use Raviart-Thomas elements for divergence-free velocity fields
u_space = fem.make_polynomial_space(
    geo=adaptive_geo,
    element_basis=fem.ElementBasis.RAVIART_THOMAS,
    degree=degree,
    dtype=wp.vec3,
)

# =============================================================================
# HANDLING T-JUNCTIONS (Resolution Boundaries)
# =============================================================================

# At resolution boundaries, need side integrals for jumps
@fem.integrand
def side_divergence_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, psi: fem.Field):
    # Normal velocity jump (non-zero at resolution boundaries)
    return -wp.dot(fem.jump(u, s), fem.normal(domain, s)) * fem.average(psi, s)

# Integrate over all interior sides
sides = fem.Sides(geo)  # all interior sides
divergence_matrix += fem.integrate(
    side_divergence_form,
    fields={"u": u_side_trial, "psi": p_side_test},
)

# =============================================================================
# KEY FEM OPERATORS FOR ADAPTIVE GRIDS
# =============================================================================

# fem.jump(field, sample)    - jump across cell interface
# fem.average(field, sample) - average across cell interface
# fem.Sides(geo)             - domain of all interior cell interfaces
