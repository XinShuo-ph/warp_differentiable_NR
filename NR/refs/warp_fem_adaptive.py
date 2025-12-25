# Warp FEM Adaptive Grid Reference
# NOTE: Requires CUDA for NanoVDB volume operations

# 1. ADAPTIVE NANOGRID
# From adaptivity.py - creates grids with varying resolution levels

# Creating adaptive grid from refinement field:
"""
refinement = fem.ImplicitField(
    domain=fem.Cells(fem.Nanogrid(sim_vol)), 
    func=refinement_field, 
    values={"volume": collider.id}
)
geo = fem.adaptive_nanogrid_from_field(
    sim_vol, 
    level_count, 
    refinement_field=refinement, 
    grading="face"  # or "vertex" or None
)
"""

# 2. REFINEMENT FIELD FUNCTION
# Returns scalar: negative = carve out, positive = desired level (0=finest, 1=coarsest)
"""
@wp.func
def refinement_field(xyz: wp.vec3, volume: wp.uint64):
    uvw = wp.volume_world_to_index(volume, xyz)
    sdf = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)
    
    if sdf < 0.0:
        return sdf  # carve out
    
    # Distance-based refinement
    return 0.5 * wp.max(wp.length(xyz) - 20.0, sdf)
"""

# 3. GRADING OPTIONS
# - None: no grading enforcement
# - "face": face grading (2:1 ratio at faces)
# - "vertex": vertex grading (2:1 ratio at vertices)

# 4. CREATING FROM HIERARCHY
"""
geo = fem.adaptive_nanogrid_from_hierarchy(
    grids=[fine_vol, medium_vol, coarse_vol],  # finest to coarsest
    grading="face"
)
"""

# 5. HANDLING T-JUNCTIONS AT RESOLUTION BOUNDARIES
# Must account for discontinuities at resolution boundaries
"""
p_side_test = fem.make_test(p_space, domain=fem.Sides(geo))
u_side_trial = fem.make_trial(u_space, domain=fem.Sides(geo))

@fem.integrand
def side_divergence_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, psi: fem.Field):
    # normal velocity jump (non-zero at resolution boundaries)
    return -wp.dot(fem.jump(u, s), fem.normal(domain, s)) * fem.average(psi, s)

divergence_matrix += fem.integrate(
    side_divergence_form,
    fields={"u": u_side_trial, "psi": p_side_test},
)
"""

# 6. VOLUME UTILITIES
# Creating regular volume for base grid
"""
sim_vol = fem_example_utils.gen_volume(
    res=wp.vec3i(nx, ny, nz), 
    bounds_lo=wp.vec3(x0, y0, z0), 
    bounds_hi=wp.vec3(x1, y1, z1)
)
"""

# 7. DOMAINS FOR ADAPTIVE GRIDS
# Cells - interior cells
domain = fem.Cells(geometry=geo)

# BoundarySides - exterior faces
boundary = fem.BoundarySides(geo)

# Sides - all faces (including internal resolution boundaries)
sides = fem.Sides(geo)
