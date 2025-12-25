# Warp Adaptive Grid / Refinement API Reference
# Key patterns from example_adaptive_grid.py
# NOTE: Nanogrid/Volume features require CUDA

import warp as wp
import warp.fem as fem

# ============ NANOGRID (Sparse Volume) ============
# Nanogrid: Sparse hierarchical grid (based on NanoVDB)
# Requires CUDA for Volume operations

# Create from dense volume:
# sim_vol = fem_example_utils.gen_volume(res=res, bounds_lo=bounds_lo, bounds_hi=bounds_hi)
# base_geo = fem.Nanogrid(sim_vol)

# ============ ADAPTIVE REFINEMENT ============
# adaptive_nanogrid_from_field: Creates refined grid based on a field

# 1. Define refinement field (function that returns distance/error)
# @wp.func
# def refinement_field(xyz: wp.vec3, volume: wp.uint64):
#     sdf = ...  # signed distance or error metric
#     return sdf  # negative values -> refine more

# 2. Wrap as ImplicitField
# refinement = fem.ImplicitField(
#     domain=fem.Cells(base_geo),
#     func=refinement_field,
#     values={"volume": collider.id}
# )

# 3. Create adaptive grid
# adaptive_geo = fem.adaptive_nanogrid_from_field(
#     base_volume,        # coarse base grid
#     level_count,        # number of refinement levels
#     refinement_field=refinement,
#     grading="face"      # grading options: "face", "edge", "vertex", None
# )

# ============ HANDLING DISCONTINUITIES ============
# At resolution boundaries (T-junctions), need special handling:

# fem.Sides(geo) - all interior faces including level transitions
# fem.jump(u, s) - jump in field value across face
# fem.average(psi, s) - average value across face

# Divergence correction at T-junctions:
# @fem.integrand
# def side_divergence_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, psi: fem.Field):
#     return -wp.dot(fem.jump(u, s), fem.normal(domain, s)) * fem.average(psi, s)

# ============ ELEMENT BASES FOR FLUID ============
# fem.ElementBasis.RAVIART_THOMAS - H(div)-conforming for incompressible flow
# Ensures continuous normal velocity across elements

# u_space = fem.make_polynomial_space(
#     geo=geo,
#     element_basis=fem.ElementBasis.RAVIART_THOMAS,
#     degree=degree,
#     dtype=wp.vec3,
# )
