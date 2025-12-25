# Extracted (and slightly trimmed) from:
# warp/examples/fem/example_adaptive_grid.py

import os.path

import warp as wp
import warp.examples
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem


def refinement_api_slice(base_resolution: int, level_count: int):
    # Start from a coarse, dense grid (as a wp.Volume)
    res = wp.vec3i(2 * base_resolution, base_resolution // 2, base_resolution)
    bounds_lo = wp.vec3(-50.0, 0.0, -17.5)
    bounds_hi = wp.vec3(50.0, 12.5, 17.5)
    sim_vol = fem_example_utils.gen_volume(res=res, bounds_lo=bounds_lo, bounds_hi=bounds_hi)

    # Load an NVDB collider as wp.Volume
    collider_path = os.path.join(warp.examples.get_asset_directory(), "rocks.nvdb")
    with open(collider_path, "rb") as file:
        collider = wp.Volume.load_from_nvdb(file)

    # Adaptive grid from coarse base + a refinement scalar field
    refinement = fem.ImplicitField(
        domain=fem.Cells(fem.Nanogrid(sim_vol)),
        func=refinement_field,
        values={"volume": collider.id},
    )
    geo = fem.adaptivity.adaptive_nanogrid_from_field(
        sim_vol,
        level_count,
        refinement_field=refinement,
        grading="face",
    )

    return geo

