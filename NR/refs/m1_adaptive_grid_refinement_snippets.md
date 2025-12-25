```python
# excerpt from: warp/warp/examples/fem/example_adaptive_grid.py

@wp.func
def refinement_field(xyz: wp.vec3, volume: wp.uint64):
    # use distance to collider as refinement function
    uvw = wp.volume_world_to_index(volume, xyz)
    sdf = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)

    if sdf < 0.0:
        return sdf

    # combine with  heuristical distance to keep coarsening past nvdb narrowband
    return 0.5 * wp.max(wp.length(xyz) - 20.0, sdf)


class Example:
    def __init__(
        self, quiet=False, degree=2, div_conforming=False, base_resolution=8, level_count=4, headless: bool = False
    ):
        # Start from a coarse, dense grid
        res = wp.vec3i(2 * base_resolution, base_resolution // 2, base_resolution)
        bounds_lo = wp.vec3(-50.0, 0.0, -17.5)
        bounds_hi = wp.vec3(50.0, 12.5, 17.5)
        sim_vol = fem_example_utils.gen_volume(res=res, bounds_lo=bounds_lo, bounds_hi=bounds_hi)

        # load collision volume
        collider_path = os.path.join(warp.examples.get_asset_directory(), "rocks.nvdb")
        with open(collider_path, "rb") as file:
            # create Volume object
            collider = wp.Volume.load_from_nvdb(file)

        # Make adaptive grid from coarse base and refinement field
        refinement = fem.ImplicitField(
            domain=fem.Cells(fem.Nanogrid(sim_vol)), func=refinement_field, values={"volume": collider.id}
        )
        self._geo = fem.adaptive_nanogrid_from_field(sim_vol, level_count, refinement_field=refinement, grading="face")
```

