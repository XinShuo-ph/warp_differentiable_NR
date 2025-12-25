import warp as wp
import warp.fem as fem


@wp.func
def refinement_field(xyz: wp.vec3, volume: wp.uint64):
    uvw = wp.volume_world_to_index(volume, xyz)
    sdf = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)

    if sdf < 0.0:
        return sdf

    return 0.5 * wp.max(wp.length(xyz) - 20.0, sdf)


def build_adaptive_nanogrid(sim_vol: wp.Volume, collider: wp.Volume, level_count: int):
    refinement = fem.ImplicitField(
        domain=fem.Cells(fem.Nanogrid(sim_vol)), func=refinement_field, values={"volume": collider.id}
    )
    geo = fem.adaptivity.adaptive_nanogrid_from_field(sim_vol, level_count, refinement_field=refinement, grading="face")
    return geo

