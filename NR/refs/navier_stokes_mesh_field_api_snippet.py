# Extracted (and slightly trimmed) from:
# warp/examples/fem/example_navier_stokes.py

import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils


def fem_api_slice(resolution: int, degree: int, top_velocity: float, Re: float, mesh: str):
    viscosity = top_velocity / Re

    if mesh == "tri":
        positions, tri_vidx = fem_example_utils.gen_trimesh(res=wp.vec2i(resolution))
        geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions, build_bvh=True)
    elif mesh == "quad":
        positions, quad_vidx = fem_example_utils.gen_quadmesh(res=wp.vec2i(resolution))
        geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions, build_bvh=True)
    else:
        geo = fem.Grid2D(res=wp.vec2i(resolution))

    domain = fem.Cells(geometry=geo)
    boundary = fem.BoundarySides(geo)

    # Function spaces (velocity: vec2, pressure: scalar)
    u_space = fem.make_polynomial_space(geo, degree=degree, dtype=wp.vec2)
    p_space = fem.make_polynomial_space(geo, degree=degree - 1)

    # Weak forms via test/trial fields
    u_test = fem.make_test(space=u_space, domain=domain)
    u_trial = fem.make_trial(space=u_space, domain=domain)
    p_test = fem.make_test(space=p_space, domain=domain)

    u_matrix = fem.integrate(viscosity_and_inertia_form, fields={"u": u_trial, "v": u_test}, values={"nu": viscosity, "dt": 1.0 / resolution})
    div_matrix = fem.integrate(div_form, fields={"u": u_trial, "q": p_test})

    # Hard Dirichlet BC projector assembled on boundary
    u_bd_test = fem.make_test(space=u_space, domain=boundary)
    u_bd_trial = fem.make_trial(space=u_space, domain=boundary)
    u_bd_projector = fem.integrate(mass_form, fields={"u": u_bd_trial, "v": u_bd_test}, assembly="nodal")

    u_bd_field = fem.ImplicitField(domain=boundary, func=u_boundary_value, values={"top_velocity": top_velocity, "box_height": 1.0})
    u_bd_value = fem.integrate(mass_form, fields={"u": u_bd_field, "v": u_bd_test}, assembly="nodal", output_dtype=wp.vec2d)
    fem.normalize_dirichlet_projector(u_bd_projector, u_bd_value)

    # Discrete fields
    u_field = u_space.make_field()
    p_field = p_space.make_field()

