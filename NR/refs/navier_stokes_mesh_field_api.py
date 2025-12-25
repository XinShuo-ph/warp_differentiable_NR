import warp as wp
import warp.fem as fem


@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


def build_mesh_and_spaces(resolution: int, degree: int):
    # geometry (grid / tri / quad)
    geo = fem.Grid2D(res=wp.vec2i(resolution))

    # domains
    domain = fem.Cells(geometry=geo)
    boundary = fem.BoundarySides(geo)

    # mixed function spaces (Q(d) - Q(d-1))
    u_space = fem.make_polynomial_space(geo, degree=degree, dtype=wp.vec2)
    p_space = fem.make_polynomial_space(geo, degree=degree - 1)

    # discrete fields
    u_field = u_space.make_field()
    p_field = p_space.make_field()

    return geo, domain, boundary, u_space, p_space, u_field, p_field


def dirichlet_bc_projector(u_space, boundary, u_boundary_value, top_velocity: float):
    # build nodal boundary mass matrix (projector)
    u_bd_test = fem.make_test(space=u_space, domain=boundary)
    u_bd_trial = fem.make_trial(space=u_space, domain=boundary)
    u_bd_projector = fem.integrate(mass_form, fields={"u": u_bd_trial, "v": u_bd_test}, assembly="nodal")

    # implicit boundary value field, then integrate to get prescribed nodal values
    u_bd_field = fem.ImplicitField(
        domain=boundary, func=u_boundary_value, values={"top_velocity": top_velocity, "box_height": 1.0}
    )
    u_bd_value = fem.integrate(
        mass_form,
        fields={"u": u_bd_field, "v": u_bd_test},
        assembly="nodal",
        output_dtype=wp.vec2d,
    )

    fem.normalize_dirichlet_projector(u_bd_projector, u_bd_value)
    return u_bd_projector, u_bd_value

