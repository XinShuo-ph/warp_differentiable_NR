```python
# excerpt from: warp/warp/examples/fem/example_navier_stokes.py

class Example:
    def __init__(self, quiet=False, degree=2, resolution=25, Re=1000.0, top_velocity=1.0, mesh: str = "grid"):
        self._quiet = quiet

        self.sim_dt = 1.0 / resolution
        self.current_frame = 0

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

        # Functions spaces: Q(d)-Q(d-1)
        u_degree = degree
        u_space = fem.make_polynomial_space(geo, degree=u_degree, dtype=wp.vec2)
        p_space = fem.make_polynomial_space(geo, degree=u_degree - 1)

        # Viscosity and inertia
        u_test = fem.make_test(space=u_space, domain=domain)
        u_trial = fem.make_trial(space=u_space, domain=domain)

        u_matrix = fem.integrate(
            viscosity_and_inertia_form,
            fields={"u": u_trial, "v": u_test},
            values={"nu": viscosity, "dt": self.sim_dt},
        )

        # Pressure-velocity coupling
        p_test = fem.make_test(space=p_space, domain=domain)
        div_matrix = fem.integrate(div_form, fields={"u": u_trial, "q": p_test})

        # Enforcing the Dirichlet boundary condition the hard way;
        # build projector for velocity left- and right-hand-sides
        u_bd_test = fem.make_test(space=u_space, domain=boundary)
        u_bd_trial = fem.make_trial(space=u_space, domain=boundary)
        u_bd_projector = fem.integrate(mass_form, fields={"u": u_bd_trial, "v": u_bd_test}, assembly="nodal")

        # Define an implicit field for our boundary condition value and integrate
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

        u_bd_rhs = wp.zeros_like(u_bd_value)
        fem.project_linear_system(u_matrix, u_bd_rhs, u_bd_projector, u_bd_value, normalize_projector=False)

        div_bd_rhs = -div_matrix @ u_bd_value
        div_matrix -= div_matrix @ u_bd_projector

        # Assemble saddle system
        self._saddle_system = fem_example_utils.SaddleSystem(u_matrix, div_matrix)

        # Velocitiy and pressure fields
        self._u_field = u_space.make_field()
        self._p_field = p_space.make_field()
```

