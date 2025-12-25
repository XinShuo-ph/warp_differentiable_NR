Source: `NR/warp/warp/examples/fem/example_navier_stokes.py`

```python
# Geometry + domains
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

# Function spaces + fields
u_space = fem.make_polynomial_space(geo, degree=u_degree, dtype=wp.vec2)
p_space = fem.make_polynomial_space(geo, degree=u_degree - 1)
self._u_field = u_space.make_field()
self._p_field = p_space.make_field()

# Boundary condition as an implicit field over a boundary domain
u_bd_test = fem.make_test(space=u_space, domain=boundary)
u_bd_field = fem.ImplicitField(
    domain=boundary, func=u_boundary_value, values={"top_velocity": top_velocity, "box_height": 1.0}
)
u_bd_value = fem.integrate(
    mass_form,
    fields={"u": u_bd_field, "v": u_bd_test},
    assembly="nodal",
    output_dtype=wp.vec2d,
)
```

```python
# Time step RHS assembly + saddle solve
u_rhs = fem.integrate(
    transported_inertia_form,
    fields={"u": self._u_field, "v": self._u_test},
    values={"dt": self.sim_dt},
    output_dtype=wp.vec2d,
)

wp.sparse.bsr_mv(self._u_bd_projector, x=u_rhs, y=u_rhs, alpha=-1.0, beta=1.0)
array_axpy(x=self._u_bd_rhs, y=u_rhs, alpha=1.0, beta=1.0)

x_u = wp.empty_like(u_rhs)
x_p = wp.empty_like(p_rhs)
wp.utils.array_cast(out_array=x_u, in_array=self._u_field.dof_values)
wp.utils.array_cast(out_array=x_p, in_array=self._p_field.dof_values)
```

