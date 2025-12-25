# Extracted (and slightly trimmed) from:
# warp/examples/fem/example_navier_stokes.py

import warp as wp
import warp.fem as fem
from warp.fem.linalg import array_axpy
import warp.examples.fem.utils as fem_example_utils


def timestep_slice(u_field, u_test, sim_dt, u_bd_projector, u_bd_rhs, div_bd_rhs, saddle_system, quiet: bool):
    u_rhs = fem.integrate(
        transported_inertia_form,
        fields={"u": u_field, "v": u_test},
        values={"dt": sim_dt},
        output_dtype=wp.vec2d,
    )

    # Apply boundary conditions: u_rhs = (I - P) * u_rhs + u_bd_rhs
    wp.sparse.bsr_mv(u_bd_projector, x=u_rhs, y=u_rhs, alpha=-1.0, beta=1.0)
    array_axpy(x=u_bd_rhs, y=u_rhs, alpha=1.0, beta=1.0)

    x_u = wp.empty_like(u_rhs)
    x_p = wp.empty_like(div_bd_rhs)
    wp.utils.array_cast(out_array=x_u, in_array=u_field.dof_values)

    fem_example_utils.bsr_solve_saddle(
        saddle_system=saddle_system,
        tol=1.0e-6,
        x_u=x_u,
        x_p=x_p,
        b_u=u_rhs,
        b_p=div_bd_rhs,
        quiet=quiet,
    )

    wp.utils.array_cast(in_array=x_u, out_array=u_field.dof_values)

