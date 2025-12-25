# Extracted (and slightly trimmed) from:
# warp/examples/fem/example_darcy_ls_optimization.py

import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils


def step_like(p_matrix, p_rhs, p, bd_projector, bd_prescribed_value, quiet: bool):
    # Forward step, record adjoint tape for forces
    p_rhs = wp.empty(p_rhs.shape[0], dtype=wp.float32, requires_grad=True)

    tape = wp.Tape()
    with tape:
        fem.integrate(
            diffusion_form,
            fields={"level_set": advected_level_set, "p": p_field, "q": p_test},
            values={"smoothing": smoothing, "scale": -1.0},
            output=p_rhs,
        )

    fem.project_linear_system(p_matrix, p_rhs, bd_projector, bd_prescribed_value, normalize_projector=False)
    fem_example_utils.bsr_cg(p_matrix, b=p_rhs, x=p, quiet=quiet, tol=1e-6, max_iters=1000)

    # Record adjoint of linear solve (implicit function theorem / custom adjoint)
    def solve_linear_system():
        fem_example_utils.bsr_cg(p_matrix, b=p.grad, x=p_rhs.grad, quiet=quiet, tol=1e-6, max_iters=1000)
        p_rhs.grad -= bd_projector @ p_rhs.grad

    tape.record_func(solve_linear_system, arrays=(p_rhs, p))

    loss = wp.empty(shape=1, dtype=wp.float32, requires_grad=True)
    with tape:
        fem.integrate(
            inflow_velocity,
            fields={"level_set": advected_level_set.trace(), "p": p_field.trace()},
            values={"smoothing": smoothing},
            domain=inflow_domain,
            output=loss,
        )

    tape.backward(loss=loss)
    tape.zero()

