import math
from dataclasses import dataclass

import numpy as np

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem


PI_F32 = wp.float32(math.pi)
TWO_PI2_F32 = wp.float32(2.0 * math.pi * math.pi)


@fem.integrand
def poisson_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def rhs_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    x = domain(s)
    f = TWO_PI2_F32 * wp.sin(PI_F32 * x[0]) * wp.sin(PI_F32 * x[1])
    return f * v(s)


@fem.integrand
def dirichlet_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


@fem.integrand
def l2_error_form(s: fem.Sample, domain: fem.Domain, u: fem.Field):
    x = domain(s)
    u_exact = wp.sin(PI_F32 * x[0]) * wp.sin(PI_F32 * x[1])
    d = u(s) - u_exact
    return d * d


@dataclass(frozen=True)
class PoissonResult:
    u_field: fem.DiscreteField
    l2_error: float


def solve_poisson_sin_dirichlet(
    *,
    resolution: int = 32,
    degree: int = 2,
    device: str = "cpu",
    tol: float = 1.0e-10,
    max_iters: int = 10_000,
) -> PoissonResult:
    wp.set_module_options({"enable_backward": False})

    with wp.ScopedDevice(device):
        geo = fem.Grid2D(res=wp.vec2i(resolution))
        domain = fem.Cells(geo)
        boundary = fem.BoundarySides(geo)

        space = fem.make_polynomial_space(geo, degree=degree, dtype=wp.float32)
        u_field = space.make_field()

        test = fem.make_test(space=space, domain=domain)
        trial = fem.make_trial(space=space, domain=domain)

        rhs = fem.integrate(rhs_form, fields={"v": test}, domain=domain, output_dtype=wp.float32)
        matrix = fem.integrate(poisson_form, fields={"u": trial, "v": test}, domain=domain, output_dtype=wp.float32)

        bd_test = fem.make_test(space=space, domain=boundary)
        bd_trial = fem.make_trial(space=space, domain=boundary)
        bd_matrix = fem.integrate(
            dirichlet_projector_form,
            fields={"u": bd_trial, "v": bd_test},
            assembly="nodal",
            output_dtype=wp.float32,
        )

        bd_rhs = wp.zeros(shape=rhs.shape[0], dtype=wp.float32)
        fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)

        x = wp.zeros_like(rhs)
        fem_example_utils.bsr_cg(matrix, b=rhs, x=x, quiet=True, tol=tol, max_iters=max_iters)
        u_field.dof_values = x

        err = wp.zeros(shape=1, dtype=wp.float32)
        fem.integrate(l2_error_form, fields={"u": u_field}, domain=domain, output=err)

        l2_error = float(np.sqrt(err.numpy()[0]))
        return PoissonResult(u_field=u_field, l2_error=l2_error)

