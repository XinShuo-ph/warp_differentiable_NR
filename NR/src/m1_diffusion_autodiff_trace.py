import numpy as np

import warp as wp
import warp.fem as fem


@fem.integrand
def y_boundary_value_form_param(s: fem.Sample, domain: fem.Domain, v: fem.Field, val: wp.array(dtype=wp.float32)):
    nor = fem.normal(domain, s)
    return wp.float32(val[0]) * v(s) * wp.abs(nor[0])


@wp.kernel
def sum_reduce(x: wp.array(dtype=wp.float32), n: int, out: wp.array(dtype=wp.float32)):
    # single-thread deterministic reduction (keeps backward simple)
    acc = wp.float32(0.0)
    for i in range(n):
        acc += x[i]
    out[0] = acc


def main() -> None:
    wp.set_module_options({"enable_backward": True})

    with wp.ScopedDevice("cpu"):
        geo = fem.Grid2D(res=wp.vec2i(10))
        space = fem.make_polynomial_space(geo, degree=2)
        boundary = fem.BoundarySides(geo)
        test = fem.make_test(space=space, domain=boundary)

        # baseline: loss(val=1) = sum(bd_rhs)
        val_unit = wp.array([1.0], dtype=wp.float32)
        bd_rhs_unit = fem.integrate(
            y_boundary_value_form_param,
            fields={"v": test},
            values={"val": val_unit},
            assembly="nodal",
            output_dtype=float,
        )
        baseline = float(np.sum(bd_rhs_unit.numpy()))

        # autodiff: d/dval sum(bd_rhs) == baseline
        val = wp.array([5.0], dtype=wp.float32, requires_grad=True)
        loss = wp.zeros(shape=1, dtype=wp.float32, requires_grad=True)

        tape = wp.Tape()
        with tape:
            bd_rhs = wp.empty(shape=space.node_count(), dtype=wp.float32, requires_grad=True)
            fem.integrate(
                y_boundary_value_form_param,
                fields={"v": test},
                values={"val": val},
                assembly="nodal",
                output_dtype=float,
                output=bd_rhs,
            )
            wp.launch(sum_reduce, dim=1, inputs=(bd_rhs, bd_rhs.shape[0], loss))

        tape.backward(loss=loss)

        grad = float(val.grad.numpy()[0])
        loss_val = float(loss.numpy()[0])

        # print with rounding so CPU scheduling doesn't affect validation
        print(f"baseline={baseline:.8e} loss={loss_val:.8e} grad={grad:.8e}")


if __name__ == "__main__":
    main()

