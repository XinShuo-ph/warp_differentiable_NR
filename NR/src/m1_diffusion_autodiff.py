import argparse

import numpy as np
import warp as wp
import warp.fem as fem


@fem.integrand
def grad_energy_density(s: fem.Sample, u: fem.Field, nu: float):
    g = fem.grad(u, s)
    return nu * wp.dot(g, g)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=10, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree of shape functions.")
    parser.add_argument("--nu", type=float, default=2.0, help="Diffusion coefficient.")
    args = parser.parse_args()

    wp.set_module_options({"enable_backward": True})

    with wp.ScopedDevice(args.device):
        geo = fem.Grid2D(res=wp.vec2i(args.resolution))
        space = fem.make_polynomial_space(geo, degree=args.degree)

        # Differentiable input field (DOFs)
        u = space.make_field()
        rng = np.random.default_rng(0)
        u_np = rng.standard_normal(space.node_count(), dtype=np.float32)
        u.dof_values = wp.array(u_np, dtype=wp.float32, requires_grad=True)

        domain = fem.Cells(geo)

        energy = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        tape = wp.Tape()
        with tape:
            fem.integrate(
                grad_energy_density,
                fields={"u": u},
                values={"nu": float(args.nu)},
                domain=domain,
                output=energy,
            )

        tape.backward(energy)

        grad = u.dof_values.grad.numpy()
        print("energy:", float(energy.numpy()[0]))
        print("grad_u: l2 =", float(np.linalg.norm(grad)), "max_abs =", float(np.max(np.abs(grad))))


if __name__ == "__main__":
    main()

