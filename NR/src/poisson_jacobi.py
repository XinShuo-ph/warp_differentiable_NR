import argparse
from typing import Optional

import numpy as np
import warp as wp


@wp.kernel
def init_sin_sin_rhs(
    f: wp.array2d(dtype=wp.float32),
    u_exact: wp.array2d(dtype=wp.float32),
    n: int,
    h: float,
):
    i, j = wp.tid()
    if i >= n or j >= n:
        return

    x = float(i) * h
    y = float(j) * h
    pi = 3.141592653589793

    u = wp.sin(pi * x) * wp.sin(pi * y)
    u_exact[i, j] = u

    # -Δu = f  => f = 2*pi^2*sin(pi x)*sin(pi y)
    f[i, j] = 2.0 * pi * pi * u


@wp.kernel
def jacobi_step_dirichlet0(
    u: wp.array2d(dtype=wp.float32),
    u_new: wp.array2d(dtype=wp.float32),
    f: wp.array2d(dtype=wp.float32),
    n: int,
    h2: float,
):
    i, j = wp.tid()
    if i >= n or j >= n:
        return

    if i == 0 or j == 0 or i == n - 1 or j == n - 1:
        u_new[i, j] = 0.0
        return

    u_new[i, j] = 0.25 * (
        u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] + h2 * f[i, j]
    )


def solve_poisson_dirichlet0_jacobi(
    f: wp.array2d, iters: int, u0: Optional[wp.array2d] = None
) -> wp.array2d:
    n = int(f.shape[0])
    h = 1.0 / float(n - 1)
    h2 = h * h

    u = wp.zeros((n, n), dtype=wp.float32) if u0 is None else u0
    u_new = wp.zeros((n, n), dtype=wp.float32)

    for _ in range(iters):
        wp.launch(jacobi_step_dirichlet0, dim=(n, n), inputs=(u, u_new, f, n, h2))
        u, u_new = u_new, u

    return u


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--n", type=int, default=64, help="Grid nodes per dimension (includes boundaries).")
    parser.add_argument("--iters", type=int, default=500, help="Jacobi iterations.")
    args = parser.parse_args()

    with wp.ScopedDevice(args.device):
        n = int(args.n)
        h = 1.0 / float(n - 1)

        f = wp.zeros((n, n), dtype=wp.float32)
        u_exact = wp.zeros((n, n), dtype=wp.float32)
        wp.launch(init_sin_sin_rhs, dim=(n, n), inputs=(f, u_exact, n, h))

        u = solve_poisson_dirichlet0_jacobi(f=f, iters=int(args.iters))

        u_np = u.numpy()
        u_exact_np = u_exact.numpy()
        err_l2 = float(np.linalg.norm(u_np - u_exact_np) / np.linalg.norm(u_exact_np))
        print("rel_l2_error:", err_l2)


if __name__ == "__main__":
    main()

