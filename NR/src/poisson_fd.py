from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.kernel
def _rbgs_update(
    u: wp.array2d(dtype=wp.float32),
    f: wp.array2d(dtype=wp.float32),
    h2: float,
    parity: int,
    omega: float,
):
    i, j = wp.tid()

    n = u.shape[0]
    m = u.shape[1]

    if i == 0 or j == 0 or i == n - 1 or j == m - 1:
        return

    if ((i + j) & 1) != parity:
        return

    nb = u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1]
    u_gs = 0.25 * (nb - h2 * f[i, j])  # solves Δu = f
    u[i, j] = (1.0 - omega) * u[i, j] + omega * u_gs


@dataclass(frozen=True)
class PoissonProblem:
    n: int
    f: np.ndarray  # (n, n) float32, includes boundary entries
    u_exact: np.ndarray  # (n, n) float32, includes boundary entries


def make_sin_sin_problem(n: int) -> PoissonProblem:
    if n < 3:
        raise ValueError("n must be >= 3")

    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y = np.linspace(0.0, 1.0, n, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    u = np.sin(math.pi * xx) * np.sin(math.pi * yy)
    f = -(2.0 * (math.pi**2)) * u  # Δu = f

    return PoissonProblem(n=n, f=f.astype(np.float32, copy=False), u_exact=u.astype(np.float32, copy=False))


def solve_poisson_dirichlet_rbgs(
    f: wp.array,
    num_iters: int,
    omega: float = 1.9,
    device: str | None = None,
) -> wp.array:
    if num_iters < 1:
        raise ValueError("num_iters must be >= 1")

    wp.init()

    with wp.ScopedDevice(device):
        n = int(f.shape[0])
        if int(f.shape[1]) != n:
            raise ValueError("f must be square (n, n)")

        u = wp.zeros((n, n), dtype=wp.float32)
        h = 1.0 / float(n - 1)
        h2 = float(h * h)

        for _k in range(num_iters):
            wp.launch(_rbgs_update, dim=(n, n), inputs=[u, f, h2, 0, float(omega)])
            wp.launch(_rbgs_update, dim=(n, n), inputs=[u, f, h2, 1, float(omega)])

        return u

