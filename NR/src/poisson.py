import math
from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.kernel
def rbgs_sor_step(u: wp.array(dtype=wp.float32), f: wp.array(dtype=wp.float32), n: int, h2: float, omega: float, color: int):
    tid = wp.tid()

    m = n - 2
    i = 1 + (tid % m)
    j = 1 + (tid // m)

    if ((i + j) & 1) != color:
        return

    idx = j * n + i

    ue = u[idx + 1]
    uw = u[idx - 1]
    un = u[idx + n]
    us = u[idx - n]

    u_new = 0.25 * (ue + uw + un + us + h2 * f[idx])
    u[idx] = (1.0 - omega) * u[idx] + omega * u_new


@dataclass(frozen=True)
class PoissonSolveResult:
    u: np.ndarray
    n: int
    iters: int
    omega: float


def solve_poisson_dirichlet_zero(
    f: np.ndarray,
    *,
    n: int,
    iters: int = 800,
    omega: float = 1.8,
    device: str = "cpu",
) -> PoissonSolveResult:
    """
    Solve -Δu = f on [0,1]^2 using a 5-pt stencil with u=0 on the boundary.

    Grid is n x n including boundary.
    """
    if f.shape != (n, n):
        raise ValueError(f"expected f.shape == ({n}, {n}), got {f.shape}")
    if n < 3:
        raise ValueError("n must be >= 3")
    if not (0.0 < omega < 2.0):
        raise ValueError("omega must be in (0, 2)")

    wp.init()

    h = 1.0 / (n - 1)
    h2 = float(h * h)

    u0 = np.zeros((n, n), dtype=np.float32)

    with wp.ScopedDevice(device):
        u = wp.array(u0.reshape(-1), dtype=wp.float32)
        ff = wp.array(f.reshape(-1).astype(np.float32, copy=False), dtype=wp.float32)

        dim = (n - 2) * (n - 2)
        for _ in range(iters):
            wp.launch(rbgs_sor_step, dim=dim, inputs=[u, ff, n, h2, float(omega), 0])
            wp.launch(rbgs_sor_step, dim=dim, inputs=[u, ff, n, h2, float(omega), 1])

        u_host = u.numpy().reshape(n, n)

    return PoissonSolveResult(u=u_host, n=n, iters=iters, omega=omega)


def exact_sine_solution(n: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    y = np.linspace(0.0, 1.0, n, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.sin(math.pi * xx) * np.sin(math.pi * yy)


def rhs_for_exact_sine(n: int) -> np.ndarray:
    u = exact_sine_solution(n)
    return 2.0 * (math.pi**2) * u

