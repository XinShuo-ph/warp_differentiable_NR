import numpy as np
import warp as wp

from NR.src.poisson_fd import make_sin_sin_problem, solve_poisson_dirichlet_rbgs


def test_poisson_rbgs_matches_sin_sin():
    wp.init()

    prob = make_sin_sin_problem(n=65)
    f = wp.array(prob.f, dtype=wp.float32)

    u = solve_poisson_dirichlet_rbgs(f, num_iters=400, omega=1.9)
    u_np = u.numpy()

    err = u_np - prob.u_exact
    l2 = float(np.sqrt(np.mean(err[1:-1, 1:-1] ** 2)))

    # Iterative error + discretization error at this resolution
    assert l2 < 5.0e-3

