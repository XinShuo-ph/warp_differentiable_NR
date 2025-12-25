import unittest

import numpy as np

from NR.src.poisson import exact_sine_solution, rhs_for_exact_sine, solve_poisson_dirichlet_zero


class TestPoissonDirichletZero(unittest.TestCase):
    def test_sine_solution_matches(self):
        n = 33
        f = rhs_for_exact_sine(n)
        exact = exact_sine_solution(n)

        res = solve_poisson_dirichlet_zero(f, n=n, iters=1200, omega=1.8, device="cpu")

        err = res.u - exact
        l2 = float(np.sqrt(np.mean(err * err)))
        linf = float(np.max(np.abs(err)))

        self.assertLess(l2, 5e-3)
        self.assertLess(linf, 2e-2)


if __name__ == "__main__":
    unittest.main()

