import unittest

import numpy as np
import warp as wp

from NR.src.poisson_jacobi import init_sin_sin_rhs, solve_poisson_dirichlet0_jacobi


class TestPoissonJacobi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 32
        cls.iters = 800
        cls.h = 1.0 / float(cls.n - 1)

        with wp.ScopedDevice(None):
            cls.f = wp.zeros((cls.n, cls.n), dtype=wp.float32)
            cls.u_exact = wp.zeros((cls.n, cls.n), dtype=wp.float32)
            wp.launch(init_sin_sin_rhs, dim=(cls.n, cls.n), inputs=(cls.f, cls.u_exact, cls.n, cls.h))

    def test_converges_to_known_solution(self):
        with wp.ScopedDevice(None):
            u = solve_poisson_dirichlet0_jacobi(self.f, iters=self.iters)

        u_np = u.numpy()
        u_exact_np = self.u_exact.numpy()
        rel_l2 = float(np.linalg.norm(u_np - u_exact_np) / np.linalg.norm(u_exact_np))
        self.assertLess(rel_l2, 0.03)

    def test_deterministic_two_runs(self):
        with wp.ScopedDevice(None):
            u1 = solve_poisson_dirichlet0_jacobi(self.f, iters=self.iters)
            u2 = solve_poisson_dirichlet0_jacobi(self.f, iters=self.iters)

        max_abs = float(np.max(np.abs(u1.numpy() - u2.numpy())))
        self.assertEqual(max_abs, 0.0)


if __name__ == "__main__":
    unittest.main()

