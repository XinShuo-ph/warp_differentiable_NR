import sys
from pathlib import Path

import pytest


NR_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(NR_SRC))


@pytest.mark.parametrize(
    ("resolution", "degree", "max_l2_error"),
    [
        (8, 2, 2.0e-3),
        (16, 2, 2.0e-4),
    ],
)
def test_poisson_sin_dirichlet_l2_error(resolution: int, degree: int, max_l2_error: float):
    from poisson import solve_poisson_sin_dirichlet

    result = solve_poisson_sin_dirichlet(resolution=resolution, degree=degree)
    assert result.l2_error < max_l2_error

