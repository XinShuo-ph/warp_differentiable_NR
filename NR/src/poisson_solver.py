"""
Poisson equation solver from scratch using Warp FEM

Solves: -Laplace(u) = f on [0,1]^2
with u = 0 on boundary
"""

import warp as wp
import warp.fem as fem
import numpy as np

@fem.integrand
def laplacian_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Bilinear form: integral of grad(u) . grad(v)"""
    return wp.dot(fem.grad(u, s), fem.grad(v, s))

@fem.integrand
def source_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    """Linear form: integral of f*v, where f = 2*pi^2*sin(pi*x)*sin(pi*y)"""
    pos = domain(s)
    x, y = pos[0], pos[1]
    pi2 = wp.pi * wp.pi
    f = 2.0 * pi2 * wp.sin(wp.pi * x) * wp.sin(wp.pi * y)
    return f * v(s)

@fem.integrand
def zero_bc_projector(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Projector for homogeneous Dirichlet BC"""
    return u(s) * v(s)

def solve_poisson(resolution=32, degree=2):
    """
    Solve Poisson equation with manufactured solution
    
    Args:
        resolution: grid resolution
        degree: polynomial degree
    
    Returns:
        field: solution field
        geo: geometry
    """
    geo = fem.Grid2D(res=wp.vec2i(resolution))
    
    scalar_space = fem.make_polynomial_space(geo, degree=degree, dtype=wp.float32)
    
    domain = fem.Cells(geometry=geo)
    test = fem.make_test(space=scalar_space, domain=domain)
    trial = fem.make_trial(space=scalar_space, domain=domain)
    
    matrix = fem.integrate(laplacian_form, fields={"u": trial, "v": test})
    rhs = fem.integrate(source_form, fields={"v": test})
    
    boundary = fem.BoundarySides(geo)
    bd_test = fem.make_test(space=scalar_space, domain=boundary)
    bd_trial = fem.make_trial(space=scalar_space, domain=boundary)
    
    bd_projector = fem.integrate(
        zero_bc_projector,
        fields={"u": bd_trial, "v": bd_test},
        assembly="nodal"
    )
    bd_rhs = wp.zeros_like(rhs)
    
    fem.project_linear_system(matrix, rhs, bd_projector, bd_rhs)
    
    x = wp.zeros_like(rhs)
    
    from warp.optim.linear import bicgstab
    bicgstab(matrix, b=rhs, x=x, tol=1.0e-8)
    
    field = scalar_space.make_field()
    wp.utils.array_cast(in_array=x, out_array=field.dof_values)
    
    return field, geo

def test_twice():
    """Run solver twice to validate consistency"""
    print("Run 1:")
    field1, geo1 = solve_poisson(resolution=32, degree=2)
    sol1 = field1.dof_values.numpy()
    print(f"Solution norm: {np.linalg.norm(sol1)}")
    print(f"Solution max: {np.max(np.abs(sol1))}")
    
    print("\nRun 2:")
    field2, geo2 = solve_poisson(resolution=32, degree=2)
    sol2 = field2.dof_values.numpy()
    print(f"Solution norm: {np.linalg.norm(sol2)}")
    print(f"Solution max: {np.max(np.abs(sol2))}")
    
    diff = np.linalg.norm(sol1 - sol2)
    print(f"\nDifference between runs: {diff}")
    print("Consistent!" if diff < 1e-10 else "INCONSISTENT!")
    
    return field1, geo1

if __name__ == "__main__":
    wp.init()
    with wp.ScopedDevice("cpu"):
        field, geo = test_twice()
