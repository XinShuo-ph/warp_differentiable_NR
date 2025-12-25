"""
Poisson equation solver from scratch using Warp FEM.

Solves: -Laplacian(u) = f
with Dirichlet boundary conditions u = g on boundary.

Test problem: u(x,y) = sin(pi*x) * sin(pi*y) on [0,1]^2
Then: f = 2*pi^2 * sin(pi*x) * sin(pi*y)
BC: u = 0 on boundary
"""

import warp as wp
import warp.fem as fem
import numpy as np

wp.init()


@fem.integrand
def poisson_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Bilinear form for Poisson equation: (grad u, grad v)"""
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand  
def rhs_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    """RHS forcing term: f = 2*pi^2 * sin(pi*x) * sin(pi*y)"""
    pos = domain(s)
    x = pos[0]
    y = pos[1]
    f = 2.0 * wp.pi * wp.pi * wp.sin(wp.pi * x) * wp.sin(wp.pi * y)
    return f * v(s)


@fem.integrand
def boundary_value_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    """Boundary condition: u = 0 on boundary"""
    return 0.0 * v(s)


@fem.integrand
def boundary_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Boundary condition projector"""
    return u(s) * v(s)


def solve_poisson_2d(resolution=32, degree=2):
    """
    Solve 2D Poisson equation on unit square.
    
    Args:
        resolution: Grid resolution
        degree: Polynomial degree for finite elements
    
    Returns:
        field: Solution field
        geo: Geometry
        space: Function space
    """
    
    print(f"Solving 2D Poisson equation")
    print(f"Resolution: {resolution}x{resolution}, degree: {degree}")
    print("=" * 60)
    
    # Create geometry: unit square [0,1]^2
    geo = fem.Grid2D(res=wp.vec2i(resolution))
    
    # Create function space
    space = fem.make_polynomial_space(geo, degree=degree)
    
    # Define domains
    domain = fem.Cells(geometry=geo)
    boundary = fem.BoundarySides(geo)
    
    # Create test and trial functions
    test = fem.make_test(space=space, domain=domain)
    trial = fem.make_trial(space=space, domain=domain)
    
    # Assemble stiffness matrix
    print("Assembling stiffness matrix...")
    matrix = fem.integrate(
        poisson_form,
        fields={"u": trial, "v": test}
    )
    print(f"Matrix: {matrix.nrow} x {matrix.ncol}, nnz: {matrix.nnz}")
    
    # Assemble RHS vector
    print("Assembling RHS vector...")
    rhs = fem.integrate(
        rhs_form,
        fields={"v": test}
    )
    print(f"RHS size: {rhs.shape[0]}")
    
    # Apply Dirichlet boundary conditions
    print("Applying boundary conditions...")
    bd_test = fem.make_test(space=space, domain=boundary)
    bd_trial = fem.make_trial(space=space, domain=boundary)
    
    bd_projector = fem.integrate(
        boundary_projector_form,
        fields={"u": bd_trial, "v": bd_test},
        assembly="nodal"
    )
    
    bd_value = fem.integrate(
        boundary_value_form,
        fields={"v": bd_test},
        assembly="nodal"
    )
    
    # Project linear system
    fem.project_linear_system(matrix, rhs, bd_projector, bd_value)
    
    # Solve linear system using Conjugate Gradient
    print("Solving linear system...")
    x = wp.zeros_like(rhs)
    
    # Use the same CG solver as in the examples
    import warp.examples.fem.utils as fem_example_utils
    
    residual, iters = fem_example_utils.bsr_cg(
        matrix,
        b=rhs,
        x=x,
        tol=1.0e-6,
        max_iters=1000,
        quiet=False
    )
    
    print(f"CG solve complete: {iters} iterations, residual: {residual:.2e}")
    
    # Create field with solution
    field = space.make_field()
    field.dof_values = x
    
    return field, geo, space


if __name__ == "__main__":
    # Solve Poisson equation
    field, geo, space = solve_poisson_2d(resolution=32, degree=2)
    
    print("\n" + "=" * 60)
    print("Poisson solver test complete")
    print("=" * 60)
    
    # Check solution norm
    dof_values = field.dof_values.numpy()
    print(f"Solution L2 norm: {np.linalg.norm(dof_values):.6f}")
    print(f"Solution max: {np.max(np.abs(dof_values)):.6f}")
    print(f"DOF count: {len(dof_values)}")
