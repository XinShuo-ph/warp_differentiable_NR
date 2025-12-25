"""
Poisson Equation Solver using NVIDIA Warp FEM

Solves: -∇²u = f on [0,1]×[0,1]
With homogeneous Dirichlet BC: u = 0 on boundary

Uses analytical solution u = sin(πx)sin(πy) for verification.
This gives f = 2π²sin(πx)sin(πy)
"""

import math

import warp as wp
import warp.fem as fem

wp.init()


@fem.integrand
def laplacian_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Bilinear form: ∫ ∇u · ∇v dx (weak form of -∇²u)"""
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def source_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    """Linear form: ∫ f v dx where f = 2π²sin(πx)sin(πy)"""
    pos = domain(s)
    pi = 3.141592653589793
    f = 2.0 * pi * pi * wp.sin(pi * pos[0]) * wp.sin(pi * pos[1])
    return f * v(s)


@fem.integrand
def boundary_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Bilinear form for Dirichlet BC projector"""
    return u(s) * v(s)


@fem.integrand
def analytical_solution(s: fem.Sample, domain: fem.Domain):
    """Analytical solution: u = sin(πx)sin(πy)"""
    pos = domain(s)
    pi = 3.141592653589793
    return wp.sin(pi * pos[0]) * wp.sin(pi * pos[1])


@fem.integrand
def l2_error_integrand(s: fem.Sample, domain: fem.Domain, u: fem.Field):
    """L2 error integrand: (u - u_exact)^2"""
    pos = domain(s)
    pi = 3.141592653589793
    u_exact = wp.sin(pi * pos[0]) * wp.sin(pi * pos[1])
    diff = u(s) - u_exact
    return diff * diff


@fem.integrand  
def l2_norm_integrand(s: fem.Sample, u: fem.Field):
    """L2 norm integrand: u^2"""
    val = u(s)
    return val * val


def solve_poisson(resolution: int = 32, degree: int = 2, quiet: bool = False):
    """
    Solve Poisson equation on unit square.
    
    Args:
        resolution: Grid resolution
        degree: Polynomial degree for finite elements
        quiet: Suppress solver output
        
    Returns:
        Tuple of (solution_field, l2_error, linf_error)
    """
    # Create 2D grid geometry
    geo = fem.Grid2D(res=wp.vec2i(resolution))
    
    # Create scalar function space
    scalar_space = fem.make_polynomial_space(geo, degree=degree)
    
    # Create solution field
    u_field = scalar_space.make_field()
    
    # Define domain (interior cells)
    domain = fem.Cells(geometry=geo)
    
    # Create test and trial functions
    test = fem.make_test(space=scalar_space, domain=domain)
    trial = fem.make_trial(space=scalar_space, domain=domain)
    
    # Assemble stiffness matrix (Laplacian bilinear form)
    stiffness_matrix = fem.integrate(laplacian_form, fields={"u": trial, "v": test}, output_dtype=float)
    
    # Assemble RHS (source term)
    rhs = fem.integrate(source_form, fields={"v": test}, output_dtype=float)
    
    # Apply Dirichlet boundary conditions (u = 0 on boundary)
    boundary = fem.BoundarySides(geo)
    bd_test = fem.make_test(space=scalar_space, domain=boundary)
    bd_trial = fem.make_trial(space=scalar_space, domain=boundary)
    
    # Boundary projector
    bd_projector = fem.integrate(
        boundary_projector_form, 
        fields={"u": bd_trial, "v": bd_test}, 
        assembly="nodal",
        output_dtype=float
    )
    
    # Zero boundary value (must match matrix scalar type)
    bd_value = wp.zeros(bd_projector.nrow, dtype=bd_projector.scalar_type)
    
    # Project linear system to enforce Dirichlet BC
    fem.project_linear_system(stiffness_matrix, rhs, bd_projector, bd_value)
    
    # Solve using Conjugate Gradient
    x = wp.zeros_like(rhs)
    
    from warp.optim.linear import cg
    
    if quiet:
        cg(stiffness_matrix, b=rhs, x=x, tol=1e-8)
    else:
        def print_residual(iteration, err, tol):
            if iteration % 10 == 0:
                print(f"  CG iteration {iteration}: residual = {err:.6e}")
        cg(stiffness_matrix, b=rhs, x=x, tol=1e-8, callback=print_residual)
    
    # Assign solution to field (cast to field's dtype if necessary)
    if x.dtype != u_field.dof_values.dtype:
        x_cast = wp.empty_like(u_field.dof_values)
        wp.utils.array_cast(in_array=x, out_array=x_cast)
        u_field.dof_values = x_cast
    else:
        u_field.dof_values = x
    
    # Compute L2 error against analytical solution using proper integration
    l2_error_squared = fem.integrate(
        l2_error_integrand,
        domain=domain,
        fields={"u": u_field},
        output_dtype=float
    )
    # integrate returns either a scalar or array depending on form type
    if hasattr(l2_error_squared, 'numpy'):
        l2_error = math.sqrt(float(l2_error_squared.numpy()))
    else:
        l2_error = math.sqrt(float(l2_error_squared))
    
    # Also compute using DOF values for comparison (Linf at DOFs)
    u_analytical = scalar_space.make_field()
    fem.interpolate(analytical_solution, dest=u_analytical)
    
    u_numerical = u_field.dof_values.numpy()
    u_exact = u_analytical.dof_values.numpy()
    diff = u_numerical - u_exact
    linf_error = abs(diff).max()
    
    return u_field, l2_error, linf_error


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Poisson Equation Solver")
    parser.add_argument("--resolution", type=int, default=32, help="Grid resolution")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree")
    parser.add_argument("--quiet", action="store_true", help="Suppress solver output")
    args = parser.parse_args()
    
    print(f"Solving Poisson equation on {args.resolution}x{args.resolution} grid, degree {args.degree}")
    
    u_field, l2_error, linf_error = solve_poisson(
        resolution=args.resolution,
        degree=args.degree,
        quiet=args.quiet
    )
    
    print(f"L2 error: {l2_error:.6e}")
    print(f"Linf error: {linf_error:.6e}")


if __name__ == "__main__":
    main()
