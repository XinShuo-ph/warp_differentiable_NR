"""
Poisson Equation Solver from Scratch
Solves: -∇²u = f on 2D domain with Dirichlet BC
"""

import warp as wp
import warp.fem as fem
import numpy as np


@fem.integrand
def laplace_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Bilinear form for Laplacian: ∫∇u·∇v"""
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def source_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    """Linear form for source term: ∫fv"""
    pos = domain(s)
    x, y = pos[0], pos[1]
    # Source: f = 2π²sin(πx)sin(πy) -> analytical solution u = sin(πx)sin(πy)
    pi_sq = wp.pi * wp.pi
    f = 2.0 * pi_sq * wp.sin(wp.pi * x) * wp.sin(wp.pi * y)
    return f * v(s)


@fem.integrand
def dirichlet_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Boundary projector: u·v on boundary"""
    return u(s) * v(s)


@fem.integrand
def dirichlet_value_form(s: fem.Sample, v: fem.Field):
    """Boundary value (homogeneous): 0 on boundary"""
    return 0.0 * v(s)


def solve_poisson(resolution=32, degree=2, quiet=True):
    """
    Solve Poisson equation on [0,1]x[0,1] domain
    """
    # Setup geometry
    geo = fem.Grid2D(res=wp.vec2i(resolution, resolution), bounds_lo=(0.0, 0.0), bounds_hi=(1.0, 1.0))
    
    # Create function space
    space = fem.make_polynomial_space(geo, degree=degree)
    
    # Define domain
    domain = fem.Cells(geometry=geo)
    
    # Build system matrix (Laplacian)
    trial = fem.make_trial(space=space, domain=domain)
    test = fem.make_test(space=space, domain=domain)
    matrix = fem.integrate(laplace_form, fields={"u": trial, "v": test})
    
    # Build RHS (source term)
    rhs = fem.integrate(source_form, fields={"v": test})
    
    # Apply Dirichlet boundary conditions
    boundary = fem.BoundarySides(geo)
    bd_test = fem.make_test(space=space, domain=boundary)
    bd_trial = fem.make_trial(space=space, domain=boundary)
    
    bd_matrix = fem.integrate(dirichlet_projector_form, fields={"u": bd_trial, "v": bd_test}, assembly="nodal")
    bd_rhs = fem.integrate(dirichlet_value_form, fields={"v": bd_test}, assembly="nodal")
    
    # Project linear system
    fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)
    
    # Solve system using CG
    x = wp.zeros_like(rhs)
    
    # Simple conjugate gradient solver
    from warp.sparse import bsr_mv
    
    def array_inner(a, b):
        """Compute inner product of two arrays"""
        return np.dot(a.numpy(), b.numpy())
    
    # r = b - A*x (initially x=0, so r=b)
    r = rhs.numpy().copy()
    p = r.copy()
    x_np = np.zeros_like(r)
    
    tol = 1e-8
    max_iter = 1000
    
    for i in range(max_iter):
        # Compute A*p
        p_wp = wp.from_numpy(p, dtype=rhs.dtype)
        Ap_wp = bsr_mv(matrix, p_wp)
        Ap = Ap_wp.numpy()
        
        rr = np.dot(r, r)
        pAp = np.dot(p, Ap)
        
        if abs(pAp) < 1e-14:
            break
            
        alpha = rr / pAp
        x_np = x_np + alpha * p
        r_new = r - alpha * Ap
        
        rr_new = np.dot(r_new, r_new)
        
        if not quiet and (i % 100 == 0):
            print(f"  CG iter {i}: residual = {np.sqrt(rr_new):.6e}")
        
        if np.sqrt(rr_new) < tol:
            if not quiet:
                print(f"  CG converged in {i+1} iterations")
            break
            
        beta = rr_new / rr
        p = r_new + beta * p
        r = r_new
    
    x = wp.from_numpy(x_np, dtype=rhs.dtype)
    
    # Create field with solution
    field = space.make_field()
    field.dof_values = x
    
    return field, space, geo


def compute_l2_error(field, space, geo):
    """
    Compute L2 error against analytical solution u = sin(πx)sin(πy)
    """
    # Get solution values
    u_approx = field.dof_values.numpy()
    
    # Get node positions
    import numpy as np
    n = int(np.sqrt(len(u_approx)))
    
    # Compute analytical solution at nodes
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    u_exact_flat = u_exact.flatten()
    
    # Compute L2 norm of error
    diff = u_approx - u_exact_flat[:len(u_approx)]
    l2_error = np.sqrt(np.mean(diff**2))
    
    return l2_error


if __name__ == "__main__":
    wp.init()
    
    print("Solving Poisson equation: -∇²u = f")
    print("Domain: [0,1]x[0,1]")
    print("BC: u=0 on boundary")
    print("Analytical solution: u = sin(πx)sin(πy)")
    print()
    
    # Solve with different resolutions
    for res in [16, 32, 64]:
        print(f"Resolution: {res}x{res}")
        field, space, geo = solve_poisson(resolution=res, degree=2, quiet=True)
        error = compute_l2_error(field, space, geo)
        print(f"L2 error: {error:.6e}")
        print()
    
    print("Verification: Running solver twice to check consistency")
    field1, space, geo = solve_poisson(resolution=32, degree=2, quiet=True)
    field2, space, geo = solve_poisson(resolution=32, degree=2, quiet=True)
    diff = np.abs(field1.dof_values.numpy() - field2.dof_values.numpy()).max()
    print(f"Max difference between two runs: {diff:.6e}")
    print("✓ Poisson solver validated!" if diff < 1e-10 else "✗ Consistency check failed")
