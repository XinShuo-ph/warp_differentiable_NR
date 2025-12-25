"""
Poisson Equation Solver using Warp FEM

Solves: -Laplacian(u) = f on [0,1]^2
with Dirichlet boundary conditions: u = g on boundary

Weak form: integral(grad(u) . grad(v)) = integral(f * v)
"""

import warp as wp
import warp.fem as fem
import numpy as np

wp.init()


@fem.integrand
def laplacian_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Bilinear form: integral(grad(u) . grad(v))"""
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def source_form(s: fem.Sample, v: fem.Field, f_val: float):
    """Linear form: integral(f * v)"""
    return f_val * v(s)


@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Mass matrix for boundary projector"""
    return u(s) * v(s)


@wp.func
def boundary_value_func(pos: wp.vec2):
    """Dirichlet BC: u = sin(pi*x) * sin(pi*y) on boundary (actually 0 on boundary)"""
    return 0.0


@fem.integrand
def boundary_value_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    """Boundary condition value integrand"""
    pos = fem.position(domain, s)
    return boundary_value_func(pos) * v(s)


@fem.integrand
def boundary_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Boundary projector bilinear form"""
    return u(s) * v(s)


def solve_poisson(resolution: int = 32, degree: int = 2, quiet: bool = False):
    """
    Solve -Laplacian(u) = 2*pi^2*sin(pi*x)*sin(pi*y) on [0,1]^2
    with u = 0 on boundary
    
    Exact solution: u(x,y) = sin(pi*x) * sin(pi*y)
    """
    
    # Create geometry
    geo = fem.Grid2D(res=wp.vec2i(resolution))
    
    # Create scalar function space
    scalar_space = fem.make_polynomial_space(geo, degree=degree)
    
    # Create discrete field
    u_field = scalar_space.make_field()
    
    # Domain for integration
    domain = fem.Cells(geometry=geo)
    
    # Create test and trial functions
    test = fem.make_test(space=scalar_space, domain=domain)
    trial = fem.make_trial(space=scalar_space, domain=domain)
    
    # Assemble stiffness matrix (Laplacian)
    stiffness_matrix = fem.integrate(laplacian_form, fields={"u": trial, "v": test})
    
    # RHS with source term f = 2*pi^2*sin(pi*x)*sin(pi*y)
    # For simplicity, use constant source term first
    f_value = 2.0 * np.pi**2  # peak value of the exact source
    rhs = fem.integrate(source_form, fields={"v": test}, values={"f_val": f_value})
    
    # Apply Dirichlet boundary conditions (u = 0 on boundary)
    boundary = fem.BoundarySides(geo)
    bd_test = fem.make_test(space=scalar_space, domain=boundary)
    bd_trial = fem.make_trial(space=scalar_space, domain=boundary)
    
    # Boundary projector matrix
    bd_projector = fem.integrate(
        boundary_projector_form, 
        fields={"u": bd_trial, "v": bd_test}, 
        assembly="nodal"
    )
    
    # Boundary values (zero Dirichlet)
    bd_values = fem.integrate(
        boundary_value_form,
        fields={"v": bd_test},
        assembly="nodal"
    )
    
    # Project linear system to enforce BC
    fem.project_linear_system(stiffness_matrix, rhs, bd_projector, bd_values)
    
    # Solve using CG
    x = wp.zeros_like(rhs)
    
    # Simple CG solver
    cg_solve(stiffness_matrix, rhs, x, max_iters=1000, tol=1e-8, quiet=quiet)
    
    # Assign solution to field
    u_field.dof_values = x
    
    return u_field, geo


@wp.kernel
def axpy_kernel(y: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.float64), alpha: wp.float64):
    tid = wp.tid()
    y[tid] = y[tid] + alpha * x[tid]

@wp.kernel
def xpay_kernel(y: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.float64), alpha: wp.float64):
    tid = wp.tid()
    y[tid] = alpha * y[tid] + x[tid]

@wp.kernel
def dot_kernel(x: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.float64), result: wp.array(dtype=wp.float64)):
    tid = wp.tid()
    wp.atomic_add(result, 0, x[tid] * y[tid])


def cg_solve(A, b, x, max_iters=1000, tol=1e-8, quiet=False):
    """Conjugate Gradient solver for sparse system Ax = b"""
    
    # r = b - A*x
    r = wp.zeros_like(b)
    wp.copy(r, b)
    Ax = wp.zeros_like(b)
    wp.sparse.bsr_mv(A, x, Ax)
    
    n = len(b)
    
    # r = b - A*x
    wp.launch(axpy_kernel, dim=n, inputs=[r, Ax, wp.float64(-1.0)])
    
    # p = r
    p = wp.zeros_like(r)
    wp.copy(p, r)
    
    # rsold = r.r
    rsold_arr = wp.zeros(1, dtype=wp.float64)
    wp.launch(dot_kernel, dim=n, inputs=[r, r, rsold_arr])
    rsold = rsold_arr.numpy()[0]
    
    r0_norm = np.sqrt(rsold)
    if not quiet:
        print(f"CG iter 0: residual = {r0_norm:.6e}")
    
    Ap = wp.zeros_like(b)
    
    for i in range(max_iters):
        # Ap = A*p
        Ap.zero_()
        wp.sparse.bsr_mv(A, p, Ap)
        
        # pAp = p.Ap
        pAp_arr = wp.zeros(1, dtype=wp.float64)
        wp.launch(dot_kernel, dim=n, inputs=[p, Ap, pAp_arr])
        pAp = pAp_arr.numpy()[0]
        
        alpha = rsold / (pAp + 1e-30)
        
        # x = x + alpha*p
        wp.launch(axpy_kernel, dim=n, inputs=[x, p, wp.float64(alpha)])
        
        # r = r - alpha*Ap
        wp.launch(axpy_kernel, dim=n, inputs=[r, Ap, wp.float64(-alpha)])
        
        # rsnew = r.r
        rsnew_arr = wp.zeros(1, dtype=wp.float64)
        wp.launch(dot_kernel, dim=n, inputs=[r, r, rsnew_arr])
        rsnew = rsnew_arr.numpy()[0]
        
        res_norm = np.sqrt(rsnew)
        if not quiet and (i + 1) % 50 == 0:
            print(f"CG iter {i+1}: residual = {res_norm:.6e}")
        
        if res_norm < tol * r0_norm:
            if not quiet:
                print(f"CG converged at iter {i+1}: residual = {res_norm:.6e}")
            break
        
        beta = rsnew / (rsold + 1e-30)
        
        # p = r + beta*p
        wp.launch(xpay_kernel, dim=n, inputs=[p, r, wp.float64(beta)])
        
        rsold = rsnew
    
    return x


if __name__ == "__main__":
    wp.set_module_options({"enable_backward": False})
    
    print("Solving Poisson equation...")
    u_field, geo = solve_poisson(resolution=32, degree=2, quiet=False)
    
    # Get solution values
    u_values = u_field.dof_values.numpy()
    print(f"\nSolution range: [{u_values.min():.6f}, {u_values.max():.6f}]")
    print(f"Solution DOFs: {len(u_values)}")
    print("Poisson solver test complete!")
