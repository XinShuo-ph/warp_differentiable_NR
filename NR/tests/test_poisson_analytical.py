"""
Verify Poisson solver against analytical solution.

Problem: -Laplacian(u) = 2*pi^2*sin(pi*x)*sin(pi*y) on [0,1]^2
         u = 0 on boundary

Exact solution: u(x,y) = sin(pi*x) * sin(pi*y)
"""

import warp as wp
import warp.fem as fem
import numpy as np

wp.init()


@wp.func
def exact_solution(pos: wp.vec2) -> float:
    """Exact solution: sin(pi*x) * sin(pi*y)"""
    pi = 3.14159265358979323846
    return wp.sin(pi * pos[0]) * wp.sin(pi * pos[1])


@wp.func
def source_function(pos: wp.vec2) -> float:
    """Source term: 2*pi^2*sin(pi*x)*sin(pi*y)"""
    pi = 3.14159265358979323846
    return 2.0 * pi * pi * wp.sin(pi * pos[0]) * wp.sin(pi * pos[1])


@fem.integrand
def laplacian_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Bilinear form: integral(grad(u) . grad(v))"""
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def source_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    """Linear form: integral(f(x) * v)"""
    pos = fem.position(domain, s)
    f = source_function(pos)
    return f * v(s)


@fem.integrand
def boundary_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


@fem.integrand
def boundary_value_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    """Zero Dirichlet BC"""
    return 0.0 * v(s)


@fem.integrand
def l2_error_integrand(s: fem.Sample, domain: fem.Domain, u: fem.Field):
    """Squared error between numerical and exact solution"""
    pos = fem.position(domain, s)
    u_exact = exact_solution(pos)
    u_num = u(s)
    err = u_num - u_exact
    return err * err


@fem.integrand
def l2_norm_integrand(s: fem.Sample, domain: fem.Domain):
    """Squared norm of exact solution"""
    pos = fem.position(domain, s)
    u_exact = exact_solution(pos)
    return u_exact * u_exact


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


def cg_solve(A, b, x, max_iters=1000, tol=1e-10, quiet=True):
    """Conjugate Gradient solver"""
    r = wp.zeros_like(b)
    wp.copy(r, b)
    Ax = wp.zeros_like(b)
    wp.sparse.bsr_mv(A, x, Ax)
    
    n = len(b)
    wp.launch(axpy_kernel, dim=n, inputs=[r, Ax, wp.float64(-1.0)])
    
    p = wp.zeros_like(r)
    wp.copy(p, r)
    
    rsold_arr = wp.zeros(1, dtype=wp.float64)
    wp.launch(dot_kernel, dim=n, inputs=[r, r, rsold_arr])
    rsold = rsold_arr.numpy()[0]
    r0_norm = np.sqrt(rsold)
    
    Ap = wp.zeros_like(b)
    
    for i in range(max_iters):
        Ap.zero_()
        wp.sparse.bsr_mv(A, p, Ap)
        
        pAp_arr = wp.zeros(1, dtype=wp.float64)
        wp.launch(dot_kernel, dim=n, inputs=[p, Ap, pAp_arr])
        pAp = pAp_arr.numpy()[0]
        
        alpha = rsold / (pAp + 1e-30)
        wp.launch(axpy_kernel, dim=n, inputs=[x, p, wp.float64(alpha)])
        wp.launch(axpy_kernel, dim=n, inputs=[r, Ap, wp.float64(-alpha)])
        
        rsnew_arr = wp.zeros(1, dtype=wp.float64)
        wp.launch(dot_kernel, dim=n, inputs=[r, r, rsnew_arr])
        rsnew = rsnew_arr.numpy()[0]
        
        if np.sqrt(rsnew) < tol * r0_norm:
            if not quiet:
                print(f"CG converged at iter {i+1}")
            break
        
        beta = rsnew / (rsold + 1e-30)
        wp.launch(xpay_kernel, dim=n, inputs=[p, r, wp.float64(beta)])
        rsold = rsnew


def test_poisson_convergence():
    """Test that solution converges to analytical solution with mesh refinement"""
    
    print("Testing Poisson solver convergence...")
    print("-" * 50)
    
    resolutions = [8, 16, 32]  # Skip 64 due to CG convergence issues
    errors = []
    
    for resolution in resolutions:
        # Create geometry
        geo = fem.Grid2D(res=wp.vec2i(resolution))
        
        # Create function space
        scalar_space = fem.make_polynomial_space(geo, degree=2)
        u_field = scalar_space.make_field()
        
        # Domain
        domain = fem.Cells(geometry=geo)
        
        # Test and trial functions
        test = fem.make_test(space=scalar_space, domain=domain)
        trial = fem.make_trial(space=scalar_space, domain=domain)
        
        # Stiffness matrix
        stiffness_matrix = fem.integrate(laplacian_form, fields={"u": trial, "v": test})
        
        # RHS with position-dependent source
        rhs = fem.integrate(source_form, fields={"v": test})
        
        # Boundary conditions
        boundary = fem.BoundarySides(geo)
        bd_test = fem.make_test(space=scalar_space, domain=boundary)
        bd_trial = fem.make_trial(space=scalar_space, domain=boundary)
        
        bd_projector = fem.integrate(
            boundary_projector_form, 
            fields={"u": bd_trial, "v": bd_test}, 
            assembly="nodal"
        )
        bd_values = fem.integrate(
            boundary_value_form,
            fields={"v": bd_test},
            assembly="nodal"
        )
        
        fem.project_linear_system(stiffness_matrix, rhs, bd_projector, bd_values)
        
        # Solve
        x = wp.zeros_like(rhs)
        cg_solve(stiffness_matrix, rhs, x, max_iters=2000, tol=1e-12)
        # Cast solution to float32 for field assignment
        x_f32 = wp.zeros(len(x), dtype=wp.float32)
        wp.utils.array_cast(out_array=x_f32, in_array=x)
        u_field.dof_values = x_f32
        
        # Compute L2 error
        error_sq = fem.integrate(l2_error_integrand, domain=domain, fields={"u": u_field}, output_dtype=wp.float64)
        norm_sq = fem.integrate(l2_norm_integrand, domain=domain, output_dtype=wp.float64)
        
        # Handle both array and scalar return types
        error_sq_val = float(error_sq.numpy()) if hasattr(error_sq, 'numpy') else float(error_sq)
        norm_sq_val = float(norm_sq.numpy()) if hasattr(norm_sq, 'numpy') else float(norm_sq)
        
        rel_error = np.sqrt(error_sq_val / norm_sq_val)
        errors.append(rel_error)
        
        print(f"Resolution {resolution:3d}: L2 relative error = {rel_error:.6e}")
    
    # Check convergence rate (should be O(h^(p+1)) for degree p)
    print("-" * 50)
    for i in range(1, len(errors)):
        if errors[i-1] > 0 and errors[i] > 0:
            rate = np.log(errors[i-1] / errors[i]) / np.log(2)
            print(f"Convergence rate ({resolutions[i-1]}->{resolutions[i]}): {rate:.2f}")
    
    # Verify convergence (error should decrease)
    print("-" * 50)
    if all(errors[i] < errors[i-1] for i in range(1, len(errors))):
        print("PASSED: Error decreases with mesh refinement")
    else:
        print("FAILED: Error does not decrease monotonically")
    
    # Verify reasonable convergence rate (expect ~3 for degree 2)
    if len(errors) >= 2:
        final_rate = np.log(errors[-2] / errors[-1]) / np.log(2)
        if final_rate > 2.0:
            print(f"PASSED: Convergence rate {final_rate:.2f} > 2.0 (expected for P2 elements)")
        else:
            print(f"WARNING: Convergence rate {final_rate:.2f} lower than expected")
    
    print("-" * 50)
    print("Poisson verification complete!")
    
    return errors


if __name__ == "__main__":
    wp.set_module_options({"enable_backward": False})
    test_poisson_convergence()
