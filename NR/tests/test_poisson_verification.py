"""
Verify Poisson solver against analytical solution.

Test problem:
    -Laplacian(u) = f  on [0,1]^2
    u = 0 on boundary

Analytical solution: u(x,y) = sin(pi*x) * sin(pi*y)
Forcing term: f = 2*pi^2 * sin(pi*x) * sin(pi*y)
"""

import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils
import numpy as np

wp.init()


@fem.integrand
def poisson_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Bilinear form: (grad u, grad v)"""
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand  
def rhs_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    """RHS forcing: f = 2*pi^2 * sin(pi*x) * sin(pi*y)"""
    pos = domain(s)
    x = pos[0]
    y = pos[1]
    f = 2.0 * wp.pi * wp.pi * wp.sin(wp.pi * x) * wp.sin(wp.pi * y)
    return f * v(s)


@fem.integrand
def boundary_value_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    """BC: u = 0"""
    return 0.0 * v(s)


@fem.integrand
def boundary_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """BC projector"""
    return u(s) * v(s)


@fem.integrand
def analytical_solution(s: fem.Sample, domain: fem.Domain):
    """Analytical solution: sin(pi*x) * sin(pi*y)"""
    pos = domain(s)
    x = pos[0]
    y = pos[1]
    return wp.sin(wp.pi * x) * wp.sin(wp.pi * y)


@fem.integrand
def l2_error_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field):
    """L2 error integrand: (u_h - u_exact)^2 * v"""
    u_h = u(s)
    u_exact = analytical_solution(s, domain)
    diff = u_h - u_exact
    return diff * diff * v(s)


@fem.integrand
def l2_norm_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    """L2 norm of exact solution: u_exact^2 * v"""
    u_exact = analytical_solution(s, domain)
    return u_exact * u_exact * v(s)


def compute_convergence_rate(resolutions, errors):
    """Compute convergence rate from error data"""
    if len(resolutions) < 2:
        return None
    
    log_h = np.log([1.0/r for r in resolutions])
    log_e = np.log(errors)
    
    # Linear fit: log(e) = rate * log(h) + c
    rate = np.polyfit(log_h, log_e, 1)[0]
    return rate


def verify_poisson_solver():
    """Verify solver against analytical solution with convergence study"""
    
    print("=" * 70)
    print("Poisson Solver Verification")
    print("=" * 70)
    print("\nTest problem: -Laplacian(u) = 2*pi^2*sin(pi*x)*sin(pi*y)")
    print("Analytical solution: u(x,y) = sin(pi*x)*sin(pi*y)")
    print("Boundary condition: u = 0 on all boundaries")
    print()
    
    # Test multiple resolutions and polynomial degrees
    test_configs = [
        {"resolutions": [8, 16, 32], "degree": 1},
        {"resolutions": [8, 16, 32, 64], "degree": 2},
    ]
    
    for config in test_configs:
        degree = config["degree"]
        resolutions = config["resolutions"]
        
        print(f"\n{'='*70}")
        print(f"Polynomial degree: {degree}")
        print(f"{'='*70}")
        print(f"{'Resolution':<12} {'DOFs':<10} {'L2 Error':<15} {'L2 Rel Error':<15}")
        print("-" * 70)
        
        errors = []
        
        for resolution in resolutions:
            # Create geometry
            geo = fem.Grid2D(res=wp.vec2i(resolution))
            space = fem.make_polynomial_space(geo, degree=degree)
            
            # Define domains
            domain = fem.Cells(geometry=geo)
            boundary = fem.BoundarySides(geo)
            
            # Assemble system
            test = fem.make_test(space=space, domain=domain)
            trial = fem.make_trial(space=space, domain=domain)
            
            matrix = fem.integrate(poisson_form, fields={"u": trial, "v": test})
            rhs = fem.integrate(rhs_form, fields={"v": test})
            
            # Apply BCs
            bd_test = fem.make_test(space=space, domain=boundary)
            bd_trial = fem.make_trial(space=space, domain=boundary)
            bd_projector = fem.integrate(boundary_projector_form, 
                                         fields={"u": bd_trial, "v": bd_test},
                                         assembly="nodal")
            bd_value = fem.integrate(boundary_value_form, 
                                    fields={"v": bd_test},
                                    assembly="nodal")
            
            fem.project_linear_system(matrix, rhs, bd_projector, bd_value)
            
            # Solve
            x = wp.zeros_like(rhs)
            fem_example_utils.bsr_cg(matrix, b=rhs, x=x, tol=1.0e-10, quiet=True)
            
            # Create solution field with matching dtype
            u_field = space.make_field()
            # Convert to float32 if needed
            if x.dtype == wp.float64:
                x_f32 = wp.empty(shape=x.shape, dtype=wp.float32)
                wp.utils.array_cast(in_array=x, out_array=x_f32)
                u_field.dof_values = x_f32
            else:
                u_field.dof_values = x
            
            # Compute L2 error
            test_scalar = fem.make_test(space=space, domain=domain)
            l2_error_sq = fem.integrate(l2_error_form, fields={"u": u_field, "v": test_scalar})
            l2_error_sq_val = l2_error_sq.numpy().sum()
            l2_error = np.sqrt(l2_error_sq_val)
            
            # Compute L2 norm of exact solution
            l2_norm_sq = fem.integrate(l2_norm_form, fields={"v": test_scalar})
            l2_norm_sq_val = l2_norm_sq.numpy().sum()
            l2_norm = np.sqrt(l2_norm_sq_val)
            
            rel_error = l2_error / l2_norm if l2_norm > 0 else l2_error
            errors.append(rel_error)
            
            dof_count = len(x)
            print(f"{resolution:<12} {dof_count:<10} {l2_error:<15.6e} {rel_error:<15.6e}")
        
        # Compute convergence rate
        if len(errors) >= 2:
            rate = compute_convergence_rate(resolutions, errors)
            expected_rate = degree + 1
            print(f"\nConvergence rate: {rate:.2f} (expected: {expected_rate} for degree {degree})")
            
            # Check if convergence rate is close to expected
            if abs(rate - expected_rate) < 0.5:
                print(f"✓ Convergence rate matches theory!")
            else:
                print(f"✗ Convergence rate differs from theory")
    
    print("\n" + "=" * 70)
    print("Verification complete")
    print("=" * 70)


if __name__ == "__main__":
    verify_poisson_solver()
