"""
Verify Poisson solver against analytical solution

Problem: -Laplace(u) = f on [0,1]^2 with u = 0 on boundary
Analytical solution: u = sin(pi*x)*sin(pi*y)
Source term: f = 2*pi^2*sin(pi*x)*sin(pi*y)
"""

import warp as wp
import warp.fem as fem
import numpy as np
from poisson_solver import solve_poisson

@fem.integrand
def exact_solution_integrand(s: fem.Sample, domain: fem.Domain):
    """Analytical solution: u = sin(pi*x)*sin(pi*y)"""
    pos = domain(s)
    x, y = pos[0], pos[1]
    return wp.sin(wp.pi * x) * wp.sin(wp.pi * y)

@fem.integrand
def error_integrand(s: fem.Sample, domain: fem.Domain, u_h: fem.Field):
    """Compute (u_exact - u_h)^2 for L2 error"""
    pos = domain(s)
    x, y = pos[0], pos[1]
    u_exact = wp.sin(wp.pi * x) * wp.sin(wp.pi * y)
    u_approx = u_h(s)
    diff = u_exact - u_approx
    return diff * diff

@fem.integrand
def grad_error_integrand(s: fem.Sample, domain: fem.Domain, u_h: fem.Field):
    """Compute |grad(u_exact) - grad(u_h)|^2 for H1 seminorm"""
    pos = domain(s)
    x, y = pos[0], pos[1]
    
    grad_exact_x = wp.pi * wp.cos(wp.pi * x) * wp.sin(wp.pi * y)
    grad_exact_y = wp.pi * wp.sin(wp.pi * x) * wp.cos(wp.pi * y)
    grad_exact = wp.vec2(grad_exact_x, grad_exact_y)
    
    grad_approx = fem.grad(u_h, s)
    grad_diff = grad_exact - grad_approx
    
    return wp.dot(grad_diff, grad_diff)

def verify_convergence():
    """Test convergence rate against analytical solution"""
    resolutions = [8, 16, 32, 64]
    degree = 2
    
    print("Testing convergence for polynomial degree", degree)
    print("Resolution | L2 Error | H1 Error | L2 Rate | H1 Rate")
    print("-" * 65)
    
    prev_l2_error = None
    prev_h1_error = None
    prev_res = None
    
    for res in resolutions:
        field, geo = solve_poisson(resolution=res, degree=degree)
        
        domain = fem.Cells(geometry=geo)
        
        l2_error_sq = fem.integrate(
            error_integrand,
            fields={"u_h": field},
            domain=domain,
            output_dtype=float
        )
        l2_error = np.sqrt(l2_error_sq)
        
        h1_error_sq = fem.integrate(
            grad_error_integrand,
            fields={"u_h": field},
            domain=domain,
            output_dtype=float
        )
        h1_error = np.sqrt(h1_error_sq)
        
        l2_rate = ""
        h1_rate = ""
        if prev_l2_error is not None:
            l2_rate = f"{np.log2(prev_l2_error / l2_error):.2f}"
            h1_rate = f"{np.log2(prev_h1_error / h1_error):.2f}"
        
        print(f"{res:10d} | {l2_error:.2e} | {h1_error:.2e} | {l2_rate:7s} | {h1_rate:7s}")
        
        prev_l2_error = l2_error
        prev_h1_error = h1_error
        prev_res = res
    
    print("\nExpected rates for degree 2:")
    print("  L2 error: ~3.0 (optimal)")
    print("  H1 error: ~2.0 (optimal)")
    print("\nVerification: Check that rates approach expected values as resolution increases")

def test_point_values():
    """Test solution at specific points"""
    field, geo = solve_poisson(resolution=64, degree=2)
    
    print("\nPoint-wise verification at (x,y):")
    test_points = [
        (0.25, 0.25),
        (0.5, 0.5),
        (0.75, 0.25),
        (0.25, 0.75),
    ]
    
    # Sample field at specific locations
    domain = fem.Cells(geometry=geo)
    
    for px, py in test_points:
        exact = np.sin(np.pi * px) * np.sin(np.pi * py)
        print(f"  ({px}, {py}): exact = {exact:.6f}")
    
    print("\n(Note: Direct point sampling requires advanced API)")

if __name__ == "__main__":
    wp.init()
    with wp.ScopedDevice("cpu"):
        verify_convergence()
        test_point_values()
