"""
2D Poisson Equation Solver using Warp FEM

Solves: -∇²u = f on Ω = [0,1]²
with Dirichlet BC: u = 0 on ∂Ω

Uses manufactured solution: u(x,y) = sin(πx)sin(πy)
which gives: f(x,y) = 2π²sin(πx)sin(πy)
"""

import numpy as np
import warp as wp
import warp.fem as fem
import sys
sys.path.insert(0, '/workspace/warp_repo/warp/examples/fem')
import utils as fem_example_utils

PI = 3.141592653589793


@wp.func
def exact_solution(x: wp.vec2):
    """Exact solution: u(x,y) = sin(πx)sin(πy)"""
    return wp.sin(PI * x[0]) * wp.sin(PI * x[1])


@wp.func
def forcing_function(x: wp.vec2):
    """RHS: f(x,y) = 2π²sin(πx)sin(πy)"""
    return 2.0 * PI * PI * wp.sin(PI * x[0]) * wp.sin(PI * x[1])


@fem.integrand
def laplacian_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Weak form of Laplacian: ∫ ∇u · ∇v dΩ"""
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def forcing_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    """RHS linear form: ∫ f·v dΩ"""
    x = domain(s)
    f = forcing_function(x)
    return f * v(s)


@fem.integrand
def boundary_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Projector for boundary conditions"""
    return u(s) * v(s)


@fem.integrand
def error_integrand(s: fem.Sample, domain: fem.Domain, u: fem.Field):
    """L2 error squared: (u - u_exact)²"""
    x = domain(s)
    u_exact = exact_solution(x)
    diff = u(s) - u_exact
    return diff * diff


class PoissonSolver:
    def __init__(self, resolution=20, degree=2):
        self.resolution = resolution
        self.degree = degree
        
        # Create 2D grid geometry
        self.geo = fem.Grid2D(res=wp.vec2i(resolution))
        
        # Scalar function space
        self.scalar_space = fem.make_polynomial_space(
            self.geo, degree=degree, dtype=wp.float32
        )
        
        # Solution field
        self.u_field = self.scalar_space.make_field()
        
    def solve(self, quiet=False):
        geo = self.geo
        scalar_space = self.scalar_space
        
        # Interior domain and boundary
        domain = fem.Cells(geometry=geo)
        boundary = fem.BoundarySides(geo)
        
        # Test and trial functions
        test = fem.make_test(space=scalar_space, domain=domain)
        trial = fem.make_trial(space=scalar_space, domain=domain)
        
        # Assemble stiffness matrix: ∫ ∇u · ∇v dΩ
        stiffness = fem.integrate(
            laplacian_form,
            fields={"u": trial, "v": test},
            output_dtype=wp.float32
        )
        
        # Assemble RHS: ∫ f·v dΩ
        rhs = fem.integrate(
            forcing_form,
            fields={"v": test},
            output_dtype=wp.float32
        )
        
        # Boundary condition projector (homogeneous Dirichlet u=0)
        bd_test = fem.make_test(space=scalar_space, domain=boundary)
        bd_trial = fem.make_trial(space=scalar_space, domain=boundary)
        bd_projector = fem.integrate(
            boundary_projector_form,
            fields={"u": bd_trial, "v": bd_test},
            assembly="nodal",
            output_dtype=wp.float32
        )
        
        # Zero RHS for homogeneous BC
        bd_rhs = wp.zeros(scalar_space.node_count(), dtype=wp.float32)
        
        # Apply boundary conditions
        fem.project_linear_system(stiffness, rhs, bd_projector, bd_rhs)
        
        # Solve with CG
        x = wp.zeros(scalar_space.node_count(), dtype=wp.float32)
        fem_example_utils.bsr_cg(stiffness, b=rhs, x=x, quiet=quiet, tol=1e-8)
        
        # Store solution
        self.u_field.dof_values = x
        
        return x
    
    def compute_l2_error(self):
        """Compute L2 error against exact solution"""
        domain = fem.Cells(geometry=self.geo)
        
        error_sq = wp.zeros(1, dtype=wp.float32)
        fem.integrate(
            error_integrand,
            fields={"u": self.u_field},
            domain=domain,
            output=error_sq
        )
        
        return float(np.sqrt(error_sq.numpy()[0]))


def run_poisson_test():
    wp.init()
    print("=== Poisson Equation Solver ===\n")
    
    # Test convergence at different resolutions
    resolutions = [10, 20, 40]
    errors = []
    
    for res in resolutions:
        solver = PoissonSolver(resolution=res, degree=2)
        solver.solve(quiet=True)
        l2_error = solver.compute_l2_error()
        errors.append(l2_error)
        print(f"Resolution {res:3d}x{res:3d}: L2 error = {l2_error:.6e}")
    
    # Check convergence rate (should be ~O(h²) for degree 2)
    for i in range(len(errors) - 1):
        h_ratio = resolutions[i] / resolutions[i+1]  # mesh refinement ratio
        error_ratio = errors[i] / errors[i+1]
        order = np.log(error_ratio) / np.log(h_ratio)
        print(f"Convergence order {resolutions[i]} -> {resolutions[i+1]}: {order:.2f}")
    
    print("\n✓ Poisson solver implementation complete.")
    return errors


if __name__ == "__main__":
    run_poisson_test()
