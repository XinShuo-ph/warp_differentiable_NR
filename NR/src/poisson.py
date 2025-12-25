import warp as wp
import warp.fem as fem
try:
    import warp.examples.fem.utils as fem_utils
except ImportError:
    # Fallback if module structure is different
    # Since we are in /workspace/NR/src, and warp is in /workspace/warp
    # We might need to adjust path
    import sys
    sys.path.append("/workspace/warp")
    import warp.examples.fem.utils as fem_utils

import numpy as np

@fem.integrand
def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    # Weak form of -Laplacian u: dot(grad(u), grad(v))
    return wp.dot(
        fem.grad(u, s),
        fem.grad(v, s)
    )

@fem.integrand
def source_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    # f = 2 * pi^2 * sin(pi * x) * sin(pi * y)
    x = fem.position(domain, s)
    val = 2.0 * wp.PI * wp.PI * wp.sin(wp.PI * x[0]) * wp.sin(wp.PI * x[1])
    return val * v(s)

@fem.integrand
def l2_error_form(s: fem.Sample, domain: fem.Domain, u: fem.Field):
    x = fem.position(domain, s)
    exact = wp.sin(wp.PI * x[0]) * wp.sin(wp.PI * x[1])
    diff = u(s) - exact
    return diff * diff

@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)

class PoissonSolver:
    def __init__(self, resolution=50, degree=2):
        self.resolution = resolution
        self.degree = degree
        
        # Geometry: Unit square
        self.geo = fem.Grid2D(res=wp.vec2i(resolution))
        
        # Space
        self.space = fem.make_polynomial_space(self.geo, degree=degree, dtype=float)
        self.field = self.space.make_field()

    def solve(self):
        domain = fem.Cells(geometry=self.geo)
        
        # Test and Trial
        test = fem.make_test(space=self.space, domain=domain)
        trial = fem.make_trial(space=self.space, domain=domain)
        
        # Matrix and RHS
        matrix = fem.integrate(diffusion_form, fields={"u": trial, "v": test}, output_dtype=float)
        rhs = fem.integrate(source_form, fields={"v": test}, domain=domain, output_dtype=float)
        
        # Boundary Conditions: u = 0 on all boundaries
        boundary = fem.BoundarySides(self.geo)
        bd_test = fem.make_test(space=self.space, domain=boundary)
        bd_trial = fem.make_trial(space=self.space, domain=boundary)
        
        # Projector for Homogeneous Dirichlet BC (val = 0)
        bd_projector = fem.integrate(mass_form, fields={"u": bd_trial, "v": bd_test}, assembly="nodal", output_dtype=float)
        
        # Values are zero
        bd_values = wp.zeros_like(rhs) 
        
        fem.normalize_dirichlet_projector(bd_projector)
        fem.project_linear_system(matrix, rhs, bd_projector, bd_values, normalize_projector=False)
        
        # Solve
        x = wp.zeros_like(rhs)
        fem_utils.bsr_cg(matrix, b=rhs, x=x, tol=1e-8, quiet=False)
        
        self.field.dof_values = x

    def compute_error(self):
        domain = fem.Cells(geometry=self.geo)
        
        # Let's try defining output as a size 1 array.
        err_val = wp.zeros(1, dtype=float)
        
        fem.integrate(l2_error_form, fields={"u": self.field}, domain=domain, output=err_val, output_dtype=float)
        
        return np.sqrt(err_val.numpy()[0])

if __name__ == "__main__":
    wp.init()
    solver = PoissonSolver(resolution=64)
    solver.solve()
    error = solver.compute_error()
    print(f"L2 Error: {error}")
