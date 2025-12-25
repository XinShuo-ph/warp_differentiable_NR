import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils
import math

wp.init()
wp.set_module_options({"enable_backward": False})

@fem.integrand
def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))

@fem.integrand
def source_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    x = domain(s)
    # f = 2 * pi^2 * sin(pi*x) * sin(pi*y)
    # Precompute constant factor
    factor = 19.739208802178716 # 2 * pi^2
    pi = 3.141592653589793
    
    return factor * wp.sin(pi * x[0]) * wp.sin(pi * x[1]) * v(s)

@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)

class PoissonSolver:
    def __init__(self, res=50, degree=2):
        self.res = res
        self.degree = degree
        
        self.geo = fem.Grid2D(res=wp.vec2i(res), bounds_lo=wp.vec2(0.0, 0.0), bounds_hi=wp.vec2(1.0, 1.0))
        
        self.scalar_space = fem.make_polynomial_space(self.geo, degree=degree)
        self.domain = fem.Cells(geometry=self.geo)
        
        self.u_field = self.scalar_space.make_field()
        
    def solve(self):
        # Test/Trial
        u = fem.make_trial(self.scalar_space, domain=self.domain)
        v = fem.make_test(self.scalar_space, domain=self.domain)
        
        # Matrix
        matrix = fem.integrate(diffusion_form, fields={"u": u, "v": v})
        
        # RHS
        rhs = fem.integrate(source_form, fields={"v": v})
        
        # BCs: Homogeneous Dirichlet u=0 on boundary
        boundary = fem.BoundarySides(self.geo)
        bd_trial = fem.make_trial(self.scalar_space, domain=boundary)
        bd_test = fem.make_test(self.scalar_space, domain=boundary)
        
        # Projector for BCs
        bd_projector = fem.integrate(mass_form, fields={"u": bd_trial, "v": bd_test}, assembly="nodal")
        
        # Target values (0)
        bd_values = wp.zeros_like(rhs) 
        
        fem.normalize_dirichlet_projector(bd_projector, bd_values)
        fem.project_linear_system(matrix, rhs, bd_projector, bd_values)
        
        # Solve
        x = wp.zeros_like(rhs)
        fem_example_utils.bsr_cg(matrix, b=rhs, x=x, tol=1e-8, quiet=True)
        
        self.u_field.dof_values = x

    def verify(self):
        # Verify by checking value at center node
        # For Grid2D with degree k, nodes are at (i/k*res_x, j/k*res_y)
        # Center (0.5, 0.5) corresponds to i = k*res_x / 2
        
        kx = self.degree * self.res
        ky = self.degree * self.res
        
        # Assuming resolution is even so center is a node
        cx = kx // 2
        cy = ky // 2
        
        # Number of nodes in X
        nx = kx + 1
        
        idx = cy * nx + cx
        
        dofs = self.u_field.dof_values.numpy()
        val = dofs[idx]
        
        print(f"Index: {idx}, Grid dims: {nx}x{ky+1}")
        print(f"Value at center: {val}")
        print(f"Analytical value: 1.0")
        error = abs(val - 1.0)
        print(f"Error: {error}")
        
        if error < 1e-2:
            print("Verification PASSED")
        else:
            print("Verification FAILED")

if __name__ == "__main__":
    solver = PoissonSolver(res=64, degree=2)
    solver.solve()
    solver.verify()
