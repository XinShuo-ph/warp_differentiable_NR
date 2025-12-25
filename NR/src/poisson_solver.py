import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils
import math

wp.init()

@fem.integrand
def laplacian_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    # Weak form of -Laplacian u = f
    # Integrate grad(u) . grad(v)
    return wp.dot(fem.grad(u, s), fem.grad(v, s))

PI = math.pi
PI_SQ = PI * PI

@fem.integrand
def rhs_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    x = fem.position(domain, s)
    # Cast to float (float32 in warp) to match x type
    pi_f = float(PI)
    val = 2.0 * float(PI_SQ) * wp.sin(pi_f * x[0]) * wp.sin(pi_f * x[1])
    return val * v(s)

@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)

class PoissonSolver:
    def __init__(self, resolution=50, degree=2):
        self.resolution = resolution
        self.degree = degree
        self.geo = fem.Grid2D(res=wp.vec2i(resolution))
        self.space = fem.make_polynomial_space(self.geo, degree=degree)
        self.u_field = self.space.make_field()
        
    def solve(self):
        domain = fem.Cells(self.geo)
        
        # Trial and Test functions
        u = fem.make_trial(self.space, domain=domain)
        v = fem.make_test(self.space, domain=domain)
        
        # Assemble Matrix (Stiffness)
        matrix = fem.integrate(laplacian_form, fields={"u": u, "v": v})
        
        # Assemble RHS
        rhs = fem.integrate(rhs_form, fields={"v": v})
        
        # Boundary Conditions (Homogeneous Dirichlet u=0)
        boundary = fem.BoundarySides(self.geo)
        bd_trial = fem.make_trial(self.space, domain=boundary)
        bd_test = fem.make_test(self.space, domain=boundary)
        
        # Projector for BCs
        # For u=0, we project the system such that rows/cols corresponding to boundary are identity/zero
        # fem.project_linear_system handles this.
        # We need a projector matrix which is Mass matrix on boundary?
        # See example_diffusion.py:
        # bd_matrix = fem.integrate(y_boundary_projector_form, ...)
        # Here we want u=0 everywhere on boundary.
        
        # We can use a mass form on boundary to identify boundary nodes
        bd_projector = fem.integrate(mass_form, fields={"u": bd_trial, "v": bd_test}, assembly="nodal")
        
        # RHS for boundary (0)
        bd_rhs = wp.zeros(self.space.node_count(), dtype=float) 
        # Actually fem.integrate returns a vector of size space.node_count() usually?
        # Wait, fem.integrate for linear form returns array of size space.dof_count().
        
        # We want bd_rhs to be zero.
        bd_rhs = wp.zeros_like(rhs)
        
        # Apply BCs
        fem.project_linear_system(matrix, rhs, bd_projector, bd_rhs)
        
        # Solve
        x = wp.zeros_like(rhs)
        fem_example_utils.bsr_cg(matrix, b=rhs, x=x, quiet=True, tol=1e-8, max_iters=2000)
        
        self.u_field.dof_values = x
        
    def compute_error(self):
        # Compute L2 error against analytical solution on nodes
        
        positions = self.space.node_positions().numpy()
        values = self.u_field.dof_values.numpy()
        
        max_error = 0.0
        l2_error_sq = 0.0
        
        count = len(values)
        for i in range(count):
            x = positions[i]
            u_ana = math.sin(math.pi * x[0]) * math.sin(math.pi * x[1])
            diff = values[i] - u_ana
            max_error = max(max_error, abs(diff))
            l2_error_sq += diff * diff
            
        l2_error = math.sqrt(l2_error_sq / count)
        
        return l2_error, max_error

if __name__ == "__main__":
    solver = PoissonSolver(resolution=64, degree=2)
    solver.solve()
    l2_err, max_err = solver.compute_error()
    print(f"L2 Error (Nodal): {l2_err}")
    print(f"Max Error (Nodal): {max_err}")
    
    if max_err < 1e-3:
        print("Verification SUCCESS")
    else:
        print("Verification FAILED")
