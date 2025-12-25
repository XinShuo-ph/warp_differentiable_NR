import warp as wp
import warp.fem as fem

@fem.integrand
def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))

@fem.integrand
def source_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    x = fem.position(domain, s)
    # f = 2 * pi^2 * sin(pi*x) * sin(pi*y)
    # Assuming domain is [0,1]x[0,1] which is default for Grid2D?
    # Grid2D default bounds are usually determined by res and cell size? 
    # Grid2D constructor: res, bounds_lo, bounds_hi. Default lo=(0,0), hi=(1,1)?
    # Checking warp docs or examples. example_diffusion does not specify bounds, so (0,0) to (1,1) presumably?
    # Actually Grid2D(res=...) might default to cell_size=1?
    # Let's assume standard [0,1] for now or check cell size.
    
    # Let's enforce domain [0,1] by checking coordinates.
    # But wait, if I use Grid2D(res=N), and it defaults to unit box?
    pi = 3.14159265359
    val = 2.0 * pi * pi * wp.sin(pi * x[0]) * wp.sin(pi * x[1])
    return val * v(s)

@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)

@fem.integrand
def zero_form(s: fem.Sample, v: fem.Field):
    return 0.0 * v(s)

class PoissonSolver:
    def __init__(self, resolution=50, degree=2):
        self.resolution = resolution
        self.degree = degree
        
        self.geo = fem.Grid2D(res=wp.vec2i(resolution))
        self.space = fem.make_polynomial_space(self.geo, degree=degree)
        self.field = self.space.make_field()
        
    def solve(self):
        domain = fem.Cells(geometry=self.geo)
        
        # Test and trial functions
        test = fem.make_test(space=self.space, domain=domain)
        trial = fem.make_trial(space=self.space, domain=domain)
        
        # Assemble matrix (LHS)
        matrix = fem.integrate(diffusion_form, fields={"u": trial, "v": test})
        
        # Assemble vector (RHS)
        # Use higher order quadrature for source term
        quad = fem.RegularQuadrature(domain, order=4)
        rhs = fem.integrate(source_form, fields={"v": test}, quadrature=quad)
        
        # Boundary conditions (Homogeneous Dirichlet u=0)
        boundary = fem.BoundarySides(self.geo)
        bd_test = fem.make_test(space=self.space, domain=boundary)
        bd_trial = fem.make_trial(space=self.space, domain=boundary)
        
        # Projector for Dirichlet BCs
        bd_projector = fem.integrate(mass_form, fields={"u": bd_trial, "v": bd_test}, assembly="nodal")
        bd_rhs = fem.integrate(zero_form, fields={"v": bd_test}, assembly="nodal")
        
        # Apply BCs
        fem.project_linear_system(matrix, rhs, bd_projector, bd_rhs)
        
        # Solve
        x = wp.zeros_like(rhs)
        from warp.examples.fem.utils import bsr_cg
        bsr_cg(matrix, b=rhs, x=x, tol=1.0e-8, max_iters=1000)
        
        self.field.dof_values = x

if __name__ == "__main__":
    wp.init()
    solver = PoissonSolver(resolution=64)
    solver.solve()
    print("Solved Poisson equation.")
