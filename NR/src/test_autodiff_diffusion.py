import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils
import numpy as np

wp.set_module_options({"enable_backward": True})

@fem.integrand
def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field, nu: float):
    return nu * wp.dot(fem.grad(u, s), fem.grad(v, s))

@fem.integrand
def linear_form(s: fem.Sample, v: fem.Field):
    return v(s)

def test_autodiff():
    geo = fem.Grid2D(res=wp.vec2i(10))
    scalar_space = fem.make_polynomial_space(geo, degree=2)
    
    domain = fem.Cells(geometry=geo)
    test = fem.make_test(space=scalar_space, domain=domain)
    trial = fem.make_trial(space=scalar_space, domain=domain)
    
    viscosity = 1.0
    
    tape = wp.Tape()
    with tape:
        matrix = fem.integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": viscosity})
        rhs = fem.integrate(linear_form, fields={"v": test})
    
    print("Diffusion matrix shape:", matrix.shape)
    print("RHS vector shape:", rhs.shape)
    print("Autodiff tape recorded:", len(tape.gradients) > 0)
    print("Tape operations count:", len(tape.gradients))
    
    x = wp.zeros_like(rhs)
    fem_example_utils.bsr_cg(matrix, b=rhs, x=x, quiet=False, tol=1.0e-4)
    
    print("Solution computed, norm:", np.linalg.norm(x.numpy()))
    
    tape.backward()
    print("Backward pass completed successfully")

if __name__ == "__main__":
    with wp.ScopedDevice("cpu"):
        test_autodiff()
