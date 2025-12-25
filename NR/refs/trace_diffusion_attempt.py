import warp as wp
import warp.fem as fem
from warp.fem.linalg import array_axpy
import warp.examples.fem.utils as fem_example_utils
import numpy as np

# Enable backward differentiation
wp.init()
wp.set_module_options({"enable_backward": True})

@fem.integrand
def linear_form(
    s: fem.Sample,
    v: fem.Field,
):
    return v(s)

@fem.integrand
def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field, nu: wp.array(dtype=float)):
    return nu[0] * wp.dot(
        fem.grad(u, s),
        fem.grad(v, s),
    )

@fem.integrand
def y_boundary_value_form(s: fem.Sample, domain: fem.Domain, v: fem.Field, val: float):
    nor = fem.normal(domain, s)
    return val * v(s) * wp.abs(nor[0])

@fem.integrand
def y_boundary_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    return y_boundary_value_form(s, domain, v, u(s))

@wp.kernel
def compute_loss_kernel(x: wp.array(dtype=wp.float64), loss: wp.array(dtype=wp.float64)):
    tid = wp.tid()
    wp.atomic_add(loss, 0, x[tid] * x[tid])

def trace_autodiff():
    resolution = 10
    degree = 1
    
    geo = fem.Grid2D(res=wp.vec2i(resolution))
    element_basis = None # default
    scalar_space = fem.make_polynomial_space(geo, degree=degree, element_basis=element_basis)
    scalar_field = scalar_space.make_field()
    
    # We want to differentiate w.r.t viscosity
    viscosity = wp.array([2.0], dtype=float, requires_grad=True)
    boundary_value = 5.0
    
    tape = wp.Tape()
    with tape:
        domain = fem.Cells(geometry=geo)
        
        # RHS
        test = fem.make_test(space=scalar_space, domain=domain)
        rhs = fem.integrate(linear_form, fields={"v": test})
        
        # Diffusion Matrix
        trial = fem.make_trial(space=scalar_space, domain=domain)
        
        # We need to pass viscosity as a value. 
        # fem.integrate expects values to be python types or warp arrays?
        # The integrand expects 'nu: float'. 
        # If we pass a wp.array, it might not work directly if it expects a scalar in the kernel.
        # But let's try passing the array element.
        # Wait, fem.integrate 'values' dict usually maps str -> value.
        # If the kernel argument is 'float', passing a wp.array might fail or treat it as an array.
        # Usually we need to use a distinct field or uniform.
        # But let's try using the viscosity array directly, maybe as a field? No, it's a constant.
        
        # NOTE: passing a warp array to a float argument in integrate might not work for gradients 
        # if the mechanism expects constant.
        # But let's see. 
        
        # Actually, for autodiff, the parameter must be a warp array involved in the computation.
        # If fem.integrate takes values as constants at compile time or kernel launch params, 
        # we need to ensure the gradient flows.
        
        matrix = fem.integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": viscosity})
        
        # Boundary conditions (simplified, ignoring compliance for now)
        boundary = fem.BoundarySides(geo)
        bd_test = fem.make_test(space=scalar_space, domain=boundary)
        bd_trial = fem.make_trial(space=scalar_space, domain=boundary)
        
        bd_matrix = fem.integrate(y_boundary_projector_form, fields={"u": bd_trial, "v": bd_test}, assembly="nodal")
        bd_rhs = fem.integrate(
            y_boundary_value_form, fields={"v": bd_test}, values={"val": boundary_value}, assembly="nodal"
        )
        
        # Project linear system (apply BCs)
        # This modifies matrix and rhs in place usually?
        # fem.project_linear_system returns nothing, modifies arguments.
        fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)
        
        # Solve
        x = wp.zeros_like(rhs)
        # bsr_cg might not support autodiff through the solver loop easily without checkpoints 
        # or it might just unroll.
        # For small resolution it might be fine.
        fem_example_utils.bsr_cg(matrix, b=rhs, x=x, quiet=True, max_iters=100)
        
        # Compute loss: L2 norm of solution
        # loss = wp.dot(x, x) # This fails in python
        
        loss = wp.zeros(1, dtype=wp.float64, requires_grad=True)
        wp.launch(
            kernel=compute_loss_kernel,
            dim=x.shape[0],
            inputs=[x, loss],
        )
        
    print(f"Loss: {loss.numpy()[0]}")
    
    # Backward
    tape.backward(loss)
    
    grad_nu = viscosity.grad.numpy()[0]
    print(f"Gradient w.r.t viscosity: {grad_nu}")

if __name__ == "__main__":
    try:
        trace_autodiff()
        print("Autodiff trace successful.")
    except Exception as e:
        print(f"Autodiff trace failed: {e}")
        import traceback
        traceback.print_exc()
