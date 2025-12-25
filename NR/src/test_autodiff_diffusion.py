"""
Demonstration of warp autodiff with FEM diffusion.

Key autodiff concepts:
1. wp.Tape() - records computation graph
2. array.requires_grad = True - marks arrays for gradient computation
3. with tape: ... - records operations within context
4. tape.backward(loss=loss) - backpropagates through recorded operations
5. tape.zero() - resets gradient accumulation
"""

import warp as wp
import warp.fem as fem
from warp.fem.linalg import array_axpy
import sys
sys.path.insert(0, '/workspace/warp_repo/warp/examples/fem')
import utils as fem_example_utils


@fem.integrand
def linear_form(s: fem.Sample, v: fem.Field):
    return v(s)


@fem.integrand
def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field, nu: float):
    return nu * wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def loss_integrand(s: fem.Sample, u: fem.Field):
    """Squared L2 norm of solution - a differentiable loss function"""
    val = u(s)
    return val * val


def run_autodiff_test():
    wp.init()
    print("=== Warp Autodiff FEM Test ===\n")

    # Create geometry
    resolution = 20
    geo = fem.Grid2D(res=wp.vec2i(resolution))
    
    # Scalar function space (use float32 for autodiff compatibility)
    scalar_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.float32)
    
    domain = fem.Cells(geometry=geo)
    test = fem.make_test(space=scalar_space, domain=domain)
    trial = fem.make_trial(space=scalar_space, domain=domain)
    
    # Viscosity parameter we want gradients w.r.t.
    nu_val = 2.0
    
    # Build RHS (use float32)
    rhs = fem.integrate(linear_form, fields={"v": test}, output_dtype=wp.float32)
    
    # Build matrix
    matrix = fem.integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": nu_val}, output_dtype=wp.float32)
    
    # Create solution array with gradient tracking BEFORE solving
    x = wp.zeros(scalar_space.node_count(), dtype=wp.float32, requires_grad=True)
    fem_example_utils.bsr_cg(matrix, b=rhs, x=x, quiet=True)
    
    # Create field with the solution array
    scalar_field = scalar_space.make_field()
    scalar_field.dof_values = x
    
    print(f"Solution computed. Max value: {x.numpy().max():.6f}")
    
    # Now demonstrate autodiff by computing gradient of a loss
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    
    tape = wp.Tape()
    with tape:
        # Integrate the squared L2 norm
        fem.integrate(
            loss_integrand,
            fields={"u": scalar_field},
            domain=domain,
            output=loss,
        )
    
    print(f"Loss (L2 norm squared): {loss.numpy()[0]:.6f}")
    
    # Backpropagate
    tape.backward(loss=loss)
    
    # Check gradients exist
    grad = scalar_field.dof_values.grad
    print(f"Gradient computed. Grad shape: {grad.shape}, max abs grad: {abs(grad.numpy()).max():.6f}")
    
    # Verify gradient is non-zero
    grad_max = abs(grad.numpy()).max()
    assert grad_max > 0, "Gradient should be non-zero"
    print("\n✓ Autodiff works! Gradients successfully computed through FEM integration.")
    
    # Run again to verify consistency
    tape.zero()
    loss2 = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    with tape:
        fem.integrate(
            loss_integrand,
            fields={"u": scalar_field},
            domain=domain,
            output=loss2,
        )
    tape.backward(loss=loss2)
    
    print(f"\nSecond run loss: {loss2.numpy()[0]:.6f}")
    assert abs(loss.numpy()[0] - loss2.numpy()[0]) < 1e-6, "Two runs should give same result"
    print("✓ Validation: Two runs give consistent results.\n")
    

if __name__ == "__main__":
    run_autodiff_test()
