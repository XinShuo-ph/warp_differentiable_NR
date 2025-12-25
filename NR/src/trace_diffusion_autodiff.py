import warp as wp
import warp.fem as fem
import warp.tests.unittest_utils as test_utils

wp.init()
wp.set_module_options({"enable_backward": True})

@fem.integrand
def scaled_linear_form(s: fem.Sample, u: fem.Field, scale: wp.array(dtype=float)):
    return u(s) * scale[0]

@wp.kernel
def atomic_sum(v: wp.array(dtype=float), sum: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(sum, 0, v[i])

def test_autodiff():
    # Grid geometry
    geo = fem.Grid2D(res=wp.vec2i(5))
    domain = fem.Cells(geometry=geo)
    quadrature = fem.RegularQuadrature(domain=domain, order=3)
    scalar_space = fem.make_polynomial_space(geo, degree=3)

    test_field = fem.make_test(space=scalar_space, domain=domain)

    # Output array (Vector)
    # The size of the vector is the number of DOFs in the space
    n_dofs = scalar_space.node_count() # or dof_count?
    # scalar_space.node_count() might differ from dof_count if vector field, but here scalar.
    # Actually fem.integrate allocates output if not provided. Let's provide it to be safe/consistent with test.
    
    # We need to know the size. 
    # Let's let fem.integrate allocate it first to see size, or use what test did.
    # The test used u_adj = wp.empty_like(u.dof_values).
    # We need to create a field to get dof_values shape.
    dummy_u = scalar_space.make_field()
    u_adj = wp.zeros_like(dummy_u.dof_values, requires_grad=True)
    
    scale = wp.array([2.0], dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    tape = wp.Tape()
    with tape:
        # Integrate Linear Form -> Vector
        fem.integrate(
            scaled_linear_form,
            quadrature=quadrature,
            fields={"u": test_field},
            values={"scale": scale},
            output=u_adj,
        )
        
        # Sum the vector
        loss.zero_()
        wp.launch(atomic_sum, dim=u_adj.shape, inputs=[u_adj, loss])

    loss_val = loss.numpy()[0]
    print("Loss:", loss_val)
    
    tape.backward(loss)
    
    grad_scale = scale.grad.numpy()[0]
    print("Gradient of scale:", grad_scale)
    
    # Verify
    expected_grad = loss_val / scale.numpy()[0]
    print("Expected gradient:", expected_grad)

if __name__ == "__main__":
    test_autodiff()
