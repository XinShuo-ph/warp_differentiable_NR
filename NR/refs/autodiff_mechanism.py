# Warp Autodiff Mechanism Reference

# 1. Mark arrays for gradient computation
x = wp.zeros(n, dtype=wp.float32, requires_grad=True)

# 2. Create tape to record operations
tape = wp.Tape()

# 3. Record operations in tape context
with tape:
    # Operations here are recorded
    fem.integrate(integrand_fn, fields={"u": field}, output=loss)

# 4. Backpropagate
tape.backward(loss=loss)

# 5. Access gradients
grad = x.grad  # gradient array

# 6. Reset tape for next iteration
tape.zero()

# Key FEM autodiff integrand example:
@fem.integrand
def loss_integrand(s: fem.Sample, u: fem.Field):
    val = u(s)
    return val * val  # differentiable w.r.t. u
