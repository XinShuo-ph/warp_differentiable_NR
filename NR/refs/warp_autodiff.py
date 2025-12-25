# Warp Autodiff Reference
# Key patterns extracted from warp examples

import warp as wp

# 1. Arrays that need gradients must have requires_grad=True
# x = wp.array([1.0, 2.0], dtype=float, requires_grad=True)

# 2. Use wp.Tape() to record operations for backward pass
# tape = wp.Tape()
# with tape:
#     wp.launch(kernel, dim=N, inputs=[x, y])
# tape.backward(loss)  # loss must be a wp.array with single element

# 3. Retrieve gradients
# grad_x = tape.gradients[x]

# 4. Zero gradients for next iteration
# tape.zero()

# 5. @wp.kernel decorator for GPU/CPU kernels
# 6. @wp.func for helper functions callable from kernels
# 7. @fem.integrand for FEM integrands (used with fem.integrate)

# Simple autodiff example:
@wp.kernel
def loss_kernel(x: wp.array(dtype=float), loss: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(loss, 0, x[tid] * x[tid])  # loss = sum(x^2)

def test_autodiff():
    wp.init()
    
    x = wp.array([1.0, 2.0, 3.0], dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    
    tape = wp.Tape()
    with tape:
        wp.launch(loss_kernel, dim=3, inputs=[x, loss])
    
    tape.backward(loss)
    
    # grad should be 2*x = [2.0, 4.0, 6.0]
    grad_x = tape.gradients[x]
    print("x:", x.numpy())
    print("loss:", loss.numpy())  
    print("grad_x:", grad_x.numpy())  # Should be [2, 4, 6]
    
    return grad_x.numpy()

if __name__ == "__main__":
    test_autodiff()
