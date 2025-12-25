# Warp Autodiff Reference
# Key mechanism: wp.Tape for recording operations and computing gradients

import warp as wp
import numpy as np

wp.init()

# 1. Arrays that need gradients must specify requires_grad=True
# x = wp.array(..., requires_grad=True)

# 2. wp.Tape() records operations for backward pass
# tape = wp.Tape()
# with tape:
#     forward()  # all kernel launches recorded
# tape.backward(loss)  # computes gradients

# 3. Access gradients after backward: tape.gradients[array]
# 4. Clear gradients: tape.zero()

@wp.kernel
def compute_loss(x: wp.array(dtype=float), loss: wp.array(dtype=float)):
    tid = wp.tid()
    # Simple quadratic loss: sum(x^2)
    wp.atomic_add(loss, 0, x[tid] * x[tid])

@wp.kernel
def gradient_descent_step(x: wp.array(dtype=float), grad: wp.array(dtype=float), lr: float):
    tid = wp.tid()
    x[tid] = x[tid] - lr * grad[tid]

def test_autodiff():
    n = 10
    x = wp.array(np.random.randn(n).astype(np.float32), dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    print(f"Initial x: {x.numpy()}")
    print(f"Initial loss: {np.sum(x.numpy()**2):.4f}")

    lr = 0.1
    for i in range(5):
        loss.zero_()
        
        tape = wp.Tape()
        with tape:
            wp.launch(compute_loss, dim=n, inputs=[x, loss])
        
        tape.backward(loss)
        x_grad = tape.gradients[x]
        
        print(f"Iter {i}: loss={loss.numpy()[0]:.4f}, grad_norm={np.linalg.norm(x_grad.numpy()):.4f}")
        
        wp.launch(gradient_descent_step, dim=n, inputs=[x, x_grad, lr])
        tape.zero()

    print(f"Final x: {x.numpy()}")
    print(f"Final loss: {np.sum(x.numpy()**2):.4f}")

if __name__ == "__main__":
    test_autodiff()
