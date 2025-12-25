import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import warp as wp
import numpy as np
from bssn import BSSNState
from integrator import RK4Integrator

def test_autodiff():
    wp.init()
    
    # Enable autodiff
    wp.set_module_options({"enable_backward": True})
    
    res = (16, 16, 16)
    state = BSSNState(res, (0,0,0), (1,1,1))
    
    # Initialize with some noise to make derivatives non-zero
    wp.launch(
        kernel=init_noise_kernel,
        dim=state.shape,
        inputs=[state.phi, state.K]
    )
    
    state.phi.requires_grad = True
    state.K.requires_grad = True
    
    integrator = RK4Integrator(state)
    
    tape = wp.Tape()
    with tape:
        # One step
        dt = 0.01
        integrator.step(dt)
        
        # Loss function: sum of phi
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        wp.launch(
            kernel=loss_kernel,
            dim=state.shape,
            inputs=[state.phi, loss]
        )
        
    # Backward
    tape.backward(loss)
    
    # Check if we have gradients in initial state
    grad_phi = state.phi.grad.numpy()
    
    # We expect non-zero gradients because phi(t+dt) depends on phi(t)
    grad_norm = np.linalg.norm(grad_phi)
    print(f"Gradient norm for phi: {grad_norm}")
    
    if grad_norm > 1e-10:
        print("Autodiff verified.")
    else:
        print("Autodiff failed (zero gradient).")
        exit(1)

@wp.kernel
def init_noise_kernel(phi: wp.array(dtype=float, ndim=3), K: wp.array(dtype=float, ndim=3)):
    i, j, k = wp.tid()
    # Deterministic noise based on index
    val = float(i + j*10 + k*100) * 1e-5
    phi[i, j, k] = val
    K[i, j, k] = val

@wp.kernel
def loss_kernel(phi: wp.array(dtype=float, ndim=3), loss: wp.array(dtype=float, ndim=1)):
    i, j, k = wp.tid()
    wp.atomic_add(loss, 0, phi[i, j, k])

if __name__ == "__main__":
    test_autodiff()
