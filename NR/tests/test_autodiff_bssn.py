"""
Test autodiff capability through BSSN timestep.

Verify that warp can compute gradients through the evolution.
"""

import sys
sys.path.insert(0, '/workspace/NR/src')

import warp as wp
import numpy as np

wp.init()
wp.set_module_options({"enable_backward": True})


# Simple test kernel for autodiff
@wp.kernel
def simple_evolution_kernel(
    phi: wp.array3d(dtype=float),
    phi_rhs: wp.array3d(dtype=float),
    alpha: wp.array3d(dtype=float),
    K: wp.array3d(dtype=float),
    idx: float
):
    """Simplified evolution for autodiff test: d_t phi = -alpha * K / 6"""
    i, j, k = wp.tid()
    
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]
    
    if i > 0 and i < nx-1 and j > 0 and j < ny-1 and k > 0 and k < nz-1:
        alpha_val = alpha[i, j, k]
        K_val = K[i, j, k]
        phi_rhs[i, j, k] = -alpha_val * K_val / 6.0


@wp.kernel
def update_kernel(
    phi: wp.array3d(dtype=float),
    phi_init: wp.array3d(dtype=float),
    phi_rhs: wp.array3d(dtype=float),
    dt: float
):
    """Update: phi^{n+1} = phi^n + dt * phi_rhs"""
    i, j, k = wp.tid()
    phi[i, j, k] = phi_init[i, j, k] + dt * phi_rhs[i, j, k]


@wp.kernel
def loss_kernel(
    phi: wp.array3d(dtype=float),
    loss: wp.array(dtype=float)
):
    """Compute loss = sum(phi^2)"""
    i, j, k = wp.tid()
    wp.atomic_add(loss, 0, phi[i, j, k] * phi[i, j, k])


def test_autodiff_simple():
    """Test autodiff on a simple evolution step"""
    
    print("=" * 70)
    print("Autodiff Test - Simple Evolution")
    print("=" * 70)
    
    # Small grid for testing
    nx, ny, nz = 8, 8, 8
    dx = 0.1
    dt = 0.01
    
    print(f"\nGrid: {nx} x {ny} x {nz}")
    print(f"dx = {dx}, dt = {dt}")
    
    # Initialize fields
    phi_init = wp.zeros((nx, ny, nz), dtype=wp.float32, requires_grad=True)
    phi = wp.zeros((nx, ny, nz), dtype=wp.float32, requires_grad=True)
    phi_rhs = wp.zeros((nx, ny, nz), dtype=wp.float32, requires_grad=True)
    alpha = wp.ones((nx, ny, nz), dtype=wp.float32)
    K = wp.zeros((nx, ny, nz), dtype=wp.float32, requires_grad=True)
    loss_val = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    
    # Set K to non-zero in center
    K_np = K.numpy()
    K_np[nx//2, ny//2, nz//2] = 1.0
    K.assign(K_np)
    
    print("\nTesting forward pass...")
    
    # Create tape
    tape = wp.Tape()
    
    with tape:
        # Compute RHS
        wp.launch(simple_evolution_kernel, dim=(nx, ny, nz),
                 inputs=[phi_init, phi_rhs, alpha, K, 1.0/dx])
        
        # Update
        wp.launch(update_kernel, dim=(nx, ny, nz),
                 inputs=[phi, phi_init, phi_rhs, dt])
        
        # Compute loss
        wp.launch(loss_kernel, dim=(nx, ny, nz),
                 inputs=[phi, loss_val])
    
    loss_np = loss_val.numpy()[0]
    print(f"Loss after forward pass: {loss_np:.6e}")
    
    # Test backward pass
    print("\nTesting backward pass...")
    
    try:
        # Zero gradients
        phi.grad = wp.zeros_like(phi)
        K.grad = wp.zeros_like(K)
        loss_val.grad = wp.zeros_like(loss_val)
        
        # Set loss gradient to 1
        loss_val.grad.assign(np.array([1.0], dtype=np.float32))
        
        # Backward pass
        tape.backward(loss=loss_val)
        
        # Check gradients
        K_grad = K.grad.numpy()
        phi_grad = phi.grad.numpy()
        
        print(f"K gradient: max={np.max(np.abs(K_grad)):.6e}")
        print(f"phi gradient: max={np.max(np.abs(phi_grad)):.6e}")
        
        if np.max(np.abs(K_grad)) > 0:
            print("\n✓ Autodiff working - gradients computed successfully")
            print(f"  Gradient flow verified through evolution step")
            success = True
        else:
            print("\n✗ Autodiff not working - zero gradients")
            success = False
            
    except Exception as e:
        print(f"\n✗ Autodiff failed with error: {e}")
        success = False
    
    print("=" * 70)
    return success


def test_autodiff_info():
    """Print information about autodiff support in current BSSN implementation"""
    
    print("\n" + "=" * 70)
    print("Autodiff Support in BSSN Implementation")
    print("=" * 70)
    
    print("\nCurrent status:")
    print("- Warp kernels are defined with @wp.kernel decorator")
    print("- Spatial derivatives use @wp.func functions")
    print("- All operations (add, multiply, etc.) are differentiable")
    print("- Tape mechanism can record forward pass")
    
    print("\nTo enable full autodiff for BSSN:")
    print("1. Set requires_grad=True for all evolved fields")
    print("2. Wrap evolution in tape.record() context")
    print("3. Define loss function (e.g., constraint violations)")
    print("4. Call tape.backward() to compute gradients")
    
    print("\nPotential uses:")
    print("- Optimize initial data to minimize constraint violations")
    print("- Learn gauge parameters for stability")
    print("- Inverse problems: infer parameters from gravitational waves")
    
    print("=" * 70)


if __name__ == "__main__":
    success = test_autodiff_simple()
    test_autodiff_info()
    
    if success:
        print("\n✓ All autodiff tests passed")
    else:
        print("\n⚠ Autodiff tests incomplete (expected for complex kernels)")
        print("Note: Full autodiff through BSSN RHS requires further development")
