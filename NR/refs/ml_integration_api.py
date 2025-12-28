"""
BSSN ML Integration API Reference

This file documents the API for using the differentiable BSSN implementation
for machine learning applications.

Key modules:
- bssn_vars.py: Grid and variable definitions
- bssn_initial_data.py: Initial data constructors
- bssn_rhs_full.py: BSSN evolution equations
- bssn_losses.py: Differentiable loss functions
- bssn_ml_pipeline.py: End-to-end differentiable pipeline
"""

import warp as wp
import numpy as np

# ============================================================================
# BASIC USAGE
# ============================================================================

def example_basic_evolution():
    """
    Basic example: evolve Schwarzschild black hole.
    """
    import sys
    sys.path.insert(0, '/workspace/NR/src')
    from bssn_ml_pipeline import DifferentiableBSSNPipeline
    
    # Create pipeline
    pipeline = DifferentiableBSSNPipeline(
        nx=32, ny=32, nz=32,   # Grid dimensions
        domain_size=16.0,       # Physical size in M
        cfl=0.1,                # CFL number
        requires_grad=True      # Enable gradients
    )
    
    # Set initial data
    pipeline.set_schwarzschild_initial_data(bh_mass=1.0)
    
    # Evolve
    result = pipeline.evolve(n_steps=50)
    
    # Access final state
    print(f"Final time: {result['time']:.2f}M")
    print(f"Constraint: H_L2 = {result['H_L2']:.4e}")


# ============================================================================
# GRADIENT COMPUTATION
# ============================================================================

def example_gradient_computation():
    """
    Compute gradients of loss with respect to initial data.
    """
    import sys
    sys.path.insert(0, '/workspace/NR/src')
    from bssn_vars import BSSNGrid
    from bssn_initial_data import set_schwarzschild_puncture
    from bssn_losses import asymptotic_flatness_loss_kernel
    
    wp.init()
    
    # Create grid with gradient tracking
    grid = BSSNGrid(32, 32, 32, dx=0.5, requires_grad=True)
    set_schwarzschild_puncture(grid, bh_mass=1.0)
    
    # Mark variables for gradient computation
    grid.alpha.requires_grad = True
    
    # Create loss array
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    
    # Record operations with tape
    tape = wp.Tape()
    with tape:
        wp.launch(
            asymptotic_flatness_loss_kernel,
            dim=grid.n_points,
            inputs=[grid.phi, grid.alpha, grid.gt11, grid.gt22, grid.gt33,
                    grid.nx, grid.ny, grid.nz, grid.dx, loss]
        )
    
    # Backward pass
    tape.backward(loss=loss)
    
    # Access gradients
    grad_alpha = grid.alpha.grad.numpy()
    print(f"∂L/∂α max: {np.abs(grad_alpha).max():.6e}")
    
    tape.zero()


# ============================================================================
# OPTIMIZATION LOOP
# ============================================================================

def example_optimization():
    """
    Optimize lapse function to minimize loss.
    """
    import sys
    sys.path.insert(0, '/workspace/NR/src')
    from bssn_vars import BSSNGrid
    from bssn_initial_data import set_schwarzschild_puncture
    from bssn_losses import asymptotic_flatness_loss_kernel
    
    wp.init()
    
    # Setup
    grid = BSSNGrid(24, 24, 24, dx=0.5, requires_grad=True)
    set_schwarzschild_puncture(grid, bh_mass=1.0)
    
    # Optimization parameters
    lr = 0.01
    n_iters = 50
    
    for i in range(n_iters):
        # Forward pass
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        grid.alpha.requires_grad = True
        
        tape = wp.Tape()
        with tape:
            wp.launch(
                asymptotic_flatness_loss_kernel,
                dim=grid.n_points,
                inputs=[grid.phi, grid.alpha, grid.gt11, grid.gt22, grid.gt33,
                        grid.nx, grid.ny, grid.nz, grid.dx, loss]
            )
        
        # Backward pass
        tape.backward(loss=loss)
        
        # Gradient descent update
        if grid.alpha.grad is not None:
            alpha_np = grid.alpha.numpy()
            grad_np = grid.alpha.grad.numpy()
            alpha_np -= lr * grad_np
            grid.alpha = wp.array(alpha_np, dtype=wp.float32, requires_grad=True)
        
        tape.zero()
        
        if i % 10 == 0:
            print(f"Iter {i}: loss = {float(loss.numpy()[0]):.6e}")


# ============================================================================
# ML TRAINING INTEGRATION
# ============================================================================

def example_pytorch_integration():
    """
    Conceptual example of PyTorch integration.
    
    NOTE: Requires additional wrapper code for PyTorch<->Warp tensors.
    """
    # Pseudo-code for PyTorch integration:
    #
    # class BSSNLayer(torch.nn.Module):
    #     def __init__(self, grid_size, domain_size):
    #         super().__init__()
    #         self.pipeline = DifferentiableBSSNPipeline(...)
    #         
    #     def forward(self, initial_alpha):
    #         # Convert PyTorch tensor to Warp array
    #         alpha_warp = wp.from_torch(initial_alpha)
    #         self.pipeline.grid.alpha = alpha_warp
    #         
    #         # Evolve with autodiff
    #         tape = wp.Tape()
    #         with tape:
    #             self.pipeline.evolve(n_steps=10)
    #             loss = self.pipeline.compute_loss()
    #         
    #         # Backward
    #         tape.backward(loss=loss)
    #         
    #         # Convert gradients back to PyTorch
    #         return wp.to_torch(self.pipeline.grid.alpha.grad)
    pass


# ============================================================================
# API REFERENCE
# ============================================================================

"""
API Summary
===========

Core Classes
------------

BSSNGrid(nx, ny, nz, dx, requires_grad=False)
    Grid container for BSSN variables.
    
    Attributes:
        phi, gt11-gt33, trK, At11-At33, Xt1-Xt3, alpha, beta1-beta3
        *_rhs: Right-hand side arrays for each variable
    
    Methods:
        set_flat_spacetime(): Initialize to Minkowski
        

DifferentiableBSSNPipeline(nx, ny, nz, domain_size, cfl, requires_grad)
    End-to-end differentiable evolution pipeline.
    
    Methods:
        set_schwarzschild_initial_data(bh_mass, bh_pos, pre_collapse_lapse)
        set_brill_lindquist_initial_data(m1, pos1, m2, pos2)
        evolve(n_steps, extract_waveform, verbose) -> result dict
        compute_loss(loss_type) -> loss array
        step(): Single time step
        

ConstraintMonitor(grid)
    Compute Hamiltonian and momentum constraints.
    
    Methods:
        compute(): Calculate H, M1, M2, M3
        get_norms() -> {'H_L2', 'H_Linf', 'M_L2', 'M_Linf'}
        

WaveformExtractor(grid, r_extract)
    Extract gravitational waveforms.
    
    Methods:
        extract(t): Extract at time t
        get_waveform() -> (times, psi4_real, psi4_imag)
        get_strain() -> (times, h_real, h_imag)


Loss Functions
--------------

constraint_loss_kernel: L = Σ(H² + M²)
stability_loss_kernel: Penalize small α, large φ, large K
asymptotic_flatness_loss_kernel: Enforce flat boundaries

DifferentiableLoss(grid, requires_grad)
    compute_constraint_loss(H, M1, M2, M3)
    compute_stability_loss()
    compute_asymptotic_loss()
    get_loss() -> float


Autodiff Pattern
----------------

# 1. Create arrays with requires_grad=True
grid = BSSNGrid(..., requires_grad=True)
loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

# 2. Mark input for gradients
grid.alpha.requires_grad = True

# 3. Record operations
tape = wp.Tape()
with tape:
    # ... operations ...
    wp.launch(loss_kernel, ...)

# 4. Backward pass
tape.backward(loss=loss)

# 5. Access gradients
gradients = grid.alpha.grad.numpy()

# 6. Clean up
tape.zero()
"""

if __name__ == "__main__":
    print("BSSN ML Integration API Reference")
    print("=" * 50)
    print("\nSee docstrings for examples and API documentation.")
    print("\nKey modules:")
    print("  - bssn_vars.py: Grid and variables")
    print("  - bssn_initial_data.py: Initial data")
    print("  - bssn_rhs_full.py: Evolution equations")
    print("  - bssn_losses.py: Loss functions")
    print("  - bssn_ml_pipeline.py: End-to-end pipeline")
    print("  - bssn_optimization.py: Optimization examples")
    print("  - bssn_autodiff_evolution_test.py: Gradient verification")
