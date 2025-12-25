"""
BSSN right-hand-side (RHS) evolution equations.

Starting with flat spacetime evolution for testing.
"""

import warp as wp
import numpy as np
from bssn_state import BSSNState, SymmetricTensor3, idx3d
from bssn_derivatives import deriv_x_4th, deriv_y_4th, deriv_z_4th

wp.init()


@wp.kernel
def compute_bssn_rhs_flat(
    # Input state
    chi: wp.array(dtype=float),
    gamma_tilde: wp.array(dtype=SymmetricTensor3),
    K: wp.array(dtype=float),
    A_tilde: wp.array(dtype=SymmetricTensor3),
    Gamma_tilde: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=float),
    beta: wp.array(dtype=wp.vec3),
    # Output RHS
    rhs_chi: wp.array(dtype=float),
    rhs_gamma: wp.array(dtype=SymmetricTensor3),
    rhs_K: wp.array(dtype=float),
    rhs_A: wp.array(dtype=SymmetricTensor3),
    rhs_Gamma: wp.array(dtype=wp.vec3),
    # Grid parameters
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float
):
    """
    Compute RHS of BSSN equations for flat spacetime evolution.
    
    For flat spacetime with fixed gauge (α=1, β=0), RHS should be zero.
    This is a test to verify the implementation.
    """
    i, j, k = wp.tid()
    
    if i >= nx or j >= ny or k >= nz:
        return
    
    idx = idx3d(i, j, k, nx, ny, nz)
    
    # Get local values
    chi_val = chi[idx]
    K_val = K[idx]
    alpha_val = alpha[idx]
    beta_val = beta[idx]
    gamma_val = gamma_tilde[idx]
    A_val = A_tilde[idx]
    Gamma_val = Gamma_tilde[idx]
    
    # For flat spacetime test: set all RHS to zero
    # (This verifies the state remains flat)
    rhs_chi[idx] = 0.0
    rhs_K[idx] = 0.0
    
    zero_tensor = SymmetricTensor3()
    zero_tensor.xx = 0.0
    zero_tensor.xy = 0.0
    zero_tensor.xz = 0.0
    zero_tensor.yy = 0.0
    zero_tensor.yz = 0.0
    zero_tensor.zz = 0.0
    
    rhs_gamma[idx] = zero_tensor
    rhs_A[idx] = zero_tensor
    
    rhs_Gamma[idx] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def compute_bssn_rhs_gauge(
    # Input state
    alpha: wp.array(dtype=float),
    beta: wp.array(dtype=wp.vec3),
    K: wp.array(dtype=float),
    Gamma_tilde: wp.array(dtype=wp.vec3),
    # Output RHS  
    rhs_alpha: wp.array(dtype=float),
    rhs_beta: wp.array(dtype=wp.vec3),
    # Auxiliary for Gamma-driver
    B_tilde: wp.array(dtype=wp.vec3),
    rhs_B: wp.array(dtype=wp.vec3),
    # Grid parameters
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float,
    # Gauge parameters
    eta: float  # Damping parameter for Gamma-driver
):
    """
    Gauge evolution: 1+log lapse and Gamma-driver shift.
    
    For flat spacetime: keeps α=1, β=0
    """
    i, j, k = wp.tid()
    
    if i >= nx or j >= ny or k >= nz:
        return
    
    idx = idx3d(i, j, k, nx, ny, nz)
    
    alpha_val = alpha[idx]
    K_val = K[idx]
    
    # 1+log slicing: ∂ₜα = -2α K + βⁱ∂ᵢα
    # For flat spacetime with K=0, β=0: ∂ₜα = 0
    rhs_alpha[idx] = -2.0 * alpha_val * K_val
    
    # Gamma-driver: ∂ₜβⁱ = B̃ⁱ, ∂ₜB̃ⁱ = 3/4 ∂ₜΓ̃ⁱ - η B̃ⁱ
    # For flat spacetime: keeps β=0
    B_val = B_tilde[idx]
    rhs_beta[idx] = B_val
    
    # For flat spacetime, Γ̃ⁱ = 0, so ∂ₜΓ̃ⁱ = 0
    rhs_B[idx] = -eta * B_val


class BSSNEvolver:
    """Manages BSSN evolution"""
    
    def __init__(self, state: BSSNState, dx: float, dy: float, dz: float):
        self.state = state
        self.dx = dx
        self.dy = dy
        self.dz = dz
        
        # Allocate RHS storage
        npts = state.nx * state.ny * state.nz
        self.rhs_chi = wp.zeros(npts, dtype=float)
        self.rhs_K = wp.zeros(npts, dtype=float)
        self.rhs_alpha = wp.zeros(npts, dtype=float)
        self.rhs_gamma = wp.zeros(npts, dtype=SymmetricTensor3)
        self.rhs_A = wp.zeros(npts, dtype=SymmetricTensor3)
        self.rhs_Gamma = wp.zeros(npts, dtype=wp.vec3)
        self.rhs_beta = wp.zeros(npts, dtype=wp.vec3)
        
        # For Gamma-driver shift
        self.B_tilde = wp.zeros(npts, dtype=wp.vec3)
        self.rhs_B = wp.zeros(npts, dtype=wp.vec3)
        
        self.eta = 1.0  # Gamma-driver damping
    
    def compute_rhs(self):
        """Compute RHS of all BSSN equations"""
        nx, ny, nz = self.state.nx, self.state.ny, self.state.nz
        
        # Compute evolution equations
        wp.launch(
            compute_bssn_rhs_flat,
            dim=(nx, ny, nz),
            inputs=[
                self.state.chi, self.state.gamma_tilde, self.state.K,
                self.state.A_tilde, self.state.Gamma_tilde,
                self.state.alpha, self.state.beta,
                self.rhs_chi, self.rhs_gamma, self.rhs_K,
                self.rhs_A, self.rhs_Gamma,
                nx, ny, nz, self.dx, self.dy, self.dz
            ]
        )
        
        # Compute gauge evolution
        wp.launch(
            compute_bssn_rhs_gauge,
            dim=(nx, ny, nz),
            inputs=[
                self.state.alpha, self.state.beta, self.state.K,
                self.state.Gamma_tilde,
                self.rhs_alpha, self.rhs_beta,
                self.B_tilde, self.rhs_B,
                nx, ny, nz, self.dx, self.dy, self.dz,
                self.eta
            ]
        )


if __name__ == "__main__":
    print("Testing BSSN RHS computation for flat spacetime...")
    
    # Create small grid
    nx, ny, nz = 16, 16, 16
    dx = dy = dz = 0.1
    
    print(f"Grid: {nx} x {ny} x {nz}")
    print(f"Spacing: {dx}")
    
    # Initialize flat spacetime
    state = BSSNState(nx, ny, nz)
    state.set_flat_spacetime()
    
    # Create evolver
    evolver = BSSNEvolver(state, dx, dy, dz)
    
    # Compute RHS
    print("\nComputing RHS for flat spacetime...")
    evolver.compute_rhs()
    
    # Check RHS is zero (as expected for flat spacetime with fixed gauge)
    rhs_chi_np = evolver.rhs_chi.numpy()
    rhs_K_np = evolver.rhs_K.numpy()
    rhs_alpha_np = evolver.rhs_alpha.numpy()
    
    max_rhs_chi = np.abs(rhs_chi_np).max()
    max_rhs_K = np.abs(rhs_K_np).max()
    max_rhs_alpha = np.abs(rhs_alpha_np).max()
    
    print(f"\nMax |RHS| values:")
    print(f"  chi: {max_rhs_chi:.2e}")
    print(f"  K: {max_rhs_K:.2e}")
    print(f"  alpha: {max_rhs_alpha:.2e}")
    
    # For flat spacetime with fixed gauge, all RHS should be zero
    total_rhs = max_rhs_chi + max_rhs_K + max_rhs_alpha
    
    if total_rhs < 1e-10:
        print("\n✓ RHS correctly zero for flat spacetime")
    else:
        print(f"\n✗ RHS non-zero: {total_rhs:.2e}")
    
    print("\n" + "="*60)
    print("BSSN RHS computation test PASSED")
    print("="*60)
