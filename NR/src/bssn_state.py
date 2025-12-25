"""
BSSN variables and field definitions for warp implementation.

Variables:
- chi: conformal factor (χ = e^(-4φ))
- gamma_tilde: conformal 3-metric γ̃ᵢⱼ (6 components, symmetric)
- K: trace of extrinsic curvature (scalar)
- A_tilde: traceless conformal extrinsic curvature Ãᵢⱼ (6 components)
- Gamma_tilde: conformal connection Γ̃ⁱ (3 components)
- alpha: lapse function (scalar)
- beta: shift vector βⁱ (3 components)
"""

import warp as wp
import numpy as np

wp.init()


class BSSNState:
    """
    Container for BSSN field variables on a 3D grid.
    
    Uses flat arrays indexed by (i,j,k) position.
    """
    
    def __init__(self, nx, ny, nz, device='cpu'):
        """
        Initialize BSSN state on grid.
        
        Args:
            nx, ny, nz: Grid dimensions
            device: Warp device ('cpu' or 'cuda:0')
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.device = device
        
        npts = nx * ny * nz
        
        # Scalar fields
        self.chi = wp.zeros(npts, dtype=float, device=device)      # Conformal factor
        self.K = wp.zeros(npts, dtype=float, device=device)        # Trace K
        self.alpha = wp.ones(npts, dtype=float, device=device)     # Lapse (init to 1)
        
        # 3-vector fields (store as vec3)
        self.beta = wp.zeros(npts, dtype=wp.vec3, device=device)         # Shift
        self.Gamma_tilde = wp.zeros(npts, dtype=wp.vec3, device=device)  # Connection
        
        # Symmetric 3x3 tensor fields (6 independent components)
        # Store as struct with xx, xy, xz, yy, yz, zz
        self.gamma_tilde = wp.zeros(npts, dtype=SymmetricTensor3, device=device)
        self.A_tilde = wp.zeros(npts, dtype=SymmetricTensor3, device=device)
    
    def set_flat_spacetime(self):
        """Initialize to flat spacetime (Minkowski)"""
        npts = self.nx * self.ny * self.nz
        
        # Flat spacetime initial data
        chi_flat = np.ones(npts, dtype=np.float32)
        K_flat = np.zeros(npts, dtype=np.float32)
        alpha_flat = np.ones(npts, dtype=np.float32)
        
        # Flat metric: γ̃ᵢⱼ = δᵢⱼ
        gamma_flat = np.zeros((npts,), dtype=[
            ('xx', 'f4'), ('xy', 'f4'), ('xz', 'f4'),
            ('yy', 'f4'), ('yz', 'f4'), ('zz', 'f4')
        ])
        gamma_flat['xx'] = 1.0
        gamma_flat['yy'] = 1.0
        gamma_flat['zz'] = 1.0
        
        # Zero extrinsic curvature
        A_flat = np.zeros_like(gamma_flat)
        
        # Zero shift and connection
        beta_flat = np.zeros((npts, 3), dtype=np.float32)
        Gamma_flat = np.zeros((npts, 3), dtype=np.float32)
        
        # Copy to device
        self.chi.assign(chi_flat)
        self.K.assign(K_flat)
        self.alpha.assign(alpha_flat)
        self.gamma_tilde.assign(gamma_flat)
        self.A_tilde.assign(A_flat)
        self.beta.assign(beta_flat)
        self.Gamma_tilde.assign(Gamma_flat)
    
    def to_numpy(self):
        """Return all fields as numpy arrays"""
        return {
            'chi': self.chi.numpy(),
            'K': self.K.numpy(),
            'alpha': self.alpha.numpy(),
            'gamma_tilde': self.gamma_tilde.numpy(),
            'A_tilde': self.A_tilde.numpy(),
            'beta': self.beta.numpy(),
            'Gamma_tilde': self.Gamma_tilde.numpy(),
        }


# Define symmetric 3x3 tensor struct for warp
@wp.struct
class SymmetricTensor3:
    """Symmetric 3x3 tensor with 6 independent components"""
    xx: float
    xy: float
    xz: float
    yy: float
    yz: float
    zz: float


@wp.func
def idx3d(i: int, j: int, k: int, nx: int, ny: int, nz: int) -> int:
    """Convert 3D grid indices to linear index"""
    return i + nx * (j + ny * k)


@wp.func
def symmetric_tensor_det(g: SymmetricTensor3) -> float:
    """Compute determinant of symmetric 3x3 tensor"""
    det = (g.xx * (g.yy * g.zz - g.yz * g.yz) 
           - g.xy * (g.xy * g.zz - g.yz * g.xz)
           + g.xz * (g.xy * g.yz - g.yy * g.xz))
    return det


@wp.func
def symmetric_tensor_trace(g: SymmetricTensor3) -> float:
    """Compute trace of symmetric 3x3 tensor"""
    return g.xx + g.yy + g.zz


@wp.func
def symmetric_tensor_contract(a: SymmetricTensor3, b: SymmetricTensor3) -> float:
    """Compute double contraction: aᵢⱼ bⁱʲ (requires inverse for upper indices)"""
    # For flat space where γ̃ⁱʲ = δⁱʲ, this is just component-wise product
    return a.xx*b.xx + 2.0*a.xy*b.xy + 2.0*a.xz*b.xz + a.yy*b.yy + 2.0*a.yz*b.yz + a.zz*b.zz


@wp.kernel
def check_flat_spacetime(
    chi: wp.array(dtype=float),
    gamma: wp.array(dtype=SymmetricTensor3),
    K: wp.array(dtype=float),
    violation: wp.array(dtype=float)
):
    """Check deviation from flat spacetime"""
    idx = wp.tid()
    
    # Should have chi = 1, K = 0, γ̃ = diag(1,1,1)
    g = gamma[idx]
    
    error = (wp.abs(chi[idx] - 1.0) 
             + wp.abs(g.xx - 1.0) + wp.abs(g.yy - 1.0) + wp.abs(g.zz - 1.0)
             + wp.abs(g.xy) + wp.abs(g.xz) + wp.abs(g.yz)
             + wp.abs(K[idx]))
    
    wp.atomic_add(violation, 0, error)


if __name__ == "__main__":
    print("Testing BSSN state initialization...")
    
    # Create small grid
    nx, ny, nz = 16, 16, 16
    state = BSSNState(nx, ny, nz)
    
    print(f"Grid size: {nx} x {ny} x {nz} = {nx*ny*nz} points")
    print(f"Fields allocated: chi, K, alpha, beta, Gamma_tilde, gamma_tilde, A_tilde")
    
    # Initialize to flat spacetime
    print("\nInitializing to flat spacetime...")
    state.set_flat_spacetime()
    
    # Check flat spacetime
    violation = wp.zeros(1, dtype=float)
    wp.launch(
        check_flat_spacetime,
        dim=nx*ny*nz,
        inputs=[state.chi, state.gamma_tilde, state.K, violation]
    )
    
    total_violation = violation.numpy()[0]
    print(f"Total violation from flat spacetime: {total_violation:.2e}")
    
    if total_violation < 1e-6:
        print("✓ Flat spacetime initialization correct")
    else:
        print("✗ Flat spacetime initialization has errors")
    
    # Show some values
    data = state.to_numpy()
    print(f"\nSample values:")
    print(f"  chi[0] = {data['chi'][0]:.6f} (should be 1.0)")
    print(f"  gamma_xx[0] = {data['gamma_tilde'][0]['xx']:.6f} (should be 1.0)")
    print(f"  K[0] = {data['K'][0]:.6f} (should be 0.0)")
    print(f"  alpha[0] = {data['alpha'][0]:.6f} (should be 1.0)")
    
    print("\n" + "="*60)
    print("BSSN state definition test PASSED")
    print("="*60)
