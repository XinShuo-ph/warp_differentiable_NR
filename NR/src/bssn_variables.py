"""
BSSN Variable Definitions for Warp

Variables evolved:
- chi: exp(-4*phi), conformal factor (W formulation also common)
- gamma_tilde_ij: conformal 3-metric (6 components, symmetric)
- K: trace of extrinsic curvature
- A_tilde_ij: traceless conformal extrinsic curvature (6 components, symmetric)
- Gamma_tilde^i: contracted Christoffel symbols (3 components)
- alpha: lapse function
- beta^i: shift vector (3 components)
- B^i: auxiliary shift variable for Gamma-driver (3 components)
"""

import warp as wp
import numpy as np

wp.init()


@wp.struct
class BSSNState:
    """BSSN state variables stored as 3D arrays"""
    # Conformal factor (chi = exp(-4*phi))
    chi: wp.array3d(dtype=float)
    
    # Conformal metric (6 independent components)
    gamma_xx: wp.array3d(dtype=float)
    gamma_xy: wp.array3d(dtype=float)
    gamma_xz: wp.array3d(dtype=float)
    gamma_yy: wp.array3d(dtype=float)
    gamma_yz: wp.array3d(dtype=float)
    gamma_zz: wp.array3d(dtype=float)
    
    # Trace of extrinsic curvature
    K: wp.array3d(dtype=float)
    
    # Traceless conformal extrinsic curvature (6 components)
    A_xx: wp.array3d(dtype=float)
    A_xy: wp.array3d(dtype=float)
    A_xz: wp.array3d(dtype=float)
    A_yy: wp.array3d(dtype=float)
    A_yz: wp.array3d(dtype=float)
    A_zz: wp.array3d(dtype=float)
    
    # Contracted Christoffel symbols
    Gamma_x: wp.array3d(dtype=float)
    Gamma_y: wp.array3d(dtype=float)
    Gamma_z: wp.array3d(dtype=float)
    
    # Lapse
    alpha: wp.array3d(dtype=float)
    
    # Shift
    beta_x: wp.array3d(dtype=float)
    beta_y: wp.array3d(dtype=float)
    beta_z: wp.array3d(dtype=float)
    
    # Auxiliary shift (for Gamma-driver)
    B_x: wp.array3d(dtype=float)
    B_y: wp.array3d(dtype=float)
    B_z: wp.array3d(dtype=float)


class BSSNFields:
    """Container for BSSN field arrays with helper methods"""
    
    def __init__(self, nx: int, ny: int, nz: int, 
                 dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
                 device: str = "cpu"):
        """
        Initialize BSSN field arrays.
        
        Args:
            nx, ny, nz: Grid dimensions (including ghost zones)
            dx, dy, dz: Grid spacing
            device: Warp device ("cpu" or "cuda:N")
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.device = device
        
        # Total number of evolved fields
        # chi(1) + gamma_tilde(6) + K(1) + A_tilde(6) + Gamma_tilde(3) 
        # + alpha(1) + beta(3) + B(3) = 24 fields
        self.num_fields = 24
        
        # Create arrays for current state
        self._init_arrays()
        
        # Set flat spacetime initial data
        self.set_flat_spacetime()
    
    def _init_arrays(self):
        """Initialize all field arrays"""
        shape = (self.nx, self.ny, self.nz)
        
        # Conformal factor
        self.chi = wp.zeros(shape, dtype=float, device=self.device)
        
        # Conformal metric
        self.gamma_xx = wp.zeros(shape, dtype=float, device=self.device)
        self.gamma_xy = wp.zeros(shape, dtype=float, device=self.device)
        self.gamma_xz = wp.zeros(shape, dtype=float, device=self.device)
        self.gamma_yy = wp.zeros(shape, dtype=float, device=self.device)
        self.gamma_yz = wp.zeros(shape, dtype=float, device=self.device)
        self.gamma_zz = wp.zeros(shape, dtype=float, device=self.device)
        
        # Extrinsic curvature
        self.K = wp.zeros(shape, dtype=float, device=self.device)
        
        # Traceless extrinsic curvature
        self.A_xx = wp.zeros(shape, dtype=float, device=self.device)
        self.A_xy = wp.zeros(shape, dtype=float, device=self.device)
        self.A_xz = wp.zeros(shape, dtype=float, device=self.device)
        self.A_yy = wp.zeros(shape, dtype=float, device=self.device)
        self.A_yz = wp.zeros(shape, dtype=float, device=self.device)
        self.A_zz = wp.zeros(shape, dtype=float, device=self.device)
        
        # Contracted Christoffels
        self.Gamma_x = wp.zeros(shape, dtype=float, device=self.device)
        self.Gamma_y = wp.zeros(shape, dtype=float, device=self.device)
        self.Gamma_z = wp.zeros(shape, dtype=float, device=self.device)
        
        # Gauge
        self.alpha = wp.zeros(shape, dtype=float, device=self.device)
        self.beta_x = wp.zeros(shape, dtype=float, device=self.device)
        self.beta_y = wp.zeros(shape, dtype=float, device=self.device)
        self.beta_z = wp.zeros(shape, dtype=float, device=self.device)
        self.B_x = wp.zeros(shape, dtype=float, device=self.device)
        self.B_y = wp.zeros(shape, dtype=float, device=self.device)
        self.B_z = wp.zeros(shape, dtype=float, device=self.device)
    
    def set_flat_spacetime(self):
        """Initialize to flat (Minkowski) spacetime"""
        # chi = 1 (psi = 1)
        self.chi.fill_(1.0)
        
        # gamma_tilde_ij = delta_ij
        self.gamma_xx.fill_(1.0)
        self.gamma_yy.fill_(1.0)
        self.gamma_zz.fill_(1.0)
        self.gamma_xy.fill_(0.0)
        self.gamma_xz.fill_(0.0)
        self.gamma_yz.fill_(0.0)
        
        # K = 0, A_ij = 0
        self.K.fill_(0.0)
        self.A_xx.fill_(0.0)
        self.A_xy.fill_(0.0)
        self.A_xz.fill_(0.0)
        self.A_yy.fill_(0.0)
        self.A_yz.fill_(0.0)
        self.A_zz.fill_(0.0)
        
        # Gamma^i = 0 for flat space
        self.Gamma_x.fill_(0.0)
        self.Gamma_y.fill_(0.0)
        self.Gamma_z.fill_(0.0)
        
        # alpha = 1, beta^i = 0
        self.alpha.fill_(1.0)
        self.beta_x.fill_(0.0)
        self.beta_y.fill_(0.0)
        self.beta_z.fill_(0.0)
        self.B_x.fill_(0.0)
        self.B_y.fill_(0.0)
        self.B_z.fill_(0.0)
    
    def copy_from(self, other: 'BSSNFields'):
        """Copy all fields from another BSSNFields instance"""
        wp.copy(self.chi, other.chi)
        wp.copy(self.gamma_xx, other.gamma_xx)
        wp.copy(self.gamma_xy, other.gamma_xy)
        wp.copy(self.gamma_xz, other.gamma_xz)
        wp.copy(self.gamma_yy, other.gamma_yy)
        wp.copy(self.gamma_yz, other.gamma_yz)
        wp.copy(self.gamma_zz, other.gamma_zz)
        wp.copy(self.K, other.K)
        wp.copy(self.A_xx, other.A_xx)
        wp.copy(self.A_xy, other.A_xy)
        wp.copy(self.A_xz, other.A_xz)
        wp.copy(self.A_yy, other.A_yy)
        wp.copy(self.A_yz, other.A_yz)
        wp.copy(self.A_zz, other.A_zz)
        wp.copy(self.Gamma_x, other.Gamma_x)
        wp.copy(self.Gamma_y, other.Gamma_y)
        wp.copy(self.Gamma_z, other.Gamma_z)
        wp.copy(self.alpha, other.alpha)
        wp.copy(self.beta_x, other.beta_x)
        wp.copy(self.beta_y, other.beta_y)
        wp.copy(self.beta_z, other.beta_z)
        wp.copy(self.B_x, other.B_x)
        wp.copy(self.B_y, other.B_y)
        wp.copy(self.B_z, other.B_z)
    
    def get_all_fields_numpy(self) -> dict:
        """Return all fields as numpy arrays for inspection"""
        return {
            'chi': self.chi.numpy(),
            'gamma_xx': self.gamma_xx.numpy(),
            'gamma_xy': self.gamma_xy.numpy(),
            'gamma_xz': self.gamma_xz.numpy(),
            'gamma_yy': self.gamma_yy.numpy(),
            'gamma_yz': self.gamma_yz.numpy(),
            'gamma_zz': self.gamma_zz.numpy(),
            'K': self.K.numpy(),
            'A_xx': self.A_xx.numpy(),
            'A_xy': self.A_xy.numpy(),
            'A_xz': self.A_xz.numpy(),
            'A_yy': self.A_yy.numpy(),
            'A_yz': self.A_yz.numpy(),
            'A_zz': self.A_zz.numpy(),
            'Gamma_x': self.Gamma_x.numpy(),
            'Gamma_y': self.Gamma_y.numpy(),
            'Gamma_z': self.Gamma_z.numpy(),
            'alpha': self.alpha.numpy(),
            'beta_x': self.beta_x.numpy(),
            'beta_y': self.beta_y.numpy(),
            'beta_z': self.beta_z.numpy(),
            'B_x': self.B_x.numpy(),
            'B_y': self.B_y.numpy(),
            'B_z': self.B_z.numpy(),
        }


def test_bssn_variables():
    """Test BSSN variable initialization"""
    print("Testing BSSN variable initialization...")
    
    # Create a small test grid (8^3 with 3 ghost zones each side)
    ng = 3  # ghost zones
    n_interior = 8
    n_total = n_interior + 2 * ng
    
    fields = BSSNFields(n_total, n_total, n_total, dx=0.1, dy=0.1, dz=0.1)
    
    # Verify flat spacetime values
    all_fields = fields.get_all_fields_numpy()
    
    assert np.allclose(all_fields['chi'], 1.0), "chi should be 1"
    assert np.allclose(all_fields['gamma_xx'], 1.0), "gamma_xx should be 1"
    assert np.allclose(all_fields['gamma_yy'], 1.0), "gamma_yy should be 1"
    assert np.allclose(all_fields['gamma_zz'], 1.0), "gamma_zz should be 1"
    assert np.allclose(all_fields['gamma_xy'], 0.0), "gamma_xy should be 0"
    assert np.allclose(all_fields['alpha'], 1.0), "alpha should be 1"
    assert np.allclose(all_fields['K'], 0.0), "K should be 0"
    
    print(f"Grid shape: {all_fields['chi'].shape}")
    print(f"chi: {all_fields['chi'][ng, ng, ng]}")
    print(f"gamma_xx: {all_fields['gamma_xx'][ng, ng, ng]}")
    print(f"alpha: {all_fields['alpha'][ng, ng, ng]}")
    
    print("BSSN variables test PASSED!")
    return fields


if __name__ == "__main__":
    wp.set_module_options({"enable_backward": True})  # Enable for autodiff
    test_bssn_variables()
