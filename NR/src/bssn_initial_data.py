"""
BSSN Initial Data: Brill-Lindquist and Puncture Data

Brill-Lindquist (time-symmetric, single BH):
  psi = 1 + M/(2*r)     # conformal factor
  chi = 1/psi^4
  gamma_tilde_ij = delta_ij
  K = 0
  A_tilde_ij = 0

For BSSN with chi formulation:
  chi -> 0 at puncture (regularized)
"""

import warp as wp
import numpy as np

wp.init()


@wp.kernel
def set_brill_lindquist_kernel(
    chi: wp.array3d(dtype=float),
    gamma_xx: wp.array3d(dtype=float),
    gamma_xy: wp.array3d(dtype=float),
    gamma_xz: wp.array3d(dtype=float),
    gamma_yy: wp.array3d(dtype=float),
    gamma_yz: wp.array3d(dtype=float),
    gamma_zz: wp.array3d(dtype=float),
    K: wp.array3d(dtype=float),
    A_xx: wp.array3d(dtype=float),
    A_xy: wp.array3d(dtype=float),
    A_xz: wp.array3d(dtype=float),
    A_yy: wp.array3d(dtype=float),
    A_yz: wp.array3d(dtype=float),
    A_zz: wp.array3d(dtype=float),
    Gamma_x: wp.array3d(dtype=float),
    Gamma_y: wp.array3d(dtype=float),
    Gamma_z: wp.array3d(dtype=float),
    alpha: wp.array3d(dtype=float),
    beta_x: wp.array3d(dtype=float),
    beta_y: wp.array3d(dtype=float),
    beta_z: wp.array3d(dtype=float),
    B_x: wp.array3d(dtype=float),
    B_y: wp.array3d(dtype=float),
    B_z: wp.array3d(dtype=float),
    # Grid parameters
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float,
    x_min: float, y_min: float, z_min: float,
    # Puncture parameters
    M: float,  # BH mass
    x_p: float, y_p: float, z_p: float  # puncture position
):
    """Set Brill-Lindquist initial data for single puncture"""
    i, j, k = wp.tid()
    
    if i >= nx or j >= ny or k >= nz:
        return
    
    # Grid coordinates
    x = x_min + float(i) * dx
    y = y_min + float(j) * dy
    z = z_min + float(k) * dz
    
    # Distance from puncture (with small regularization)
    rx = x - x_p
    ry = y - y_p
    rz = z - z_p
    r = wp.sqrt(rx*rx + ry*ry + rz*rz + 1.0e-10)
    
    # Conformal factor: psi = 1 + M/(2*r)
    psi = 1.0 + M / (2.0 * r)
    
    # chi = 1/psi^4
    chi_val = 1.0 / (psi * psi * psi * psi)
    chi[i, j, k] = chi_val
    
    # Conformal metric = flat
    gamma_xx[i, j, k] = 1.0
    gamma_yy[i, j, k] = 1.0
    gamma_zz[i, j, k] = 1.0
    gamma_xy[i, j, k] = 0.0
    gamma_xz[i, j, k] = 0.0
    gamma_yz[i, j, k] = 0.0
    
    # Time-symmetric: K = 0, A_ij = 0
    K[i, j, k] = 0.0
    A_xx[i, j, k] = 0.0
    A_xy[i, j, k] = 0.0
    A_xz[i, j, k] = 0.0
    A_yy[i, j, k] = 0.0
    A_yz[i, j, k] = 0.0
    A_zz[i, j, k] = 0.0
    
    # Gamma^i: for conformally flat, Gamma^i = -d_j(gamma^ij) = 0
    Gamma_x[i, j, k] = 0.0
    Gamma_y[i, j, k] = 0.0
    Gamma_z[i, j, k] = 0.0
    
    # Pre-collapsed lapse: alpha = psi^(-2) or alpha = chi^(1/2)
    # This avoids slice hitting singularity
    alpha[i, j, k] = wp.sqrt(chi_val)
    
    # Zero shift initially
    beta_x[i, j, k] = 0.0
    beta_y[i, j, k] = 0.0
    beta_z[i, j, k] = 0.0
    B_x[i, j, k] = 0.0
    B_y[i, j, k] = 0.0
    B_z[i, j, k] = 0.0


class BrillLindquistData:
    """Container for Brill-Lindquist initial data"""
    
    def __init__(self, nx: int, ny: int, nz: int,
                 x_min: float, x_max: float,
                 y_min: float, y_max: float,
                 z_min: float, z_max: float,
                 M: float = 1.0,
                 x_p: float = 0.0, y_p: float = 0.0, z_p: float = 0.0):
        """
        Initialize Brill-Lindquist data.
        
        Args:
            nx, ny, nz: Grid dimensions
            x_min, x_max, etc.: Domain bounds
            M: Black hole mass
            x_p, y_p, z_p: Puncture position
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        self.dx = (x_max - x_min) / (nx - 1) if nx > 1 else 1.0
        self.dy = (y_max - y_min) / (ny - 1) if ny > 1 else 1.0
        self.dz = (z_max - z_min) / (nz - 1) if nz > 1 else 1.0
        
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        
        self.M = M
        self.x_p = x_p
        self.y_p = y_p
        self.z_p = z_p
        
        shape = (nx, ny, nz)
        
        # Allocate fields
        self.chi = wp.zeros(shape, dtype=float)
        self.gamma_xx = wp.zeros(shape, dtype=float)
        self.gamma_xy = wp.zeros(shape, dtype=float)
        self.gamma_xz = wp.zeros(shape, dtype=float)
        self.gamma_yy = wp.zeros(shape, dtype=float)
        self.gamma_yz = wp.zeros(shape, dtype=float)
        self.gamma_zz = wp.zeros(shape, dtype=float)
        self.K = wp.zeros(shape, dtype=float)
        self.A_xx = wp.zeros(shape, dtype=float)
        self.A_xy = wp.zeros(shape, dtype=float)
        self.A_xz = wp.zeros(shape, dtype=float)
        self.A_yy = wp.zeros(shape, dtype=float)
        self.A_yz = wp.zeros(shape, dtype=float)
        self.A_zz = wp.zeros(shape, dtype=float)
        self.Gamma_x = wp.zeros(shape, dtype=float)
        self.Gamma_y = wp.zeros(shape, dtype=float)
        self.Gamma_z = wp.zeros(shape, dtype=float)
        self.alpha = wp.zeros(shape, dtype=float)
        self.beta_x = wp.zeros(shape, dtype=float)
        self.beta_y = wp.zeros(shape, dtype=float)
        self.beta_z = wp.zeros(shape, dtype=float)
        self.B_x = wp.zeros(shape, dtype=float)
        self.B_y = wp.zeros(shape, dtype=float)
        self.B_z = wp.zeros(shape, dtype=float)
        
        # Set initial data
        self._set_initial_data()
    
    def _set_initial_data(self):
        """Apply Brill-Lindquist initial data"""
        wp.launch(
            set_brill_lindquist_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self.chi, self.gamma_xx, self.gamma_xy, self.gamma_xz,
                self.gamma_yy, self.gamma_yz, self.gamma_zz,
                self.K, self.A_xx, self.A_xy, self.A_xz,
                self.A_yy, self.A_yz, self.A_zz,
                self.Gamma_x, self.Gamma_y, self.Gamma_z,
                self.alpha, self.beta_x, self.beta_y, self.beta_z,
                self.B_x, self.B_y, self.B_z,
                self.nx, self.ny, self.nz,
                self.dx, self.dy, self.dz,
                self.x_min, self.y_min, self.z_min,
                self.M, self.x_p, self.y_p, self.z_p
            ]
        )
    
    def get_chi_slice(self, k_idx: int = None):
        """Get chi on z=0 slice for visualization"""
        if k_idx is None:
            k_idx = self.nz // 2
        return self.chi.numpy()[:, :, k_idx]
    
    def get_alpha_slice(self, k_idx: int = None):
        """Get alpha on z=0 slice for visualization"""
        if k_idx is None:
            k_idx = self.nz // 2
        return self.alpha.numpy()[:, :, k_idx]


def test_brill_lindquist():
    """Test Brill-Lindquist initial data"""
    print("Testing Brill-Lindquist initial data...")
    
    # Grid setup
    n = 64
    L = 10.0  # Domain: [-L, L]^3
    M = 1.0   # BH mass
    
    data = BrillLindquistData(
        nx=n, ny=n, nz=n,
        x_min=-L, x_max=L,
        y_min=-L, y_max=L,
        z_min=-L, z_max=L,
        M=M,
        x_p=0.0, y_p=0.0, z_p=0.0
    )
    
    # Check values at different radii
    chi_np = data.chi.numpy()
    alpha_np = data.alpha.numpy()
    
    # Center index
    ic = n // 2
    
    # At center (puncture): chi should be very small, alpha should be small
    print(f"At puncture (center):")
    print(f"  chi = {chi_np[ic, ic, ic]:.6f}")
    print(f"  alpha = {alpha_np[ic, ic, ic]:.6f}")
    
    # At r = 5M (away from puncture): should approach flat space
    # Index offset for r = 5
    dx = 2.0 * L / (n - 1)
    offset = int(5.0 / dx)
    
    print(f"At r ≈ 5M:")
    print(f"  chi = {chi_np[ic + offset, ic, ic]:.6f}")
    print(f"  alpha = {alpha_np[ic + offset, ic, ic]:.6f}")
    
    # Check expected values
    # At r = 5, psi = 1 + 1/(2*5) = 1.1, chi = 1/1.1^4 ≈ 0.683
    r_test = 5.0
    psi_expected = 1.0 + M / (2.0 * r_test)
    chi_expected = 1.0 / psi_expected**4
    print(f"  Expected chi at r=5: {chi_expected:.6f}")
    
    # Verify conformal metric is flat
    gamma_xx_np = data.gamma_xx.numpy()
    gamma_xy_np = data.gamma_xy.numpy()
    
    assert np.allclose(gamma_xx_np, 1.0), "gamma_xx should be 1"
    assert np.allclose(gamma_xy_np, 0.0), "gamma_xy should be 0"
    
    # Verify K = 0
    K_np = data.K.numpy()
    assert np.allclose(K_np, 0.0), "K should be 0"
    
    print("Brill-Lindquist initial data test PASSED!")
    
    return data


if __name__ == "__main__":
    wp.set_module_options({"enable_backward": True})
    test_brill_lindquist()
