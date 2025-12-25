"""
BSSN Constraint Monitors

Hamiltonian constraint: H = R + K^2 - K_ij*K^ij - 16*pi*rho = 0
Momentum constraint: M^i = D_j(K^ij - gamma^ij*K) - 8*pi*S^i = 0

In vacuum (rho=0, S^i=0):
H = R + K^2 - A_ij*A^ij - 2/3*K^2 = R + K^2/3 - A_ij*A^ij
"""

import warp as wp
import numpy as np
from bssn_derivatives import d1_x, d1_y, d1_z, d2_xx, d2_yy, d2_zz

wp.init()


@wp.kernel
def compute_hamiltonian_constraint_kernel(
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
    H_out: wp.array3d(dtype=float),
    nx: int, ny: int, nz: int,
    inv_12h: float,
    inv_12h2: float
):
    """
    Compute Hamiltonian constraint.
    Simplified: H = K^2/3 - A_ij*A^ij (ignoring Ricci scalar for now)
    """
    i, j, k = wp.tid()
    
    if i < 2 or i >= nx - 2:
        return
    if j < 2 or j >= ny - 2:
        return
    if k < 2 or k >= nz - 2:
        return
    
    K_ijk = K[i, j, k]
    
    # A_ij * A^ij (flat metric approximation)
    A_sq = (
        A_xx[i, j, k] * A_xx[i, j, k]
        + A_yy[i, j, k] * A_yy[i, j, k]
        + A_zz[i, j, k] * A_zz[i, j, k]
        + 2.0 * (A_xy[i, j, k] * A_xy[i, j, k]
                + A_xz[i, j, k] * A_xz[i, j, k]
                + A_yz[i, j, k] * A_yz[i, j, k])
    )
    
    # Simplified Hamiltonian (without Ricci tensor computation)
    H_out[i, j, k] = K_ijk * K_ijk / 3.0 - A_sq


@wp.kernel
def compute_constraint_norms_kernel(
    H: wp.array3d(dtype=float),
    L2_norm: wp.array(dtype=float),
    Linf_norm: wp.array(dtype=float),
    count: wp.array(dtype=int),
    nx: int, ny: int, nz: int,
    ng: int
):
    """Compute L2 and Linf norms of constraint"""
    i, j, k = wp.tid()
    
    if i < ng or i >= nx - ng:
        return
    if j < ng or j >= ny - ng:
        return
    if k < ng or k >= nz - ng:
        return
    
    h_val = H[i, j, k]
    h_sq = h_val * h_val
    h_abs = wp.abs(h_val)
    
    wp.atomic_add(L2_norm, 0, h_sq)
    wp.atomic_max(Linf_norm, 0, h_abs)
    wp.atomic_add(count, 0, 1)


class ConstraintMonitor:
    """Monitor BSSN constraints during evolution"""
    
    def __init__(self, nx, ny, nz, dx, ng=3):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.ng = ng
        
        self.inv_12h = 1.0 / (12.0 * dx)
        self.inv_12h2 = 1.0 / (12.0 * dx * dx)
        
        shape = (nx, ny, nz)
        self.H = wp.zeros(shape, dtype=float)
        
        self.L2_norm = wp.zeros(1, dtype=float)
        self.Linf_norm = wp.zeros(1, dtype=float)
        self.count = wp.zeros(1, dtype=wp.int32)
    
    def compute_hamiltonian(self, chi, gamma_xx, gamma_xy, gamma_xz,
                            gamma_yy, gamma_yz, gamma_zz,
                            K, A_xx, A_xy, A_xz, A_yy, A_yz, A_zz):
        """Compute Hamiltonian constraint field"""
        wp.launch(
            compute_hamiltonian_constraint_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                chi, gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz,
                K, A_xx, A_xy, A_xz, A_yy, A_yz, A_zz,
                self.H, self.nx, self.ny, self.nz, self.inv_12h, self.inv_12h2
            ]
        )
    
    def compute_norms(self):
        """Compute L2 and Linf norms of Hamiltonian constraint"""
        self.L2_norm.zero_()
        self.Linf_norm.zero_()
        self.count.zero_()
        
        wp.launch(
            compute_constraint_norms_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self.H, self.L2_norm, self.Linf_norm, self.count,
                self.nx, self.ny, self.nz, self.ng
            ]
        )
        
        count = self.count.numpy()[0]
        L2 = np.sqrt(self.L2_norm.numpy()[0] / max(count, 1))
        Linf = self.Linf_norm.numpy()[0]
        
        return L2, Linf


def test_constraint_monitor():
    """Test constraint monitor on flat and puncture data"""
    print("Testing constraint monitor...")
    
    from bssn_initial_data import BrillLindquistData
    
    # Test on flat spacetime
    n = 32
    L = 10.0
    dx = 2.0 * L / (n - 1)
    
    shape = (n, n, n)
    chi = wp.zeros(shape, dtype=float); chi.fill_(1.0)
    gamma_xx = wp.zeros(shape, dtype=float); gamma_xx.fill_(1.0)
    gamma_xy = wp.zeros(shape, dtype=float)
    gamma_xz = wp.zeros(shape, dtype=float)
    gamma_yy = wp.zeros(shape, dtype=float); gamma_yy.fill_(1.0)
    gamma_yz = wp.zeros(shape, dtype=float)
    gamma_zz = wp.zeros(shape, dtype=float); gamma_zz.fill_(1.0)
    K = wp.zeros(shape, dtype=float)
    A_xx = wp.zeros(shape, dtype=float)
    A_xy = wp.zeros(shape, dtype=float)
    A_xz = wp.zeros(shape, dtype=float)
    A_yy = wp.zeros(shape, dtype=float)
    A_yz = wp.zeros(shape, dtype=float)
    A_zz = wp.zeros(shape, dtype=float)
    
    monitor = ConstraintMonitor(n, n, n, dx)
    
    monitor.compute_hamiltonian(
        chi, gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz,
        K, A_xx, A_xy, A_xz, A_yy, A_yz, A_zz
    )
    
    L2, Linf = monitor.compute_norms()
    print(f"Flat spacetime: H_L2 = {L2:.6e}, H_Linf = {Linf:.6e}")
    
    # Test on puncture data
    data = BrillLindquistData(
        nx=n, ny=n, nz=n,
        x_min=-L, x_max=L,
        y_min=-L, y_max=L,
        z_min=-L, z_max=L,
        M=1.0
    )
    
    monitor.compute_hamiltonian(
        data.chi, data.gamma_xx, data.gamma_xy, data.gamma_xz,
        data.gamma_yy, data.gamma_yz, data.gamma_zz,
        data.K, data.A_xx, data.A_xy, data.A_xz,
        data.A_yy, data.A_yz, data.A_zz
    )
    
    L2, Linf = monitor.compute_norms()
    print(f"Puncture data: H_L2 = {L2:.6e}, H_Linf = {Linf:.6e}")
    
    print("Constraint monitor test complete!")


if __name__ == "__main__":
    wp.set_module_options({"enable_backward": True})
    test_constraint_monitor()
