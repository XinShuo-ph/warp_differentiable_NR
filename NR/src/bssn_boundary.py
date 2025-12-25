"""
Sommerfeld (Radiative) Boundary Conditions for BSSN

Assumes outgoing waves at the boundary:
(d_t + d_r + (f - f0)/r) = 0

Implemented as extrapolation with falloff correction.
"""

import warp as wp
import numpy as np

wp.init()


@wp.kernel
def apply_sommerfeld_bc_kernel(
    f: wp.array3d(dtype=float),
    f0: float,  # asymptotic value
    falloff: float,  # 1/r^n falloff power
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float,
    x_min: float, y_min: float, z_min: float,
    x_c: float, y_c: float, z_c: float  # center for radial distance
):
    """Apply Sommerfeld-like BC via extrapolation at boundaries"""
    i, j, k = wp.tid()
    
    # Only apply at boundary ghost zones
    is_boundary = False
    
    if i < 3 or i >= nx - 3:
        is_boundary = True
    if j < 3 or j >= ny - 3:
        is_boundary = True
    if k < 3 or k >= nz - 3:
        is_boundary = True
    
    if not is_boundary:
        return
    
    # Grid position
    x = x_min + float(i) * dx
    y = y_min + float(j) * dy
    z = z_min + float(k) * dz
    
    # Distance from center
    rx = x - x_c
    ry = y - y_c
    rz = z - z_c
    r = wp.sqrt(rx*rx + ry*ry + rz*rz + 1.0e-10)
    
    # Simple extrapolation with falloff
    # f -> f0 + (f_interior - f0) * (r_interior/r)^falloff
    
    # Find interior point (clamp indices)
    ii = wp.clamp(i, 3, nx - 4)
    jj = wp.clamp(j, 3, ny - 4)
    kk = wp.clamp(k, 3, nz - 4)
    
    f_int = f[ii, jj, kk]
    
    # Interior position
    x_int = x_min + float(ii) * dx
    y_int = y_min + float(jj) * dy
    z_int = z_min + float(kk) * dz
    
    rx_int = x_int - x_c
    ry_int = y_int - y_c
    rz_int = z_int - z_c
    r_int = wp.sqrt(rx_int*rx_int + ry_int*ry_int + rz_int*rz_int + 1.0e-10)
    
    # Extrapolate
    if falloff > 0.0:
        ratio = wp.pow(r_int / r, falloff)
        f[i, j, k] = f0 + (f_int - f0) * ratio
    else:
        f[i, j, k] = f_int


def apply_sommerfeld_bc(field, f0, falloff, nx, ny, nz, dx, dy, dz, x_min, y_min, z_min, x_c=0.0, y_c=0.0, z_c=0.0):
    """Apply Sommerfeld BC to a field"""
    wp.launch(
        apply_sommerfeld_bc_kernel,
        dim=(nx, ny, nz),
        inputs=[field, f0, falloff, nx, ny, nz, dx, dy, dz, x_min, y_min, z_min, x_c, y_c, z_c]
    )


class BSSNBoundaryConditions:
    """Apply appropriate BCs to all BSSN fields"""
    
    def __init__(self, nx, ny, nz, dx, dy, dz, x_min, y_min, z_min):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        
        # Center (usually origin)
        self.x_c = 0.0
        self.y_c = 0.0
        self.z_c = 0.0
    
    def apply(self, chi, gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz,
              K, A_xx, A_xy, A_xz, A_yy, A_yz, A_zz,
              Gamma_x, Gamma_y, Gamma_z,
              alpha, beta_x, beta_y, beta_z, B_x, B_y, B_z):
        """Apply Sommerfeld BCs to all fields"""
        
        n = self.nx
        d = self.dx
        xm = self.x_min
        ym = self.y_min
        zm = self.z_min
        xc = self.x_c
        yc = self.y_c
        zc = self.z_c
        
        # chi: asymptotes to 1, falloff ~1/r^4
        apply_sommerfeld_bc(chi, 1.0, 4.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        
        # gamma_tilde_ij: asymptotes to delta_ij
        apply_sommerfeld_bc(gamma_xx, 1.0, 2.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(gamma_xy, 0.0, 2.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(gamma_xz, 0.0, 2.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(gamma_yy, 1.0, 2.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(gamma_yz, 0.0, 2.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(gamma_zz, 1.0, 2.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        
        # K, A_ij: asymptote to 0
        apply_sommerfeld_bc(K, 0.0, 2.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(A_xx, 0.0, 3.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(A_xy, 0.0, 3.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(A_xz, 0.0, 3.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(A_yy, 0.0, 3.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(A_yz, 0.0, 3.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(A_zz, 0.0, 3.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        
        # Gamma^i: asymptote to 0
        apply_sommerfeld_bc(Gamma_x, 0.0, 2.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(Gamma_y, 0.0, 2.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(Gamma_z, 0.0, 2.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        
        # alpha: asymptotes to 1
        apply_sommerfeld_bc(alpha, 1.0, 1.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        
        # beta, B: asymptote to 0
        apply_sommerfeld_bc(beta_x, 0.0, 1.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(beta_y, 0.0, 1.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(beta_z, 0.0, 1.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(B_x, 0.0, 1.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(B_y, 0.0, 1.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)
        apply_sommerfeld_bc(B_z, 0.0, 1.0, n, n, n, d, d, d, xm, ym, zm, xc, yc, zc)


def test_sommerfeld_bc():
    """Test Sommerfeld BC on a simple field"""
    print("Testing Sommerfeld boundary conditions...")
    
    n = 32
    L = 10.0
    dx = 2.0 * L / (n - 1)
    
    shape = (n, n, n)
    f = wp.zeros(shape, dtype=float)
    
    # Set interior to some value
    f_np = np.ones((n, n, n))
    f_np[n//2, n//2, n//2] = 2.0  # peak at center
    wp.copy(f, wp.array(f_np.astype(np.float32), dtype=float))
    
    # Apply BC
    apply_sommerfeld_bc(f, 1.0, 2.0, n, n, n, dx, dx, dx, -L, -L, -L, 0.0, 0.0, 0.0)
    
    f_result = f.numpy()
    
    # Check that boundary values are close to asymptotic
    print(f"Corner value (should be ~1.0): {f_result[0, 0, 0]:.4f}")
    print(f"Center value (should be ~2.0): {f_result[n//2, n//2, n//2]:.4f}")
    
    print("Sommerfeld BC test complete!")


if __name__ == "__main__":
    wp.set_module_options({"enable_backward": True})
    test_sommerfeld_bc()
