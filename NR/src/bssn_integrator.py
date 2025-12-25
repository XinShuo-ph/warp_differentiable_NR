"""
RK4 Time Integration for BSSN Evolution

Classic 4th-order Runge-Kutta:
k1 = dt * f(t, y)
k2 = dt * f(t + dt/2, y + k1/2)
k3 = dt * f(t + dt/2, y + k2/2)
k4 = dt * f(t + dt, y + k3)
y_new = y + (k1 + 2*k2 + 2*k3 + k4) / 6
"""

import warp as wp
import numpy as np
from bssn_derivatives import d1_x, d1_y, d1_z, d2_xx, d2_yy, d2_zz, ko_dissipation

wp.init()


@wp.kernel
def rk_update_kernel(
    y: wp.array3d(dtype=float),
    k: wp.array3d(dtype=float),
    y_out: wp.array3d(dtype=float),
    coeff: float,
    nx: int, ny: int, nz: int
):
    """y_out = y + coeff * k"""
    i, j, k_idx = wp.tid()
    
    if i >= nx or j >= ny or k_idx >= nz:
        return
    
    y_out[i, j, k_idx] = y[i, j, k_idx] + coeff * k[i, j, k_idx]


@wp.kernel
def rk4_combine_kernel(
    y: wp.array3d(dtype=float),
    k1: wp.array3d(dtype=float),
    k2: wp.array3d(dtype=float),
    k3: wp.array3d(dtype=float),
    k4: wp.array3d(dtype=float),
    y_out: wp.array3d(dtype=float),
    dt: float,
    nx: int, ny: int, nz: int
):
    """y_out = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6"""
    i, j, k_idx = wp.tid()
    
    if i >= nx or j >= ny or k_idx >= nz:
        return
    
    y_out[i, j, k_idx] = y[i, j, k_idx] + dt * (
        k1[i, j, k_idx] + 2.0*k2[i, j, k_idx] + 2.0*k3[i, j, k_idx] + k4[i, j, k_idx]
    ) / 6.0


# Simplified BSSN RHS for testing (just chi and K)
@wp.kernel
def compute_simple_rhs(
    chi: wp.array3d(dtype=float),
    K_in: wp.array3d(dtype=float),
    alpha: wp.array3d(dtype=float),
    rhs_chi: wp.array3d(dtype=float),
    rhs_K: wp.array3d(dtype=float),
    nx: int, ny: int, nz: int,
    inv_12h: float,
    inv_12h2: float,
    sigma_ko: float,
    dt: float
):
    """Simplified RHS for chi and K (no shift, flat metric)"""
    i, j, k = wp.tid()
    
    if i < 3 or i >= nx - 3:
        return
    if j < 3 or j >= ny - 3:
        return
    if k < 3 or k >= nz - 3:
        return
    
    chi_ijk = chi[i, j, k]
    alpha_ijk = alpha[i, j, k]
    K_ijk = K_in[i, j, k]
    
    # RHS chi = (2/3)*chi*alpha*K (no shift)
    rhs_chi_val = (2.0/3.0) * chi_ijk * alpha_ijk * K_ijk
    rhs_chi_val = rhs_chi_val - ko_dissipation(chi, i, j, k, sigma_ko / dt)
    rhs_chi[i, j, k] = rhs_chi_val
    
    # RHS K = -D^2(alpha) + alpha*K^2/3 (simplified, no A terms)
    d2alpha_xx = d2_xx(alpha, i, j, k, inv_12h2)
    d2alpha_yy = d2_yy(alpha, i, j, k, inv_12h2)
    d2alpha_zz = d2_zz(alpha, i, j, k, inv_12h2)
    lap_alpha = d2alpha_xx + d2alpha_yy + d2alpha_zz
    
    rhs_K_val = -lap_alpha + alpha_ijk * K_ijk * K_ijk / 3.0
    rhs_K_val = rhs_K_val - ko_dissipation(K_in, i, j, k, sigma_ko / dt)
    rhs_K[i, j, k] = rhs_K_val


class SimpleRK4Integrator:
    """RK4 integrator for simplified BSSN (chi and K only)"""
    
    def __init__(self, nx: int, ny: int, nz: int, h: float, sigma_ko: float = 0.1):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.h = h
        self.sigma_ko = sigma_ko
        
        self.inv_12h = 1.0 / (12.0 * h)
        self.inv_12h2 = 1.0 / (12.0 * h * h)
        
        shape = (nx, ny, nz)
        
        # Scratch arrays for RK4 stages
        self.chi_tmp = wp.zeros(shape, dtype=float)
        self.K_tmp = wp.zeros(shape, dtype=float)
        
        self.k1_chi = wp.zeros(shape, dtype=float)
        self.k1_K = wp.zeros(shape, dtype=float)
        
        self.k2_chi = wp.zeros(shape, dtype=float)
        self.k2_K = wp.zeros(shape, dtype=float)
        
        self.k3_chi = wp.zeros(shape, dtype=float)
        self.k3_K = wp.zeros(shape, dtype=float)
        
        self.k4_chi = wp.zeros(shape, dtype=float)
        self.k4_K = wp.zeros(shape, dtype=float)
    
    def step(self, chi: wp.array3d, K: wp.array3d, alpha: wp.array3d, dt: float):
        """Perform one RK4 step"""
        
        n = self.nx
        dim = (n, n, n)
        
        # Stage 1: k1 = f(y)
        wp.launch(
            compute_simple_rhs, dim=dim,
            inputs=[chi, K, alpha, self.k1_chi, self.k1_K,
                    n, n, n, self.inv_12h, self.inv_12h2, self.sigma_ko, dt]
        )
        
        # Stage 2: y_tmp = y + dt/2 * k1
        wp.launch(rk_update_kernel, dim=dim,
                  inputs=[chi, self.k1_chi, self.chi_tmp, dt * 0.5, n, n, n])
        wp.launch(rk_update_kernel, dim=dim,
                  inputs=[K, self.k1_K, self.K_tmp, dt * 0.5, n, n, n])
        
        # k2 = f(y_tmp)
        wp.launch(
            compute_simple_rhs, dim=dim,
            inputs=[self.chi_tmp, self.K_tmp, alpha, self.k2_chi, self.k2_K,
                    n, n, n, self.inv_12h, self.inv_12h2, self.sigma_ko, dt]
        )
        
        # Stage 3: y_tmp = y + dt/2 * k2
        wp.launch(rk_update_kernel, dim=dim,
                  inputs=[chi, self.k2_chi, self.chi_tmp, dt * 0.5, n, n, n])
        wp.launch(rk_update_kernel, dim=dim,
                  inputs=[K, self.k2_K, self.K_tmp, dt * 0.5, n, n, n])
        
        # k3 = f(y_tmp)
        wp.launch(
            compute_simple_rhs, dim=dim,
            inputs=[self.chi_tmp, self.K_tmp, alpha, self.k3_chi, self.k3_K,
                    n, n, n, self.inv_12h, self.inv_12h2, self.sigma_ko, dt]
        )
        
        # Stage 4: y_tmp = y + dt * k3
        wp.launch(rk_update_kernel, dim=dim,
                  inputs=[chi, self.k3_chi, self.chi_tmp, dt, n, n, n])
        wp.launch(rk_update_kernel, dim=dim,
                  inputs=[K, self.k3_K, self.K_tmp, dt, n, n, n])
        
        # k4 = f(y_tmp)
        wp.launch(
            compute_simple_rhs, dim=dim,
            inputs=[self.chi_tmp, self.K_tmp, alpha, self.k4_chi, self.k4_K,
                    n, n, n, self.inv_12h, self.inv_12h2, self.sigma_ko, dt]
        )
        
        # Final combination: y_new = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        wp.launch(rk4_combine_kernel, dim=dim,
                  inputs=[chi, self.k1_chi, self.k2_chi, self.k3_chi, self.k4_chi,
                          chi, dt, n, n, n])
        wp.launch(rk4_combine_kernel, dim=dim,
                  inputs=[K, self.k1_K, self.k2_K, self.k3_K, self.k4_K,
                          K, dt, n, n, n])


def test_rk4_flat_spacetime():
    """Test RK4 evolution preserves flat spacetime"""
    print("Testing RK4 integration on flat spacetime...")
    
    # Grid parameters
    ng = 3
    n_interior = 16
    n = n_interior + 2 * ng
    h = 0.1
    dt = 0.001
    n_steps = 100
    
    shape = (n, n, n)
    
    # Initialize flat spacetime
    chi = wp.zeros(shape, dtype=float)
    chi.fill_(1.0)
    
    K = wp.zeros(shape, dtype=float)  # K = 0
    
    alpha = wp.zeros(shape, dtype=float)
    alpha.fill_(1.0)
    
    # Create integrator
    integrator = SimpleRK4Integrator(n, n, n, h, sigma_ko=0.1)
    
    # Evolve
    print(f"Evolving for {n_steps} steps with dt = {dt}...")
    
    for step in range(n_steps):
        integrator.step(chi, K, alpha, dt)
    
    # Check final state (should still be flat)
    sl = slice(ng, n - ng)
    chi_np = chi.numpy()[sl, sl, sl]
    K_np = K.numpy()[sl, sl, sl]
    
    chi_deviation = np.max(np.abs(chi_np - 1.0))
    K_max = np.max(np.abs(K_np))
    
    print(f"After {n_steps} steps:")
    print(f"  Max |chi - 1|: {chi_deviation:.6e}")
    print(f"  Max |K|: {K_max:.6e}")
    
    tol = 1e-10
    if chi_deviation < tol and K_max < tol:
        print("RK4 flat spacetime test PASSED!")
    else:
        print("RK4 flat spacetime test FAILED!")
    
    return chi_deviation, K_max


if __name__ == "__main__":
    wp.set_module_options({"enable_backward": True})
    test_rk4_flat_spacetime()
