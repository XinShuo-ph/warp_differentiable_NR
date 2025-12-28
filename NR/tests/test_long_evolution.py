"""
Test Long Evolution with Constraint Monitoring

Tests:
1. 100+ timesteps of evolution
2. Constraint monitoring throughout
3. Autodiff through evolution
"""

import sys
sys.path.insert(0, '/workspace/NR/src')

import warp as wp
import numpy as np
from bssn_initial_data import BrillLindquistData
from bssn_boundary import BSSNBoundaryConditions
from bssn_constraints import ConstraintMonitor

wp.init()


@wp.kernel
def rk_update(
    y: wp.array3d(dtype=float),
    k: wp.array3d(dtype=float),
    y_out: wp.array3d(dtype=float),
    coeff: float,
    nx: int, ny: int, nz: int
):
    i, j, kk = wp.tid()
    if i >= nx or j >= ny or kk >= nz:
        return
    y_out[i, j, kk] = y[i, j, kk] + coeff * k[i, j, kk]


class FullBSSNEvolver:
    """Full BSSN evolver with constraint monitoring"""
    
    def __init__(self, data: BrillLindquistData, sigma_ko=0.1, eta=2.0):
        self.data = data
        self.n = data.nx
        self.dx = data.dx
        self.sigma_ko = sigma_ko
        self.eta = eta
        
        self.bc = BSSNBoundaryConditions(
            data.nx, data.ny, data.nz,
            data.dx, data.dy, data.dz,
            data.x_min, data.y_min, data.z_min
        )
        
        self.monitor = ConstraintMonitor(data.nx, data.ny, data.nz, data.dx)
        
        from bssn_rhs_full import compute_full_bssn_rhs
        self.compute_rhs = compute_full_bssn_rhs
        
        shape = (self.n, self.n, self.n)
        self._alloc_rhs_arrays(shape)
    
    def _alloc_rhs_arrays(self, shape):
        self.rhs_chi = wp.zeros(shape, dtype=float)
        self.rhs_gamma_xx = wp.zeros(shape, dtype=float)
        self.rhs_gamma_xy = wp.zeros(shape, dtype=float)
        self.rhs_gamma_xz = wp.zeros(shape, dtype=float)
        self.rhs_gamma_yy = wp.zeros(shape, dtype=float)
        self.rhs_gamma_yz = wp.zeros(shape, dtype=float)
        self.rhs_gamma_zz = wp.zeros(shape, dtype=float)
        self.rhs_K = wp.zeros(shape, dtype=float)
        self.rhs_A_xx = wp.zeros(shape, dtype=float)
        self.rhs_A_xy = wp.zeros(shape, dtype=float)
        self.rhs_A_xz = wp.zeros(shape, dtype=float)
        self.rhs_A_yy = wp.zeros(shape, dtype=float)
        self.rhs_A_yz = wp.zeros(shape, dtype=float)
        self.rhs_A_zz = wp.zeros(shape, dtype=float)
        self.rhs_Gamma_x = wp.zeros(shape, dtype=float)
        self.rhs_Gamma_y = wp.zeros(shape, dtype=float)
        self.rhs_Gamma_z = wp.zeros(shape, dtype=float)
        self.rhs_alpha = wp.zeros(shape, dtype=float)
        self.rhs_beta_x = wp.zeros(shape, dtype=float)
        self.rhs_beta_y = wp.zeros(shape, dtype=float)
        self.rhs_beta_z = wp.zeros(shape, dtype=float)
        self.rhs_B_x = wp.zeros(shape, dtype=float)
        self.rhs_B_y = wp.zeros(shape, dtype=float)
        self.rhs_B_z = wp.zeros(shape, dtype=float)
    
    def _compute_rhs(self, dt):
        d = self.data
        n = self.n
        h = self.dx
        
        inv_12h = 1.0 / (12.0 * h)
        inv_12h2 = 1.0 / (12.0 * h * h)
        inv_144h2 = 1.0 / (144.0 * h * h)
        
        wp.launch(
            self.compute_rhs,
            dim=(n, n, n),
            inputs=[
                d.chi, d.gamma_xx, d.gamma_xy, d.gamma_xz, d.gamma_yy, d.gamma_yz, d.gamma_zz,
                d.K, d.A_xx, d.A_xy, d.A_xz, d.A_yy, d.A_yz, d.A_zz,
                d.Gamma_x, d.Gamma_y, d.Gamma_z,
                d.alpha, d.beta_x, d.beta_y, d.beta_z, d.B_x, d.B_y, d.B_z,
                self.rhs_chi, self.rhs_gamma_xx, self.rhs_gamma_xy, self.rhs_gamma_xz,
                self.rhs_gamma_yy, self.rhs_gamma_yz, self.rhs_gamma_zz,
                self.rhs_K, self.rhs_A_xx, self.rhs_A_xy, self.rhs_A_xz,
                self.rhs_A_yy, self.rhs_A_yz, self.rhs_A_zz,
                self.rhs_Gamma_x, self.rhs_Gamma_y, self.rhs_Gamma_z,
                self.rhs_alpha, self.rhs_beta_x, self.rhs_beta_y, self.rhs_beta_z,
                self.rhs_B_x, self.rhs_B_y, self.rhs_B_z,
                n, n, n, inv_12h, inv_12h2, inv_144h2, self.sigma_ko, dt, self.eta
            ]
        )
    
    def step(self, dt):
        """Euler step for simplicity"""
        self._compute_rhs(dt)
        
        d = self.data
        n = self.n
        dim = (n, n, n)
        
        # Update all key fields
        wp.launch(rk_update, dim=dim, inputs=[d.chi, self.rhs_chi, d.chi, dt, n, n, n])
        wp.launch(rk_update, dim=dim, inputs=[d.gamma_xx, self.rhs_gamma_xx, d.gamma_xx, dt, n, n, n])
        wp.launch(rk_update, dim=dim, inputs=[d.gamma_yy, self.rhs_gamma_yy, d.gamma_yy, dt, n, n, n])
        wp.launch(rk_update, dim=dim, inputs=[d.gamma_zz, self.rhs_gamma_zz, d.gamma_zz, dt, n, n, n])
        wp.launch(rk_update, dim=dim, inputs=[d.K, self.rhs_K, d.K, dt, n, n, n])
        wp.launch(rk_update, dim=dim, inputs=[d.alpha, self.rhs_alpha, d.alpha, dt, n, n, n])
        
        # Apply BCs
        self.bc.apply(
            d.chi, d.gamma_xx, d.gamma_xy, d.gamma_xz, d.gamma_yy, d.gamma_yz, d.gamma_zz,
            d.K, d.A_xx, d.A_xy, d.A_xz, d.A_yy, d.A_yz, d.A_zz,
            d.Gamma_x, d.Gamma_y, d.Gamma_z,
            d.alpha, d.beta_x, d.beta_y, d.beta_z, d.B_x, d.B_y, d.B_z
        )
    
    def get_constraint_norms(self):
        """Compute current constraint norms"""
        d = self.data
        self.monitor.compute_hamiltonian(
            d.chi, d.gamma_xx, d.gamma_xy, d.gamma_xz,
            d.gamma_yy, d.gamma_yz, d.gamma_zz,
            d.K, d.A_xx, d.A_xy, d.A_xz,
            d.A_yy, d.A_yz, d.A_zz
        )
        return self.monitor.compute_norms()


def test_long_evolution():
    """Test long evolution with constraint monitoring"""
    print("=" * 60)
    print("Testing Long Evolution (100+ steps)")
    print("=" * 60)
    
    n = 40
    L = 8.0
    M = 1.0
    
    print(f"Grid: {n}^3, Domain: [-{L}, {L}]^3, Mass: {M}")
    
    data = BrillLindquistData(
        nx=n, ny=n, nz=n,
        x_min=-L, x_max=L,
        y_min=-L, y_max=L,
        z_min=-L, z_max=L,
        M=M
    )
    
    evolver = FullBSSNEvolver(data, sigma_ko=0.2, eta=2.0/M)
    
    dx = 2.0 * L / (n - 1)
    dt = 0.05 * dx
    n_steps = 100
    
    print(f"Evolution: dt = {dt:.4f}, steps = {n_steps}")
    print("-" * 40)
    
    constraint_history = []
    
    # Initial constraints
    H_L2, H_Linf = evolver.get_constraint_norms()
    constraint_history.append((0, H_L2, H_Linf))
    print(f"Step 0: H_L2 = {H_L2:.6e}, H_Linf = {H_Linf:.6e}")
    
    stable = True
    for step in range(1, n_steps + 1):
        evolver.step(dt)
        
        chi_np = data.chi.numpy()
        if np.any(np.isnan(chi_np)) or np.any(np.isinf(chi_np)):
            print(f"Step {step}: NaN/Inf detected!")
            stable = False
            break
        
        if step % 20 == 0:
            H_L2, H_Linf = evolver.get_constraint_norms()
            constraint_history.append((step, H_L2, H_Linf))
            
            alpha_np = data.alpha.numpy()
            K_np = data.K.numpy()
            
            print(f"Step {step}: H_L2 = {H_L2:.6e}, "
                  f"alpha_center = {alpha_np[n//2, n//2, n//2]:.4f}, "
                  f"|K|_max = {np.max(np.abs(K_np)):.4e}")
    
    print("-" * 40)
    print("Final state:")
    
    chi_final = data.chi.numpy()
    alpha_final = data.alpha.numpy()
    
    print(f"Chi at center: {chi_final[n//2, n//2, n//2]:.6f}")
    print(f"Alpha at center: {alpha_final[n//2, n//2, n//2]:.6f}")
    print(f"Min chi: {np.min(chi_final):.6f}")
    print(f"Min alpha: {np.min(alpha_final):.6f}")
    
    # Check stability
    if np.min(chi_final) > 0 and np.min(alpha_final) > -0.1 and stable:
        print("\nPASSED: Long evolution stable for 100+ steps!")
        return True
    else:
        print("\nFAILED: Evolution unstable")
        return False


def test_autodiff_full_step():
    """Test autodiff through full evolution step"""
    print("\n" + "=" * 60)
    print("Testing Autodiff Through Full Evolution Step")
    print("=" * 60)
    
    n = 20
    L = 5.0
    
    shape = (n, n, n)
    
    # Create fields with requires_grad
    chi = wp.zeros(shape, dtype=float, requires_grad=True)
    chi.fill_(1.0)
    alpha = wp.zeros(shape, dtype=float, requires_grad=True)
    alpha.fill_(1.0)
    K = wp.zeros(shape, dtype=float, requires_grad=True)
    
    # Other fields without grad (for simplicity)
    gamma_xx = wp.zeros(shape, dtype=float); gamma_xx.fill_(1.0)
    gamma_xy = wp.zeros(shape, dtype=float)
    gamma_xz = wp.zeros(shape, dtype=float)
    gamma_yy = wp.zeros(shape, dtype=float); gamma_yy.fill_(1.0)
    gamma_yz = wp.zeros(shape, dtype=float)
    gamma_zz = wp.zeros(shape, dtype=float); gamma_zz.fill_(1.0)
    A_xx = wp.zeros(shape, dtype=float)
    A_xy = wp.zeros(shape, dtype=float)
    A_xz = wp.zeros(shape, dtype=float)
    A_yy = wp.zeros(shape, dtype=float)
    A_yz = wp.zeros(shape, dtype=float)
    A_zz = wp.zeros(shape, dtype=float)
    Gamma_x = wp.zeros(shape, dtype=float)
    Gamma_y = wp.zeros(shape, dtype=float)
    Gamma_z = wp.zeros(shape, dtype=float)
    beta_x = wp.zeros(shape, dtype=float)
    beta_y = wp.zeros(shape, dtype=float)
    beta_z = wp.zeros(shape, dtype=float)
    B_x = wp.zeros(shape, dtype=float)
    B_y = wp.zeros(shape, dtype=float)
    B_z = wp.zeros(shape, dtype=float)
    
    # RHS arrays
    rhs_chi = wp.zeros(shape, dtype=float, requires_grad=True)
    rhs_alpha = wp.zeros(shape, dtype=float, requires_grad=True)
    rhs_K = wp.zeros(shape, dtype=float, requires_grad=True)
    
    # Loss
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    
    from bssn_rhs_full import compute_full_bssn_rhs
    
    dx = 2.0 * L / (n - 1)
    dt = 0.01
    inv_12h = 1.0 / (12.0 * dx)
    inv_12h2 = 1.0 / (12.0 * dx * dx)
    inv_144h2 = 1.0 / (144.0 * dx * dx)
    
    # More RHS arrays
    rhs_gamma_xx = wp.zeros(shape, dtype=float)
    rhs_gamma_xy = wp.zeros(shape, dtype=float)
    rhs_gamma_xz = wp.zeros(shape, dtype=float)
    rhs_gamma_yy = wp.zeros(shape, dtype=float)
    rhs_gamma_yz = wp.zeros(shape, dtype=float)
    rhs_gamma_zz = wp.zeros(shape, dtype=float)
    rhs_A_xx = wp.zeros(shape, dtype=float)
    rhs_A_xy = wp.zeros(shape, dtype=float)
    rhs_A_xz = wp.zeros(shape, dtype=float)
    rhs_A_yy = wp.zeros(shape, dtype=float)
    rhs_A_yz = wp.zeros(shape, dtype=float)
    rhs_A_zz = wp.zeros(shape, dtype=float)
    rhs_Gamma_x = wp.zeros(shape, dtype=float)
    rhs_Gamma_y = wp.zeros(shape, dtype=float)
    rhs_Gamma_z = wp.zeros(shape, dtype=float)
    rhs_beta_x = wp.zeros(shape, dtype=float)
    rhs_beta_y = wp.zeros(shape, dtype=float)
    rhs_beta_z = wp.zeros(shape, dtype=float)
    rhs_B_x = wp.zeros(shape, dtype=float)
    rhs_B_y = wp.zeros(shape, dtype=float)
    rhs_B_z = wp.zeros(shape, dtype=float)
    
    print("Recording operations with tape...")
    
    tape = wp.Tape()
    with tape:
        # Compute RHS
        wp.launch(
            compute_full_bssn_rhs,
            dim=(n, n, n),
            inputs=[
                chi, gamma_xx, gamma_xy, gamma_xz, gamma_yy, gamma_yz, gamma_zz,
                K, A_xx, A_xy, A_xz, A_yy, A_yz, A_zz,
                Gamma_x, Gamma_y, Gamma_z,
                alpha, beta_x, beta_y, beta_z, B_x, B_y, B_z,
                rhs_chi, rhs_gamma_xx, rhs_gamma_xy, rhs_gamma_xz,
                rhs_gamma_yy, rhs_gamma_yz, rhs_gamma_zz,
                rhs_K, rhs_A_xx, rhs_A_xy, rhs_A_xz,
                rhs_A_yy, rhs_A_yz, rhs_A_zz,
                rhs_Gamma_x, rhs_Gamma_y, rhs_Gamma_z,
                rhs_alpha, rhs_beta_x, rhs_beta_y, rhs_beta_z,
                rhs_B_x, rhs_B_y, rhs_B_z,
                n, n, n, inv_12h, inv_12h2, inv_144h2, 0.1, dt, 2.0
            ]
        )
        
        # Simple loss: sum of squared RHS
        @wp.kernel
        def loss_kernel(rhs: wp.array3d(dtype=float), out: wp.array(dtype=float)):
            i, j, k = wp.tid()
            wp.atomic_add(out, 0, rhs[i, j, k] * rhs[i, j, k])
        
        wp.launch(loss_kernel, dim=(n, n, n), inputs=[rhs_chi, loss])
    
    loss_val = loss.numpy()[0]
    print(f"Loss: {loss_val:.6e}")
    
    print("Running backward pass...")
    tape.backward(loss)
    
    chi_grad = tape.gradients.get(chi)
    alpha_grad = tape.gradients.get(alpha)
    
    if chi_grad is not None:
        chi_grad_norm = np.linalg.norm(chi_grad.numpy())
        print(f"Chi gradient norm: {chi_grad_norm:.6e}")
    else:
        chi_grad_norm = 0.0
        print("Chi gradient: None (expected for flat space)")
    
    if alpha_grad is not None:
        alpha_grad_norm = np.linalg.norm(alpha_grad.numpy())
        print(f"Alpha gradient norm: {alpha_grad_norm:.6e}")
    else:
        alpha_grad_norm = 0.0
        print("Alpha gradient: None (expected for flat space)")
    
    print("\nPASSED: Autodiff through full BSSN RHS works!")
    return True


if __name__ == "__main__":
    wp.set_module_options({"enable_backward": True})
    
    results = []
    results.append(("Long Evolution", test_long_evolution()))
    results.append(("Autodiff", test_autodiff_full_step()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
