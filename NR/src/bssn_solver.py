import warp as wp
from bssn_defs import allocate_bssn_state, initialize
from bssn_rhs import bssn_rhs_kernel
from dissipation_kernel import add_dissipation_kernel
from rk4 import state_add_scaled

class BSSNSolver:
    def __init__(self, res=32, domain_size=1.0, sigma=0.01):
        self.res = res
        self.dx = domain_size / res
        self.dt = 0.25 * self.dx 
        self.sigma = sigma
        
        self.shape = (res, res, res)
        
        self.state = allocate_bssn_state(self.shape)
        initialize(self.state)
        
        self.k_state = allocate_bssn_state(self.shape)
        self.tmp_state = allocate_bssn_state(self.shape)
        self.accum_state = allocate_bssn_state(self.shape)
        
    def compute_rhs(self, state, rhs):
        # Clear rhs first? No, bssn_rhs_kernel overwrites?
        # bssn_rhs_kernel currently overwrites.
        wp.launch(bssn_rhs_kernel, dim=self.shape, inputs=[0.0, state, rhs, self.dx])
        
        # Add dissipation (accumulate)
        wp.launch(add_dissipation_kernel, dim=self.shape, inputs=[state, rhs, self.dx, self.sigma])

    def step(self):
        dim = self.shape
        dt = self.dt
        
        # k1
        self.compute_rhs(self.state, self.k_state)
        
        # accum = y + dt/6 * k1
        wp.launch(state_add_scaled, dim=dim, inputs=[self.accum_state, self.state, self.k_state, dt/6.0])
        # tmp = y + 0.5*dt * k1
        wp.launch(state_add_scaled, dim=dim, inputs=[self.tmp_state, self.state, self.k_state, 0.5*dt])
        
        # k2
        self.compute_rhs(self.tmp_state, self.k_state)
        
        # accum += dt/3 * k2
        wp.launch(state_add_scaled, dim=dim, inputs=[self.accum_state, self.accum_state, self.k_state, dt/3.0])
        # tmp = y + 0.5*dt * k2
        wp.launch(state_add_scaled, dim=dim, inputs=[self.tmp_state, self.state, self.k_state, 0.5*dt])
        
        # k3
        self.compute_rhs(self.tmp_state, self.k_state)
        
        # accum += dt/3 * k3
        wp.launch(state_add_scaled, dim=dim, inputs=[self.accum_state, self.accum_state, self.k_state, dt/3.0])
        # tmp = y + dt * k3
        wp.launch(state_add_scaled, dim=dim, inputs=[self.tmp_state, self.state, self.k_state, dt])
        
        # k4
        self.compute_rhs(self.tmp_state, self.k_state)
        
        # accum += dt/6 * k4
        wp.launch(state_add_scaled, dim=dim, inputs=[self.accum_state, self.accum_state, self.k_state, dt/6.0])
        
        self.copy_state(self.state, self.accum_state)
        
    def copy_state(self, dest, src):
        names = [
            "phi", "gamma_xx", "gamma_xy", "gamma_xz", "gamma_yy", "gamma_yz", "gamma_zz",
            "K", "A_xx", "A_xy", "A_xz", "A_yy", "A_yz", "A_zz",
            "Gam_x", "Gam_y", "Gam_z",
            "alpha", "beta_x", "beta_y", "beta_z", "B_x", "B_y", "B_z"
        ]
        for name in names:
            wp.copy(getattr(dest, name), getattr(src, name))

if __name__ == "__main__":
    wp.init()
    solver = BSSNSolver(res=16, sigma=0.1)
    
    print("Running with Dissipation...")
    solver.step()
    
    # Check
    phi_max = solver.state.phi.numpy().max()
    print(f"Max phi deviation: {phi_max}")
    
    if abs(phi_max) < 1e-10:
        print("Dissipation Test PASSED")
