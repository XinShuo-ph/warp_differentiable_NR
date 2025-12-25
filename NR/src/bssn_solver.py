import warp as wp
from NR.src.bssn_defs import BSSNState, allocate_bssn_state, init_bssn_state
from NR.src.bssn_rhs import bssn_rhs_kernel

class BSSNSolver:
    def __init__(self, resolution=64, device=None):
        self.res = (resolution, resolution, resolution)
        self.device = device
        
        # Allocate State
        self.state = allocate_bssn_state(self.res, device=device)
        self.init_state()
        
        # Allocate RHS buffers (k1, k2, k3, k4 can reuse same buffers if we step sequentially)
        # But for RK4 standard:
        # k1 = f(y)
        # k2 = f(y + 0.5*dt*k1)
        # k3 = f(y + 0.5*dt*k2)
        # k4 = f(y + dt*k3)
        # y_next = y + dt/6 * (k1 + 2k2 + 2k3 + k4)
        
        # We need at least one intermediate state buffer 'y_temp'
        # And we need to accumulate the update 'y_next' or 'dy_total'.
        
        self.state_temp = allocate_bssn_state(self.res, device=device)
        self.rhs = allocate_bssn_state(self.res, device=device) # stores k
        
        # To avoid allocating 4 k-buffers, we can accumulate result into `state` directly? 
        # No, we need original `state` for each step.
        # So we need `state_next` or use `state` as source and a buffer for accumulation.
        
        # Let's allocate:
        # - state (current n)
        # - state_temp (argument for RHS)
        # - rhs (output of RHS)
        # - state_out (result n+1, can be swapped with state)
        
        self.state_out = allocate_bssn_state(self.res, device=device)
        
    def init_state(self):
        init_bssn_state(self.state)
        
    def compute_rhs(self, state_in, rhs_out):
        wp.launch(
            kernel=bssn_rhs_kernel,
            dim=self.res,
            inputs=[
                state_in.phi, state_in.gamma_tilde, state_in.K, state_in.A_tilde, state_in.Gamma_tilde, state_in.alpha, state_in.beta, state_in.B,
                rhs_out.phi, rhs_out.gamma_tilde, rhs_out.K, rhs_out.A_tilde, rhs_out.Gamma_tilde, rhs_out.alpha, rhs_out.beta, rhs_out.B,
                state_in.dx, state_in.dy, state_in.dz
            ],
            device=self.device
        )

    def rk4_step(self, dt):
        # 1. k1 = f(y)
        self.compute_rhs(self.state, self.rhs) # rhs has k1
        
        # 2. y_temp = y + 0.5 * dt * k1
        self.update_step(self.state_temp, self.state, self.rhs, 0.5 * dt)
        
        # Accumulate final update: dy = k1
        # Actually standard way: y_out = y + dt/6 * k1
        self.update_step(self.state_out, self.state, self.rhs, dt / 6.0)
        
        # 3. k2 = f(y_temp)
        self.compute_rhs(self.state_temp, self.rhs) # rhs has k2
        
        # 4. y_temp = y + 0.5 * dt * k2
        self.update_step(self.state_temp, self.state, self.rhs, 0.5 * dt)
        
        # Accumulate: y_out += dt/3 * k2  (since 2/6 = 1/3)
        self.accumulate_step(self.state_out, self.rhs, dt / 3.0)
        
        # 5. k3 = f(y_temp)
        self.compute_rhs(self.state_temp, self.rhs) # rhs has k3
        
        # 6. y_temp = y + dt * k3
        self.update_step(self.state_temp, self.state, self.rhs, dt)
        
        # Accumulate: y_out += dt/3 * k3
        self.accumulate_step(self.state_out, self.rhs, dt / 3.0)
        
        # 7. k4 = f(y_temp)
        self.compute_rhs(self.state_temp, self.rhs) # rhs has k4
        
        # Accumulate: y_out += dt/6 * k4
        self.accumulate_step(self.state_out, self.rhs, dt / 6.0)
        
        # Swap state and state_out
        self.state, self.state_out = self.state_out, self.state
        
    def update_step(self, dest, src, k, scale):
        wp.launch(
            kernel=update_kernel,
            dim=self.res,
            inputs=[
                dest.phi, dest.gamma_tilde, dest.K, dest.A_tilde, dest.Gamma_tilde, dest.alpha, dest.beta, dest.B,
                src.phi, src.gamma_tilde, src.K, src.A_tilde, src.Gamma_tilde, src.alpha, src.beta, src.B,
                k.phi, k.gamma_tilde, k.K, k.A_tilde, k.Gamma_tilde, k.alpha, k.beta, k.B,
                scale
            ],
            device=self.device
        )

    def accumulate_step(self, dest, k, scale):
        wp.launch(
            kernel=accumulate_kernel,
            dim=self.res,
            inputs=[
                dest.phi, dest.gamma_tilde, dest.K, dest.A_tilde, dest.Gamma_tilde, dest.alpha, dest.beta, dest.B,
                k.phi, k.gamma_tilde, k.K, k.A_tilde, k.Gamma_tilde, k.alpha, k.beta, k.B,
                scale
            ],
            device=self.device
        )

@wp.kernel
def update_kernel(
    # dest fields
    d_phi: wp.array(dtype=float, ndim=3), d_g: wp.array(dtype=wp.mat33, ndim=3), d_K: wp.array(dtype=float, ndim=3), 
    d_A: wp.array(dtype=wp.mat33, ndim=3), d_Gam: wp.array(dtype=wp.vec3, ndim=3),
    d_alp: wp.array(dtype=float, ndim=3), d_bet: wp.array(dtype=wp.vec3, ndim=3), d_B: wp.array(dtype=wp.vec3, ndim=3),
    # src fields
    s_phi: wp.array(dtype=float, ndim=3), s_g: wp.array(dtype=wp.mat33, ndim=3), s_K: wp.array(dtype=float, ndim=3), 
    s_A: wp.array(dtype=wp.mat33, ndim=3), s_Gam: wp.array(dtype=wp.vec3, ndim=3),
    s_alp: wp.array(dtype=float, ndim=3), s_bet: wp.array(dtype=wp.vec3, ndim=3), s_B: wp.array(dtype=wp.vec3, ndim=3),
    # k fields
    k_phi: wp.array(dtype=float, ndim=3), k_g: wp.array(dtype=wp.mat33, ndim=3), k_K: wp.array(dtype=float, ndim=3), 
    k_A: wp.array(dtype=wp.mat33, ndim=3), k_Gam: wp.array(dtype=wp.vec3, ndim=3),
    k_alp: wp.array(dtype=float, ndim=3), k_bet: wp.array(dtype=wp.vec3, ndim=3), k_B: wp.array(dtype=wp.vec3, ndim=3),
    scale: float
):
    i, j, k = wp.tid()
    d_phi[i,j,k] = s_phi[i,j,k] + scale * k_phi[i,j,k]
    d_g[i,j,k] = s_g[i,j,k] + scale * k_g[i,j,k]
    d_K[i,j,k] = s_K[i,j,k] + scale * k_K[i,j,k]
    d_A[i,j,k] = s_A[i,j,k] + scale * k_A[i,j,k]
    d_Gam[i,j,k] = s_Gam[i,j,k] + scale * k_Gam[i,j,k]
    d_alp[i,j,k] = s_alp[i,j,k] + scale * k_alp[i,j,k]
    d_bet[i,j,k] = s_bet[i,j,k] + scale * k_bet[i,j,k]
    d_B[i,j,k] = s_B[i,j,k] + scale * k_B[i,j,k]

@wp.kernel
def accumulate_kernel(
    # dest fields (accumulate in place)
    d_phi: wp.array(dtype=float, ndim=3), d_g: wp.array(dtype=wp.mat33, ndim=3), d_K: wp.array(dtype=float, ndim=3), 
    d_A: wp.array(dtype=wp.mat33, ndim=3), d_Gam: wp.array(dtype=wp.vec3, ndim=3),
    d_alp: wp.array(dtype=float, ndim=3), d_bet: wp.array(dtype=wp.vec3, ndim=3), d_B: wp.array(dtype=wp.vec3, ndim=3),
    # k fields
    k_phi: wp.array(dtype=float, ndim=3), k_g: wp.array(dtype=wp.mat33, ndim=3), k_K: wp.array(dtype=float, ndim=3), 
    k_A: wp.array(dtype=wp.mat33, ndim=3), k_Gam: wp.array(dtype=wp.vec3, ndim=3),
    k_alp: wp.array(dtype=float, ndim=3), k_bet: wp.array(dtype=wp.vec3, ndim=3), k_B: wp.array(dtype=wp.vec3, ndim=3),
    scale: float
):
    i, j, k = wp.tid()
    d_phi[i,j,k] += scale * k_phi[i,j,k]
    d_g[i,j,k] += scale * k_g[i,j,k]
    d_K[i,j,k] += scale * k_K[i,j,k]
    d_A[i,j,k] += scale * k_A[i,j,k]
    d_Gam[i,j,k] += scale * k_Gam[i,j,k]
    d_alp[i,j,k] += scale * k_alp[i,j,k]
    d_bet[i,j,k] += scale * k_bet[i,j,k]
    d_B[i,j,k] += scale * k_B[i,j,k]
