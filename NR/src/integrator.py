import warp as wp
import copy
from bssn import BSSNState
from rhs import bssn_rhs_kernel

class RK4Integrator:
    def __init__(self, state: BSSNState):
        self.state = state
        self.k1 = BSSNState(state.res, (0,0,0), (1,1,1)) # Bounds don't matter for allocation
        self.k2 = BSSNState(state.res, (0,0,0), (1,1,1))
        self.k3 = BSSNState(state.res, (0,0,0), (1,1,1))
        self.k4 = BSSNState(state.res, (0,0,0), (1,1,1))
        self.temp_state = BSSNState(state.res, (0,0,0), (1,1,1))
        
    def step(self, dt):
        # k1 = RHS(U)
        self.compute_rhs(self.state, self.k1)
        
        # k2 = RHS(U + 0.5*dt*k1)
        self.update_state(self.temp_state, self.state, self.k1, 0.5*dt)
        self.compute_rhs(self.temp_state, self.k2)
        
        # k3 = RHS(U + 0.5*dt*k2)
        self.update_state(self.temp_state, self.state, self.k2, 0.5*dt)
        self.compute_rhs(self.temp_state, self.k3)
        
        # k4 = RHS(U + dt*k3)
        self.update_state(self.temp_state, self.state, self.k3, dt)
        self.compute_rhs(self.temp_state, self.k4)
        
        # U = U + dt/6 * (k1 + 2k2 + 2k3 + k4)
        self.final_update(self.state, self.k1, self.k2, self.k3, self.k4, dt)
        
    def compute_rhs(self, u: BSSNState, k: BSSNState):
        wp.launch(
            kernel=bssn_rhs_kernel,
            dim=u.shape,
            inputs=[
                u.phi, u.gamma_tilde, u.K, u.A_tilde, u.Gam_tilde, u.alpha, u.beta, u.B,
                k.phi, k.gamma_tilde, k.K, k.A_tilde, k.Gam_tilde, k.alpha, k.beta, k.B,
                u.dx, u.dx, u.dx # Assuming isotropic grid for now
            ]
        )
        
    def update_state(self, out: BSSNState, u: BSSNState, k: BSSNState, dt: float):
        wp.launch(
            kernel=update_kernel,
            dim=u.shape,
            inputs=[
                out.phi, out.gamma_tilde, out.K, out.A_tilde, out.Gam_tilde, out.alpha, out.beta, out.B,
                u.phi, u.gamma_tilde, u.K, u.A_tilde, u.Gam_tilde, u.alpha, u.beta, u.B,
                k.phi, k.gamma_tilde, k.K, k.A_tilde, k.Gam_tilde, k.alpha, k.beta, k.B,
                dt
            ]
        )

    def final_update(self, u: BSSNState, k1: BSSNState, k2: BSSNState, k3: BSSNState, k4: BSSNState, dt: float):
        wp.launch(
            kernel=final_update_kernel,
            dim=u.shape,
            inputs=[
                u.phi, u.gamma_tilde, u.K, u.A_tilde, u.Gam_tilde, u.alpha, u.beta, u.B,
                k1.phi, k1.gamma_tilde, k1.K, k1.A_tilde, k1.Gam_tilde, k1.alpha, k1.beta, k1.B,
                k2.phi, k2.gamma_tilde, k2.K, k2.A_tilde, k2.Gam_tilde, k2.alpha, k2.beta, k2.B,
                k3.phi, k3.gamma_tilde, k3.K, k3.A_tilde, k3.Gam_tilde, k3.alpha, k3.beta, k3.B,
                k4.phi, k4.gamma_tilde, k4.K, k4.A_tilde, k4.Gam_tilde, k4.alpha, k4.beta, k4.B,
                dt
            ]
        )

@wp.kernel
def update_kernel(
    # out
    out_phi: wp.array(dtype=float, ndim=3),
    out_gamma_tilde: wp.array(dtype=float, ndim=4),
    out_K: wp.array(dtype=float, ndim=3),
    out_A_tilde: wp.array(dtype=float, ndim=4),
    out_Gam_tilde: wp.array(dtype=float, ndim=4),
    out_alpha: wp.array(dtype=float, ndim=3),
    out_beta: wp.array(dtype=float, ndim=4),
    out_B: wp.array(dtype=float, ndim=4),
    # u
    u_phi: wp.array(dtype=float, ndim=3),
    u_gamma_tilde: wp.array(dtype=float, ndim=4),
    u_K: wp.array(dtype=float, ndim=3),
    u_A_tilde: wp.array(dtype=float, ndim=4),
    u_Gam_tilde: wp.array(dtype=float, ndim=4),
    u_alpha: wp.array(dtype=float, ndim=3),
    u_beta: wp.array(dtype=float, ndim=4),
    u_B: wp.array(dtype=float, ndim=4),
    # k
    k_phi: wp.array(dtype=float, ndim=3),
    k_gamma_tilde: wp.array(dtype=float, ndim=4),
    k_K: wp.array(dtype=float, ndim=3),
    k_A_tilde: wp.array(dtype=float, ndim=4),
    k_Gam_tilde: wp.array(dtype=float, ndim=4),
    k_alpha: wp.array(dtype=float, ndim=3),
    k_beta: wp.array(dtype=float, ndim=4),
    k_B: wp.array(dtype=float, ndim=4),
    dt: float
):
    i, j, k = wp.tid()
    out_phi[i, j, k] = u_phi[i, j, k] + dt * k_phi[i, j, k]
    out_K[i, j, k] = u_K[i, j, k] + dt * k_K[i, j, k]
    out_alpha[i, j, k] = u_alpha[i, j, k] + dt * k_alpha[i, j, k]
    
    for c in range(6):
        out_gamma_tilde[i, j, k, c] = u_gamma_tilde[i, j, k, c] + dt * k_gamma_tilde[i, j, k, c]
        out_A_tilde[i, j, k, c] = u_A_tilde[i, j, k, c] + dt * k_A_tilde[i, j, k, c]
        
    for c in range(3):
        out_Gam_tilde[i, j, k, c] = u_Gam_tilde[i, j, k, c] + dt * k_Gam_tilde[i, j, k, c]
        out_beta[i, j, k, c] = u_beta[i, j, k, c] + dt * k_beta[i, j, k, c]
        out_B[i, j, k, c] = u_B[i, j, k, c] + dt * k_B[i, j, k, c]

@wp.kernel
def final_update_kernel(
    # u (in/out)
    u_phi: wp.array(dtype=float, ndim=3),
    u_gamma_tilde: wp.array(dtype=float, ndim=4),
    u_K: wp.array(dtype=float, ndim=3),
    u_A_tilde: wp.array(dtype=float, ndim=4),
    u_Gam_tilde: wp.array(dtype=float, ndim=4),
    u_alpha: wp.array(dtype=float, ndim=3),
    u_beta: wp.array(dtype=float, ndim=4),
    u_B: wp.array(dtype=float, ndim=4),
    # k1
    k1_phi: wp.array(dtype=float, ndim=3),
    k1_gamma_tilde: wp.array(dtype=float, ndim=4),
    k1_K: wp.array(dtype=float, ndim=3),
    k1_A_tilde: wp.array(dtype=float, ndim=4),
    k1_Gam_tilde: wp.array(dtype=float, ndim=4),
    k1_alpha: wp.array(dtype=float, ndim=3),
    k1_beta: wp.array(dtype=float, ndim=4),
    k1_B: wp.array(dtype=float, ndim=4),
    # k2
    k2_phi: wp.array(dtype=float, ndim=3),
    k2_gamma_tilde: wp.array(dtype=float, ndim=4),
    k2_K: wp.array(dtype=float, ndim=3),
    k2_A_tilde: wp.array(dtype=float, ndim=4),
    k2_Gam_tilde: wp.array(dtype=float, ndim=4),
    k2_alpha: wp.array(dtype=float, ndim=3),
    k2_beta: wp.array(dtype=float, ndim=4),
    k2_B: wp.array(dtype=float, ndim=4),
    # k3
    k3_phi: wp.array(dtype=float, ndim=3),
    k3_gamma_tilde: wp.array(dtype=float, ndim=4),
    k3_K: wp.array(dtype=float, ndim=3),
    k3_A_tilde: wp.array(dtype=float, ndim=4),
    k3_Gam_tilde: wp.array(dtype=float, ndim=4),
    k3_alpha: wp.array(dtype=float, ndim=3),
    k3_beta: wp.array(dtype=float, ndim=4),
    k3_B: wp.array(dtype=float, ndim=4),
    # k4
    k4_phi: wp.array(dtype=float, ndim=3),
    k4_gamma_tilde: wp.array(dtype=float, ndim=4),
    k4_K: wp.array(dtype=float, ndim=3),
    k4_A_tilde: wp.array(dtype=float, ndim=4),
    k4_Gam_tilde: wp.array(dtype=float, ndim=4),
    k4_alpha: wp.array(dtype=float, ndim=3),
    k4_beta: wp.array(dtype=float, ndim=4),
    k4_B: wp.array(dtype=float, ndim=4),
    dt: float
):
    i, j, k = wp.tid()
    
    fac = dt / 6.0
    
    u_phi[i, j, k] += fac * (k1_phi[i, j, k] + 2.0*k2_phi[i, j, k] + 2.0*k3_phi[i, j, k] + k4_phi[i, j, k])
    u_K[i, j, k] += fac * (k1_K[i, j, k] + 2.0*k2_K[i, j, k] + 2.0*k3_K[i, j, k] + k4_K[i, j, k])
    u_alpha[i, j, k] += fac * (k1_alpha[i, j, k] + 2.0*k2_alpha[i, j, k] + 2.0*k3_alpha[i, j, k] + k4_alpha[i, j, k])
    
    for c in range(6):
        u_gamma_tilde[i, j, k, c] += fac * (k1_gamma_tilde[i, j, k, c] + 2.0*k2_gamma_tilde[i, j, k, c] + 2.0*k3_gamma_tilde[i, j, k, c] + k4_gamma_tilde[i, j, k, c])
        u_A_tilde[i, j, k, c] += fac * (k1_A_tilde[i, j, k, c] + 2.0*k2_A_tilde[i, j, k, c] + 2.0*k3_A_tilde[i, j, k, c] + k4_A_tilde[i, j, k, c])
        
    for c in range(3):
        u_Gam_tilde[i, j, k, c] += fac * (k1_Gam_tilde[i, j, k, c] + 2.0*k2_Gam_tilde[i, j, k, c] + 2.0*k3_Gam_tilde[i, j, k, c] + k4_Gam_tilde[i, j, k, c])
        u_beta[i, j, k, c] += fac * (k1_beta[i, j, k, c] + 2.0*k2_beta[i, j, k, c] + 2.0*k3_beta[i, j, k, c] + k4_beta[i, j, k, c])
        u_B[i, j, k, c] += fac * (k1_B[i, j, k, c] + 2.0*k2_B[i, j, k, c] + 2.0*k3_B[i, j, k, c] + k4_B[i, j, k, c])
