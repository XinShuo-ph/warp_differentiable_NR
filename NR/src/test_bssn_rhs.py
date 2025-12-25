import warp as wp
from bssn_defs import BSSNState, allocate_bssn_state, initialize
from bssn_rhs import bssn_rhs_kernel

# RK4 Integrator
def rk4_step(state, rhs_state, dt, dx):
    # k1 = f(y)
    # k2 = f(y + 0.5*dt*k1)
    # k3 = f(y + 0.5*dt*k2)
    # k4 = f(y + dt*k3)
    # y_new = y + dt/6 * (k1 + 2k2 + 2k3 + k4)
    
    # We need intermediate buffers.
    # Allocating them every step is slow. Should persist.
    # For M3, just allocate once in main.
    pass

def compute_rhs(state, rhs, dx):
    wp.launch(
        kernel=bssn_rhs_kernel,
        dim=state.phi.shape,
        inputs=[0.0, state, rhs, dx]
    )

if __name__ == "__main__":
    wp.init()
    res = 32
    dx = 1.0/res
    state = allocate_bssn_state((res, res, res))
    initialize(state)
    
    rhs = allocate_bssn_state((res, res, res))
    
    compute_rhs(state, rhs, dx)
    
    # Check that for flat spacetime, RHS is zero
    print("Checking RHS for flat spacetime...")
    max_val = 0.0
    
    # Check a few fields
    # Use simple loop or max reduction
    rhs_phi = rhs.phi.numpy()
    if abs(rhs_phi).max() > 1e-10:
        print("RHS phi non-zero:", abs(rhs_phi).max())
    else:
        print("RHS phi is zero.")
        
    rhs_gxx = rhs.gamma_xx.numpy()
    if abs(rhs_gxx).max() > 1e-10:
        print("RHS gamma_xx non-zero")
    else:
        print("RHS gamma_xx is zero.")
        
    print("Test Complete")
