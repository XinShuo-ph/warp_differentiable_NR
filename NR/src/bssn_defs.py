import warp as wp
import math

# BSSN variables structure
# We need 24 evolution variables per grid point:
# phi (1)
# gamma_ij (6, symmetric) -> xx, xy, xz, yy, yz, zz
# K (1)
# A_ij (6, symmetric, traceless) -> xx, xy, xz, yy, yz, zz (tracelessness enforced algebraically)
# Gamma_i (3) -> x, y, z
#
# Gauge variables (evolved):
# alpha (1)
# beta_i (3)
# B_i (3)
#
# Total: 1+6+1+6+3 + 1+3+3 = 24.

@wp.struct
class BSSNState:
    # Metric variables
    phi: wp.array(dtype=float, ndim=3)
    gamma_xx: wp.array(dtype=float, ndim=3)
    gamma_xy: wp.array(dtype=float, ndim=3)
    gamma_xz: wp.array(dtype=float, ndim=3)
    gamma_yy: wp.array(dtype=float, ndim=3)
    gamma_yz: wp.array(dtype=float, ndim=3)
    gamma_zz: wp.array(dtype=float, ndim=3)
    
    # Curvature variables
    K: wp.array(dtype=float, ndim=3)
    A_xx: wp.array(dtype=float, ndim=3)
    A_xy: wp.array(dtype=float, ndim=3)
    A_xz: wp.array(dtype=float, ndim=3)
    A_yy: wp.array(dtype=float, ndim=3)
    A_yz: wp.array(dtype=float, ndim=3)
    A_zz: wp.array(dtype=float, ndim=3)
    
    # Connection variables
    Gam_x: wp.array(dtype=float, ndim=3)
    Gam_y: wp.array(dtype=float, ndim=3)
    Gam_z: wp.array(dtype=float, ndim=3)
    
    # Gauge variables
    alpha: wp.array(dtype=float, ndim=3)
    beta_x: wp.array(dtype=float, ndim=3)
    beta_y: wp.array(dtype=float, ndim=3)
    beta_z: wp.array(dtype=float, ndim=3)
    B_x: wp.array(dtype=float, ndim=3)
    B_y: wp.array(dtype=float, ndim=3)
    B_z: wp.array(dtype=float, ndim=3)

def allocate_bssn_state(shape, device=None):
    # Helper to allocate all fields
    fields = {}
    names = [
        "phi", 
        "gamma_xx", "gamma_xy", "gamma_xz", "gamma_yy", "gamma_yz", "gamma_zz",
        "K", 
        "A_xx", "A_xy", "A_xz", "A_yy", "A_yz", "A_zz",
        "Gam_x", "Gam_y", "Gam_z",
        "alpha", 
        "beta_x", "beta_y", "beta_z", 
        "B_x", "B_y", "B_z"
    ]
    
    # Create struct instance
    state = BSSNState()
    
    for name in names:
        arr = wp.zeros(shape, dtype=float, device=device, requires_grad=True)
        setattr(state, name, arr)
        
    return state

@wp.kernel
def init_flat_spacetime(
    phi: wp.array(dtype=float, ndim=3),
    gamma_xx: wp.array(dtype=float, ndim=3),
    gamma_yy: wp.array(dtype=float, ndim=3),
    gamma_zz: wp.array(dtype=float, ndim=3),
    alpha: wp.array(dtype=float, ndim=3),
    # Implicitly others are zero from allocation
):
    i, j, k = wp.tid()
    
    # Flat spacetime:
    # phi = 0
    # gamma_ij = delta_ij
    # K = 0
    # A_ij = 0
    # Gam_i = 0
    # alpha = 1
    # beta_i = 0
    
    phi[i, j, k] = 0.0
    gamma_xx[i, j, k] = 1.0
    gamma_yy[i, j, k] = 1.0
    gamma_zz[i, j, k] = 1.0
    alpha[i, j, k] = 1.0

def initialize(state, device=None):
    shape = state.phi.shape
    wp.launch(
        kernel=init_flat_spacetime,
        dim=shape,
        inputs=[state.phi, state.gamma_xx, state.gamma_yy, state.gamma_zz, state.alpha],
        device=device
    )

if __name__ == "__main__":
    wp.init()
    res = 32
    shape = (res, res, res)
    state = allocate_bssn_state(shape)
    initialize(state)
    print("BSSN state allocated and initialized to flat spacetime.")
    
    # Verify values
    phi_np = state.phi.numpy()
    assert (phi_np == 0).all()
    alpha_np = state.alpha.numpy()
    assert (alpha_np == 1).all()
    print("Verification passed.")
