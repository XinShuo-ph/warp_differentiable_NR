import warp as wp

# BSSN Variables Structure
# We will use a grid-based approach.
# For simplicity in Warp, we can store fields as separate arrays or a struct of arrays.

@wp.struct
class BSSNState:
    # Evolution variables
    phi: wp.array(dtype=float)        # Conformal factor
    gamma_tilde: wp.array(dtype=wp.mat33) # Conformal metric
    K: wp.array(dtype=float)          # Trace of extrinsic curvature
    A_tilde: wp.array(dtype=wp.mat33) # Conformal traceless extrinsic curvature
    Gamma_tilde: wp.array(dtype=wp.vec3) # Conformal connection functions
    
    # Gauge variables
    alpha: wp.array(dtype=float)      # Lapse
    beta: wp.array(dtype=wp.vec3)     # Shift
    B: wp.array(dtype=wp.vec3)        # Gamma-driver B
    
    # Grid info
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float

def allocate_bssn_state(res, device=None):
    nx, ny, nz = res
    shape = (nx, ny, nz)
    
    state = BSSNState()
    state.nx = nx
    state.ny = ny
    state.nz = nz
    state.dx = 1.0/nx # Standard unit box
    state.dy = 1.0/ny
    state.dz = 1.0/nz
    
    state.phi = wp.zeros(shape, dtype=float, device=device)
    state.gamma_tilde = wp.zeros(shape, dtype=wp.mat33, device=device)
    state.K = wp.zeros(shape, dtype=float, device=device)
    state.A_tilde = wp.zeros(shape, dtype=wp.mat33, device=device)
    state.Gamma_tilde = wp.zeros(shape, dtype=wp.vec3, device=device)
    state.alpha = wp.zeros(shape, dtype=float, device=device)
    state.beta = wp.zeros(shape, dtype=wp.vec3, device=device)
    state.B = wp.zeros(shape, dtype=wp.vec3, device=device)
    
    return state

@wp.kernel
def initialize_flat_spacetime(
    phi: wp.array(dtype=float, ndim=3),
    gamma_tilde: wp.array(dtype=wp.mat33, ndim=3),
    K: wp.array(dtype=float, ndim=3),
    A_tilde: wp.array(dtype=wp.mat33, ndim=3),
    Gamma_tilde: wp.array(dtype=wp.vec3, ndim=3),
    alpha: wp.array(dtype=float, ndim=3),
    beta: wp.array(dtype=wp.vec3, ndim=3),
    B: wp.array(dtype=wp.vec3, ndim=3)
):
    i, j, k = wp.tid()
    
    # Flat spacetime (Minkowski)
    # phi = 0
    # gamma_tilde = identity
    # K = 0
    # A_tilde = 0
    # Gamma_tilde = 0
    # alpha = 1
    # beta = 0
    # B = 0
    
    phi[i,j,k] = 0.0
    gamma_tilde[i,j,k] = wp.mat33(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )
    K[i,j,k] = 0.0
    A_tilde[i,j,k] = wp.mat33(0.0)
    Gamma_tilde[i,j,k] = wp.vec3(0.0)
    alpha[i,j,k] = 1.0
    beta[i,j,k] = wp.vec3(0.0)
    B[i,j,k] = wp.vec3(0.0)

def init_bssn_state(state):
    dim = (state.nx, state.ny, state.nz)
    wp.launch(
        kernel=initialize_flat_spacetime,
        dim=dim,
        inputs=[
            state.phi,
            state.gamma_tilde,
            state.K,
            state.A_tilde,
            state.Gamma_tilde,
            state.alpha,
            state.beta,
            state.B
        ]
    )
