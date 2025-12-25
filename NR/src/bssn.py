import warp as wp

# Indices for symmetric tensors stored as 6 components
# xx=0, xy=1, xz=2, yy=3, yz=4, zz=5
IDX_XX = 0
IDX_XY = 1
IDX_XZ = 2
IDX_YY = 3
IDX_YZ = 4
IDX_ZZ = 5

class BSSNState:
    def __init__(self, res, bounds_lo, bounds_hi):
        self.res = res
        self.shape = (res[0], res[1], res[2])
        self.dx = (bounds_hi[0] - bounds_lo[0]) / res[0]
        
        # Allocate fields
        # Conformal factor phi
        self.phi = wp.zeros(self.shape, dtype=float)
        
        # Conformal metric gamma_tilde (symmetric 3x3 -> 6 components)
        # We could use wp.mat33, but symmetric storage saves memory? 
        # For simplicity in kernels, let's use separate arrays or a last dim of 6.
        # Let's use last dim 6 for tensors.
        self.gamma_tilde = wp.zeros(self.shape + (6,), dtype=float)
        
        # Trace of extrinsic curvature K
        self.K = wp.zeros(self.shape, dtype=float)
        
        # Conformal traceless extrinsic curvature A_tilde (symmetric 3x3 -> 6)
        self.A_tilde = wp.zeros(self.shape + (6,), dtype=float)
        
        # Conformal connection functions Gamma_tilde (vector -> 3)
        self.Gam_tilde = wp.zeros(self.shape + (3,), dtype=float)
        
        # Gauge variables
        self.alpha = wp.zeros(self.shape, dtype=float)
        self.beta = wp.zeros(self.shape + (3,), dtype=float)
        self.B = wp.zeros(self.shape + (3,), dtype=float) # For Gamma-driver shift
        
    def init_flat_spacetime(self):
        wp.launch(
            kernel=init_flat_kernel,
            dim=self.shape,
            inputs=[self.phi, self.gamma_tilde, self.K, self.A_tilde, self.Gam_tilde, self.alpha, self.beta, self.B]
        )

@wp.kernel
def init_flat_kernel(
    phi: wp.array(dtype=float, ndim=3),
    gamma_tilde: wp.array(dtype=float, ndim=4),
    K: wp.array(dtype=float, ndim=3),
    A_tilde: wp.array(dtype=float, ndim=4),
    Gam_tilde: wp.array(dtype=float, ndim=4),
    alpha: wp.array(dtype=float, ndim=3),
    beta: wp.array(dtype=float, ndim=4),
    B: wp.array(dtype=float, ndim=4)
):
    i, j, k = wp.tid()
    
    phi[i, j, k] = 0.0
    K[i, j, k] = 0.0
    alpha[i, j, k] = 1.0
    
    # gamma_tilde = delta_ij
    gamma_tilde[i, j, k, IDX_XX] = 1.0
    gamma_tilde[i, j, k, IDX_XY] = 0.0
    gamma_tilde[i, j, k, IDX_XZ] = 0.0
    gamma_tilde[i, j, k, IDX_YY] = 1.0
    gamma_tilde[i, j, k, IDX_YZ] = 0.0
    gamma_tilde[i, j, k, IDX_ZZ] = 1.0
    
    # A_tilde = 0
    for c in range(6):
        A_tilde[i, j, k, c] = 0.0
        
    # Gam_tilde = 0
    for c in range(3):
        Gam_tilde[i, j, k, c] = 0.0
        beta[i, j, k, c] = 0.0
        B[i, j, k, c] = 0.0
