import warp as wp
import math

@wp.kernel
def brill_lindquist_kernel(
    phi: wp.array(dtype=float, ndim=3),
    # Puncture params
    m1: float, x1: wp.vec3,
    m2: float, x2: wp.vec3,
    dx: float, dy: float, dz: float,
    nx: int, ny: int, nz: int
):
    i, j, k = wp.tid()
    
    # Grid coordinates (centered at 0?)
    # Assume domain is [-L, L]
    # Let's say domain is defined by dx.
    # We need an origin. Assume index (nx/2, ny/2, nz/2) is (0,0,0) or passed as param.
    # Let's assume (0,0,0) index is at coordinate (ox, oy, oz).
    # For now, let's assume domain is centered.
    
    # Coordinate of current point
    x = (float(i) - float(nx)*0.5) * dx
    y = (float(j) - float(ny)*0.5) * dy
    z = (float(k) - float(nz)*0.5) * dz
    
    pos = wp.vec3(x, y, z)
    
    # Distance to punctures
    r1 = wp.length(pos - x1)
    r2 = wp.length(pos - x2)
    
    # Regularize r to avoid division by zero if on grid point
    eps = 1e-6
    r1 = wp.max(r1, eps)
    r2 = wp.max(r2, eps)
    
    # Psi = 1 + m1/(2r1) + m2/(2r2)
    psi = 1.0 + m1 / (2.0 * r1) + m2 / (2.0 * r2)
    
    # Phi = ln(Psi)
    phi[i,j,k] = wp.log(psi)

def setup_brill_lindquist(state, m1, pos1, m2, pos2):
    # Initializes phi for Brill-Lindquist data
    # Other variables initialized to flat/zero by default in init_bssn_state, 
    # but gamma_tilde should be identity (conformally flat).
    
    wp.launch(
        kernel=brill_lindquist_kernel,
        dim=state.phi.shape,
        inputs=[
            state.phi,
            m1, wp.vec3(*pos1),
            m2, wp.vec3(*pos2),
            state.dx, state.dy, state.dz,
            state.nx, state.ny, state.nz
        ],
        device=state.phi.device
    )
    
    # Initialize alpha = psi^-2 (pre-collapsed lapse) or alpha = 1 (geodesic)
    # Moving Punctures usually starts with alpha = psi^-2 or similar "pre-collapsed" lapse
    # alpha = exp(-2 phi)
    wp.launch(
        kernel=initialize_lapse_kernel,
        dim=state.alpha.shape,
        inputs=[state.alpha, state.phi],
        device=state.alpha.device
    )

@wp.kernel
def initialize_lapse_kernel(alpha: wp.array(dtype=float, ndim=3), phi: wp.array(dtype=float, ndim=3)):
    i,j,k = wp.tid()
    # Pre-collapsed lapse: alpha = psi^-2 = exp(-2 phi)
    alpha[i,j,k] = wp.exp(-2.0 * phi[i,j,k])

@wp.kernel
def bowen_york_kernel(
    A_tilde: wp.array(dtype=wp.mat33, ndim=3),
    # Puncture 1
    x1: wp.vec3, P1: wp.vec3,
    # Puncture 2
    x2: wp.vec3, P2: wp.vec3,
    dx: float, dy: float, dz: float,
    nx: int, ny: int, nz: int
):
    i, j, k = wp.tid()
    
    x = (float(i) - float(nx)*0.5) * dx
    y = (float(j) - float(ny)*0.5) * dy
    z = (float(k) - float(nz)*0.5) * dz
    pos = wp.vec3(x, y, z)
    
    # Contribution from P1
    r1_vec = pos - x1
    r1 = wp.length(r1_vec) + 1e-6
    n1 = r1_vec / r1
    
    # Term: 3/(2r^2) * (P_i n_j + P_j n_i - (delta_ij - n_i n_j) P.n)
    # Let's compute term for P1
    P1_dot_n1 = wp.dot(P1, n1)
    factor1 = 1.5 / (r1*r1)
    
    # Contribution from P2
    r2_vec = pos - x2
    r2 = wp.length(r2_vec) + 1e-6
    n2 = r2_vec / r2
    P2_dot_n2 = wp.dot(P2, n2)
    factor2 = 1.5 / (r2*r2)
    
    # Construct Matrix
    # A_ij = sum_k (factor_k * ...)
    
    # Warp doesn't support constructing matrix from loop easily inside kernel unless unrolled.
    # Just compute components.
    
    val = wp.mat33(0.0)
    
    # We can iterate 0..2 for rows and cols?
    # Or just write it out. 
    # Let's use a helper func? No, inline is fine for 3x3.
    
    # A_ab for P1
    # P_a n_b + P_b n_a - (delta_ab - n_a n_b) * P_dot_n
    # = P_a n_b + P_b n_a - delta_ab * P_dot_n + n_a n_b * P_dot_n
    
    # Total
    for r in range(3):
        for c in range(3):
            delta = 1.0 if r == c else 0.0
            
            # P1 term
            term1 = P1[r]*n1[c] + P1[c]*n1[r] - delta*P1_dot_n1 + n1[r]*n1[c]*P1_dot_n1
            
            # P2 term
            term2 = P2[r]*n2[c] + P2[c]*n2[r] - delta*P2_dot_n2 + n2[r]*n2[c]*P2_dot_n2
            
            val[r,c] = factor1 * term1 + factor2 * term2
            
    A_tilde[i,j,k] = val

def setup_bowen_york(state, pos1, P1, pos2, P2):
    wp.launch(
        kernel=bowen_york_kernel,
        dim=state.A_tilde.shape,
        inputs=[
            state.A_tilde,
            wp.vec3(*pos1), wp.vec3(*P1),
            wp.vec3(*pos2), wp.vec3(*P2),
            state.dx, state.dy, state.dz,
            state.nx, state.ny, state.nz
        ],
        device=state.A_tilde.device
    )

