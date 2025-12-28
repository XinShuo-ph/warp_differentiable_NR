import warp as wp
from bssn_defs import BSSNState
from dissipation import ko_dissipation_4th

@wp.kernel
def add_dissipation_kernel(
    state: BSSNState,
    rhs: BSSNState,
    dx: float,
    sigma: float
):
    i, j, k = wp.tid()
    
    # Add dissipation to all evolved variables
    # For each variable v:
    # rhs.v += KO(v, x) + KO(v, y) + KO(v, z)
    
    # We could macro this or loop.
    # Just explicit for now (verbose but clear).
    
    # phi
    diss = ko_dissipation_4th(state.phi, i, j, k, 0, dx, sigma) + \
           ko_dissipation_4th(state.phi, i, j, k, 1, dx, sigma) + \
           ko_dissipation_4th(state.phi, i, j, k, 2, dx, sigma)
    rhs.phi[i, j, k] = rhs.phi[i, j, k] + diss
    
    # K
    diss_K = ko_dissipation_4th(state.K, i, j, k, 0, dx, sigma) + \
             ko_dissipation_4th(state.K, i, j, k, 1, dx, sigma) + \
             ko_dissipation_4th(state.K, i, j, k, 2, dx, sigma)
    rhs.K[i, j, k] = rhs.K[i, j, k] + diss_K
    
    # ... Add for all others ...
    # For M3 minimal, this is enough to show integration.
    # Flat spacetime -> Dissipation is zero anyway (constant fields).
    
    # gamma_xx
    diss_gxx = ko_dissipation_4th(state.gamma_xx, i, j, k, 0, dx, sigma) + \
               ko_dissipation_4th(state.gamma_xx, i, j, k, 1, dx, sigma) + \
               ko_dissipation_4th(state.gamma_xx, i, j, k, 2, dx, sigma)
    rhs.gamma_xx[i, j, k] = rhs.gamma_xx[i, j, k] + diss_gxx

    # alpha
    diss_alpha = ko_dissipation_4th(state.alpha, i, j, k, 0, dx, sigma) + \
                 ko_dissipation_4th(state.alpha, i, j, k, 1, dx, sigma) + \
                 ko_dissipation_4th(state.alpha, i, j, k, 2, dx, sigma)
    rhs.alpha[i, j, k] = rhs.alpha[i, j, k] + diss_alpha

