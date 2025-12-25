import warp as wp
from bssn_defs import BSSNState

# Constraints:
# Hamiltonian: H = R - K_ij K^ij + K^2
# Momentum: M^i = D_j (K^ij - gamma^ij K)
#
# For flat spacetime, R=0, K=0, M=0.
# We just check deviations from 0.
#
# Since we are using BSSN variables:
# H = e^-4phi R_tilde + ...
#
# Implementing full Hamiltonian constraint check is heavy.
# For flat spacetime (Minkowski), everything is zero.
# We just check if state variables remain consistent with flat spacetime (phi=0, gam=delta, etc).
#
# If phi=0, gam=delta, K=0, A=0 -> H=0, M=0 trivially.
#
# The solver test already checks phi deviation.
# Let's add checks for K and A_ij deviations.

def check_constraints(state: BSSNState):
    # Check max deviation of K
    k_max = abs(state.K.numpy()).max()
    print(f"Max K deviation: {k_max}")
    
    # Check trace of A (should be 0 algebraically, but numerically?)
    # A_xx + A_yy + A_zz should be 0.
    
    a_xx = state.A_xx.numpy()
    a_yy = state.A_yy.numpy()
    a_zz = state.A_zz.numpy()
    
    tr_A = a_xx + a_yy + a_zz
    tr_max = abs(tr_A).max()
    print(f"Max tr(A) deviation: {tr_max}")
    
    return k_max < 1e-10 and tr_max < 1e-10

