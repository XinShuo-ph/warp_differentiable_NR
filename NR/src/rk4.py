import warp as wp
from bssn_defs import BSSNState

@wp.kernel
def state_add_scaled(
    y_out: BSSNState,
    y_in: BSSNState,
    dy: BSSNState,
    scale: float
):
    # y_out = y_in + scale * dy
    i, j, k = wp.tid()
    
    # Macro or loop over fields? 
    # Must be explicit in kernel
    
    y_out.phi[i, j, k] = y_in.phi[i, j, k] + scale * dy.phi[i, j, k]
    y_out.gamma_xx[i, j, k] = y_in.gamma_xx[i, j, k] + scale * dy.gamma_xx[i, j, k]
    y_out.gamma_xy[i, j, k] = y_in.gamma_xy[i, j, k] + scale * dy.gamma_xy[i, j, k]
    y_out.gamma_xz[i, j, k] = y_in.gamma_xz[i, j, k] + scale * dy.gamma_xz[i, j, k]
    y_out.gamma_yy[i, j, k] = y_in.gamma_yy[i, j, k] + scale * dy.gamma_yy[i, j, k]
    y_out.gamma_yz[i, j, k] = y_in.gamma_yz[i, j, k] + scale * dy.gamma_yz[i, j, k]
    y_out.gamma_zz[i, j, k] = y_in.gamma_zz[i, j, k] + scale * dy.gamma_zz[i, j, k]
    
    y_out.K[i, j, k] = y_in.K[i, j, k] + scale * dy.K[i, j, k]
    y_out.A_xx[i, j, k] = y_in.A_xx[i, j, k] + scale * dy.A_xx[i, j, k]
    y_out.A_xy[i, j, k] = y_in.A_xy[i, j, k] + scale * dy.A_xy[i, j, k]
    y_out.A_xz[i, j, k] = y_in.A_xz[i, j, k] + scale * dy.A_xz[i, j, k]
    y_out.A_yy[i, j, k] = y_in.A_yy[i, j, k] + scale * dy.A_yy[i, j, k]
    y_out.A_yz[i, j, k] = y_in.A_yz[i, j, k] + scale * dy.A_yz[i, j, k]
    y_out.A_zz[i, j, k] = y_in.A_zz[i, j, k] + scale * dy.A_zz[i, j, k]
    
    y_out.Gam_x[i, j, k] = y_in.Gam_x[i, j, k] + scale * dy.Gam_x[i, j, k]
    y_out.Gam_y[i, j, k] = y_in.Gam_y[i, j, k] + scale * dy.Gam_y[i, j, k]
    y_out.Gam_z[i, j, k] = y_in.Gam_z[i, j, k] + scale * dy.Gam_z[i, j, k]
    
    y_out.alpha[i, j, k] = y_in.alpha[i, j, k] + scale * dy.alpha[i, j, k]
    y_out.beta_x[i, j, k] = y_in.beta_x[i, j, k] + scale * dy.beta_x[i, j, k]
    y_out.beta_y[i, j, k] = y_in.beta_y[i, j, k] + scale * dy.beta_y[i, j, k]
    y_out.beta_z[i, j, k] = y_in.beta_z[i, j, k] + scale * dy.beta_z[i, j, k]
    y_out.B_x[i, j, k] = y_in.B_x[i, j, k] + scale * dy.B_x[i, j, k]
    y_out.B_y[i, j, k] = y_in.B_y[i, j, k] + scale * dy.B_y[i, j, k]
    y_out.B_z[i, j, k] = y_in.B_z[i, j, k] + scale * dy.B_z[i, j, k]
