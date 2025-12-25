"""
BSSN variable definitions for Warp implementation.

BSSN evolved variables (3+1 formulation):
- phi: conformal factor (or W = exp(-2*phi))
- gt_xx, gt_xy, gt_xz, gt_yy, gt_yz, gt_zz: conformal metric (6 components)
- At_xx, At_xy, At_xz, At_yy, At_yz, At_zz: traceless conformal extrinsic curvature (6 components)
- Gamma_x, Gamma_y, Gamma_z: contracted Christoffel symbols (3 components)
- K: trace of extrinsic curvature (1 component)
- alpha: lapse function (1 component)
- beta_x, beta_y, beta_z: shift vector (3 components)

Total: 25 evolved variables
"""

import warp as wp
import numpy as np


class BSSNVariables:
    """Container for BSSN field variables on a 3D grid"""
    
    def __init__(self, nx, ny, nz):
        """
        Initialize BSSN variables on a 3D grid.
        
        Args:
            nx, ny, nz: Grid dimensions
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.shape = (nx, ny, nz)
        
        # Conformal factor (log conformal factor, phi = ln(psi))
        self.phi = wp.zeros(self.shape, dtype=wp.float32)
        
        # Conformal metric (6 independent components, symmetric)
        self.gt_xx = wp.zeros(self.shape, dtype=wp.float32)
        self.gt_xy = wp.zeros(self.shape, dtype=wp.float32)
        self.gt_xz = wp.zeros(self.shape, dtype=wp.float32)
        self.gt_yy = wp.zeros(self.shape, dtype=wp.float32)
        self.gt_yz = wp.zeros(self.shape, dtype=wp.float32)
        self.gt_zz = wp.zeros(self.shape, dtype=wp.float32)
        
        # Traceless conformal extrinsic curvature (6 components)
        self.At_xx = wp.zeros(self.shape, dtype=wp.float32)
        self.At_xy = wp.zeros(self.shape, dtype=wp.float32)
        self.At_xz = wp.zeros(self.shape, dtype=wp.float32)
        self.At_yy = wp.zeros(self.shape, dtype=wp.float32)
        self.At_yz = wp.zeros(self.shape, dtype=wp.float32)
        self.At_zz = wp.zeros(self.shape, dtype=wp.float32)
        
        # Contracted Christoffel symbols (3 components)
        self.Gamma_x = wp.zeros(self.shape, dtype=wp.float32)
        self.Gamma_y = wp.zeros(self.shape, dtype=wp.float32)
        self.Gamma_z = wp.zeros(self.shape, dtype=wp.float32)
        
        # Trace of extrinsic curvature
        self.K = wp.zeros(self.shape, dtype=wp.float32)
        
        # Lapse function
        self.alpha = wp.zeros(self.shape, dtype=wp.float32)
        
        # Shift vector (3 components)
        self.beta_x = wp.zeros(self.shape, dtype=wp.float32)
        self.beta_y = wp.zeros(self.shape, dtype=wp.float32)
        self.beta_z = wp.zeros(self.shape, dtype=wp.float32)
    
    def set_flat_spacetime(self):
        """Initialize variables to flat spacetime (Minkowski in Cartesian coords)"""
        # Conformal factor: phi = 0 (psi = 1)
        self.phi.fill_(0.0)
        
        # Conformal metric: flat Euclidean metric
        self.gt_xx.fill_(1.0)
        self.gt_xy.fill_(0.0)
        self.gt_xz.fill_(0.0)
        self.gt_yy.fill_(1.0)
        self.gt_yz.fill_(0.0)
        self.gt_zz.fill_(1.0)
        
        # Extrinsic curvature: zero (no time evolution initially)
        self.At_xx.fill_(0.0)
        self.At_xy.fill_(0.0)
        self.At_xz.fill_(0.0)
        self.At_yy.fill_(0.0)
        self.At_yz.fill_(0.0)
        self.At_zz.fill_(0.0)
        
        # Christoffel symbols: zero (flat space)
        self.Gamma_x.fill_(0.0)
        self.Gamma_y.fill_(0.0)
        self.Gamma_z.fill_(0.0)
        
        # Trace of K: zero
        self.K.fill_(0.0)
        
        # Lapse: alpha = 1 (normal time)
        self.alpha.fill_(1.0)
        
        # Shift: beta = 0 (Cartesian coordinates)
        self.beta_x.fill_(0.0)
        self.beta_y.fill_(0.0)
        self.beta_z.fill_(0.0)
    
    def get_all_vars(self):
        """Return list of all variable arrays"""
        return [
            self.phi,
            self.gt_xx, self.gt_xy, self.gt_xz, self.gt_yy, self.gt_yz, self.gt_zz,
            self.At_xx, self.At_xy, self.At_xz, self.At_yy, self.At_yz, self.At_zz,
            self.Gamma_x, self.Gamma_y, self.Gamma_z,
            self.K,
            self.alpha,
            self.beta_x, self.beta_y, self.beta_z
        ]
    
    def copy_from(self, other):
        """Copy all variables from another BSSNVariables instance"""
        for var_self, var_other in zip(self.get_all_vars(), other.get_all_vars()):
            wp.copy(var_self, var_other)


class BSSNRHSVariables:
    """Container for RHS (time derivatives) of BSSN variables"""
    
    def __init__(self, nx, ny, nz):
        """Initialize RHS variables (same structure as BSSNVariables)"""
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.shape = (nx, ny, nz)
        
        # RHS for each evolved variable
        self.phi_rhs = wp.zeros(self.shape, dtype=wp.float32)
        
        self.gt_xx_rhs = wp.zeros(self.shape, dtype=wp.float32)
        self.gt_xy_rhs = wp.zeros(self.shape, dtype=wp.float32)
        self.gt_xz_rhs = wp.zeros(self.shape, dtype=wp.float32)
        self.gt_yy_rhs = wp.zeros(self.shape, dtype=wp.float32)
        self.gt_yz_rhs = wp.zeros(self.shape, dtype=wp.float32)
        self.gt_zz_rhs = wp.zeros(self.shape, dtype=wp.float32)
        
        self.At_xx_rhs = wp.zeros(self.shape, dtype=wp.float32)
        self.At_xy_rhs = wp.zeros(self.shape, dtype=wp.float32)
        self.At_xz_rhs = wp.zeros(self.shape, dtype=wp.float32)
        self.At_yy_rhs = wp.zeros(self.shape, dtype=wp.float32)
        self.At_yz_rhs = wp.zeros(self.shape, dtype=wp.float32)
        self.At_zz_rhs = wp.zeros(self.shape, dtype=wp.float32)
        
        self.Gamma_x_rhs = wp.zeros(self.shape, dtype=wp.float32)
        self.Gamma_y_rhs = wp.zeros(self.shape, dtype=wp.float32)
        self.Gamma_z_rhs = wp.zeros(self.shape, dtype=wp.float32)
        
        self.K_rhs = wp.zeros(self.shape, dtype=wp.float32)
        
        self.alpha_rhs = wp.zeros(self.shape, dtype=wp.float32)
        
        self.beta_x_rhs = wp.zeros(self.shape, dtype=wp.float32)
        self.beta_y_rhs = wp.zeros(self.shape, dtype=wp.float32)
        self.beta_z_rhs = wp.zeros(self.shape, dtype=wp.float32)
    
    def zero(self):
        """Set all RHS variables to zero"""
        for var in self.get_all_vars():
            var.fill_(0.0)
    
    def get_all_vars(self):
        """Return list of all RHS variable arrays"""
        return [
            self.phi_rhs,
            self.gt_xx_rhs, self.gt_xy_rhs, self.gt_xz_rhs, 
            self.gt_yy_rhs, self.gt_yz_rhs, self.gt_zz_rhs,
            self.At_xx_rhs, self.At_xy_rhs, self.At_xz_rhs,
            self.At_yy_rhs, self.At_yz_rhs, self.At_zz_rhs,
            self.Gamma_x_rhs, self.Gamma_y_rhs, self.Gamma_z_rhs,
            self.K_rhs,
            self.alpha_rhs,
            self.beta_x_rhs, self.beta_y_rhs, self.beta_z_rhs
        ]


class GridParameters:
    """Grid and physical parameters for BSSN evolution"""
    
    def __init__(self, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax):
        """
        Initialize grid parameters.
        
        Args:
            nx, ny, nz: Number of grid points
            xmin, xmax, ymin, ymax, zmin, zmax: Physical domain bounds
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        
        # Grid spacing
        self.dx = (xmax - xmin) / (nx - 1) if nx > 1 else 1.0
        self.dy = (ymax - ymin) / (ny - 1) if ny > 1 else 1.0
        self.dz = (zmax - zmin) / (nz - 1) if nz > 1 else 1.0
        
        # Inverse grid spacing (for derivatives)
        self.idx = 1.0 / self.dx
        self.idy = 1.0 / self.dy
        self.idz = 1.0 / self.dz
        
    def get_coords(self, i, j, k):
        """Get physical coordinates for grid point (i,j,k)"""
        x = self.xmin + i * self.dx
        y = self.ymin + j * self.dy
        z = self.zmin + k * self.dz
        return x, y, z
