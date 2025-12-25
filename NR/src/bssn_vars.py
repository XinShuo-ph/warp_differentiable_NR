"""
BSSN variables and data structures for Warp

Defines evolved fields and auxiliary variables for BSSN formulation
"""

import warp as wp
import numpy as np

# BSSN evolved variables (12 variables in 3D)
# Conformal factor: W (1 component)
# Conformal metric: gamt_ij (6 components, symmetric)
# Trace of extrinsic curvature: exKh (1 component)
# Tracefree extrinsic curvature: exAt_ij (6 components, symmetric, tracefree)
# Conformal connection: trGt_i (3 components)
# Gauge: alpha (lapse), beta_i (shift, 3 components)

class BSSNVars:
    """Container for BSSN field variables"""
    
    def __init__(self, resolution: int):
        """
        Initialize BSSN variables on a 3D grid
        
        Args:
            resolution: grid resolution (cubic grid)
        """
        self.res = resolution
        self.shape = (resolution, resolution, resolution)
        
        # Conformal factor
        self.W = wp.zeros(self.shape, dtype=wp.float32)
        
        # Conformal metric (symmetric 3x3)
        self.gamt_xx = wp.zeros(self.shape, dtype=wp.float32)
        self.gamt_xy = wp.zeros(self.shape, dtype=wp.float32)
        self.gamt_xz = wp.zeros(self.shape, dtype=wp.float32)
        self.gamt_yy = wp.zeros(self.shape, dtype=wp.float32)
        self.gamt_yz = wp.zeros(self.shape, dtype=wp.float32)
        self.gamt_zz = wp.zeros(self.shape, dtype=wp.float32)
        
        # Trace extrinsic curvature
        self.exKh = wp.zeros(self.shape, dtype=wp.float32)
        
        # Tracefree conformal extrinsic curvature (symmetric 3x3, tracefree)
        self.exAt_xx = wp.zeros(self.shape, dtype=wp.float32)
        self.exAt_xy = wp.zeros(self.shape, dtype=wp.float32)
        self.exAt_xz = wp.zeros(self.shape, dtype=wp.float32)
        self.exAt_yy = wp.zeros(self.shape, dtype=wp.float32)
        self.exAt_yz = wp.zeros(self.shape, dtype=wp.float32)
        self.exAt_zz = wp.zeros(self.shape, dtype=wp.float32)
        
        # Conformal connection functions
        self.trGt_x = wp.zeros(self.shape, dtype=wp.float32)
        self.trGt_y = wp.zeros(self.shape, dtype=wp.float32)
        self.trGt_z = wp.zeros(self.shape, dtype=wp.float32)
        
        # Lapse
        self.alpha = wp.zeros(self.shape, dtype=wp.float32)
        
        # Shift
        self.beta_x = wp.zeros(self.shape, dtype=wp.float32)
        self.beta_y = wp.zeros(self.shape, dtype=wp.float32)
        self.beta_z = wp.zeros(self.shape, dtype=wp.float32)
        
    def set_flat_spacetime(self):
        """Initialize to flat spacetime (Minkowski)"""
        # W = 1 (no conformal factor)
        self.W.fill_(1.0)
        
        # gamt_ij = delta_ij (flat conformal metric)
        self.gamt_xx.fill_(1.0)
        self.gamt_xy.fill_(0.0)
        self.gamt_xz.fill_(0.0)
        self.gamt_yy.fill_(1.0)
        self.gamt_yz.fill_(0.0)
        self.gamt_zz.fill_(1.0)
        
        # exKh = 0 (no extrinsic curvature)
        self.exKh.fill_(0.0)
        
        # exAt_ij = 0 (no extrinsic curvature)
        self.exAt_xx.fill_(0.0)
        self.exAt_xy.fill_(0.0)
        self.exAt_xz.fill_(0.0)
        self.exAt_yy.fill_(0.0)
        self.exAt_yz.fill_(0.0)
        self.exAt_zz.fill_(0.0)
        
        # trGt_i = 0 (flat connection)
        self.trGt_x.fill_(0.0)
        self.trGt_y.fill_(0.0)
        self.trGt_z.fill_(0.0)
        
        # alpha = 1 (unit lapse)
        self.alpha.fill_(1.0)
        
        # beta_i = 0 (zero shift)
        self.beta_x.fill_(0.0)
        self.beta_y.fill_(0.0)
        self.beta_z.fill_(0.0)

class BSSNGrid:
    """Grid specification for BSSN evolution"""
    
    def __init__(self, resolution: int, xmin: float, xmax: float):
        """
        Define computational grid
        
        Args:
            resolution: number of grid points per dimension
            xmin: minimum coordinate (cubic domain)
            xmax: maximum coordinate (cubic domain)
        """
        self.res = resolution
        self.xmin = xmin
        self.xmax = xmax
        self.dx = (xmax - xmin) / (resolution - 1)
        self.dy = self.dx
        self.dz = self.dx
        
        # Inverse grid spacing for derivatives
        self.idx = 1.0 / self.dx
        self.idy = 1.0 / self.dy
        self.idz = 1.0 / self.dz

def get_coordinate(i: int, j: int, k: int, grid: BSSNGrid) -> wp.vec3:
    """
    Get physical coordinates for grid index
    
    Args:
        i, j, k: grid indices
        grid: grid specification
    
    Returns:
        (x, y, z) coordinates
    """
    x = grid.xmin + i * grid.dx
    y = grid.xmin + j * grid.dy
    z = grid.xmin + k * grid.dz
    return wp.vec3(x, y, z)
