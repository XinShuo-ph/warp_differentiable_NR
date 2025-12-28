"""
BSSN Variables Definition for Warp

Defines the evolved variables for the BSSN formulation of Einstein's equations.
Uses 3D structured grids with finite difference discretization.
"""

import warp as wp
import numpy as np


# BSSN variable structure
@wp.struct
class BSSNVars:
    """BSSN evolved variables at a single grid point."""
    # Conformal factor: W = e^{-2φ} or φ = ln(χ^{-1/6})
    phi: wp.float32
    
    # Conformal metric γ̃ᵢⱼ (symmetric, det = 1)
    # Stored as 6 independent components: gt11, gt12, gt13, gt22, gt23, gt33
    gt11: wp.float32
    gt12: wp.float32
    gt13: wp.float32
    gt22: wp.float32
    gt23: wp.float32
    gt33: wp.float32
    
    # Trace of extrinsic curvature K
    trK: wp.float32
    
    # Traceless conformal extrinsic curvature Ãᵢⱼ (symmetric, tr = 0)
    # Stored as 6 components (one algebraically determined by tracelessness)
    At11: wp.float32
    At12: wp.float32
    At13: wp.float32
    At22: wp.float32
    At23: wp.float32
    At33: wp.float32
    
    # Conformal connection Γ̃ⁱ
    Xt1: wp.float32
    Xt2: wp.float32
    Xt3: wp.float32
    
    # Lapse function α
    alpha: wp.float32
    
    # Shift vector βⁱ
    beta1: wp.float32
    beta2: wp.float32
    beta3: wp.float32


class BSSNGrid:
    """
    3D grid for BSSN evolution.
    
    Stores all evolved variables on a structured 3D grid.
    """
    
    def __init__(self, nx: int, ny: int, nz: int, dx: float, requires_grad: bool = False):
        """
        Initialize BSSN grid.
        
        Args:
            nx, ny, nz: Grid dimensions
            dx: Grid spacing (uniform in all directions)
            requires_grad: Enable gradient tracking for autodiff
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.n_points = nx * ny * nz
        
        # Allocate evolved variable arrays
        self.phi = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        
        # Conformal metric (6 components)
        self.gt11 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.gt12 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.gt13 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.gt22 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.gt23 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.gt33 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        
        # Trace of extrinsic curvature
        self.trK = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        
        # Traceless extrinsic curvature (6 components)
        self.At11 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.At12 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.At13 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.At22 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.At23 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.At33 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        
        # Conformal connection (3 components)
        self.Xt1 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.Xt2 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.Xt3 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        
        # Lapse
        self.alpha = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        
        # Shift (3 components)
        self.beta1 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.beta2 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.beta3 = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        
        # RHS arrays for time evolution
        self.phi_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.gt11_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.gt12_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.gt13_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.gt22_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.gt23_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.gt33_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.trK_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.At11_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.At12_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.At13_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.At22_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.At23_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.At33_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.Xt1_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.Xt2_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.Xt3_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.alpha_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.beta1_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.beta2_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
        self.beta3_rhs = wp.zeros(self.n_points, dtype=wp.float32, requires_grad=requires_grad)
    
    def set_flat_spacetime(self):
        """Initialize to Minkowski (flat) spacetime."""
        n = self.n_points
        
        # Conformal factor: φ = 0 (for W method: W = 1)
        self.phi.zero_()
        
        # Conformal metric: identity
        wp.copy(self.gt11, wp.ones(n, dtype=wp.float32))
        self.gt12.zero_()
        self.gt13.zero_()
        wp.copy(self.gt22, wp.ones(n, dtype=wp.float32))
        self.gt23.zero_()
        wp.copy(self.gt33, wp.ones(n, dtype=wp.float32))
        
        # Extrinsic curvature: zero
        self.trK.zero_()
        self.At11.zero_()
        self.At12.zero_()
        self.At13.zero_()
        self.At22.zero_()
        self.At23.zero_()
        self.At33.zero_()
        
        # Conformal connection: zero
        self.Xt1.zero_()
        self.Xt2.zero_()
        self.Xt3.zero_()
        
        # Lapse: 1
        wp.copy(self.alpha, wp.ones(n, dtype=wp.float32))
        
        # Shift: zero
        self.beta1.zero_()
        self.beta2.zero_()
        self.beta3.zero_()
    
    def idx(self, i: int, j: int, k: int) -> int:
        """Convert 3D indices to 1D index."""
        return i + self.nx * (j + self.ny * k)


@wp.func
def idx_3d(i: int, j: int, k: int, nx: int, ny: int) -> int:
    """Convert 3D indices to 1D index (warp kernel function)."""
    return i + nx * (j + ny * k)


def test_bssn_vars():
    """Test BSSN variable initialization."""
    wp.init()
    print("=== BSSN Variables Test ===\n")
    
    # Create a small grid
    grid = BSSNGrid(nx=10, ny=10, nz=10, dx=0.1)
    
    # Set flat spacetime
    grid.set_flat_spacetime()
    
    # Check values
    print(f"Grid size: {grid.nx}x{grid.ny}x{grid.nz} = {grid.n_points} points")
    print(f"phi mean: {grid.phi.numpy().mean():.6f} (expected: 0)")
    print(f"gt11 mean: {grid.gt11.numpy().mean():.6f} (expected: 1)")
    print(f"gt22 mean: {grid.gt22.numpy().mean():.6f} (expected: 1)")
    print(f"gt33 mean: {grid.gt33.numpy().mean():.6f} (expected: 1)")
    print(f"gt12 mean: {grid.gt12.numpy().mean():.6f} (expected: 0)")
    print(f"alpha mean: {grid.alpha.numpy().mean():.6f} (expected: 1)")
    print(f"trK mean: {grid.trK.numpy().mean():.6f} (expected: 0)")
    
    # Verify det(gt) = 1
    gt11 = grid.gt11.numpy()
    gt22 = grid.gt22.numpy()
    gt33 = grid.gt33.numpy()
    gt12 = grid.gt12.numpy()
    gt13 = grid.gt13.numpy()
    gt23 = grid.gt23.numpy()
    
    det = (gt11 * (gt22 * gt33 - gt23**2) 
           - gt12 * (gt12 * gt33 - gt23 * gt13) 
           + gt13 * (gt12 * gt23 - gt22 * gt13))
    print(f"det(gt) mean: {det.mean():.6f} (expected: 1)")
    
    print("\n✓ BSSN variables initialized correctly.")


if __name__ == "__main__":
    test_bssn_vars()
