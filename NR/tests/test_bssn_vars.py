"""
Test BSSN variable initialization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import warp as wp
from bssn_vars import BSSNVars, BSSNGrid

def test_flat_spacetime():
    """Test flat spacetime initialization"""
    wp.init()
    
    with wp.ScopedDevice("cpu"):
        grid = BSSNGrid(resolution=16, xmin=-1.0, xmax=1.0)
        vars = BSSNVars(resolution=16)
        
        vars.set_flat_spacetime()
        
        # Check conformal factor
        W_np = vars.W.numpy()
        assert abs(W_np[8, 8, 8] - 1.0) < 1e-6, "W should be 1.0"
        
        # Check conformal metric
        gxx_np = vars.gamt_xx.numpy()
        gxy_np = vars.gamt_xy.numpy()
        gyy_np = vars.gamt_yy.numpy()
        
        assert abs(gxx_np[8, 8, 8] - 1.0) < 1e-6, "gamt_xx should be 1.0"
        assert abs(gxy_np[8, 8, 8] - 0.0) < 1e-6, "gamt_xy should be 0.0"
        assert abs(gyy_np[8, 8, 8] - 1.0) < 1e-6, "gamt_yy should be 1.0"
        
        # Check lapse
        alpha_np = vars.alpha.numpy()
        assert abs(alpha_np[8, 8, 8] - 1.0) < 1e-6, "alpha should be 1.0"
        
        # Check extrinsic curvature
        exKh_np = vars.exKh.numpy()
        assert abs(exKh_np[8, 8, 8] - 0.0) < 1e-6, "exKh should be 0.0"
        
        print("Flat spacetime initialization: PASS")
        print(f"Grid: {grid.res}^3, dx = {grid.dx:.4f}")
        print(f"Domain: [{grid.xmin}, {grid.xmax}]^3")

if __name__ == "__main__":
    test_flat_spacetime()
