"""
Test spatial derivatives
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import warp as wp
import numpy as np
from bssn_derivatives import deriv_x_4th, deriv_y_4th, deriv_z_4th
from bssn_vars import BSSNGrid

@wp.kernel
def compute_deriv_x_kernel(
    f: wp.array3d(dtype=float),
    df: wp.array3d(dtype=float),
    idx: float,
    nx: int,
    ny: int,
    nz: int
):
    i, j, k = wp.tid()
    if i < nx and j < ny and k < nz:
        df[i,j,k] = deriv_x_4th(f, i, j, k, idx, nx)

def test_derivatives():
    """Test 4th order derivatives on a known function"""
    wp.init()
    
    with wp.ScopedDevice("cpu"):
        res = 32
        grid = BSSNGrid(resolution=res, xmin=-1.0, xmax=1.0)
        
        # Create test function: f(x,y,z) = sin(2*pi*x)
        f_np = np.zeros((res, res, res), dtype=np.float32)
        df_exact_np = np.zeros((res, res, res), dtype=np.float32)
        
        for i in range(res):
            x = grid.xmin + i * grid.dx
            # f = sin(2*pi*x), df/dx = 2*pi*cos(2*pi*x)
            f_np[i, :, :] = np.sin(2.0 * np.pi * x)
            df_exact_np[i, :, :] = 2.0 * np.pi * np.cos(2.0 * np.pi * x)
        
        f = wp.array(f_np, dtype=wp.float32)
        df = wp.zeros((res, res, res), dtype=wp.float32)
        
        # Compute derivative
        wp.launch(
            kernel=compute_deriv_x_kernel,
            dim=(res, res, res),
            inputs=[f, df, grid.idx, res, res, res]
        )
        
        df_np = df.numpy()
        
        # Check error (exclude boundary points)
        error = np.abs(df_np[4:-4, 4:-4, 4:-4] - df_exact_np[4:-4, 4:-4, 4:-4])
        max_error = np.max(error)
        mean_error = np.mean(error)
        
        print(f"Derivative test (sin function):")
        print(f"  Max error: {max_error:.2e}")
        print(f"  Mean error: {mean_error:.2e}")
        print(f"  Expected: O(dx^4) = O({grid.dx**4:.2e})")
        
        # For 4th order, error should be ~ C * dx^4
        assert max_error < 0.01, f"Error too large: {max_error}"
        
        print("Derivative test: PASS")

if __name__ == "__main__":
    test_derivatives()
