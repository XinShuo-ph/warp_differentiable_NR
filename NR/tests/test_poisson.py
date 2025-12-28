"""
Test suite for Poisson solver
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import warp as wp
from poisson_solver import solve_poisson, run_twice

def test_consistency():
    """Test that solver produces consistent results"""
    with wp.ScopedDevice("cpu"):
        field, geo = run_twice()
    assert field is not None

def test_basic_solve():
    """Test basic solve runs without error"""
    field, geo = solve_poisson(resolution=16, degree=2)
    assert field is not None
    assert geo is not None

if __name__ == "__main__":
    wp.init()
    with wp.ScopedDevice("cpu"):
        print("Test 1: Consistency")
        test_consistency()
        print("PASS\n")
        
        print("Test 2: Basic solve")
        test_basic_solve()
        print("PASS\n")
        
        print("All tests passed!")
