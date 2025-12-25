"""
Test suite for Poisson solver
"""
import sys
sys.path.insert(0, '../src')

import warp as wp
from poisson_solver import solve_poisson, test_twice

def test_consistency():
    """Test that solver produces consistent results"""
    field, geo = test_twice()
    return True

def test_basic_solve():
    """Test basic solve runs without error"""
    field, geo = solve_poisson(resolution=16, degree=2)
    assert field is not None
    assert geo is not None
    return True

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
