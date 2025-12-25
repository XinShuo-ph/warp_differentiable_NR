import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import warp as wp
import numpy as np
from poisson import PoissonSolver

def test_poisson_convergence():
    wp.init()
    
    # Run with different resolutions and check convergence
    resolutions = [16, 32, 64]
    errors = []
    
    for res in resolutions:
        solver = PoissonSolver(resolution=res, degree=1)
        solver.solve()
        
        # Get node positions
        positions = solver.space.node_positions().numpy()
        values = solver.field.dof_values.numpy()
        
        pi = np.pi
        analytical = np.sin(pi * positions[:, 0]) * np.sin(pi * positions[:, 1])
        
        diff = values - analytical
        l2_error = np.sqrt(np.mean(diff**2))
        
        print(f"Res {res}, L2 error: {l2_error}")
        
        errors.append(l2_error)
        
    print(f"Errors: {errors}")
    
    # Check if error decreases
    if errors[-1] < errors[0]:
        print("Convergence verified.")
    else:
        print("Convergence check failed.")
        exit(1)

if __name__ == "__main__":
    test_poisson_convergence()
