import warp as wp
from bssn_solver import BSSNSolver
from constraints import check_constraints

if __name__ == "__main__":
    wp.init()
    solver = BSSNSolver(res=16, sigma=0.01)
    
    print("Evolving flat spacetime...")
    for i in range(10):
        solver.step()
        
    passed = check_constraints(solver.state)
    
    if passed:
        print("Constraint Test PASSED")
    else:
        print("Constraint Test FAILED")
