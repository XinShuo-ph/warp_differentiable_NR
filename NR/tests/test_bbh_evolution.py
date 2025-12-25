import warp as wp
import numpy as np
from NR.src.bssn_solver import BSSNSolver
from NR.src.initial_data import setup_brill_lindquist, setup_bowen_york
import time

wp.init()

def test_bbh_headon():
    res = 48
    solver = BSSNSolver(resolution=res)
    # Domain [-5, 5] approx if dx is large?
    # allocate_bssn_state sets dx = 1.0/nx. Domain is [0, 1].
    # Punctures at +/- 2 would be outside!
    # We need to adjust dx or the interpretation of coordinates.
    # If dx = 1/48 ~ 0.02.
    # To fit punctures at distance d=1, we need domain > 1.
    # Standard code usually defines domain size.
    # allocate_bssn_state currently hardcodes dx=1/nx -> Domain [0, 1].
    
    # Let's verify bssn_defs.py
    # state.dx = 1.0/nx
    
    # We should override dx/dy/dz to simulate a larger physical domain.
    # Let's say domain is [-5, 5]. L = 10.
    # dx = 10.0 / nx.
    
    L = 10.0
    solver.state.dx = L / res
    solver.state.dy = L / res
    solver.state.dz = L / res
    
    dt = 0.25 * solver.state.dx
    
    print(f"Grid: {res}^3. Domain size: {L}. dx: {solver.state.dx}. dt: {dt}")
    
    # Initial Data: Head-on collision
    # Punctures at +/- 1.5
    pos1 = (1.5, 0.0, 0.0)
    pos2 = (-1.5, 0.0, 0.0)
    m1 = 0.5
    m2 = 0.5
    P = 0.1
    mom1 = (-P, 0.0, 0.0) # Moving left
    mom2 = (P, 0.0, 0.0)  # Moving right
    
    print("Setting up Initial Data...")
    setup_brill_lindquist(solver.state, m1, pos1, m2, pos2)
    setup_bowen_york(solver.state, pos1, mom1, pos2, mom2)
    
    # Check initial norms
    phi_max = np.max(solver.state.phi.numpy())
    alpha_min = np.min(solver.state.alpha.numpy())
    print(f"Initial: Max Phi = {phi_max}, Min Alpha = {alpha_min}")
    
    # Evolve
    steps = 20
    print(f"Evolving for {steps} steps...")
    
    start_t = time.time()
    for i in range(steps):
        solver.rk4_step(dt)
        if i % 5 == 0:
            phi_max = np.max(solver.state.phi.numpy())
            alpha_min = np.min(solver.state.alpha.numpy())
            k_norm = np.linalg.norm(solver.state.K.numpy().flatten())
            print(f"Step {i}: Max Phi = {phi_max:.4f}, Min Alpha = {alpha_min:.4f}, K Norm = {k_norm:.4f}")
            
    end_t = time.time()
    print(f"Evolution finished in {end_t - start_t:.2f}s")

if __name__ == "__main__":
    test_bbh_headon()
