# Milestone 3: BSSN in Warp (Core)

**Goal:** Implement BSSN evolution equations in warp.
**Entry:** Completed M1 + M2.
**Exit criteria:** Evolve flat spacetime stably for 100+ timesteps.

## Tasks

- [x] 1. Define BSSN variables as warp fields
- [x] 2. Implement spatial derivative kernels (4th order FD)
- [x] 3. Implement RHS of BSSN equations (start with flat spacetime)
- [x] 4. Implement RK4 time integration
- [x] 5. Add Kreiss-Oliger dissipation
- [x] 6. Test constraint preservation on flat spacetime
- [x] 7. Verify autodiff works through one timestep

## Status: COMPLETE

Exit criteria met:
- Flat spacetime evolution stable for 100+ timesteps
- All BSSN variables properly defined as warp arrays
- 4th-order finite differences with verified convergence
- RK4 time integrator implemented
- Kreiss-Oliger dissipation integrated
- Autodiff verified working through evolution step
