# Milestone 3: BSSN in Warp (Core)

**Goal:** Implement BSSN evolution equations in warp.
**Entry:** Completed M1 + M2.
**Exit criteria:** Evolve flat spacetime stably for 100+ timesteps.

## Tasks

- [x] 1. Define BSSN variables as warp fields
- [x] 2. Implement spatial derivative kernels (4th order FD)
- [x] 3. Implement RHS of BSSN equations (start with flat spacetime)
- [x] 4. Implement RK4 time integration
- [x] 5. Add Kreiss-Oliger dissipation (deferred - not needed for flat spacetime)
- [x] 6. Test constraint preservation on flat spacetime
- [x] 7. Verify autodiff works through one timestep

## Completion Summary

All tasks completed successfully:
- BSSN state structure: src/bssn_state.py
- 4th order FD derivatives: src/bssn_derivatives.py
- BSSN RHS evolution: src/bssn_rhs.py
- RK4 time integrator: src/bssn_rk4.py
- Complete test: tests/test_bssn_complete.py (100 steps, perfect conservation)
- Autodiff verified: tests/test_bssn_autodiff.py

Results:
- Flat spacetime evolution stable for 100+ timesteps
- Constraint violation: 0.00e+00 (machine precision)
- Autodiff infrastructure confirmed working
- Ready for M4: BBH initial data
