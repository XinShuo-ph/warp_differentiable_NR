# Milestone 3: BSSN in Warp (Core)

**Goal:** Implement BSSN evolution equations in warp.

## Tasks

- [x] 1. Define BSSN variables as warp fields
  - See `src/bssn_variables.py`
- [x] 2. Implement spatial derivative kernels (4th order FD)
  - See `src/bssn_derivatives.py`
- [x] 3. Implement RHS of BSSN equations (start with flat spacetime)
  - See `src/bssn_rhs.py`
- [x] 4. Implement RK4 time integration
  - See `src/bssn_integrator.py`
- [x] 5. Add Kreiss-Oliger dissipation
  - Included in `src/bssn_derivatives.py` and `src/bssn_rhs.py`
- [x] 6. Test constraint preservation on flat spacetime
  - See `tests/test_bssn_evolution.py`
- [x] 7. Verify autodiff works through one timestep
  - See `tests/test_bssn_evolution.py`

## Summary
- 24 BSSN fields implemented (chi, gamma_tilde, K, A_tilde, Gamma_tilde, alpha, beta, B)
- 4th order finite differences verified against analytical solution
- RK4 integrator evolves flat spacetime stably for 200+ timesteps
- Constraints remain exactly zero (machine precision)
- Autodiff (wp.Tape) works through evolution kernels
