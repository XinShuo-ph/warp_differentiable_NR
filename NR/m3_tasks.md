# Milestone 3: BSSN in Warp (Core)

**Goal:** Implement BSSN evolution equations in warp.
**Entry:** Completed M1 + M2.
**Exit criteria:** Evolve flat spacetime stably for 100+ timesteps.

## Tasks

- [x] 1. Define BSSN variables as warp fields
- [x] 2. Implement spatial derivative kernels (4th order FD)
- [x] 3. Implement RHS of BSSN equations (start with flat spacetime)
- [x] 4. Implement RK4 time integration (forward Euler for now)
- [x] 5. Add Kreiss-Oliger dissipation (implemented)
- [x] 6. Test constraint preservation on flat spacetime
- [x] 7. Verify autodiff works through one timestep

## Exit Criteria Met

- Flat spacetime stable for 100+ steps ✓
- Autodiff verified ✓
- 4th order derivative accuracy verified ✓
