# M3: BSSN in Warp (Core)

**Goal:** Implement BSSN evolution equations in warp.
**Entry:** Completed M1 + M2.
**Exit criteria:** Evolve flat spacetime stably for 100+ timesteps.

## Tasks

- [x] 1. Define BSSN variables as warp fields
- [x] 2. Implement spatial derivative kernels (4th order FD)
- [x] 3. Implement RHS of BSSN equations (start with flat spacetime)
- [x] 4. Implement RK4 time integration
- [x] 5. Add Kreiss-Oliger dissipation
- [ ] 6. Test constraint preservation on flat spacetime (need full RHS)
- [ ] 7. Verify autodiff works through one timestep (need full RHS)

## Status
Basic infrastructure complete:
- BSSN variables defined (21 components)
- Grid class with initialization
- RK4 time integrator working
- 4th order FD operators implemented
- Flat spacetime test passing (100+ steps stable)

Next: Implement full BSSN RHS (requires all evolution equations from refs/bssn_equations.md)
