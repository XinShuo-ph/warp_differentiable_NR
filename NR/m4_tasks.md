# Milestone 4: BSSN in Warp (BBH)

**Goal:** Reproduce BBH-like initial data evolution.

## Tasks

- [x] 1. Implement Brill-Lindquist initial data (single puncture)
  - See `src/bssn_initial_data.py`
- [x] 2. Implement full BSSN RHS with all terms
  - See `src/bssn_rhs_full.py`
- [x] 3. Implement 1+log slicing gauge condition
  - Included in `src/bssn_rhs_full.py`
- [x] 4. Implement Gamma-driver shift condition
  - Included in `src/bssn_rhs_full.py`
- [x] 5. Add Sommerfeld boundary conditions
  - See `src/bssn_boundary.py`
- [x] 6. Test single puncture evolution stability
  - See `tests/test_puncture_evolution.py`
- [x] 7. Verify puncture stays approximately stationary
  - Chi change at center: ~0.003, evolution stable

## Summary
- Brill-Lindquist initial data: chi = 1/psi^4, psi = 1 + M/(2r)
- Full BSSN RHS with 24 evolved fields
- 1+log slicing: d_t(alpha) = -2*alpha*K
- Gamma-driver: d_t(beta^i) = (3/4)*B^i, d_t(B^i) = d_t(Gamma^i) - eta*B^i
- Sommerfeld BCs with appropriate falloff for each field
- Stable puncture evolution for 50+ timesteps
- Alpha evolves from pre-collapsed (0.14) toward asymptotic (1.0)
