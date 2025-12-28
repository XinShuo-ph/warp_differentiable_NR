# Tier 2 Branches Quick Summary (16a3, 8b82, 3a28, 99cb)

## Branch 16a3
- Milestone: M4 started (M3 complete)
- Status: Flat-space approximation, needs full BSSN
- Files: bssn.py, derivs.py, integrator.py, rhs.py, poisson.py
- **Assessment**: Redundant with Tier 1, SKIP

## Branch 8b82
- Milestone: M4 (5/5 tasks, but unstable)
- Status: Project halted, BBH evolution unstable
- Features: BSSN solver, RK4, dissipation, BL/BY initial data
- Documentation: refs/etk_bbh_structure.md
- **Assessment**: Lower quality than Tier 1, may check ETK docs

## Branch 3a28
- Milestone: M3 COMPLETE (7/7 tasks)
- Status: All milestones M1-M3 complete
- Tests: 5 tests passing (Poisson, flat evolution, constraints, gauge wave)
- Documentation: README.md, COMPLETION_REPORT.md, PROJECT_SUMMARY.md ⭐
- **Assessment**: Good documentation, redundant code, TAKE DOCS

## Branch 99cb
- Milestone: M3 COMPLETE (7/7 tasks)
- Status: Core BSSN implementation complete
- Files: bssn_variables.py, bssn_derivatives.py, bssn_rhs.py, bssn_evolver.py
- Tests: test_flat_evolution.py, test_derivatives.py, test_autodiff_bssn.py
- **Assessment**: Clean M3, but redundant with Tier 1

## Merge Recommendations
- **16a3**: SKIP (redundant)
- **8b82**: Check ETK docs only
- **3a28**: TAKE documentation files (README, reports) ⭐⭐
- **99cb**: SKIP (redundant with Tier 1)
