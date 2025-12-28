# Tier 3-4 Branches Summary

## c374 (M3 Complete)
- BSSN implemented, stable on flat space
- Autodiff working
- Files: bssn.py, bssn_rhs.py, derivatives.py, poisson.py
- **Recommendation**: Skip (similar to other M3 branches)

## 2b4b (M3 Started)
- M1/M2 complete, starting M3
- Extensive refs documentation
- Files: bssn_vars.py, bssn_derivatives.py, poisson_solver.py
- **Recommendation**: Check refs/ for documentation value

## 2eb4 (M1 Complete)
- Poisson Jacobi solver verified
- Diffusion autodiff working
- Files: poisson_jacobi.py, m1_diffusion_autodiff.py
- **Recommendation**: Skip (M1 only)

## 5800 (M1 Complete)
- FD Poisson solver (RBGS)
- ETK code snippets extracted
- Files: poisson_fd.py, various M1 examples
- **Recommendation**: Skip (M1 only)

## 7134 (M2 Started)
- Poisson solver
- Autodiff smoke test
- Files: poisson.py, autodiff_smoke.py
- **Recommendation**: Skip (M1-M2 only)

## 95d7 (M1 Complete)
- Diffusion autodiff trace
- Various API snippets in refs/
- Files: poisson.py, m1_diffusion_autodiff_trace.py
- **Recommendation**: Skip (M1 only)

## Overall Assessment
No unique features in Tier 3-4 beyond what's already in Tier 1-2.
Tier 1-2 branches provide all necessary components for merge.
