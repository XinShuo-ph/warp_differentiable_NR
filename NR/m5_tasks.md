# Milestone 5: Full Toolkit Port

**Goal:** Port remaining Einstein Toolkit core features.
**Entry:** Completed M4.

## Tasks

- [x] 1. Implement momentum constraints
- [x] 2. Implement full Gamma-driver shift evolution
- [x] 3. Add traceless At evolution equations
- [ ] 4. Add conformal metric evolution equations (integrated in step 3)
- [ ] 5. Implement gravitational wave extraction (Ψ₄)
- [x] 6. Add binary black hole initial data (two punctures)
- [ ] 7. Performance optimization with autodiff

## Completed

- Momentum constraints Mⁱ implemented ✓
- Gamma-driver shift equations (∂ₜβⁱ, ∂ₜBⁱ) implemented ✓
- Traceless At (Ãᵢⱼ) evolution RHS implemented ✓
- Binary black hole (two punctures) initial data ✓

## Notes

Focus on differentiability for ML integration as per project goal.
Current implementation supports autodiff through the RHS kernels.
