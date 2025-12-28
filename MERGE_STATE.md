# Merge State
- **Phase**: P1
- **Current Branch**: 0a7f (first in queue)
- **Branches Completed**: []
- **Status**: ready_for_next

## Next Action
1. Create merge_notes directory:
   ```bash
   mkdir -p merge_notes
   ```
2. Start analyzing branch 0a7f:
   ```bash
   git show origin/cursor/following-instructions-md-0a7f:NR/STATE.md
   git ls-tree --name-only -r origin/cursor/following-instructions-md-0a7f | grep -E '\.(py|md)$' | head -30
   ```
3. Test BSSN evolution from 0a7f
4. Document findings in `merge_notes/0a7f_notes.md`

Note: You are on a `cursor/merge-...` branch created by Cursor agent.

## Branch Queue (from branch_progresses.md)

### Tier 1 - Must Process (M4-M5, Most Complete)
- [ ] 0a7f (M5, 14 tests, full BSSN + BBH)
- [ ] 0d97 (M5, ML pipeline - unique)
- [ ] c633 (M4, BBH framework, 3300+ lines)
- [ ] 9052 (M5, puncture evolution)

### Tier 2 - Process for Features (M3-M4)
- [ ] 1183 (M5, RK4 + BCs)
- [ ] bd28 (M4, dissipation kernel - unique)
- [ ] 16a3 (M4, modular structure)
- [ ] 8b82 (M4, ETK docs)
- [ ] 3a28 (M3, README)
- [ ] 99cb (M3, derivative tests)

### Tier 3-4 - Quick Scan
- [ ] c374, 2b4b, 2eb4, 5800, 7134, 95d7

## Unique Features to Watch For
- **0d97**: ML pipeline, losses, waveforms, optimization
- **bd28**: Kreiss-Oliger dissipation kernel
- **c633**: BBH initial data, comprehensive tests
- **9052**: Puncture evolution, long-term stability

## Key Findings This Session
(none yet)

## Merge Decisions Made
(none yet)

## Session Log
- (initial): Merge workflow initialized, ready to begin P1 with branch 0a7f

