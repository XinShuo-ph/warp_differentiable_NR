#!/bin/bash
# Run all tests to verify the implementation

echo "=========================================="
echo "Running All BSSN Implementation Tests"
echo "=========================================="
echo ""

# Track results
TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    TEST_NAME=$1
    TEST_FILE=$2
    
    echo "----------------------------------------"
    echo "Running: $TEST_NAME"
    echo "----------------------------------------"
    
    if python3 "$TEST_FILE" > /tmp/test_output.txt 2>&1; then
        echo "✓ PASSED"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        # Show last few lines
        tail -5 /tmp/test_output.txt
    else
        echo "✗ FAILED"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        # Show error
        tail -20 /tmp/test_output.txt
    fi
    echo ""
}

# M1 Tests
echo "=== Milestone 1 Tests ==="
run_test "M1: Poisson Solver" "src/poisson_solver.py"
run_test "M1: Poisson Verification" "tests/test_poisson_verification.py"

# M3 Tests
echo "=== Milestone 3 Tests ==="
run_test "M3: BSSN State" "src/bssn_state.py"
run_test "M3: Spatial Derivatives" "src/bssn_derivatives.py"
run_test "M3: BSSN RHS" "src/bssn_rhs.py"
run_test "M3: RK4 Integration" "src/bssn_rk4.py"
run_test "M3: Complete Evolution" "tests/test_bssn_complete.py"
run_test "M3: Autodiff" "tests/test_bssn_autodiff.py"

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "✓✓✓ ALL TESTS PASSED ✓✓✓"
    exit 0
else
    echo "Some tests failed - see output above"
    exit 1
fi
