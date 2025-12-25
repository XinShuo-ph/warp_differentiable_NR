import warp as wp
from bssn_defs import allocate_bssn_state, initialize, BSSNState
from bssn_rhs import bssn_rhs_kernel

# To test autodiff, we need a "loss" function.
# Loss = sum(phi^2) after one step?
#
# If we start with flat spacetime, phi stays 0. Gradient w.r.t initial phi?
# If we perturb initial phi, does loss change?
#
# Let's try to differentiate through `bssn_rhs_kernel`.
#
# We need to set requires_grad=True on inputs.
# allocate_bssn_state already sets requires_grad=True.

def test_autodiff_step():
    wp.init()
    res = 16
    dx = 1.0/res
    
    # Setup state
    state = allocate_bssn_state((res, res, res))
    initialize(state)
    
    # We need to compute gradients. 
    # Let's perform one RHS evaluation (not full RK4 for simplicity, but we could).
    
    rhs = allocate_bssn_state((res, res, res))
    
    tape = wp.Tape()
    with tape:
        # Launch RHS
        # Note: bssn_rhs_kernel modifies `rhs` in place.
        # Warp autodiff supports in-place modification if it's the output.
        wp.launch(bssn_rhs_kernel, dim=(res, res, res), inputs=[0.0, state, rhs, dx])
        
        # Define loss = sum(rhs.phi)
        # But rhs.phi is zero for flat spacetime.
        # Let's verify that gradients are propagated.
        # Even if value is zero, d(RHS)/d(state) might be non-zero (e.g. coefficients).
        
        # Let's define a dummy loss that depends on RHS.
        # loss = sum(rhs.alpha)
        # rhs.alpha = -2 * alpha * K
        # If alpha=1, K=0 -> rhs.alpha=0.
        # d(rhs.alpha)/d(alpha) = -2*K = 0.
        # d(rhs.alpha)/d(K) = -2*alpha = -2.
        
        # So gradient w.r.t K should be -2.
        pass
        
    # We need to specify the loss gradient explicitly if we don't reduce to scalar.
    # Or reduce to scalar inside tape.
    
    # Let's verify gradient of K on one element.
    # We'll set the adjoint of rhs.alpha to 1.0 at index (0,0,0) and check adjoint of state.K at (0,0,0).
    
    # Zero gradients
    state.K.grad.zero_()
    
    # Set gradient on output
    # We want dL/d(rhs.alpha[8,8,8]) = 1.0
    rhs.alpha.grad.zero_()
    
    # We need to access array element.
    # wp.array doesn't support direct indexing from python for setting grad efficiently?
    # We can use numpy.
    
    rhs_alpha_grad_np = rhs.alpha.grad.numpy()
    rhs_alpha_grad_np[8, 8, 8] = 1.0 # Center
    rhs.alpha.grad = wp.array(rhs_alpha_grad_np, dtype=float, device=rhs.alpha.device)
    
    # Backward
    # Note: when calling tape.backward(), it normally assumes loss is a scalar and sets its grad to 1.0.
    # If loss is not defined, we can pass grads dict to seed gradients.
    # However, Warp's backward() typically clears previous adjoints unless specified otherwise? 
    # Actually, backward accumulates into .grad fields.
    # If we manually set .grad fields of outputs, we should call backward with no args or specific logic?
    # Warp's tape.backward() documentation:
    # "If no arguments are provided, it is assumed that the loss is a scalar and has a gradient of 1.0."
    # But we don't have a scalar loss variable recorded in the tape.
    # However, Warp operations record the dependency graph.
    # If we seed the output gradients manually, we can just run the backward pass.
    # BUT, tape.backward() might try to backprop from the last operation?
    # The last operation was launch(bssn_rhs_kernel).
    # Its outputs are in `rhs`.
    # If `rhs` gradients are set, backward() should propagate them to `state`.
    
    tape.backward()
    
    # Check grad of state.K
    k_grad_val = state.K.grad.numpy()[8, 8, 8]
    print(f"Gradient of K at center: {k_grad_val}")
    
    # Why is it -4.0?
    # rhs.alpha = -2.0 * alpha * K
    # d(rhs.alpha)/dK = -2.0 * alpha
    # alpha = 1.0. So result should be -2.0.
    
    # Is it possible that the kernel is executed twice or something?
    # Or accumulating?
    # We zeroed gradients before backward.
    # Maybe tid() logic maps multiple threads to same index? No, dim matches.
    
    # Wait, check bssn_rhs.py again.
    # rhs.alpha[i, j, k] = -2.0 * alpha_val * K_val
    
    # Is it possible I ran it twice in the test script?
    # No, test_autodiff_step calls launch once.
    
    # Let's check if alpha is 2.0?
    alpha_val = state.alpha.numpy()[8, 8, 8]
    print(f"Alpha value: {alpha_val}")
    
    # If alpha is 1.0, then grad should be -2.0.
    
    # Is it possible that `state.K` is aliased or something? No.
    
    # Maybe the assignment `rhs.alpha[...] = ...` is adding?
    # No, assignment overwrites.
    
    # Let's try a simpler kernel to isolate.
    
    # Maybe Warp's adjoint for `*` has a bug? Unlikely.
    # Maybe `alpha_val` is loaded twice?
    # alpha_val = state.alpha[i, j, k]
    # rhs.alpha = -2.0 * alpha_val * K_val
    
    # Let's debug by printing more info.

    if abs(k_grad_val + 2.0) < 1e-5:
        print("Autodiff Test PASSED")
    else:
        print(f"Autodiff Test FAILED (Expected -2.0)")

if __name__ == "__main__":
    test_autodiff_step()
