import warp as wp


def tape_context_manager_example(A_wp, B_wp, C_wp):
    # arrays must opt-in to autodiff
    A_wp.requires_grad = True
    B_wp.requires_grad = True
    C_wp.requires_grad = True

    # record kernels launched inside the context
    with wp.Tape() as tape:
        wp.launch(kernel=None, dim=1, inputs=(A_wp, B_wp), outputs=(C_wp,))

    # compute gradients w.r.t. any recorded array
    tape.backward(C_wp)


def tape_record_func_example(p_rhs, p):
    tape = wp.Tape()

    def solve_linear_system():
        # custom backward hooks can be recorded explicitly (e.g., implicit solves)
        pass

    tape.record_func(solve_linear_system, arrays=(p_rhs, p))

