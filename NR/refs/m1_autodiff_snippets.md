```python
# excerpt from: warp/warp/_src/tape.py

class Tape:
    """
    Record kernel launches within a Tape scope to enable automatic differentiation.
    Gradients can be computed after the operations have been recorded on the tape via
    :meth:`Tape.backward()`.

    Example
    -------

    .. code-block:: python

        tape = wp.Tape()

        # forward pass
        with tape:
            wp.launch(kernel=compute1, inputs=[a, b], device="cuda")
            wp.launch(kernel=compute2, inputs=[c, d], device="cuda")
            wp.launch(kernel=loss, inputs=[d, l], device="cuda")

        # reverse pass
        tape.backward(l)

    Gradients can be accessed via the ``tape.gradients`` dictionary, e.g.:

    .. code-block:: python

        print(tape.gradients[a])

    """

    def __init__(self):
        self.gradients = {}
        self.launches = []
        self.scopes = []

        self.loss = None

    def __enter__(self):
        wp._src.context.init()

        if wp._src.context.runtime.tape is not None:
            raise RuntimeError("Warp: Error, entering a tape while one is already active")

        wp._src.context.runtime.tape = self

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if wp._src.context.runtime.tape is None:
            raise RuntimeError("Warp: Error, ended tape capture, but tape not present")

        wp._src.context.runtime.tape = None

    def backward(self, loss: wp.array | None = None, grads: dict[wp.array, wp.array] | None = None):
        # if scalar loss is specified then initialize
        # a 'seed' array for it, with gradient of one
        if loss:
            if loss.size > 1 or wp._src.types.type_size(loss.dtype) > 1:
                raise RuntimeError("Can only return gradients for scalar loss functions.")

            if not loss.requires_grad:
                raise RuntimeError(
                    "Scalar loss arrays should have requires_grad=True set before calling Tape.backward()"
                )

            # set the seed grad to 1.0
            loss.grad.fill_(1.0)

        # simply apply dict grads to objects
        # this is just for backward compat. with
        # existing code before we added wp.array.grad attribute
        if grads:
            for a, g in grads.items():
                if a.grad is None:
                    a.grad = g
                else:
                    # ensure we can capture this backward pass in a CUDA graph
                    a.grad.assign(g)

        # run launches backwards
        for launch in reversed(self.launches):
            if callable(launch):
                launch()
```

```python
# excerpt from: warp/warp/examples/fem/example_darcy_ls_optimization.py

# Advected level set field, used in adjoint computations
advected_level_set = fem.make_discrete_field(space=self._ls_space)
advected_level_set.dof_values.assign(self._level_set_field.dof_values)
advected_level_set.dof_values.requires_grad = True
advected_level_set_restriction = fem.make_restriction(advected_level_set, domain=self._p_test.domain)

# Forward step, record adjoint tape for forces
p_rhs = wp.empty(self._p_space.node_count(), dtype=wp.float32, requires_grad=True)

tape = wp.Tape()
with tape:
    # Dummy advection step, so backward pass can compute adjoint w.r.t advection velocity
    self.advect_level_set(
        level_set_in=self._level_set_field,
        level_set_out=advected_level_set_restriction,
        velocity=self._level_set_velocity_field,
        dt=1.0,
    )

    # Left-hand-side of implicit solve (zero if p=0, but required for adjoint computation through implicit function theorem)
    fem.integrate(
        diffusion_form,
        fields={
            "level_set": advected_level_set,
            "p": self._p_field,
            "q": self._p_test,
        },
        values={"smoothing": self._smoothing, "scale": -1.0},
        output=p_rhs,
    )

# Diffusion matrix (inhomogeneous Poisson)
p_matrix = fem.integrate(
    diffusion_form,
    fields={
        "level_set": advected_level_set,
        "p": self._p_trial,
        "q": self._p_test,
    },
    values={"smoothing": self._smoothing, "scale": 1.0},
    output_dtype=float,
)

# Project to enforce Dirichlet boundary conditions then solve linear system
fem.project_linear_system(
    p_matrix, p_rhs, self._bd_projector, self._bd_prescribed_value, normalize_projector=False
)

fem_example_utils.bsr_cg(p_matrix, b=p_rhs, x=p, quiet=self._quiet, tol=1e-6, max_iters=1000)

# Record adjoint of linear solve
def solve_linear_system():
    fem_example_utils.bsr_cg(p_matrix, b=p.grad, x=p_rhs.grad, quiet=self._quiet, tol=1e-6, max_iters=1000)
    p_rhs.grad -= self._bd_projector @ p_rhs.grad

tape.record_func(solve_linear_system, arrays=(p_rhs, p))

# Evaluate losses
loss = wp.empty(shape=1, dtype=wp.float32, requires_grad=True)
vol = wp.empty(shape=1, dtype=wp.float32, requires_grad=True)

with tape:
    # Main objective: inflow flux
    fem.integrate(
        inflow_velocity,
        fields={"level_set": advected_level_set.trace(), "p": self._p_field.trace()},
        values={"smoothing": self._smoothing},
        domain=self._inflow,
        output=loss,
    )

    # Add penalization term enforcing constant volume
    fem.integrate(
        volume_form,
        fields={"level_set": advected_level_set},
        values={"smoothing": self._smoothing},
        domain=self._p_test.domain,
        output=vol,
    )

    vol_loss_weight = 1000.0
    wp.launch(
        combine_losses,
        dim=1,
        inputs=(loss, vol, self._target_vol, vol_loss_weight),
    )

# perform backward step
tape.backward(loss=loss)
```

```python
# excerpt from: warp/warp/examples/fem/example_diffusion.py

if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=50, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree of shape functions.")
    # ...
```

