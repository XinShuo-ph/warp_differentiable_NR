Source: `NR/warp/warp/examples/fem/example_elastic_shape_optimization.py` (Warp FEM example)

```python
tape = wp.Tape()

with tape:
    fem.integrate(
        applied_load_form,
        fields={"v": self._u_right_test},
        values={"load": self._load},
        output=u_rhs,
    )
    fem.integrate(
        hooke_elasticity_form,
        fields={"u": self._u_field, "v": self._u_test},
        values={"lame": -self._lame},
        output=u_rhs,
        add=True,
    )

# ... build stiffness matrix, project BCs, solve for u ...

def solve_linear_system():
    fem_example_utils.bsr_cg(u_matrix, b=u.grad, x=u_rhs.grad, quiet=self._quiet, tol=1e-6, max_iters=1000)
    u_rhs.grad -= self._bd_projector @ u_rhs.grad
    self._u_field.dof_values.grad.zero_()

tape.record_func(solve_linear_system, arrays=(u_rhs, u))

loss = wp.empty(shape=1, dtype=wp.float32, requires_grad=True)

with tape:
    fem.integrate(
        loss_form,
        fields={"u": self._u_field},
        values={"lame": self._lame, "quality_threshold": 0.2, "quality_weight": 20.0},
        domain=self._u_test.domain,
        output=loss,
    )

tape.backward(loss=loss)
```

