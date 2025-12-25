# Warp Autodiff Reference Snippets

# 1. Tape-based autodiff - record forward pass, call backward
# From example_diffray.py:
"""
self.tape = wp.Tape()
with self.tape:
    self.forward()
self.tape.backward(self.loss)
rot_grad = self.tape.gradients[self.render_mesh.rot]
self.tape.zero()
"""

# 2. Arrays requiring gradients
# Arrays that need gradients must be created with requires_grad=True:
"""
self.mesh = wp.Mesh(
    points=wp.array(points, dtype=wp.vec3, requires_grad=True),
    indices=wp.array(indices, dtype=int),
)
self.loss = wp.zeros(1, dtype=float, requires_grad=True)
"""

# 3. Kernels are automatically differentiable
# @wp.kernel decorated functions support autodiff through the tape

# 4. FEM integration with autodiff
# From example_diffusion.py - @fem.integrand creates differentiable forms:
"""
@fem.integrand
def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field, nu: float):
    return nu * wp.dot(fem.grad(u, s), fem.grad(v, s))
    
# fem.integrate builds matrices/vectors that can participate in autodiff
matrix = fem.integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": nu})
"""

# 5. Disable backward for performance when not needed
# wp.set_module_options({"enable_backward": False})

# 6. Key warp autodiff concepts:
# - wp.Tape() - records operations for reverse-mode autodiff
# - tape.backward(loss) - computes gradients
# - tape.gradients[array] - access gradients for specific arrays
# - tape.zero() - clear gradients
# - requires_grad=True - mark arrays for gradient computation

# 7. Optimizer integration
# From warp.optim import SGD, Adam
# optimizer.step([gradients]) applies updates
