import warp as wp
import numpy as np
from NR.src.bssn_rhs import bssn_rhs_kernel

@wp.kernel
def init_flat_spacetime(
    phi: wp.array(dtype=float, ndim=3),
    gt: wp.array(dtype=wp.mat33, ndim=3),
    K: wp.array(dtype=float, ndim=3),
    At: wp.array(dtype=wp.mat33, ndim=3),
    Xt: wp.array(dtype=wp.vec3, ndim=3),
    alpha: wp.array(dtype=float, ndim=3),
    beta: wp.array(dtype=wp.vec3, ndim=3),
    B: wp.array(dtype=wp.vec3, ndim=3)
):
    i, j, k = wp.tid()
    
    # Flat spacetime (Minkowski)
    phi[i,j,k] = 0.0
    gt[i,j,k] = wp.mat33(1.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 0.0, 1.0)
    K[i,j,k] = 0.0
    At[i,j,k] = wp.mat33(0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0)
    Xt[i,j,k] = wp.vec3(0.0, 0.0, 0.0)
    
    alpha[i,j,k] = 1.0
    beta[i,j,k] = wp.vec3(0.0, 0.0, 0.0)
    B[i,j,k] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def saxpy_scalar(
    y_out: wp.array(dtype=float, ndim=3),
    y_in: wp.array(dtype=float, ndim=3),
    x: wp.array(dtype=float, ndim=3),
    scale: float
):
    i, j, k = wp.tid()
    y_out[i,j,k] = y_in[i,j,k] + scale * x[i,j,k]

@wp.kernel
def saxpy_vec3(
    y_out: wp.array(dtype=wp.vec3, ndim=3),
    y_in: wp.array(dtype=wp.vec3, ndim=3),
    x: wp.array(dtype=wp.vec3, ndim=3),
    scale: float
):
    i, j, k = wp.tid()
    y_out[i,j,k] = y_in[i,j,k] + scale * x[i,j,k]

@wp.kernel
def saxpy_mat33(
    y_out: wp.array(dtype=wp.mat33, ndim=3),
    y_in: wp.array(dtype=wp.mat33, ndim=3),
    x: wp.array(dtype=wp.mat33, ndim=3),
    scale: float
):
    i, j, k = wp.tid()
    y_out[i,j,k] = y_in[i,j,k] + scale * x[i,j,k]

class BSSNSolver:
    def __init__(self, resolution=(32, 32, 32), extent=(10.0, 10.0, 10.0), requires_grad=False):
        self.shape = resolution
        self.extent = extent
        self.dx = extent[0] / resolution[0]
        self.dy = extent[1] / resolution[1]
        self.dz = extent[2] / resolution[2]
        self.dt = 0.25 * min(self.dx, self.dy, self.dz)
        
        # Allocate fields
        self.fields = {
            "phi": wp.zeros(self.shape, dtype=float, requires_grad=requires_grad),
            "gt": wp.zeros(self.shape, dtype=wp.mat33, requires_grad=requires_grad),
            "K": wp.zeros(self.shape, dtype=float, requires_grad=requires_grad),
            "At": wp.zeros(self.shape, dtype=wp.mat33, requires_grad=requires_grad),
            "Xt": wp.zeros(self.shape, dtype=wp.vec3, requires_grad=requires_grad),
            "alpha": wp.zeros(self.shape, dtype=float, requires_grad=requires_grad),
            "beta": wp.zeros(self.shape, dtype=wp.vec3, requires_grad=requires_grad),
            "B": wp.zeros(self.shape, dtype=wp.vec3, requires_grad=requires_grad)
        }
        
        # Temp fields for RK4 (intermediate state)
        self.temp_fields = {k: wp.zeros_like(v, requires_grad=requires_grad) for k, v in self.fields.items()}
        
        # RHS fields
        self.rhs_fields = {k: wp.zeros_like(v, requires_grad=requires_grad) for k, v in self.fields.items()}
        
        # Accumulator for RK4 final step
        self.initial_fields = {k: wp.zeros_like(v, requires_grad=requires_grad) for k, v in self.fields.items()}

        # Initialize
        wp.launch(
            kernel=init_flat_spacetime,
            dim=self.shape,
            inputs=[
                self.fields["phi"], self.fields["gt"], self.fields["K"], self.fields["At"], self.fields["Xt"],
                self.fields["alpha"], self.fields["beta"], self.fields["B"]
            ]
        )

    def calc_rhs(self, state, rhs):
        wp.launch(
            kernel=bssn_rhs_kernel,
            dim=self.shape,
            inputs=[
                state["phi"], state["gt"], state["K"], state["At"], state["Xt"],
                state["alpha"], state["beta"], state["B"],
                rhs["phi"], rhs["gt"], rhs["K"], rhs["At"], rhs["Xt"],
                rhs["alpha"], rhs["beta"], rhs["B"],
                self.dx, self.dy, self.dz
            ]
        )

    def update_state(self, target, source, rhs, scale):
        # target = source + scale * rhs
        for name in self.fields:
            arr_t = target[name]
            arr_s = source[name]
            arr_rhs = rhs[name]
            
            dtype = arr_t.dtype
            if dtype == float:
                wp.launch(saxpy_scalar, dim=self.shape, inputs=[arr_t, arr_s, arr_rhs, scale])
            elif dtype == wp.vec3:
                wp.launch(saxpy_vec3, dim=self.shape, inputs=[arr_t, arr_s, arr_rhs, scale])
            elif dtype == wp.mat33:
                wp.launch(saxpy_mat33, dim=self.shape, inputs=[arr_t, arr_s, arr_rhs, scale])

    def step(self):
        # RK4 implementation
        # k1 = f(yn)
        # y1 = yn + 0.5 * dt * k1
        # k2 = f(y1)
        # y2 = yn + 0.5 * dt * k2
        # k3 = f(y2)
        # y3 = yn + dt * k3
        # k4 = f(y3)
        # yn+1 = yn + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # But standard memory efficient way:
        # 1. calc k1 from fields -> rhs
        # 2. temp = fields + 0.5*dt*rhs
        # 3. calc k2 from temp -> rhs (reuse rhs buffer?) No, we need k1 for final sum.
        # If we want to be memory efficient, we can accumulate result.
        # acc = fields (copy)
        # acc += dt/6 * k1 ... 
        
        # Let's do simple logic first:
        # copy fields to initial
        for k, v in self.fields.items():
            wp.copy(self.initial_fields[k], v)
            
        # 1. k1
        self.calc_rhs(self.initial_fields, self.rhs_fields)
        # Accumulate k1 to final solution (using temp_fields as accumulator?)
        # Let's use fields as accumulator? No, fields is updated at the very end.
        # Let's use initial_fields as base.
        # We need an accumulator buffer. Let's use fields as accumulator if we initialize it properly?
        # Actually: y_n+1 = y_n + ...
        # So accumulator should start at 0? Or start at y_n?
        # Let's use temp_fields as intermediate y_i.
        
        # To avoid allocating 4 RHS buffers, we can accumulate.
        # y_new = y_old + dt/6 * k1 + ...
        # But k1 is needed to compute y1.
        
        # Let's assume we have enough memory for 1 RHS buffer.
        # k1 = RHS(y)
        # y_temp = y + 0.5*dt*k1
        # y_acc = y + dt/6 * k1
        
        # k2 = RHS(y_temp)
        # y_temp = y + 0.5*dt*k2
        # y_acc += dt/3 * k2
        
        # k3 = RHS(y_temp)
        # y_temp = y + dt*k3
        # y_acc += dt/3 * k3
        
        # k4 = RHS(y_temp)
        # y_acc += dt/6 * k4
        
        # Result is in y_acc. 
        # We need:
        # - fields (y)
        # - initial_fields (y_n, immutable during step)
        # - temp_fields (y_temp)
        # - rhs_fields (k)
        # We can update self.fields in place as accumulator?
        # Yes, if we copy to initial_fields first.
        
        dt = self.dt
        
        # k1
        self.calc_rhs(self.initial_fields, self.rhs_fields)
        self.update_state(self.temp_fields, self.initial_fields, self.rhs_fields, 0.5 * dt)
        self.update_state(self.fields, self.initial_fields, self.rhs_fields, dt / 6.0) # Accumulate k1
        
        # k2
        self.calc_rhs(self.temp_fields, self.rhs_fields)
        self.update_state(self.temp_fields, self.initial_fields, self.rhs_fields, 0.5 * dt)
        self.update_state(self.fields, self.fields, self.rhs_fields, dt / 3.0) # Accumulate k2 (note source is fields)
        
        # k3
        self.calc_rhs(self.temp_fields, self.rhs_fields)
        self.update_state(self.temp_fields, self.initial_fields, self.rhs_fields, dt)
        self.update_state(self.fields, self.fields, self.rhs_fields, dt / 3.0) # Accumulate k3
        
        # k4
        self.calc_rhs(self.temp_fields, self.rhs_fields)
        self.update_state(self.fields, self.fields, self.rhs_fields, dt / 6.0) # Accumulate k4
        
        # Done. self.fields now contains y_n+1

if __name__ == "__main__":
    wp.init()
    solver = BSSNSolver()
    print("BSSN initialized.")
    
    # Run 100 steps
    for i in range(100):
        solver.step()
        if i % 10 == 0:
            phi_data = solver.fields["phi"].numpy()
            print(f"Step {i}: Phi range: {np.min(phi_data)} to {np.max(phi_data)}")
            
    print("100 steps completed.")
    phi_data = solver.fields["phi"].numpy()
    print(f"Final Phi range: {np.min(phi_data)} to {np.max(phi_data)}")
    assert np.allclose(phi_data, 0.0, atol=1e-6)
    print("Flat spacetime preserved.")
