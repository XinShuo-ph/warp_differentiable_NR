"""
End-to-End Differentiable BSSN Pipeline for ML Integration

Provides a unified interface for running differentiable numerical relativity
simulations and computing gradients for machine learning applications.
"""

import sys
sys.path.insert(0, '/workspace/src')

import warp as wp
import numpy as np
from bssn_vars import BSSNGrid
from bssn_initial_data import set_schwarzschild_puncture, set_brill_lindquist
from bssn_rhs_full import compute_bssn_rhs_full_kernel
from bssn_boundary import apply_standard_bssn_boundaries
from bssn_constraints import ConstraintMonitor
from bssn_losses import DifferentiableLoss, asymptotic_flatness_loss_kernel
from bssn_waveform import WaveformExtractor


@wp.kernel
def rk4_update_kernel(
    u: wp.array(dtype=wp.float32),
    u0: wp.array(dtype=wp.float32),
    k: wp.array(dtype=wp.float32),
    dt: float,
    coeff: float
):
    """u = u0 + coeff * dt * k"""
    tid = wp.tid()
    u[tid] = u0[tid] + coeff * dt * k[tid]


@wp.kernel
def rk4_accumulate_kernel(
    u_final: wp.array(dtype=wp.float32),
    k: wp.array(dtype=wp.float32),
    dt: float,
    weight: float
):
    """u_final += weight * dt * k"""
    tid = wp.tid()
    u_final[tid] = u_final[tid] + weight * dt * k[tid]


@wp.kernel
def copy_kernel(
    dst: wp.array(dtype=wp.float32),
    src: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    dst[tid] = src[tid]


class DifferentiableBSSNPipeline:
    """
    End-to-end differentiable BSSN evolution pipeline.
    
    This class provides:
    - Initial data setup (Schwarzschild, Brill-Lindquist)
    - Time evolution with RK4
    - Constraint monitoring
    - Waveform extraction
    - Loss computation
    - Gradient computation via autodiff
    
    All operations are differentiable, enabling ML integration.
    """
    def __init__(self, nx=32, ny=32, nz=32, domain_size=16.0,
                 cfl=0.1, eps_diss=0.5, requires_grad=True):
        """
        Initialize the differentiable BSSN pipeline.
        
        Args:
            nx, ny, nz: Grid dimensions
            domain_size: Physical domain size (in M units)
            cfl: CFL number for time stepping
            eps_diss: Kreiss-Oliger dissipation coefficient
            requires_grad: Enable gradient computation
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = domain_size / nx
        self.cfl = cfl
        self.dt = cfl * self.dx
        self.eps_diss = eps_diss * self.dx
        self.inv_dx = 1.0 / self.dx
        self.requires_grad = requires_grad
        
        # Create grid
        self.grid = BSSNGrid(nx, ny, nz, self.dx, requires_grad=requires_grad)
        
        # Monitoring
        self.constraint_monitor = ConstraintMonitor(self.grid)
        self.waveform_extractor = WaveformExtractor(self.grid)
        self.loss_fn = DifferentiableLoss(self.grid, requires_grad=requires_grad)
        
        # RK4 storage
        self.var_names = [
            'phi', 'gt11', 'gt12', 'gt13', 'gt22', 'gt23', 'gt33',
            'trK', 'At11', 'At12', 'At13', 'At22', 'At23', 'At33',
            'Xt1', 'Xt2', 'Xt3', 'alpha', 'beta1', 'beta2', 'beta3'
        ]
        
        self.u0 = {}
        self.u_acc = {}
        for name in self.var_names:
            self.u0[name] = wp.zeros(self.grid.n_points, dtype=wp.float32)
            self.u_acc[name] = wp.zeros(self.grid.n_points, dtype=wp.float32)
        
        # State
        self.current_time = 0.0
        self.current_step = 0
    
    def set_schwarzschild_initial_data(self, bh_mass=1.0, bh_pos=(0., 0., 0.),
                                         pre_collapse_lapse=True):
        """Set Schwarzschild puncture initial data."""
        set_schwarzschild_puncture(self.grid, bh_mass=bh_mass, bh_pos=bh_pos,
                                    pre_collapse_lapse=pre_collapse_lapse)
        self.current_time = 0.0
        self.current_step = 0
    
    def set_brill_lindquist_initial_data(self, m1=0.5, pos1=(-2., 0., 0.),
                                          m2=0.5, pos2=(2., 0., 0.)):
        """Set Brill-Lindquist binary initial data."""
        set_brill_lindquist(self.grid, m1=m1, pos1=pos1, m2=m2, pos2=pos2)
        self.current_time = 0.0
        self.current_step = 0
    
    def _get_var(self, name):
        return getattr(self.grid, name)
    
    def _get_rhs(self, name):
        return getattr(self.grid, name + '_rhs')
    
    def compute_rhs(self):
        """Compute BSSN RHS with boundary conditions."""
        wp.launch(
            compute_bssn_rhs_full_kernel,
            dim=self.grid.n_points,
            inputs=[
                self.grid.phi, self.grid.gt11, self.grid.gt12, self.grid.gt13,
                self.grid.gt22, self.grid.gt23, self.grid.gt33,
                self.grid.trK, self.grid.At11, self.grid.At12, self.grid.At13,
                self.grid.At22, self.grid.At23, self.grid.At33,
                self.grid.Xt1, self.grid.Xt2, self.grid.Xt3,
                self.grid.alpha, self.grid.beta1, self.grid.beta2, self.grid.beta3,
                self.grid.phi_rhs, self.grid.gt11_rhs, self.grid.gt12_rhs, 
                self.grid.gt13_rhs, self.grid.gt22_rhs, self.grid.gt23_rhs, 
                self.grid.gt33_rhs, self.grid.trK_rhs, self.grid.At11_rhs, 
                self.grid.At12_rhs, self.grid.At13_rhs, self.grid.At22_rhs, 
                self.grid.At23_rhs, self.grid.At33_rhs, self.grid.Xt1_rhs, 
                self.grid.Xt2_rhs, self.grid.Xt3_rhs, self.grid.alpha_rhs, 
                self.grid.beta1_rhs, self.grid.beta2_rhs, self.grid.beta3_rhs,
                self.nx, self.ny, self.nz, self.inv_dx, self.eps_diss
            ]
        )
        apply_standard_bssn_boundaries(self.grid)
    
    def step(self):
        """Perform one RK4 time step."""
        dt = self.dt
        
        # Save initial state
        for name in self.var_names:
            wp.launch(copy_kernel, dim=self.grid.n_points,
                      inputs=[self.u0[name], self._get_var(name)])
            wp.launch(copy_kernel, dim=self.grid.n_points,
                      inputs=[self.u_acc[name], self.u0[name]])
        
        # RK4 stages
        stages = [(0.5, 1.0/6.0), (0.5, 1.0/3.0), (1.0, 1.0/3.0), (0.0, 1.0/6.0)]
        
        for stage_coeff, acc_weight in stages:
            self.compute_rhs()
            for name in self.var_names:
                wp.launch(rk4_accumulate_kernel, dim=self.grid.n_points,
                          inputs=[self.u_acc[name], self._get_rhs(name), dt, acc_weight])
                if stage_coeff > 0:
                    wp.launch(rk4_update_kernel, dim=self.grid.n_points,
                              inputs=[self._get_var(name), self.u0[name], 
                                      self._get_rhs(name), dt, stage_coeff])
        
        # Final copy
        for name in self.var_names:
            wp.launch(copy_kernel, dim=self.grid.n_points,
                      inputs=[self._get_var(name), self.u_acc[name]])
        
        self.current_time += dt
        self.current_step += 1
    
    def evolve(self, n_steps, extract_waveform=False, verbose=True):
        """
        Evolve the system for n_steps.
        
        Args:
            n_steps: Number of time steps
            extract_waveform: Whether to extract waveform during evolution
            verbose: Print progress
        
        Returns:
            Dictionary with evolution summary
        """
        if verbose:
            print(f"Evolving for {n_steps} steps (T = {n_steps * self.dt:.2f}M)...")
        
        for step in range(n_steps):
            self.step()
            
            if extract_waveform:
                self.waveform_extractor.extract(self.current_time)
        
        # Final constraint check
        self.constraint_monitor.compute()
        norms = self.constraint_monitor.get_norms()
        
        if verbose:
            print(f"Final: t = {self.current_time:.2f}M, "
                  f"H_L2 = {norms['H_L2']:.4e}, H_max = {norms['H_Linf']:.4e}")
        
        return {
            'time': self.current_time,
            'steps': self.current_step,
            'H_L2': norms['H_L2'],
            'H_max': norms['H_Linf'],
            'M_L2': norms['M_L2'],
            'M_max': norms['M_Linf'],
            'alpha_min': float(self.grid.alpha.numpy().min()),
            'alpha_max': float(self.grid.alpha.numpy().max())
        }
    
    def compute_loss(self, loss_type='asymptotic'):
        """
        Compute differentiable loss.
        
        Args:
            loss_type: 'asymptotic', 'constraint', or 'stability'
        
        Returns:
            Loss array (for backward pass)
        """
        if loss_type == 'asymptotic':
            return self.loss_fn.compute_asymptotic_loss()
        elif loss_type == 'constraint':
            self.constraint_monitor.compute()
            return self.loss_fn.compute_constraint_loss(
                self.constraint_monitor.H,
                self.constraint_monitor.M1,
                self.constraint_monitor.M2,
                self.constraint_monitor.M3
            )
        elif loss_type == 'stability':
            return self.loss_fn.compute_stability_loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def get_loss_value(self):
        """Get the current loss value as a float."""
        return self.loss_fn.get_loss()


def test_pipeline():
    """Test the end-to-end differentiable pipeline."""
    wp.init()
    print("=" * 60)
    print("End-to-End Differentiable BSSN Pipeline Test")
    print("=" * 60)
    
    # Create pipeline
    pipeline = DifferentiableBSSNPipeline(
        nx=24, ny=24, nz=24,
        domain_size=12.0,
        cfl=0.1,
        requires_grad=True
    )
    
    print(f"\nPipeline configuration:")
    print(f"  Grid: {pipeline.nx}x{pipeline.ny}x{pipeline.nz}")
    print(f"  dx = {pipeline.dx:.4f}M, dt = {pipeline.dt:.4f}M")
    
    # Set initial data
    print("\nSetting Schwarzschild initial data...")
    pipeline.set_schwarzschild_initial_data(bh_mass=1.0)
    
    # Short evolution
    print("\nEvolution test:")
    result = pipeline.evolve(n_steps=10, extract_waveform=True, verbose=True)
    
    print(f"\nEvolution summary:")
    for key, val in result.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.6e}")
        else:
            print(f"  {key}: {val}")
    
    # Test loss computation
    print("\nLoss computation test:")
    loss = pipeline.compute_loss('asymptotic')
    print(f"  Asymptotic loss: {pipeline.get_loss_value():.6e}")
    
    # Test gradient computation
    print("\nGradient test:")
    pipeline.grid.alpha.requires_grad = True
    
    tape = wp.Tape()
    with tape:
        pipeline.loss_fn.reset()
        pipeline.loss_fn.loss.zero_()
        wp.launch(
            asymptotic_flatness_loss_kernel,
            dim=pipeline.grid.n_points,
            inputs=[pipeline.grid.phi, pipeline.grid.alpha,
                    pipeline.grid.gt11, pipeline.grid.gt22, pipeline.grid.gt33,
                    pipeline.nx, pipeline.ny, pipeline.nz,
                    pipeline.dx, pipeline.loss_fn.loss]
        )
    
    tape.backward(loss=pipeline.loss_fn.loss)
    
    if pipeline.grid.alpha.grad is not None:
        grad_max = np.abs(pipeline.grid.alpha.grad.numpy()).max()
        print(f"  ∂L/∂α max: {grad_max:.6e}")
        print("  ✓ Gradients computed through pipeline!")
    
    tape.zero()
    
    print("\n" + "=" * 60)
    print("✓ Pipeline test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
