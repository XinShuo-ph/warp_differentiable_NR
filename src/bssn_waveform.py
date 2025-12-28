"""
Gravitational Waveform Extraction

Implements simplified gravitational wave extraction for BSSN evolution.
Uses the Newman-Penrose scalar ψ₄ approximation.
"""

import sys
sys.path.insert(0, '/workspace/src')

import warp as wp
import numpy as np
from bssn_vars import BSSNGrid
from bssn_initial_data import set_schwarzschild_puncture
from bssn_derivs import deriv_x_4th, deriv_y_4th, deriv_z_4th


@wp.kernel
def extract_waveform_sphere_kernel(
    # Metric components (for simplicity, using conformal factor)
    phi: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    trK: wp.array(dtype=wp.float32),
    # Output arrays
    psi4_real: wp.array(dtype=wp.float32),
    psi4_imag: wp.array(dtype=wp.float32),
    # Extraction parameters
    r_extract: float,
    nx: int, ny: int, nz: int,
    dx: float
):
    """
    Extract ψ₄-like signal at extraction sphere.
    
    For a full implementation, ψ₄ = C_{αβγδ} n^α m̄^β n^γ m̄^δ requires:
    - Computing the Weyl tensor C
    - Projecting onto null tetrad (n, m̄)
    
    This simplified version extracts an approximation based on K and its derivatives.
    """
    tid = wp.tid()
    
    k = tid // (nx * ny)
    j = (tid - k * nx * ny) // nx
    i = tid - k * nx * ny - j * nx
    
    # Grid coordinates
    x = (float(i) + 0.5) * dx - float(nx) * dx / 2.0
    y = (float(j) + 0.5) * dx - float(ny) * dx / 2.0
    z = (float(k) + 0.5) * dx - float(nz) * dx / 2.0
    
    r = wp.sqrt(x*x + y*y + z*z)
    
    # Only extract at shell around r_extract (width = 2*dx)
    dr = wp.abs(r - r_extract)
    if dr > 2.0 * dx:
        psi4_real[tid] = 0.0
        psi4_imag[tid] = 0.0
        return
    
    # Radial unit vector
    r_safe = wp.max(r, 0.1 * dx)
    nx_r = x / r_safe
    ny_r = y / r_safe
    nz_r = z / r_safe
    
    # Simplified ψ₄ approximation:
    # Uses second time derivative of quadrupole moment
    # Here we approximate with spatial derivatives of K
    inv_dx = 1.0 / dx
    
    dtrK_dx = deriv_x_4th(trK, i, j, k, nx, ny, inv_dx)
    dtrK_dy = deriv_y_4th(trK, i, j, k, nx, ny, inv_dx)
    dtrK_dz = deriv_z_4th(trK, i, j, k, nx, ny, inv_dx)
    
    # Radial derivative of K
    dK_dr = nx_r * dtrK_dx + ny_r * dtrK_dy + nz_r * dtrK_dz
    
    # Transverse components (simplified)
    # theta direction: (-sin(θ)cos(φ), -sin(θ)sin(φ), cos(θ))
    # For equatorial extraction (θ = π/2):
    rho = wp.sqrt(x*x + y*y)
    rho_safe = wp.max(rho, 0.1 * dx)
    
    cos_phi = x / rho_safe
    sin_phi = y / rho_safe
    
    # Approximate transverse component
    dK_trans = -sin_phi * dtrK_dx + cos_phi * dtrK_dy
    
    # Weight by shell membership
    shell_weight = wp.exp(-(dr * dr) / (dx * dx))
    
    # Store real and imaginary parts of ψ₄-like quantity
    # Real part ~ radial component
    # Imag part ~ azimuthal component
    psi4_real[tid] = dK_dr * shell_weight * alpha[tid]
    psi4_imag[tid] = dK_trans * shell_weight * alpha[tid]


@wp.kernel
def integrate_sphere_kernel(
    field: wp.array(dtype=wp.float32),
    r_extract: float,
    nx: int, ny: int, nz: int,
    dx: float,
    result: wp.array(dtype=wp.float32)
):
    """Integrate field over extraction sphere."""
    tid = wp.tid()
    
    k = tid // (nx * ny)
    j = (tid - k * nx * ny) // nx
    i = tid - k * nx * ny - j * nx
    
    x = (float(i) + 0.5) * dx - float(nx) * dx / 2.0
    y = (float(j) + 0.5) * dx - float(ny) * dx / 2.0
    z = (float(k) + 0.5) * dx - float(nz) * dx / 2.0
    
    r = wp.sqrt(x*x + y*y + z*z)
    
    dr = wp.abs(r - r_extract)
    if dr > 2.0 * dx:
        return
    
    shell_weight = wp.exp(-(dr * dr) / (dx * dx))
    dV = dx * dx * dx
    
    wp.atomic_add(result, 0, field[tid] * shell_weight * dV)


class WaveformExtractor:
    """
    Extracts gravitational waveforms from BSSN evolution data.
    """
    def __init__(self, grid, r_extract=None):
        self.grid = grid
        
        # Default extraction radius: half the domain size
        if r_extract is None:
            self.r_extract = min(grid.nx, grid.ny, grid.nz) * grid.dx / 4.0
        else:
            self.r_extract = r_extract
        
        # Storage for ψ₄ components
        self.psi4_real = wp.zeros(grid.n_points, dtype=wp.float32)
        self.psi4_imag = wp.zeros(grid.n_points, dtype=wp.float32)
        
        # Integrated values
        self.psi4_integrated_real = wp.zeros(1, dtype=wp.float32)
        self.psi4_integrated_imag = wp.zeros(1, dtype=wp.float32)
        
        # Time series
        self.times = []
        self.psi4_real_series = []
        self.psi4_imag_series = []
    
    def extract(self, t):
        """Extract waveform at current time."""
        wp.launch(
            extract_waveform_sphere_kernel,
            dim=self.grid.n_points,
            inputs=[
                self.grid.phi, self.grid.alpha, self.grid.trK,
                self.psi4_real, self.psi4_imag,
                self.r_extract,
                self.grid.nx, self.grid.ny, self.grid.nz,
                self.grid.dx
            ]
        )
        
        # Integrate over sphere
        self.psi4_integrated_real.zero_()
        self.psi4_integrated_imag.zero_()
        
        wp.launch(
            integrate_sphere_kernel,
            dim=self.grid.n_points,
            inputs=[
                self.psi4_real, self.r_extract,
                self.grid.nx, self.grid.ny, self.grid.nz,
                self.grid.dx, self.psi4_integrated_real
            ]
        )
        
        wp.launch(
            integrate_sphere_kernel,
            dim=self.grid.n_points,
            inputs=[
                self.psi4_imag, self.r_extract,
                self.grid.nx, self.grid.ny, self.grid.nz,
                self.grid.dx, self.psi4_integrated_imag
            ]
        )
        
        # Store in time series
        self.times.append(t)
        self.psi4_real_series.append(float(self.psi4_integrated_real.numpy()[0]))
        self.psi4_imag_series.append(float(self.psi4_integrated_imag.numpy()[0]))
    
    def get_waveform(self):
        """Return extracted waveform as numpy arrays."""
        return (np.array(self.times), 
                np.array(self.psi4_real_series),
                np.array(self.psi4_imag_series))
    
    def get_strain(self):
        """
        Compute strain from ψ₄ via double time integration.
        h = ∫∫ ψ₄ dt dt
        """
        times = np.array(self.times)
        psi4_real = np.array(self.psi4_real_series)
        psi4_imag = np.array(self.psi4_imag_series)
        
        if len(times) < 2:
            return times, np.zeros_like(psi4_real), np.zeros_like(psi4_imag)
        
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        
        # First integration
        h_dot_real = np.cumsum(psi4_real) * dt
        h_dot_imag = np.cumsum(psi4_imag) * dt
        
        # Second integration
        h_real = np.cumsum(h_dot_real) * dt
        h_imag = np.cumsum(h_dot_imag) * dt
        
        return times, h_real, h_imag


def test_waveform_extraction():
    """Test waveform extraction."""
    wp.init()
    print("=== Waveform Extraction Test ===\n")
    
    # Create grid with Schwarzschild data
    nx, ny, nz = 32, 32, 32
    domain_size = 16.0
    dx = domain_size / nx
    
    grid = BSSNGrid(nx, ny, nz, dx)
    set_schwarzschild_puncture(grid, bh_mass=1.0)
    
    print(f"Grid: {nx}x{ny}x{nz}, dx = {dx:.4f}M")
    
    # Create extractor
    r_extract = 4.0  # Extract at r = 4M
    extractor = WaveformExtractor(grid, r_extract=r_extract)
    
    print(f"Extraction radius: r = {r_extract:.1f}M")
    
    # Extract at initial time
    extractor.extract(t=0.0)
    
    times, psi4_real, psi4_imag = extractor.get_waveform()
    
    print(f"\nInitial extraction:")
    print(f"  ψ₄ (real): {psi4_real[0]:.6e}")
    print(f"  ψ₄ (imag): {psi4_imag[0]:.6e}")
    
    # For static Schwarzschild, ψ₄ should be small (no dynamics)
    # But spatial derivatives of K give non-zero values
    
    print(f"\nNote: For static Schwarzschild, gravitational waves are not expected.")
    print(f"      The extracted signal represents the static curvature, not radiation.")
    
    print("\n✓ Waveform extraction test completed.")


if __name__ == "__main__":
    test_waveform_extraction()
