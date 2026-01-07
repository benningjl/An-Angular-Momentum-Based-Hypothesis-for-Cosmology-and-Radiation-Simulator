"""
Primordial sphere model and structure formation.

Paper: "The model begins with a primordial, perfectly smooth, rotating sphere 
containing all matter in the universe in a state of uniform spin and angular 
momentum."
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Handle imports for both module and script usage
try:
    from ..core import constants as c
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core import constants as c


class PrimordialSphere:
    """
    Model of the primordial rotating sphere.
    """
    
    def __init__(self, total_mass, radius, angular_velocity):
        """
        Initialize primordial sphere.
        
        Parameters:
        -----------
        total_mass : float
            Total mass [kg]
        radius : float
            Sphere radius [m]
        angular_velocity : float
            Angular velocity [rad/s]
        """
        self.M = total_mass
        self.R = radius
        self.omega = angular_velocity
        
        # Calculate properties
        self.density = self._calculate_density()
        self.moment_of_inertia = self._calculate_moment_of_inertia()
        self.angular_momentum = self._calculate_angular_momentum()
        self.rotational_energy = self._calculate_rotational_energy()
        self.sigma_0 = self._calculate_specific_angular_momentum()
    
    def _calculate_density(self):
        """Calculate uniform density."""
        volume = (4/3) * np.pi * self.R**3
        return self.M / volume
    
    def _calculate_moment_of_inertia(self):
        """Calculate moment of inertia for uniform sphere."""
        return (2/5) * self.M * self.R**2
    
    def _calculate_angular_momentum(self):
        """Calculate total angular momentum."""
        return self.moment_of_inertia * self.omega
    
    def _calculate_rotational_energy(self):
        """Calculate rotational kinetic energy."""
        return 0.5 * self.moment_of_inertia * self.omega**2
    
    def _calculate_specific_angular_momentum(self):
        """Calculate specific angular momentum σ₀ = L/M."""
        return self.angular_momentum / self.M
    
    def surface_velocity(self, latitude=0):
        """
        Calculate surface velocity at given latitude.
        
        Parameters:
        -----------
        latitude : float
            Latitude [radians], 0 at equator, ±π/2 at poles
            
        Returns:
        --------
        v : float
            Surface velocity [m/s]
        """
        r_perp = self.R * np.cos(latitude)
        return self.omega * r_perp
    
    def centrifugal_acceleration(self, latitude=0):
        """
        Calculate centrifugal acceleration at surface.
        
        Parameters:
        -----------
        latitude : float
            Latitude [radians]
            
        Returns:
        --------
        a_centrifugal : float
            Centrifugal acceleration [m/s²]
        """
        r_perp = self.R * np.cos(latitude)
        return self.omega**2 * r_perp
    
    def stability_parameter(self, binding_energy):
        """
        Calculate stability parameter Λ = E_bind / E_rot.
        
        Sphere is stable when Λ > 1.
        
        Parameters:
        -----------
        binding_energy : float
            Internal binding energy [J]
            
        Returns:
        --------
        Lambda : float
            Stability parameter
        """
        return binding_energy / self.rotational_energy
    
    def critical_angular_velocity(self, binding_energy):
        """
        Calculate critical angular velocity for stability.
        
        Parameters:
        -----------
        binding_energy : float
            Internal binding energy [J]
            
        Returns:
        --------
        omega_crit : float
            Critical angular velocity [rad/s]
        """
        return np.sqrt(2 * binding_energy / self.moment_of_inertia)
    
    def particle_velocity_at_position(self, r, theta):
        """
        Calculate velocity of particle at (r, θ) in spherical coordinates.
        
        Parameters:
        -----------
        r : float
            Radial distance from center [m]
        theta : float
            Polar angle from rotation axis [radians]
            
        Returns:
        --------
        v : float
            Velocity magnitude [m/s]
        """
        r_perp = r * np.sin(theta)
        return self.omega * r_perp


class StructureFormation:
    """
    Model structure formation from primordial sphere breakup.
    """
    
    def __init__(self, primordial_sphere, n_particles=1000):
        """
        Initialize structure formation simulator.
        
        Parameters:
        -----------
        primordial_sphere : PrimordialSphere
            The primordial sphere
        n_particles : int
            Number of test particles to track
        """
        self.sphere = primordial_sphere
        self.n_particles = n_particles
        
        # Initialize particle positions and velocities
        self.positions = None
        self.velocities = None
        self.masses = None
        
        self._initialize_particles()
    
    def _initialize_particles(self):
        """
        Initialize particle positions uniformly in sphere.
        """
        # Generate random positions in sphere
        # Use rejection sampling for uniform distribution
        positions = []
        
        while len(positions) < self.n_particles:
            # Random point in cube
            x = np.random.uniform(-1, 1, (self.n_particles * 2, 3))
            # Keep only points inside unit sphere
            r = np.linalg.norm(x, axis=1)
            inside = x[r <= 1]
            positions.extend(inside)
        
        positions = np.array(positions[:self.n_particles])
        
        # Scale to sphere radius
        self.positions = positions * self.sphere.R
        
        # Calculate velocities (tangent to rotation)
        self.velocities = np.zeros((self.n_particles, 3))
        
        for i, pos in enumerate(self.positions):
            r = np.linalg.norm(pos[:2])  # Distance from z-axis
            if r > 0:
                # Velocity perpendicular to radius in xy-plane
                v_mag = self.sphere.omega * r
                # Direction: tangent to circle
                self.velocities[i, 0] = -pos[1] / r * v_mag
                self.velocities[i, 1] = pos[0] / r * v_mag
                self.velocities[i, 2] = 0
        
        # Equal mass particles
        self.masses = np.ones(self.n_particles) * (self.sphere.M / self.n_particles)
    
    def add_asymmetry(self, epsilon=1e-6):
        """
        Add small asymmetry to trigger breakup.
        
        Paper: "A minute asymmetry at the exact moment when this sphere 
        transitions from orbiting to linear motion triggers an explosive 
        event analogous to the Big Bang."
        
        Parameters:
        -----------
        epsilon : float
            Asymmetry parameter (relative perturbation)
        """
        # Add small random perturbation to velocities
        perturbation = np.random.randn(*self.velocities.shape) * epsilon
        perturbation *= np.linalg.norm(self.velocities, axis=1, keepdims=True)
        
        self.velocities += perturbation
        
        # Add small position offset (center of mass shift)
        com_shift = np.random.randn(3) * self.sphere.R * epsilon
        self.positions += com_shift
    
    def evolve_trajectories(self, time, n_steps=1000):
        """
        Evolve particle trajectories (ballistic, no interactions).
        
        This simulates particles moving away from the explosion.
        
        Parameters:
        -----------
        time : float
            Evolution time [s]
        n_steps : int
            Number of time steps
            
        Returns:
        --------
        trajectories : ndarray, shape (n_steps, n_particles, 3)
            Particle positions at each time step
        """
        dt = time / n_steps
        trajectories = np.zeros((n_steps, self.n_particles, 3))
        
        pos = self.positions.copy()
        vel = self.velocities.copy()
        
        for step in range(n_steps):
            trajectories[step] = pos
            pos = pos + vel * dt
        
        return trajectories
    
    def find_convergence_points(self, trajectories, distance_threshold=1e24):
        """
        Find points where particle trajectories converge.
        
        These represent proto-galaxy formation sites.
        
        Parameters:
        -----------
        trajectories : ndarray
            Particle trajectories
        distance_threshold : float
            Maximum distance to consider convergence [m]
            
        Returns:
        --------
        convergence_points : list
            List of (position, n_particles) tuples
        """
        # Use final positions
        final_positions = trajectories[-1]
        
        # Simple clustering: find regions with high density
        from scipy.spatial import cKDTree
        
        tree = cKDTree(final_positions)
        pairs = tree.query_pairs(distance_threshold)
        
        # Count particles within threshold of each other
        convergence_counts = np.zeros(self.n_particles)
        for i, j in pairs:
            convergence_counts[i] += 1
            convergence_counts[j] += 1
        
        # Find high-convergence regions
        threshold_count = np.percentile(convergence_counts, 90)
        convergence_indices = np.where(convergence_counts > threshold_count)[0]
        
        return convergence_indices, convergence_counts


def calculate_primordial_parameters():
    """
    Calculate parameters for the primordial sphere.
    """
    print("=" * 70)
    print("PRIMORDIAL SPHERE PARAMETERS")
    print("=" * 70)
    
    # Observable universe parameters
    M_universe = 1e53  # kg
    R_universe = 1e26  # m (rough estimate for initial radius)
    
    # Specific angular momentum from framework
    sigma_0_primordial = c.SIGMA_0_MACRO  # m²/s
    
    # Angular momentum
    L_primordial = M_universe * sigma_0_primordial
    
    # For a uniform sphere: L = (2/5) * M * R² * ω
    # So: ω = L / [(2/5) * M * R²] = 5L / (2MR²)
    omega = 5 * L_primordial / (2 * M_universe * R_universe**2)
    
    sphere = PrimordialSphere(M_universe, R_universe, omega)
    
    print(f"\nTotal mass: {sphere.M:.3e} kg")
    print(f"Radius: {sphere.R:.3e} m ({sphere.R/c.PARSEC:.3e} parsecs)")
    print(f"Angular velocity: {sphere.omega:.3e} rad/s")
    print(f"Period: {2*np.pi/sphere.omega:.3e} s ({2*np.pi/sphere.omega/c.YEAR:.3e} years)")
    
    print(f"\nDensity: {sphere.density:.3e} kg/m³")
    print(f"Angular momentum: {sphere.angular_momentum:.3e} kg·m²/s")
    print(f"Specific angular momentum σ₀: {sphere.sigma_0:.3e} m²/s")
    print(f"Rotational energy: {sphere.rotational_energy:.3e} J")
    
    # Surface velocity
    v_equator = sphere.surface_velocity(0)
    v_pole = sphere.surface_velocity(np.pi/2)
    
    print(f"\nSurface velocity (equator): {v_equator:.3e} m/s ({v_equator/c.C:.3e} c)")
    print(f"Surface velocity (pole): {v_pole:.3e} m/s")
    
    # Centrifugal acceleration
    a_centrifugal = sphere.centrifugal_acceleration(0)
    print(f"Centrifugal acceleration (equator): {a_centrifugal:.3e} m/s²")
    
    print("\n" + "=" * 70)
    
    return sphere


def simulate_structure_formation():
    """
    Simulate structure formation from primordial sphere.
    """
    print("\n" + "=" * 70)
    print("STRUCTURE FORMATION SIMULATION")
    print("=" * 70)
    
    # Create smaller example sphere for visualization
    M_sphere = 1e42  # kg (smaller for simulation)
    R_sphere = 1e20  # m
    sigma_0 = 1e6  # m²/s
    
    L = M_sphere * sigma_0
    omega = 5 * L / (2 * M_sphere * R_sphere**2)
    
    sphere = PrimordialSphere(M_sphere, R_sphere, omega)
    
    print(f"\nSimulation parameters:")
    print(f"  Mass: {M_sphere:.2e} kg")
    print(f"  Radius: {R_sphere:.2e} m")
    print(f"  Angular velocity: {omega:.3e} rad/s")
    
    # Create structure formation model
    print(f"\nInitializing {1000} test particles...")
    structure = StructureFormation(sphere, n_particles=1000)
    
    # Add asymmetry
    structure.add_asymmetry(epsilon=1e-4)
    print(f"Added asymmetry: ε = 1e-4")
    
    # Evolve
    time_evolution = 1e15  # seconds (~30 million years)
    print(f"\nEvolving for {time_evolution:.2e} s...")
    trajectories = structure.evolve_trajectories(time_evolution, n_steps=100)
    
    # Plot results
    fig = plt.figure(figsize=(15, 5))
    
    # Initial state
    ax1 = fig.add_subplot(131, projection='3d')
    initial_pos = trajectories[0]
    ax1.scatter(initial_pos[:, 0], initial_pos[:, 1], initial_pos[:, 2], 
               s=1, alpha=0.5)
    ax1.set_title('Initial State (Primordial Sphere)')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_zlabel('z [m]')
    
    # Mid evolution
    ax2 = fig.add_subplot(132, projection='3d')
    mid_pos = trajectories[len(trajectories)//2]
    ax2.scatter(mid_pos[:, 0], mid_pos[:, 1], mid_pos[:, 2], 
               s=1, alpha=0.5)
    ax2.set_title('Mid Evolution (Expansion)')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_zlabel('z [m]')
    
    # Final state
    ax3 = fig.add_subplot(133, projection='3d')
    final_pos = trajectories[-1]
    ax3.scatter(final_pos[:, 0], final_pos[:, 1], final_pos[:, 2], 
               s=1, alpha=0.5, c='red')
    ax3.set_title('Final State (Structure Formation)')
    ax3.set_xlabel('x [m]')
    ax3.set_ylabel('y [m]')
    ax3.set_zlabel('z [m]')
    
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'structure_formation.png'), dpi=150)
    print("\nSaved plot: structure_formation.png")
    plt.close()
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    sphere = calculate_primordial_parameters()
    simulate_structure_formation()
