"""
N-body simulator using angular momentum coupling.

This module provides tools for simulating the motion of N bodies under
angular momentum coupling (which recovers Newtonian gravity when L = m·σ₀).
"""

import numpy as np
import matplotlib.pyplot as plt
from core import constants as c
from core.coupling import AngularMomentumCoupling
from core.forces import ForceCalculator


class Body:
    """Represents a single body in the simulation."""
    
    def __init__(self, mass, position, velocity, name="Body"):
        """
        Initialize a body.
        
        Parameters:
        -----------
        mass : float
            Mass [kg]
        position : array-like, shape (3,)
            Initial position [m]
        velocity : array-like, shape (3,)
            Initial velocity [m/s]
        name : str
            Name of the body
        """
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.name = name
        
        # History for plotting
        self.position_history = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.time_history = [0.0]


class NBodySimulator:
    """
    N-body simulator using angular momentum coupling framework.
    """
    
    def __init__(self, use_angular_momentum=True, use_scale_dependence=False):
        """
        Initialize N-body simulator.
        
        Parameters:
        -----------
        use_angular_momentum : bool
            If True, use angular momentum coupling
            If False, use standard Newtonian gravity (for comparison)
        use_scale_dependence : bool
            If True, use scale-dependent σ_eff(r)
        """
        self.use_angular_momentum = use_angular_momentum
        self.bodies = []
        self.time = 0.0
        
        # Initialize force calculator
        if use_angular_momentum:
            coupling = AngularMomentumCoupling(use_scale_dependence=use_scale_dependence)
            self.force_calc = ForceCalculator(coupling)
        else:
            # Standard Newtonian (which is the same in this framework)
            self.force_calc = ForceCalculator()
        
        # Energy tracking
        self.energy_history = []
        self.angular_momentum_history = []
    
    def add_body(self, mass, position, velocity, name="Body"):
        """
        Add a body to the simulation.
        
        Parameters:
        -----------
        mass : float
            Mass [kg]
        position : array-like, shape (3,)
            Initial position [m]
        velocity : array-like, shape (3,)
            Initial velocity [m/s]
        name : str
            Name of the body
        """
        body = Body(mass, position, velocity, name)
        self.bodies.append(body)
    
    def get_state(self):
        """
        Get current state as arrays.
        
        Returns:
        --------
        positions : ndarray, shape (N, 3)
        velocities : ndarray, shape (N, 3)
        masses : ndarray, shape (N,)
        """
        N = len(self.bodies)
        positions = np.zeros((N, 3))
        velocities = np.zeros((N, 3))
        masses = np.zeros(N)
        
        for i, body in enumerate(self.bodies):
            positions[i] = body.position
            velocities[i] = body.velocity
            masses[i] = body.mass
        
        return positions, velocities, masses
    
    def set_state(self, positions, velocities):
        """
        Set state from arrays.
        
        Parameters:
        -----------
        positions : ndarray, shape (N, 3)
        velocities : ndarray, shape (N, 3)
        """
        for i, body in enumerate(self.bodies):
            body.position = positions[i].copy()
            body.velocity = velocities[i].copy()
    
    def step_rk4(self, dt):
        """
        Take one time step using 4th-order Runge-Kutta integration.
        
        Parameters:
        -----------
        dt : float
            Time step [s]
        """
        positions, velocities, masses = self.get_state()
        
        def derivatives(pos, vel):
            """Calculate derivatives: (dr/dt, dv/dt) = (v, a)"""
            accelerations = self.force_calc.accelerations(pos, masses)
            return vel, accelerations
        
        # RK4 integration
        # k1
        k1_pos, k1_vel = derivatives(positions, velocities)
        
        # k2
        pos2 = positions + 0.5 * dt * k1_pos
        vel2 = velocities + 0.5 * dt * k1_vel
        k2_pos, k2_vel = derivatives(pos2, vel2)
        
        # k3
        pos3 = positions + 0.5 * dt * k2_pos
        vel3 = velocities + 0.5 * dt * k2_vel
        k3_pos, k3_vel = derivatives(pos3, vel3)
        
        # k4
        pos4 = positions + dt * k3_pos
        vel4 = velocities + dt * k3_vel
        k4_pos, k4_vel = derivatives(pos4, vel4)
        
        # Combine
        new_positions = positions + (dt/6) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
        new_velocities = velocities + (dt/6) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)
        
        self.set_state(new_positions, new_velocities)
        self.time += dt
        
        # Store history
        for i, body in enumerate(self.bodies):
            body.position_history.append(body.position.copy())
            body.velocity_history.append(body.velocity.copy())
            body.time_history.append(self.time)
    
    def step_leapfrog(self, dt):
        """
        Take one time step using leapfrog (symplectic) integration.
        Better energy conservation for long simulations.
        
        Parameters:
        -----------
        dt : float
            Time step [s]
        """
        positions, velocities, masses = self.get_state()
        
        # Half-step velocity update
        accelerations = self.force_calc.accelerations(positions, masses)
        velocities_half = velocities + 0.5 * dt * accelerations
        
        # Full-step position update
        positions_new = positions + dt * velocities_half
        
        # Half-step velocity update (with new positions)
        accelerations_new = self.force_calc.accelerations(positions_new, masses)
        velocities_new = velocities_half + 0.5 * dt * accelerations_new
        
        self.set_state(positions_new, velocities_new)
        self.time += dt
        
        # Store history
        for i, body in enumerate(self.bodies):
            body.position_history.append(body.position.copy())
            body.velocity_history.append(body.velocity.copy())
            body.time_history.append(self.time)
    
    def run(self, duration, dt, method='leapfrog', track_energy=True):
        """
        Run the simulation.
        
        Parameters:
        -----------
        duration : float
            Simulation duration [s]
        dt : float
            Time step [s]
        method : str
            Integration method: 'rk4' or 'leapfrog'
        track_energy : bool
            If True, track energy and angular momentum
        """
        n_steps = int(duration / dt)
        
        print(f"Running simulation: {n_steps} steps, dt = {dt:.3e} s")
        print(f"Method: {method}")
        print(f"Angular momentum coupling: {self.use_angular_momentum}")
        
        # Choose integration method
        if method == 'rk4':
            step_func = self.step_rk4
        elif method == 'leapfrog':
            step_func = self.step_leapfrog
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Track initial energy
        if track_energy:
            positions, velocities, masses = self.get_state()
            E0 = self.force_calc.total_energy(positions, velocities, masses)
            L0 = self.force_calc.angular_momentum_vector(positions, velocities, masses)
            self.energy_history = [E0]
            self.angular_momentum_history = [L0]
        
        # Main loop
        for step in range(n_steps):
            step_func(dt)
            
            # Track energy periodically
            if track_energy and (step % max(1, n_steps // 100) == 0):
                positions, velocities, masses = self.get_state()
                E = self.force_calc.total_energy(positions, velocities, masses)
                L = self.force_calc.angular_momentum_vector(positions, velocities, masses)
                self.energy_history.append(E)
                self.angular_momentum_history.append(L)
            
            # Progress indicator
            if step % max(1, n_steps // 10) == 0:
                percent = 100 * step / n_steps
                print(f"  Progress: {percent:.0f}%", end='\r')
        
        print("  Progress: 100%")
        print("Simulation complete!")
        
        # Energy conservation check
        if track_energy and len(self.energy_history) > 1:
            E_final = self.energy_history[-1]
            dE = abs(E_final - E0) / abs(E0)
            print(f"Energy conservation: ΔE/E₀ = {dE:.3e}")
    
    def plot_orbits(self, plane='xy', figsize=(10, 10)):
        """
        Plot orbital trajectories.
        
        Parameters:
        -----------
        plane : str
            Projection plane: 'xy', 'xz', or 'yz'
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose axes
        if plane == 'xy':
            idx1, idx2 = 0, 1
            xlabel, ylabel = 'x', 'y'
        elif plane == 'xz':
            idx1, idx2 = 0, 2
            xlabel, ylabel = 'x', 'z'
        elif plane == 'yz':
            idx1, idx2 = 1, 2
            xlabel, ylabel = 'y', 'z'
        else:
            raise ValueError(f"Unknown plane: {plane}")
        
        # Plot each body
        for body in self.bodies:
            positions = np.array(body.position_history)
            ax.plot(positions[:, idx1], positions[:, idx2], 
                   label=body.name, linewidth=1.5)
            
            # Mark starting position
            ax.plot(positions[0, idx1], positions[0, idx2], 
                   'o', markersize=8, alpha=0.7)
        
        ax.set_xlabel(f'{xlabel} [m]')
        ax.set_ylabel(f'{ylabel} [m]')
        ax.set_title('Orbital Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_energy_conservation(self):
        """
        Plot energy and angular momentum conservation.
        """
        if len(self.energy_history) < 2:
            print("No energy history to plot. Run with track_energy=True.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Energy
        E0 = self.energy_history[0]
        E_array = np.array(self.energy_history)
        dE_rel = (E_array - E0) / abs(E0)
        
        ax1.plot(dE_rel, linewidth=1.5)
        ax1.set_ylabel('Relative Energy Change ΔE/E₀')
        ax1.set_title('Energy Conservation')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # Angular momentum
        L0 = self.angular_momentum_history[0]
        L_mag = [np.linalg.norm(L) for L in self.angular_momentum_history]
        L0_mag = np.linalg.norm(L0)
        dL_rel = (np.array(L_mag) - L0_mag) / L0_mag
        
        ax2.plot(dL_rel, linewidth=1.5)
        ax2.set_ylabel('Relative L Change ΔL/L₀')
        ax2.set_xlabel('Sample Point')
        ax2.set_title('Angular Momentum Conservation')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def get_orbital_parameters(self, body_idx=1):
        """
        Calculate orbital parameters for a body (assumes two-body problem).
        
        Parameters:
        -----------
        body_idx : int
            Index of orbiting body (0 is typically the central body)
            
        Returns:
        --------
        params : dict
            Dictionary with 'period', 'semi_major_axis', 'eccentricity'
        """
        if len(self.bodies) < 2:
            raise ValueError("Need at least 2 bodies for orbital parameters")
        
        # Get relative position and velocity
        positions = np.array(self.bodies[body_idx].position_history)
        velocities = np.array(self.bodies[body_idx].velocity_history)
        
        # Calculate orbital radius over time
        r = np.linalg.norm(positions, axis=1)
        
        # Semi-major axis (average radius for circular orbit, approximate)
        a = np.mean(r)
        
        # Period from Kepler's 3rd law (approximate)
        m_central = self.bodies[0].mass
        T = 2 * np.pi * np.sqrt(a**3 / (c.G * m_central))
        
        # Eccentricity (very approximate from r_max and r_min)
        r_max = np.max(r)
        r_min = np.min(r)
        e = (r_max - r_min) / (r_max + r_min)
        
        return {
            'period': T,
            'semi_major_axis': a,
            'eccentricity': e,
            'r_min': r_min,
            'r_max': r_max
        }


# ============================================================================
# EXAMPLE SIMULATIONS
# ============================================================================

def example_earth_sun():
    """
    Simulate Earth orbiting the Sun.
    """
    print("=" * 70)
    print("EXAMPLE: Earth-Sun System")
    print("=" * 70)
    
    # Create simulator
    sim = NBodySimulator(use_angular_momentum=True)
    
    # Add Sun at origin
    sim.add_body(
        mass=c.M_SUN,
        position=[0, 0, 0],
        velocity=[0, 0, 0],
        name="Sun"
    )
    
    # Add Earth at 1 AU with circular orbital velocity
    v_orbital = np.sqrt(c.G * c.M_SUN / c.AU)
    sim.add_body(
        mass=c.M_EARTH,
        position=[c.AU, 0, 0],
        velocity=[0, v_orbital, 0],
        name="Earth"
    )
    
    # Run for 2 years
    sim.run(duration=2*c.YEAR, dt=3600, method='leapfrog')
    
    # Calculate orbital parameters
    params = sim.get_orbital_parameters(body_idx=1)
    print(f"\nOrbital parameters:")
    print(f"  Period: {params['period']/c.YEAR:.3f} years")
    print(f"  Semi-major axis: {params['semi_major_axis']/c.AU:.6f} AU")
    print(f"  Eccentricity: {params['eccentricity']:.6f}")
    
    # Plot
    import os
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = sim.plot_orbits()
    orbit_path = os.path.join(output_dir, 'earth_sun_orbit.png')
    plt.savefig(orbit_path, dpi=150)
    print(f"\nSaved plot: {orbit_path}")
    
    sim.plot_energy_conservation()
    energy_path = os.path.join(output_dir, 'earth_sun_energy.png')
    plt.savefig(energy_path, dpi=150)
    print(f"Saved plot: {energy_path}")
    
    plt.close('all')


if __name__ == "__main__":
    example_earth_sun()
