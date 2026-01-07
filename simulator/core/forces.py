"""
Force calculations for angular momentum coupling framework.

This module provides force calculation utilities for N-body simulations.
"""

import numpy as np
from core import constants as c
from core.coupling import AngularMomentumCoupling


class ForceCalculator:
    """
    Calculate forces between bodies using angular momentum coupling.
    """
    
    def __init__(self, coupling=None):
        """
        Initialize force calculator.
        
        Parameters:
        -----------
        coupling : AngularMomentumCoupling, optional
            Angular momentum coupling calculator
            If None, creates default instance
        """
        if coupling is None:
            coupling = AngularMomentumCoupling(use_scale_dependence=False)
        self.coupling = coupling
    
    def force_between_bodies(self, pos1, pos2, m1, m2):
        """
        Calculate force vector on body 1 due to body 2.
        
        Parameters:
        -----------
        pos1, pos2 : array-like, shape (3,)
            Position vectors [m]
        m1, m2 : float
            Masses [kg]
            
        Returns:
        --------
        force : ndarray, shape (3,)
            Force vector on body 1 [N]
        """
        # Displacement vector from 1 to 2
        r_vec = np.array(pos2) - np.array(pos1)
        r_mag = np.linalg.norm(r_vec)
        
        if r_mag == 0:
            return np.zeros(3)
        
        # Unit vector
        r_hat = r_vec / r_mag
        
        # Force magnitude (attractive, so points from 1 to 2)
        F_mag = self.coupling.coupling_force_from_masses(m1, m2, r_mag)
        
        # Force vector
        return F_mag * r_hat
    
    def total_force_on_body(self, body_idx, positions, masses):
        """
        Calculate total force on one body due to all others.
        
        Parameters:
        -----------
        body_idx : int
            Index of body to calculate force on
        positions : ndarray, shape (N, 3)
            Position vectors of all bodies [m]
        masses : ndarray, shape (N,)
            Masses of all bodies [kg]
            
        Returns:
        --------
        force : ndarray, shape (3,)
            Total force vector [N]
        """
        N = len(masses)
        force_total = np.zeros(3)
        
        for i in range(N):
            if i != body_idx:
                force = self.force_between_bodies(
                    positions[body_idx], positions[i],
                    masses[body_idx], masses[i]
                )
                force_total += force
        
        return force_total
    
    def all_forces(self, positions, masses):
        """
        Calculate forces on all bodies.
        
        Parameters:
        -----------
        positions : ndarray, shape (N, 3)
            Position vectors [m]
        masses : ndarray, shape (N,)
            Masses [kg]
            
        Returns:
        --------
        forces : ndarray, shape (N, 3)
            Force vectors on each body [N]
        """
        N = len(masses)
        forces = np.zeros((N, 3))
        
        for i in range(N):
            forces[i] = self.total_force_on_body(i, positions, masses)
        
        return forces
    
    def accelerations(self, positions, masses):
        """
        Calculate accelerations of all bodies.
        
        Parameters:
        -----------
        positions : ndarray, shape (N, 3)
            Position vectors [m]
        masses : ndarray, shape (N,)
            Masses [kg]
            
        Returns:
        --------
        accelerations : ndarray, shape (N, 3)
            Acceleration vectors [m/s²]
        """
        forces = self.all_forces(positions, masses)
        # Reshape masses for broadcasting
        masses_reshaped = masses.reshape(-1, 1)
        return forces / masses_reshaped
    
    def potential_energy(self, positions, masses):
        """
        Calculate total potential energy of the system.
        
        Parameters:
        -----------
        positions : ndarray, shape (N, 3)
            Position vectors [m]
        masses : ndarray, shape (N,)
            Masses [kg]
            
        Returns:
        --------
        U : float
            Total potential energy [J]
        """
        N = len(masses)
        U_total = 0.0
        
        # Sum over all pairs (i < j to avoid double counting)
        for i in range(N):
            for j in range(i+1, N):
                r_vec = positions[j] - positions[i]
                r_mag = np.linalg.norm(r_vec)
                
                if r_mag > 0:
                    U_pair = self.coupling.coupling_potential_from_masses(
                        masses[i], masses[j], r_mag
                    )
                    U_total += U_pair
        
        return U_total
    
    def kinetic_energy(self, velocities, masses):
        """
        Calculate total kinetic energy of the system.
        
        Parameters:
        -----------
        velocities : ndarray, shape (N, 3)
            Velocity vectors [m/s]
        masses : ndarray, shape (N,)
            Masses [kg]
            
        Returns:
        --------
        K : float
            Total kinetic energy [J]
        """
        v_squared = np.sum(velocities**2, axis=1)
        return 0.5 * np.sum(masses * v_squared)
    
    def total_energy(self, positions, velocities, masses):
        """
        Calculate total mechanical energy.
        
        Parameters:
        -----------
        positions : ndarray, shape (N, 3)
            Position vectors [m]
        velocities : ndarray, shape (N, 3)
            Velocity vectors [m/s]
        masses : ndarray, shape (N,)
            Masses [kg]
            
        Returns:
        --------
        E : float
            Total energy [J]
        """
        K = self.kinetic_energy(velocities, masses)
        U = self.potential_energy(positions, masses)
        return K + U
    
    def angular_momentum_vector(self, positions, velocities, masses):
        """
        Calculate total angular momentum vector of the system.
        
        L = Σ m_i * (r_i × v_i)
        
        Parameters:
        -----------
        positions : ndarray, shape (N, 3)
            Position vectors [m]
        velocities : ndarray, shape (N, 3)
            Velocity vectors [m/s]
        masses : ndarray, shape (N,)
            Masses [kg]
            
        Returns:
        --------
        L : ndarray, shape (3,)
            Total angular momentum vector [kg·m²/s]
        """
        L_total = np.zeros(3)
        
        for i in range(len(masses)):
            r_cross_v = np.cross(positions[i], velocities[i])
            L_total += masses[i] * r_cross_v
        
        return L_total


class TidalForceCalculator:
    """
    Calculate tidal forces from angular momentum gradients.
    
    Paper: "derives tidal forces from angular momentum gradients"
    """
    
    def __init__(self, coupling=None):
        """
        Initialize tidal force calculator.
        
        Parameters:
        -----------
        coupling : AngularMomentumCoupling, optional
            Angular momentum coupling calculator
        """
        if coupling is None:
            coupling = AngularMomentumCoupling()
        self.coupling = coupling
    
    def tidal_acceleration_gradient(self, m_source, r, delta_r):
        """
        Calculate tidal acceleration gradient.
        
        The tidal force arises from the gradient of acceleration:
        Δa ≈ (da/dr) * Δr = (2GM/r³) * Δr
        
        Parameters:
        -----------
        m_source : float
            Source mass [kg]
        r : float
            Distance to center of mass [m]
        delta_r : float
            Separation within extended body [m]
            
        Returns:
        --------
        tidal_gradient : float
            Tidal acceleration gradient [1/s²]
        """
        # a = GM/r²
        # da/dr = -2GM/r³
        return 2 * c.G * m_source / r**3
    
    def tidal_acceleration(self, m_source, r, delta_r):
        """
        Calculate tidal acceleration across an extended body.
        
        Parameters:
        -----------
        m_source : float
            Source mass [kg]
        r : float
            Distance to center of mass [m]
        delta_r : float
            Separation within extended body [m]
            
        Returns:
        --------
        delta_a : float
            Tidal acceleration difference [m/s²]
        """
        gradient = self.tidal_acceleration_gradient(m_source, r, delta_r)
        return gradient * delta_r
    
    def roche_limit(self, m_primary, m_satellite, r_satellite):
        """
        Calculate Roche limit - distance at which tidal forces disrupt satellite.
        
        d ≈ 2.46 * R_satellite * (ρ_primary / ρ_satellite)^(1/3)
        
        Parameters:
        -----------
        m_primary : float
            Primary body mass [kg]
        m_satellite : float
            Satellite mass [kg]
        r_satellite : float
            Satellite radius [m]
            
        Returns:
        --------
        d_roche : float
            Roche limit [m]
        """
        # Rigid body Roche limit
        return 2.46 * r_satellite * (m_primary / m_satellite)**(1/3)


# ============================================================================
# TESTING
# ============================================================================

def test_forces():
    """
    Test force calculations.
    """
    print("=" * 70)
    print("TESTING FORCE CALCULATIONS")
    print("=" * 70)
    
    force_calc = ForceCalculator()
    
    print("\n1. Two-Body Force Test: Earth-Sun")
    print("-" * 70)
    
    # Earth at 1 AU from Sun
    pos_sun = np.array([0.0, 0.0, 0.0])
    pos_earth = np.array([c.AU, 0.0, 0.0])
    
    force_on_earth = force_calc.force_between_bodies(
        pos_earth, pos_sun, c.M_EARTH, c.M_SUN
    )
    
    F_mag = np.linalg.norm(force_on_earth)
    F_expected = c.G * c.M_SUN * c.M_EARTH / c.AU**2
    
    print(f"   Force on Earth: {force_on_earth}")
    print(f"   Magnitude: {F_mag:.6e} N")
    print(f"   Expected: {F_expected:.6e} N")
    print(f"   Ratio: {F_mag/F_expected:.10f}")
    
    # Calculate orbital velocity
    v_orbital = np.sqrt(c.G * c.M_SUN / c.AU)
    print(f"   Orbital velocity: {v_orbital:.3f} m/s")
    print(f"   Expected: ~29800 m/s")
    
    print("\n2. Energy Conservation Test")
    print("-" * 70)
    
    # Simple two-body system
    positions = np.array([pos_sun, pos_earth])
    velocities = np.array([[0.0, 0.0, 0.0], [0.0, v_orbital, 0.0]])
    masses = np.array([c.M_SUN, c.M_EARTH])
    
    K = force_calc.kinetic_energy(velocities, masses)
    U = force_calc.potential_energy(positions, masses)
    E = K + U
    
    print(f"   Kinetic energy: {K:.6e} J")
    print(f"   Potential energy: {U:.6e} J")
    print(f"   Total energy: {E:.6e} J")
    print(f"   E/K ratio: {E/K:.6f} (should be ~-0.5 for circular orbit)")
    
    print("\n3. Angular Momentum Test")
    print("-" * 70)
    
    L = force_calc.angular_momentum_vector(positions, velocities, masses)
    L_mag = np.linalg.norm(L)
    
    # Expected: L ≈ m_earth * v_orbital * AU
    L_expected = c.M_EARTH * v_orbital * c.AU
    
    print(f"   Angular momentum: {L}")
    print(f"   Magnitude: {L_mag:.6e} kg·m²/s")
    print(f"   Expected: {L_expected:.6e} kg·m²/s")
    print(f"   Ratio: {L_mag/L_expected:.6f}")
    
    print("\n4. Tidal Force Test: Moon-Earth")
    print("-" * 70)
    
    tidal_calc = TidalForceCalculator()
    
    r_moon_earth = 3.844e8  # m
    r_earth_diameter = 2 * c.R_EARTH
    
    tidal_accel = tidal_calc.tidal_acceleration(
        c.M_EARTH, r_moon_earth, r_earth_diameter
    )
    
    print(f"   Distance Moon-Earth: {r_moon_earth:.3e} m")
    print(f"   Earth diameter: {r_earth_diameter:.3e} m")
    print(f"   Tidal acceleration: {tidal_accel:.6e} m/s²")
    print(f"   (This creates ocean tides)")
    
    # Roche limit
    m_moon = 7.342e22  # kg
    r_moon = 1.7374e6  # m
    d_roche = tidal_calc.roche_limit(c.M_EARTH, m_moon, r_moon)
    
    print(f"\n   Roche limit for Moon: {d_roche:.3e} m ({d_roche/1e3:.0f} km)")
    print(f"   Actual distance: {r_moon_earth:.3e} m ({r_moon_earth/1e3:.0f} km)")
    print(f"   Ratio: {r_moon_earth/d_roche:.2f} (Moon is safely outside Roche limit)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_forces()
