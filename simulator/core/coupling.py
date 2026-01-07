"""
Angular momentum coupling implementation.

This module implements the core angular momentum coupling framework:
- U = -G·L₁·L₂/(σ₀²·r) potential
- L = m·σ₀ inherited angular momentum
- Scale-dependent σ_eff(r) transitions
"""

import numpy as np
from core import constants as c


class AngularMomentumCoupling:
    """
    Implements angular momentum coupling between bodies.
    
    The framework replaces gravitational attraction with angular momentum
    coupling: U = -G·L₁·L₂/(σ_eff²·r)
    
    When L = m·σ₀, this recovers Newtonian gravity: U = -Gm₁m₂/r
    """
    
    def __init__(self, use_scale_dependence=True, 
                 sigma_macro=c.SIGMA_0_MACRO,
                 sigma_nuclear=c.SIGMA_0_NUCLEAR,
                 r_transition=c.R_TRANSITION,
                 n_transition=c.N_TRANSITION):
        """
        Initialize angular momentum coupling calculator.
        
        Parameters:
        -----------
        use_scale_dependence : bool
            If True, use scale-dependent σ_eff(r)
            If False, use constant σ_macro for all scales
        sigma_macro : float
            Macroscopic specific angular momentum [m²/s]
        sigma_nuclear : float
            Nuclear/quantum specific angular momentum [m²/s]
        r_transition : float
            Transition scale between regimes [m]
        n_transition : float
            Transition sharpness parameter
        """
        self.use_scale_dependence = use_scale_dependence
        self.sigma_macro = sigma_macro
        self.sigma_nuclear = sigma_nuclear
        self.r_transition = r_transition
        self.n_transition = n_transition
    
    def sigma_eff(self, r):
        """
        Calculate effective specific angular momentum at scale r.
        
        Parameters:
        -----------
        r : float or array
            Separation distance [m]
            
        Returns:
        --------
        sigma_eff : float or array
            Effective specific angular momentum [m²/s]
        """
        if not self.use_scale_dependence:
            # Use constant macroscopic value
            if np.isscalar(r):
                return self.sigma_macro
            else:
                return np.full_like(r, self.sigma_macro)
        
        # Scale-dependent transition
        return c.sigma_effective(r, self.sigma_macro, self.sigma_nuclear,
                                self.r_transition, self.n_transition)
    
    def inherited_angular_momentum(self, mass, r=None):
        """
        Calculate inherited angular momentum L = m·σ_eff(r).
        
        Parameters:
        -----------
        mass : float or array
            Mass [kg]
        r : float or array, optional
            Separation scale for σ_eff(r)
            If None, uses σ_macro
            
        Returns:
        --------
        L : float or array
            Angular momentum [kg·m²/s]
        """
        if r is None:
            sigma = self.sigma_macro
        else:
            sigma = self.sigma_eff(r)
        
        return mass * sigma
    
    def coupling_potential(self, L1, L2, r):
        """
        Calculate coupling potential energy between two angular momenta.
        
        U = -G·L₁·L₂/(σ_eff²·r)
        
        Parameters:
        -----------
        L1, L2 : float or array
            Angular momenta [kg·m²/s]
        r : float or array
            Separation distance [m]
            
        Returns:
        --------
        U : float or array
            Potential energy [J]
        """
        sigma = self.sigma_eff(r)
        return -c.G * L1 * L2 / (sigma**2 * r)
    
    def coupling_potential_from_masses(self, m1, m2, r):
        """
        Calculate coupling potential using masses with L = m·σ_eff(r).
        
        This should recover Newtonian potential U = -Gm₁m₂/r
        
        Parameters:
        -----------
        m1, m2 : float or array
            Masses [kg]
        r : float or array
            Separation distance [m]
            
        Returns:
        --------
        U : float or array
            Potential energy [J]
        """
        sigma = self.sigma_eff(r)
        L1 = m1 * sigma
        L2 = m2 * sigma
        
        # U = -G·L₁·L₂/(σ²·r) = -G·(m₁σ)·(m₂σ)/(σ²·r) = -Gm₁m₂/r
        return -c.G * L1 * L2 / (sigma**2 * r)
    
    def coupling_force_magnitude(self, L1, L2, r):
        """
        Calculate magnitude of coupling force.
        
        F = -dU/dr = -G·L₁·L₂/(σ_eff²·r²) + correction terms
        
        For constant σ_eff: F = G·L₁·L₂/(σ_eff²·r²)
        For scale-dependent σ_eff: includes derivative terms
        
        Parameters:
        -----------
        L1, L2 : float or array
            Angular momenta [kg·m²/s]
        r : float or array
            Separation distance [m]
            
        Returns:
        --------
        F : float or array
            Force magnitude [N]
        """
        sigma = self.sigma_eff(r)
        
        if not self.use_scale_dependence:
            # Simple 1/r² law
            return c.G * L1 * L2 / (sigma**2 * r**2)
        
        # For scale-dependent case, compute derivative numerically
        dr = r * 1e-6  # Small perturbation
        U_plus = self.coupling_potential(L1, L2, r + dr)
        U_minus = self.coupling_potential(L1, L2, r - dr)
        return -(U_plus - U_minus) / (2 * dr)
    
    def coupling_force_from_masses(self, m1, m2, r):
        """
        Calculate force magnitude using masses.
        
        For L = m·σ_eff(r), this recovers Newton's law: F = Gm₁m₂/r²
        
        Parameters:
        -----------
        m1, m2 : float or array
            Masses [kg]
        r : float or array
            Separation distance [m]
            
        Returns:
        --------
        F : float or array
            Force magnitude [N]
        """
        sigma = self.sigma_eff(r)
        L1 = m1 * sigma
        L2 = m2 * sigma
        return self.coupling_force_magnitude(L1, L2, r)
    
    def acceleration(self, m_source, r):
        """
        Calculate acceleration of a test particle due to source mass.
        
        From the equivalence principle: a = GM_source/r²
        (independent of test particle mass)
        
        Parameters:
        -----------
        m_source : float or array
            Source mass [kg]
        r : float or array
            Distance from source [m]
            
        Returns:
        --------
        a : float or array
            Acceleration magnitude [m/s²]
        """
        # Using F = m_test · a and F = Gm₁m₂/r²
        # Cancels to a = Gm_source/r²
        return c.G * m_source / r**2
    
    def verify_newtonian_equivalence(self, m1, m2, r, tolerance=1e-10):
        """
        Verify that coupling potential matches Newtonian gravity.
        
        Parameters:
        -----------
        m1, m2 : float
            Masses [kg]
        r : float
            Separation [m]
        tolerance : float
            Relative tolerance for equivalence check
            
        Returns:
        --------
        matches : bool
            True if potentials match within tolerance
        ratio : float
            Ratio of coupling potential to Newtonian potential
        """
        U_coupling = self.coupling_potential_from_masses(m1, m2, r)
        U_newton = -c.G * m1 * m2 / r
        
        ratio = U_coupling / U_newton
        matches = np.abs(ratio - 1.0) < tolerance
        
        return matches, ratio


class TotalAngularMomentum:
    """
    Manages total angular momentum including inherited and intrinsic components.
    
    L_total = L_inherited + L_intrinsic
    L_inherited = m·σ₀
    L_intrinsic = quantum spin (s = ±ℏ/2 for fermions, etc.)
    """
    
    def __init__(self, coupling_calculator):
        """
        Initialize with an AngularMomentumCoupling instance.
        
        Parameters:
        -----------
        coupling_calculator : AngularMomentumCoupling
            Calculator for inherited angular momentum
        """
        self.coupling = coupling_calculator
    
    def total_angular_momentum(self, mass, spin=0, r=None):
        """
        Calculate total angular momentum.
        
        Parameters:
        -----------
        mass : float
            Mass [kg]
        spin : float
            Intrinsic quantum spin [J·s = kg·m²/s]
            Default is 0 (classical limit)
        r : float, optional
            Scale for σ_eff(r)
            
        Returns:
        --------
        L_total : float
            Total angular momentum [kg·m²/s]
        """
        L_inherited = self.coupling.inherited_angular_momentum(mass, r)
        return L_inherited + spin
    
    def dominance_ratio(self, mass, spin, r=None):
        """
        Calculate ratio of inherited to intrinsic angular momentum.
        
        For macroscopic objects, L_inherited >> L_intrinsic
        
        Parameters:
        -----------
        mass : float
            Mass [kg]
        spin : float
            Intrinsic quantum spin [kg·m²/s]
        r : float, optional
            Scale for σ_eff(r)
            
        Returns:
        --------
        ratio : float
            L_inherited / L_intrinsic
        """
        L_inherited = self.coupling.inherited_angular_momentum(mass, r)
        if spin == 0:
            return np.inf
        return L_inherited / spin


# ============================================================================
# TESTING AND VERIFICATION
# ============================================================================

def test_framework():
    """
    Test the angular momentum coupling framework.
    """
    print("=" * 70)
    print("TESTING ANGULAR MOMENTUM COUPLING FRAMEWORK")
    print("=" * 70)
    
    # Initialize coupling calculator
    coupling = AngularMomentumCoupling(use_scale_dependence=False)
    
    print("\n1. Testing Newtonian Equivalence")
    print("-" * 70)
    
    # Earth-Sun system
    r_earth_sun = c.AU
    matches, ratio = coupling.verify_newtonian_equivalence(
        c.M_SUN, c.M_EARTH, r_earth_sun
    )
    
    print(f"   Earth-Sun system (r = {r_earth_sun:.3e} m):")
    print(f"   U_coupling / U_newton = {ratio:.15f}")
    print(f"   Matches Newtonian: {matches}")
    
    # Earth surface
    r_earth_surface = c.R_EARTH
    test_mass = 1.0  # kg
    matches, ratio = coupling.verify_newtonian_equivalence(
        c.M_EARTH, test_mass, r_earth_surface
    )
    
    print(f"\n   Earth surface (r = {r_earth_surface:.3e} m):")
    print(f"   U_coupling / U_newton = {ratio:.15f}")
    print(f"   Matches Newtonian: {matches}")
    
    # Calculate surface gravity
    g = coupling.acceleration(c.M_EARTH, c.R_EARTH)
    print(f"   Surface acceleration: {g:.6f} m/s² (expected ~9.81)")
    
    print("\n2. Testing Scale-Dependent sigma_eff(r)")
    print("-" * 70)
    
    coupling_scale = AngularMomentumCoupling(use_scale_dependence=True)
    
    test_scales = [1e-15, 1e-10, 1e-9, 1e-5, 1.0, c.AU]
    scale_names = ["Nuclear", "Atomic", "Transition", "Microscopic", 
                   "Macroscopic", "Astronomical"]
    
    for name, r in zip(scale_names, test_scales):
        sigma = coupling_scale.sigma_eff(r)
        # Test with proton mass
        force_const = coupling.coupling_force_from_masses(c.M_PROTON, c.M_PROTON, r)
        force_scale = coupling_scale.coupling_force_from_masses(c.M_PROTON, c.M_PROTON, r)
        ratio = force_scale / force_const if force_const != 0 else 0
        
        print(f"   {name:15s} (r={r:.2e} m):")
        print(f"      sigma_eff = {sigma:.2e} m^2/s")
        print(f"      Force ratio (scale/const) = {ratio:.3e}")
    
    print("\n3. Testing Angular Momentum Dominance")
    print("-" * 70)
    
    total_am = TotalAngularMomentum(coupling)
    
    # Electron spin
    spin_electron = c.H_BAR / 2
    L_inherited_electron = coupling.inherited_angular_momentum(c.M_ELECTRON)
    ratio_electron = total_am.dominance_ratio(c.M_ELECTRON, spin_electron)
    
    print(f"   Electron:")
    print(f"      L_inherited = {L_inherited_electron:.3e} kg·m²/s")
    print(f"      L_spin = {spin_electron:.3e} kg·m²/s")
    print(f"      L_inherited/L_spin = {ratio_electron:.3e}")
    
    # Earth
    # Approximate Earth's spin angular momentum
    L_earth_spin = 0.4 * c.M_EARTH * c.R_EARTH**2 * (2*np.pi/c.DAY)
    L_inherited_earth = coupling.inherited_angular_momentum(c.M_EARTH)
    ratio_earth = L_inherited_earth / L_earth_spin
    
    print(f"\n   Earth:")
    print(f"      L_inherited = {L_inherited_earth:.3e} kg·m²/s")
    print(f"      L_spin (rotation) = {L_earth_spin:.3e} kg·m²/s")
    print(f"      L_inherited/L_spin = {ratio_earth:.3e}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_framework()
