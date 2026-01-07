"""
Physical constants and framework-specific parameters.
"""

import numpy as np

# ============================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS (SI units)
# ============================================================================

# Universal constants
C = 299792458.0  # Speed of light [m/s]
H_BAR = 1.054571817e-34  # Reduced Planck constant [J·s]
K_B = 1.380649e-23  # Boltzmann constant [J/K]
G = 6.67430e-11  # Gravitational constant [m³/(kg·s²)]

# Masses
M_ELECTRON = 9.1093837015e-31  # kg
M_PROTON = 1.67262192369e-27  # kg
M_NEUTRON = 1.67492749804e-27  # kg
M_SUN = 1.98847e30  # kg
M_EARTH = 5.972168e24  # kg
M_JUPITER = 1.8982e27  # kg

# Distances
AU = 1.495978707e11  # Astronomical unit [m]
PARSEC = 3.0857e16  # Parsec [m]
LIGHT_YEAR = 9.4607e15  # Light year [m]
R_EARTH = 6.371e6  # Earth radius [m]
R_SUN = 6.96e8  # Sun radius [m]

# Time
YEAR = 365.25 * 24 * 3600  # Seconds in a year
DAY = 24 * 3600  # Seconds in a day
HOUR = 3600  # Seconds in an hour

# ============================================================================
# ANGULAR MOMENTUM FRAMEWORK PARAMETERS
# ============================================================================

# Universal specific angular momentum [m²/s]
# These values are from the paper's scale hierarchy
SIGMA_0_MACRO = 1.0e6  # Macroscopic/astrophysical scale
SIGMA_0_NUCLEAR = 1.0e-7  # Nuclear/quantum scale
SIGMA_0_QUARK = 5.4e-27  # Quark confinement scale
SIGMA_0_PLANETARY = 8.2e5  # Earth-specific value

# Transition scale between quantum and classical regimes [m]
R_TRANSITION = 1.0e-9  # ~1 nanometer (atomic scale)

# Power-law exponent for σ_eff transition
N_TRANSITION = 2.0  # Controls transition sharpness

# Primordial sphere parameters (example values for simulation)
M_UNIVERSE = 1e53  # Total mass in primordial sphere [kg]
R_PRIMORDIAL = 1e26  # Initial radius of primordial sphere [m]

# ============================================================================
# DERIVED FRAMEWORK QUANTITIES
# ============================================================================

def sigma_effective(r, sigma_macro=SIGMA_0_MACRO, sigma_nuclear=SIGMA_0_NUCLEAR,
                   r_c=R_TRANSITION, n=N_TRANSITION):
    """
    Calculate scale-dependent effective specific angular momentum.
    
    σ_eff(r) = σ_nuclear + (σ_macro - σ_nuclear) · h(r/r_c)
    where h(x) = x^n / (1 + x^n)
    
    Parameters:
    -----------
    r : float or array
        Separation distance [m]
    sigma_macro : float
        Macroscopic σ₀ value [m²/s]
    sigma_nuclear : float
        Nuclear/quantum σ₀ value [m²/s]
    r_c : float
        Transition scale [m]
    n : float
        Transition sharpness parameter
        
    Returns:
    --------
    sigma_eff : float or array
        Effective specific angular momentum [m²/s]
    """
    x = r / r_c
    x_n = np.power(x, n)
    h = x_n / (1.0 + x_n)
    return sigma_nuclear + (sigma_macro - sigma_nuclear) * h


def coupling_constant(r, sigma_macro=SIGMA_0_MACRO, sigma_nuclear=SIGMA_0_NUCLEAR,
                     r_c=R_TRANSITION, n=N_TRANSITION):
    """
    Calculate the effective coupling constant κ = G / σ_eff(r).
    
    Parameters:
    -----------
    r : float or array
        Separation distance [m]
        
    Returns:
    --------
    kappa : float or array
        Coupling constant [m/(kg·s)]
    """
    sigma_eff = sigma_effective(r, sigma_macro, sigma_nuclear, r_c, n)
    return G / sigma_eff


def quantum_classical_crossover_mass(sigma_0=SIGMA_0_NUCLEAR):
    """
    Calculate the mass scale where quantum effects become important.
    
    m_cross ~ ℏ / σ₀
    
    Parameters:
    -----------
    sigma_0 : float
        Specific angular momentum scale [m²/s]
        
    Returns:
    --------
    m_cross : float
        Crossover mass [kg]
    """
    return H_BAR / sigma_0


def minimum_black_hole_radius(sigma_0=SIGMA_0_MACRO):
    """
    Calculate the minimum radius r_min = σ₀/c for the framework.
    
    Parameters:
    -----------
    sigma_0 : float
        Specific angular momentum [m²/s]
        
    Returns:
    --------
    r_min : float
        Minimum radius [m]
    """
    return sigma_0 / C


def minimum_black_hole_mass(sigma_0=SIGMA_0_MACRO):
    """
    Calculate the minimum black hole mass from the framework.
    
    From r_min = σ₀/c and r_min < r_s = 2GM/c²:
    M_min ~ σ₀·c/(2G)
    
    Paper predicts M_min ~ 2.4 M_Earth
    
    Parameters:
    -----------
    sigma_0 : float
        Specific angular momentum [m²/s]
        
    Returns:
    --------
    M_min : float
        Minimum black hole mass [kg]
    """
    r_min = minimum_black_hole_radius(sigma_0)
    # From Schwarzschild radius: r_s = 2GM/c²
    # Setting r_min = r_s: M_min = r_min·c²/(2G)
    return r_min * C**2 / (2 * G)


# ============================================================================
# PRINT USEFUL DERIVED QUANTITIES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ANGULAR MOMENTUM FRAMEWORK - KEY PREDICTIONS")
    print("=" * 70)
    
    print("\n1. Scale Hierarchy:")
    print(f"   σ₀,macro      = {SIGMA_0_MACRO:.2e} m²/s")
    print(f"   σ₀,planetary  = {SIGMA_0_PLANETARY:.2e} m²/s")
    print(f"   σ₀,nuclear    = {SIGMA_0_NUCLEAR:.2e} m²/s")
    print(f"   σ₀,quark      = {SIGMA_0_QUARK:.2e} m²/s")
    
    print("\n2. Transition Scale:")
    print(f"   r_c = {R_TRANSITION:.2e} m ({R_TRANSITION*1e9:.2f} nm)")
    
    print("\n3. Quantum-Classical Crossover:")
    m_cross_nuclear = quantum_classical_crossover_mass(SIGMA_0_NUCLEAR)
    m_cross_macro = quantum_classical_crossover_mass(SIGMA_0_MACRO)
    print(f"   m_cross (nuclear) = {m_cross_nuclear:.2e} kg")
    print(f"   m_cross (macro)   = {m_cross_macro:.2e} kg")
    
    print("\n4. Black Hole Predictions:")
    r_min = minimum_black_hole_radius(SIGMA_0_MACRO)
    M_min = minimum_black_hole_mass(SIGMA_0_MACRO)
    M_min_earth = M_min / M_EARTH
    print(f"   r_min = {r_min:.2e} m ({r_min/1000:.2f} km)")
    print(f"   M_min = {M_min:.2e} kg = {M_min_earth:.2f} M_Earth")
    print(f"   Paper prediction: 2.4 M_Earth (range 0.8-8 M_Earth)")
    
    print("\n5. Sample σ_eff values:")
    test_scales = [1e-15, 1e-12, 1e-10, 1e-9, 1e-7, 1e-3, 1.0, AU]
    scale_names = ["Nuclear", "Subatomic", "Atomic", "Molecular", 
                   "Microscopic", "Macromolecular", "Macroscopic", "Astronomical"]
    for name, r in zip(scale_names, test_scales):
        sig_eff = sigma_effective(r)
        print(f"   {name:15s} (r={r:.2e} m): σ_eff = {sig_eff:.2e} m²/s")
    
    print("\n" + "=" * 70)
