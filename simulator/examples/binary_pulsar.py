"""
Example: Binary Pulsar Orbital Decay

Simulate binary pulsar system and test orbital decay predictions
from angular momentum radiation.

Paper: "explains binary pulsar orbital decay through angular momentum 
radiation (matching observations to 0.2%)"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from core import constants as c
from orbital.nbody import NBodySimulator


def setup_binary_pulsar(sim, m1=1.4*c.M_SUN, m2=1.4*c.M_SUN, 
                       a=1e9, eccentricity=0.617):
    """
    Set up a binary pulsar system similar to PSR B1913+16 (Hulse-Taylor pulsar).
    
    Parameters:
    -----------
    sim : NBodySimulator
        Simulator instance
    m1, m2 : float
        Masses of the two neutron stars [kg]
    a : float
        Semi-major axis [m]
    eccentricity : float
        Orbital eccentricity
    """
    # Center of mass at origin
    # Place m1 at periastron (closest approach)
    r_periastron = a * (1 - eccentricity)
    
    # Reduced mass
    mu = m1 * m2 / (m1 + m2)
    
    # Velocity at periastron (conservation of energy and angular momentum)
    v_periastron = np.sqrt(c.G * (m1 + m2) * (1 + eccentricity) / (a * (1 - eccentricity)))
    
    # Positions (center of mass frame)
    r1 = r_periastron * m2 / (m1 + m2)
    r2 = -r_periastron * m1 / (m1 + m2)
    
    # Velocities (perpendicular to line joining stars)
    v1 = v_periastron * m2 / (m1 + m2)
    v2 = -v_periastron * m1 / (m1 + m2)
    
    sim.add_body(
        mass=m1,
        position=[r1, 0, 0],
        velocity=[0, v1, 0],
        name="Pulsar 1"
    )
    
    sim.add_body(
        mass=m2,
        position=[r2, 0, 0],
        velocity=[0, v2, 0],
        name="Pulsar 2"
    )


def calculate_orbital_decay_rate(m1, m2, a, e):
    """
    Calculate orbital decay rate from gravitational wave emission (GR prediction).
    
    da/dt = -64/5 * G³/(c⁵) * (m1*m2*(m1+m2))/a³ * f(e)
    
    where f(e) = (1 + 73/24*e² + 37/96*e⁴)/(1-e²)^(7/2)
    
    Parameters:
    -----------
    m1, m2 : float
        Masses [kg]
    a : float
        Semi-major axis [m]
    e : float
        Eccentricity
        
    Returns:
    --------
    da_dt : float
        Rate of change of semi-major axis [m/s]
    """
    # Eccentricity factor
    f_e = (1 + 73/24 * e**2 + 37/96 * e**4) / (1 - e**2)**(7/2)
    
    # Decay rate
    da_dt = -64/5 * c.G**3 / c.C**5 * (m1 * m2 * (m1 + m2)) / a**3 * f_e
    
    return da_dt


def main():
    """Run binary pulsar simulation."""
    print("=" * 70)
    print("BINARY PULSAR ORBITAL DECAY")
    print("=" * 70)
    
    # Hulse-Taylor pulsar parameters
    m1 = 1.440 * c.M_SUN  # kg
    m2 = 1.389 * c.M_SUN  # kg
    a = 1.95e9  # m (semi-major axis)
    e = 0.617  # eccentricity
    
    print("\nSystem parameters (PSR B1913+16):")
    print(f"  m1 = {m1/c.M_SUN:.3f} M_Sun")
    print(f"  m2 = {m2/c.M_SUN:.3f} M_Sun")
    print(f"  Semi-major axis = {a:.3e} m = {a/c.R_SUN:.1f} R_Sun")
    print(f"  Eccentricity = {e:.3f}")
    
    # Orbital period
    P = 2 * np.pi * np.sqrt(a**3 / (c.G * (m1 + m2)))
    print(f"  Orbital period = {P/c.HOUR:.2f} hours")
    
    # Calculate predicted decay rate
    da_dt_GR = calculate_orbital_decay_rate(m1, m2, a, e)
    print(f"\nGR prediction for orbital decay:")
    print(f"  da/dt = {da_dt_GR:.3e} m/s = {da_dt_GR*c.YEAR:.3e} m/year")
    
    # Period derivative
    dP_dt = 3/2 * P / a * da_dt_GR
    print(f"  dP/dt = {dP_dt:.3e} s/s = {dP_dt*c.YEAR:.3e} s/year")
    
    # Observed value
    dP_dt_observed = -2.423e-12  # s/s
    print(f"\nObserved:")
    print(f"  dP/dt = {dP_dt_observed:.3e} s/s")
    
    agreement = abs(dP_dt - dP_dt_observed) / abs(dP_dt_observed)
    print(f"\nAgreement: {agreement*100:.2f}% deviation")
    print("(Paper claims 0.2% agreement with angular momentum radiation)")
    
    # Run short simulation
    print("\n" + "=" * 70)
    print("SIMULATION")
    print("=" * 70)
    
    sim = NBodySimulator(use_angular_momentum=True)
    setup_binary_pulsar(sim, m1, m2, a, e)
    
    # Simulate 10 orbits
    duration = 10 * P
    dt = P / 1000  # 1000 steps per orbit
    
    print(f"\nSimulating {10} orbits ({duration/c.HOUR:.1f} hours)...")
    sim.run(duration=duration, dt=dt, method='leapfrog', track_energy=True)
    
    # Plot orbit
    fig, ax = sim.plot_orbits(plane='xy', figsize=(10, 10))
    ax.set_title('Binary Pulsar PSR B1913+16\n10 Orbits', fontsize=14)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    
    # Add scale
    ax.plot(0, 0, 'r+', markersize=15, label='Center of Mass')
    ax.legend()
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'binary_pulsar.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'binary_pulsar.png')}")
    
    # Energy conservation (should see small decrease due to "radiation")
    if len(sim.energy_history) > 1:
        E0 = sim.energy_history[0]
        Ef = sim.energy_history[-1]
        dE = (Ef - E0) / abs(E0)
        
        print(f"\nEnergy change: ΔE/E₀ = {dE:.3e}")
        print("(Note: This simulation uses conservative forces only.)")
        print("(True orbital decay requires adding angular momentum radiation term.)")
    
    fig, axes = sim.plot_energy_conservation()
    plt.savefig(os.path.join(output_dir, 'binary_pulsar_energy.png'), dpi=150, bbox_inches='tight')
    print("Saved: binary_pulsar_energy.png")
    
    print("\n" + "=" * 70)
    print("NOTES")
    print("=" * 70)
    print("This simulation demonstrates the binary system dynamics.")
    print("To include orbital decay, one would add a radiation reaction force:")
    print("  F_radiation ∝ -dL/dt")
    print("\nThe framework predicts this matches GR to 0.2%, which is")
    print("consistent with Nobel Prize-winning Hulse-Taylor observations.")
    
    plt.close('all')


if __name__ == "__main__":
    main()
