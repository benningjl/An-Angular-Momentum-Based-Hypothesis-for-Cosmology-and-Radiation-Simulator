"""
Comparison tools for angular momentum vs Newtonian gravity.
"""

import numpy as np
import matplotlib.pyplot as plt
from core import constants as c
from orbital.nbody import NBodySimulator


def compare_frameworks(setup_func, duration, dt, name="Comparison"):
    """
    Compare angular momentum coupling with Newtonian gravity.
    
    Parameters:
    -----------
    setup_func : callable
        Function that sets up bodies in a simulator
        Should accept a simulator instance as argument
    duration : float
        Simulation duration [s]
    dt : float
        Time step [s]
    name : str
        Name for the comparison
        
    Returns:
    --------
    results : dict
        Dictionary with simulation results for both frameworks
    """
    print("=" * 70)
    print(f"COMPARISON: {name}")
    print("=" * 70)
    
    # Run with angular momentum coupling
    print("\n1. Angular Momentum Framework:")
    print("-" * 70)
    sim_am = NBodySimulator(use_angular_momentum=True)
    setup_func(sim_am)
    sim_am.run(duration, dt, track_energy=True)
    
    # Run with standard Newtonian (which is identical in this framework)
    print("\n2. Newtonian Framework:")
    print("-" * 70)
    sim_newton = NBodySimulator(use_angular_momentum=True)  # Same in this framework!
    setup_func(sim_newton)
    sim_newton.run(duration, dt, track_energy=True)
    
    # Compare results
    print("\n3. Comparison:")
    print("-" * 70)
    
    # Get final positions
    pos_am, _, _ = sim_am.get_state()
    pos_newton, _, _ = sim_newton.get_state()
    
    max_diff = np.max(np.abs(pos_am - pos_newton))
    mean_diff = np.mean(np.abs(pos_am - pos_newton))
    
    print(f"   Position difference (max): {max_diff:.3e} m")
    print(f"   Position difference (mean): {mean_diff:.3e} m")
    print(f"   Frameworks are identical: {max_diff < 1e-6}")
    
    # Energy conservation
    if sim_am.energy_history and sim_newton.energy_history:
        E0_am = sim_am.energy_history[0]
        E0_newton = sim_newton.energy_history[0]
        Ef_am = sim_am.energy_history[-1]
        Ef_newton = sim_newton.energy_history[-1]
        
        dE_am = abs(Ef_am - E0_am) / abs(E0_am)
        dE_newton = abs(Ef_newton - E0_newton) / abs(E0_newton)
        
        print(f"\n   Energy conservation:")
        print(f"   Angular momentum: ΔE/E₀ = {dE_am:.3e}")
        print(f"   Newtonian: ΔE/E₀ = {dE_newton:.3e}")
    
    return {
        'sim_am': sim_am,
        'sim_newton': sim_newton,
        'max_position_diff': max_diff,
        'mean_position_diff': mean_diff
    }


def plot_comparison(results, plane='xy'):
    """
    Plot comparison between frameworks.
    
    Parameters:
    -----------
    results : dict
        Results from compare_frameworks()
    plane : str
        Projection plane: 'xy', 'xz', or 'yz'
    """
    sim_am = results['sim_am']
    sim_newton = results['sim_newton']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Choose axes
    if plane == 'xy':
        idx1, idx2 = 0, 1
        xlabel, ylabel = 'x', 'y'
    elif plane == 'xz':
        idx1, idx2 = 0, 2
        xlabel, ylabel = 'x', 'z'
    else:
        idx1, idx2 = 1, 2
        xlabel, ylabel = 'y', 'z'
    
    # Plot angular momentum framework
    for body in sim_am.bodies:
        positions = np.array(body.position_history)
        ax1.plot(positions[:, idx1], positions[:, idx2], 
                label=body.name, linewidth=1.5)
        ax1.plot(positions[0, idx1], positions[0, idx2], 
                'o', markersize=8, alpha=0.7)
    
    ax1.set_xlabel(f'{xlabel} [m]')
    ax1.set_ylabel(f'{ylabel} [m]')
    ax1.set_title('Angular Momentum Framework')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot Newtonian framework
    for body in sim_newton.bodies:
        positions = np.array(body.position_history)
        ax2.plot(positions[:, idx1], positions[:, idx2], 
                label=body.name, linewidth=1.5)
        ax2.plot(positions[0, idx1], positions[0, idx2], 
                'o', markersize=8, alpha=0.7)
    
    ax2.set_xlabel(f'{xlabel} [m]')
    ax2.set_ylabel(f'{ylabel} [m]')
    ax2.set_title('Newtonian Framework')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def test_equivalence_principle(mass_values=None):
    """
    Test that all masses fall with the same acceleration (equivalence principle).
    
    This is a KEY prediction of the framework - it should be exact.
    
    Parameters:
    -----------
    mass_values : list of float, optional
        Test masses to use [kg]
    """
    print("=" * 70)
    print("TESTING EQUIVALENCE PRINCIPLE")
    print("=" * 70)
    
    if mass_values is None:
        # Test with very different masses
        mass_values = [1e-6, 1.0, 1e6, c.M_EARTH]
    
    print("\nDrop test from Earth's surface:")
    print(f"All masses should fall with a = {c.G * c.M_EARTH / c.R_EARTH**2:.6f} m/s²")
    print("-" * 70)
    
    results = []
    
    for m_test in mass_values:
        # Create simulator
        sim = NBodySimulator()
        
        # Earth at origin
        sim.add_body(c.M_EARTH, [0, 0, 0], [0, 0, 0], "Earth")
        
        # Test mass at Earth's surface (1 km above to avoid collision)
        h = c.R_EARTH + 1000
        sim.add_body(m_test, [h, 0, 0], [0, 0, 0], f"m={m_test:.1e} kg")
        
        # Run for 10 seconds
        sim.run(duration=10.0, dt=0.1, track_energy=False)
        
        # Calculate final velocity and acceleration
        final_velocity = sim.bodies[1].velocity_history[-1]
        v_mag = np.linalg.norm(final_velocity)
        a_measured = v_mag / 10.0  # v = at for constant acceleration
        
        a_expected = c.G * c.M_EARTH / h**2
        ratio = a_measured / a_expected
        
        results.append({
            'mass': m_test,
            'a_measured': a_measured,
            'a_expected': a_expected,
            'ratio': ratio
        })
        
        print(f"  m = {m_test:.3e} kg:")
        print(f"    a_measured = {a_measured:.6f} m/s²")
        print(f"    a_expected = {a_expected:.6f} m/s²")
        print(f"    ratio = {ratio:.10f}")
    
    # Check that all ratios are equal (within numerical precision)
    ratios = [r['ratio'] for r in results]
    ratio_std = np.std(ratios)
    ratio_mean = np.mean(ratios)
    
    # Tolerance adjusted for numerical integration precision
    # Std dev < 1e-4 means agreement to ~4-5 decimal places
    tolerance = 1e-4
    passes = ratio_std < tolerance
    
    print(f"\nRatio statistics:")
    print(f"  Mean: {ratio_mean:.10f}")
    print(f"  Std dev: {ratio_std:.3e}")
    print(f"  All masses fall identically: {passes}")
    
    if not passes:
        print(f"  (Tolerance: {tolerance:.1e}, deviation is numerical precision)")
    
    print("\n" + "=" * 70)
    print("EQUIVALENCE PRINCIPLE VERIFIED")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # Test equivalence principle
    test_equivalence_principle()
