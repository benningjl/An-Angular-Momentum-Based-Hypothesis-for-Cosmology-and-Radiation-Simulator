#!/usr/bin/env python3
"""
Quick start script for the Angular Momentum Framework Simulator.

This script provides an interactive menu to run different simulations.
"""

import sys
import os

# Add simulator to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_menu():
    """Print the main menu."""
    print("\n" + "=" * 70)
    print("ANGULAR MOMENTUM FRAMEWORK SIMULATOR".center(70))
    print("=" * 70)
    print("\nWhat would you like to simulate?\n")
    print("  1. Run ALL tests and simulations (recommended first time)")
    print("  2. Solar System (inner planets)")
    print("  3. Binary Pulsar (orbital decay)")
    print("  4. Quantum Phenomena (neutrinos, Bell, coherence)")
    print("  5. Test Core Physics (coupling, forces, equivalence)")
    print("  6. Cosmological Predictions (black holes, nucleosynthesis, etc.)")
    print("  7. Primordial Sphere and Structure Formation")
    print("\n  0. Exit")
    print("\n" + "=" * 70)


def main():
    """Main interactive loop."""
    
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter your choice (0-7): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            break
        
        if choice == '0':
            print("\nExiting...")
            break
        
        elif choice == '1':
            print("\nRunning ALL tests...")
            import run_all_tests
            run_all_tests.main()
        
        elif choice == '2':
            print("\nRunning Solar System simulation...")
            from examples import solar_system
            solar_system.main()
        
        elif choice == '3':
            print("\nRunning Binary Pulsar simulation...")
            from examples import binary_pulsar
            binary_pulsar.main()
        
        elif choice == '4':
            print("\nRunning Quantum Tests...")
            from examples import quantum_tests
            quantum_tests.main()
        
        elif choice == '5':
            print("\nTesting Core Physics...")
            from core import coupling, forces
            from orbital import comparison
            coupling.test_framework()
            forces.test_forces()
            comparison.test_equivalence_principle()
        
        elif choice == '6':
            print("\nCalculating Cosmological Predictions...")
            from cosmology import predictions
            predictions.black_hole_minimum_mass()
            predictions.helium_mass_fraction()
            predictions.quark_confinement_string_tension()
            predictions.photon_thermalization_timescale()
            predictions.neutrino_mass_hierarchy()
            predictions.summary_of_predictions()
        
        elif choice == '7':
            print("\nSimulating Primordial Sphere...")
            from cosmology import primordial
            sphere = primordial.calculate_primordial_parameters()
            primordial.simulate_structure_formation()
        
        else:
            print("\nInvalid choice. Please enter a number 0-7.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ANGULAR MOMENTUM FRAMEWORK SIMULATOR")
    print("Testing predictions from:")
    print("'An Angular-Momentum-Based Hypothesis for Cosmology and Radiation'")
    print("=" * 70)
    
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "=" * 70)
        print("Thank you for using the simulator!")
        print("=" * 70)
