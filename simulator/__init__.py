"""
Angular Momentum Framework Simulator

A comprehensive simulator for testing predictions from the paper:
'An Angular-Momentum-Based Hypothesis for Cosmology and Radiation'

Modules:
--------
core : Core physics (constants, coupling, forces)
orbital : N-body simulations and orbital mechanics
quantum : Quantum phenomena (neutrinos, Bell tests, coherence)
cosmology : Primordial sphere and cosmological predictions

Quick Start:
------------
    python start.py

Or run specific tests:
    python run_all_tests.py
    python -m core.constants
    python examples/solar_system.py

See README.md for detailed documentation.
"""

__version__ = '1.0.0'
__author__ = 'Angular Momentum Framework Simulator'

# Import main modules
from . import core
from . import orbital
from . import quantum
from . import cosmology

__all__ = ['core', 'orbital', 'quantum', 'cosmology']
