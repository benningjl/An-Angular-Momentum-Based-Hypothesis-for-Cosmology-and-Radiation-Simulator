# Angular Momentum Framework Simulator

A comprehensive Python-based simulator for testing predictions from the Angular-Momentum-Based Hypothesis for Cosmology and Radiation.

## Overview

This simulator implements the core concepts from the research paper, allowing you to test predictions on a personal computer including:

- **Angular momentum coupling dynamics** (replaces gravity)
- **Scale-dependent specific angular momentum σ₀(r)**
- **Orbital mechanics** with angular momentum coupling
- **Primordial sphere evolution and structure formation**
- **Quantum phenomena** (neutrino oscillations, Bell inequality violations)
- **Black hole predictions and constraints**
- **Photon thermalization statistics**

## Structure

```
simulator/
├── core/              # Core physics engine
│   ├── constants.py   # Physical constants and parameters
│   ├── coupling.py    # Angular momentum coupling implementation
│   └── forces.py      # Force calculations
├── orbital/           # Orbital mechanics
│   ├── nbody.py       # N-body simulator
│   └── comparison.py  # Compare with Newtonian physics
├── cosmology/         # Cosmological simulations
│   ├── primordial.py  # Primordial sphere
│   ├── predictions.py # Cosmological predictions
│   └── galaxies.py    # Galaxy rotation curves
├── quantum/           # Quantum phenomena
│   ├── neutrino.py    # Neutrino oscillations
│   ├── bell.py        # Bell inequality tests
│   └── coherence.py   # Quantum coherence limits
└── examples/          # Example simulations
    ├── solar_system.py
    ├── binary_pulsar.py
    └── quantum_tests.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Graphical Interface (Recommended)

Launch the visual GUI for easy access to all simulations:

**Windows:**
```bash
launch_gui.bat
```

**Linux/macOS:**
```bash
chmod +x launch_gui.sh
./launch_gui.sh
```

Or directly:
```bash
python gui.py
```

The GUI provides:
- Visual menu for all simulations
- Live output display with syntax highlighting
- Easy access to generated plots
- One-click execution of test suites

### Option 2: Interactive Command Line

```bash
python start.py
```

### Option 3: Python API

```python
from simulator.core import constants as c
from simulator.orbital import nbody

# Create a two-body system using angular momentum coupling
sim = nbody.NBodySimulator(use_angular_momentum=True)
sim.add_body(mass=c.M_SUN, position=[0, 0, 0], velocity=[0, 0, 0])
sim.add_body(mass=c.M_EARTH, position=[c.AU, 0, 0], velocity=[0, 29.8e3, 0])
sim.run(duration=c.YEAR, dt=3600)
sim.plot_orbits()
```

## Features

### 1. Angular Momentum Coupling Framework
- Implements U = -G·L₁·L₂/(σ₀²·r) potential
- Scale-dependent σ_eff(r) transition function
- Comparison with Newtonian gravity

### 2. Orbital Mechanics
- N-body simulations with angular momentum coupling
- Solar system accuracy tests
- Binary pulsar orbital decay predictions

### 3. Primordial Sphere & Structure Formation
- Rotating sphere stability analysis
- Particle trajectory simulation from sphere breakup
- Galaxy convergence point prediction

### 4. Quantum Phenomena
- Neutrino oscillation phase evolution
- Bell inequality violation calculation (S = 2.61)
- Quantum coherence N² scaling

### 5. Cosmological Predictions
- Black hole minimum mass (2.4 M_Earth)
- Photon thermalization timescale
- Helium mass fraction from nucleosynthesis

### 6. Galaxy Rotation Curves
- Tully-Fisher relation: M ∝ v³ from angular momentum
- Flat rotation curves without dark matter
- Scale-dependent σ₀,eff(M) for galactic scales
- Universal rotation curve profile

## Testing Predictions

Run the complete test suite:

```bash
python run_all_tests.py
```

Each module includes test functions for specific predictions:

```python
# Test black hole minimum mass prediction
from simulator.cosmology.predictions import test_black_hole_mass
test_black_hole_mass()  # Should yield ~2.4 M_Earth

# Test galaxy rotation curves
from simulator.cosmology.galaxies import test_rotation_curves
test_rotation_curves()  # Validates Tully-Fisher M ~ v³

# Test neutrino oscillations
from simulator.quantum.neutrino import test_oscillation_probability
test_oscillation_probability()  # Compare with standard model

# Test Bell inequality
from simulator.quantum.bell import test_CHSH_parameter
test_CHSH_parameter()  # Should yield S = 2.61
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- (Optional) Numba for acceleration

## License

MIT License - See LICENSE file
