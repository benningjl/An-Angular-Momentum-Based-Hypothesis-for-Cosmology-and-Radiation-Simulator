# Quick Start Guide

## Installation

1. **Install dependencies:**
   ```bash
   cd /workspaces/Theory/simulator
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python -m core.constants
   ```

## Running Simulations

### Interactive Mode (Recommended)

Start the interactive menu:
```bash
cd /workspaces/Theory/simulator
python start.py
```

This provides an easy-to-use menu with all available simulations.

### Run All Tests

Execute the comprehensive test suite:
```bash
python run_all_tests.py
```

This will:
- Test all core physics components
- Run orbital mechanics simulations
- Test quantum phenomena predictions
- Calculate cosmological predictions
- Generate all plots

**Runtime:** ~2-5 minutes

### Individual Simulations

**Solar System:**
```bash
python examples/solar_system.py
```
Simulates inner planets for 10 years. Demonstrates that angular momentum coupling produces identical results to Newtonian gravity.

**Binary Pulsar:**
```bash
python examples/binary_pulsar.py
```
Simulates Hulse-Taylor pulsar system. Tests orbital decay predictions.

**Quantum Tests:**
```bash
python examples/quantum_tests.py
```
Tests neutrino oscillations, Bell inequality violations, and quantum computing limits.

## Module-Specific Tests

**Test angular momentum coupling:**
```bash
python -m core.coupling
```

**Test force calculations:**
```bash
python -m core.forces
```

**Test equivalence principle:**
```bash
python -m orbital.comparison
```

**Cosmological predictions:**
```bash
python -m cosmology.predictions
```

**Primordial sphere:**
```bash
python -m cosmology.primordial
```

**Neutrino oscillations:**
```bash
python -m quantum.neutrino
```
*Note: Derives oscillations from angular momentum phase evolution (alternative to standard quantum mixing), producing identical observable predictions.*

**Bell inequality:**
```bash
python -m quantum.bell
```

**Quantum coherence:**
```bash
python -m quantum.coherence
```

## Understanding Output

### Plots

All plots are saved to `/workspaces/Theory/simulator/` with descriptive names:
- `earth_sun_orbit.png` - Earth's orbit around Sun
- `solar_system.png` - Inner solar system
- `neutrino_atmospheric.png` - Atmospheric neutrino oscillations
- `bell_correlation.png` - Bell test correlation functions
- `quantum_coherence_scaling.png` - Qubit coherence scaling
- And more...

### Console Output

Each test prints:
- **Parameters:** Physical constants and setup
- **Predictions:** Framework-specific predictions
- **Comparisons:** How predictions match observations
- **Results:** Numerical results and error estimates

Look for:
- Confirmed predictions
- Inconsistencies
- Values with units and error bars

## Key Predictions to Test

1. **Black Hole Minimum Mass:** 2.4 M_Earth
2. **Helium Abundance:** Y_p = 0.244 (with neutron decay)
3. **Quark Confinement:** 1 GeV/fm (from QCD dynamics)
4. **Bell Inequality:** S = 2.61
5. **Quantum Computing:** ~30-100 qubit limit
6. **Neutrino Mass Ratio:** ~33:1

## Example Python Usage

```python
from simulator.core import constants as c
from simulator.orbital.nbody import NBodySimulator

# Create simulator
sim = NBodySimulator(use_angular_momentum=True)

# Add Sun
sim.add_body(c.M_SUN, [0, 0, 0], [0, 0, 0], "Sun")

# Add Earth
v_orbital = (c.G * c.M_SUN / c.AU) ** 0.5
sim.add_body(c.M_EARTH, [c.AU, 0, 0], [0, v_orbital, 0], "Earth")

# Run simulation
sim.run(duration=c.YEAR, dt=3600)

# Plot results
sim.plot_orbits()
```

## Troubleshooting

**Import errors:**
```bash
pip install -r requirements.txt
```

**Module not found:**
Make sure you're in the `/workspaces/Theory/simulator` directory.

**Plots not showing:**
Plots are saved as PNG files in the simulator directory. Use a file browser or image viewer to open them.

**Memory errors:**
Reduce simulation duration or increase time step (`dt`).

## Performance Tips

- Use `method='leapfrog'` for better energy conservation
- Increase `dt` for faster but less accurate simulations
- Reduce `n_particles` in structure formation for speed
- Use `track_energy=False` to skip energy tracking overhead

## Next Steps

1. Run `python start.py` and choose option 1 (run all tests)
2. Examine the generated plots
3. Review the console output for predictions vs observations
4. Try modifying parameters in the examples
5. Read the paper to understand the theoretical framework

## Support

For questions about the simulator:
- Check the README.md in the simulator directory
- Review the module docstrings
- Examine example scripts for usage patterns

For questions about the theory:
- Read the full research paper
- Focus on the predictions summary (Section 22)
- Review the mathematical derivations
