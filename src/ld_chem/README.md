# Lagrangian Droplets with Chemistry package (ld_chem)

The `ld_chem` package is the core module of the Lagrangian Droplets with Chemistry Model, providing a comprehensive framework for simulating aerosol-cloud microphysics and aqueous-phase chemistry.

## Purpose

This module enables detailed process-level simulations of:
- Aerosol to cloud droplet activation
- Condensation and co-condensation of water and semi-volatile organic compounds
- Aqueous-phase and gas-phase chemistry
- Gas-particle mass transfer and equilibrium
- Particle microphysical evolution in adiabatic parcel or large eddy simulation (LES) conditions

## Module Contents

### Core Modules

- **`scenario.py`** - Scenario setup and initialization functions for creating simulation configurations
- **`run.py`** - Main driver functions for executing parcel and LES simulations
- **`systems.py`** - State management and solver integration for updating particle and air properties
- **`particles.py`** - Particle population definition and management
- **`reactions.py`** - Chemical reaction definitions and rate calculations
- **`gases.py`** - Trace gas definition and properties
- **`constants.py`** - Physical and chemical constants
- **`utilities.py`** - Helper functions for mass balance and calculations
- **`write_files.py`** - Output file writing and checkpoint management

### Subdirectories

- **`mechanisms/`** - Mechanism definition files for aqueous and gas-phase chemistry
  - `aq_reactions.dat` - Default aqueous-phase reaction definitions
  - `gas_reactions.dat` - Default gas-phase reaction definitions
  
- **`processes/`** - Differential equation definitions for physical and chemical processes
  - `air_thermo.py` - Thermodynamic calculations for air parcel
  - `gas_chemistry.py` - Gas-phase chemistry rate equations
  - `aqueous_chemistry.py` - Aqueous-phase chemistry rate equations
  - `cocondensation.py` - Gas-aqueous mass transfer rate equations
  - `water_uptake.py` - Water vaopr condensation rate equations

- **`species_data/`** - Configuration data for aerosol and gas species properties

## Exports

The following are the primary public interfaces of the `ld_chem` module:

### Scenario Creation
- **`create_les_scenario()`** - Create a simulation scenario for trajectory-driven mode
- **`create_parcel_scenario()`** - Create a simulation scenario for parcel mode
- **`make_AqReactions()`** - Load and configure aqueous-phase chemistry mechanisms
- **`make_GasReactions()`** - Load and configure gas-phase chemistry mechanisms

### Core Classes
- **`ParcelState`** - State representation of an particle-containing air parcel and its thermodynamic properties
- **`Processes`** - Configuration object specifying which physical and chemical processes to simulate
- **`ParticlePopulation`** - Representation of the aerosol population

## Usage Example

```python
from ld_chem import create_les_scenario, make_AqReactions, make_GasReactions

# Create chemistry mechanisms
aq_chem = make_AqReactions(chemistry=['sulfate'])
gas_chem = make_GasReactions()

# Create a simulation scenario
scenario = create_les_scenario(
    initial_aerosol_pop=pop,
    aq_chemistry=aq_chem,
    gas_chemistry=gas_chem
)
```

## Documentation

For detailed usage information, custom mechanism definitions, and API documentation, see the main [README.md](../../README.md) in the repository root.
