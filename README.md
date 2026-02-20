[![CI](https://github.com/lfierce2/LD-Chem/actions/workflows/ci.yml/badge.svg)](https://github.com/lfierce2/LD-Chem/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lfierce2/LD-Chem/branch/main/graph/badge.svg)](https://codecov.io/gh/lfierce2/LD-Chem)

# Lagrangian Droplets with Chemistry Model (LD-Chem)

> A Python toolkit for simulating aqueous chemistry and cloud-aerosol processes in individual particles.

`LD-Chem` is a lightweight Python library that provides a framework for simulating activation of particles into cloud droplets, aqueous chemistry, gas chemistry, and gas-particle mass transfer. The framework can be run in two modes: adiabatic parcel and LES. Adiabatic parcel simulations are driven by a user-defined constant updraft velocity. LES simulations can be run at any scale (despite the name "large eddy simulation"). They are driven by time series of position, saturation ratio, temperature, pressure, and trace gas concentrations (if available). The framework enables reproducible process-level investigations if aerosol-cloud interactions. This model is an extension of the Lagrangian Droplets model, which can be found at https://github.com/lfierce2/LagrangianDroplets/.

---

## Features

- **Flexible simulation modes**: Run adiabatic parcel or LES (Large Eddy Simulation) simulations that can be run at any spatial scale
- **Particle activation**: Simulate the activation of aerosol particles into cloud droplets
- **Gas-particle mass transfer**: Resolve equilibrium partitioning and kinetic mass transfer between gas and particle phases
- **Aqueous-phase chemistry**: Simulate in-cloud chemistry with configurable reaction mechanisms
- **Gas-phase chemistry**: Model tropospheric gas-phase reactions during transport with configurable reaction mechanisms
- **Customizable aerosol populations**: Leverage `part2pop` for flexible aerosol composition and size distribution definitions
- **Efficient computation**: Numba-accelerated differential equation solvers for fast simulations
- **Process-level analysis**: Enables reproducible investigations of aerosol-cloud interactions at the process scale

## Installation

```bash
git clone git@github.com:lfierce2/LD-Chem.git
cd LD-Chem
pip install -e .
```
---

## Quick start

### Build a simple population using part2pop and run a parcel simulation

```python
from part2pop.population.builder import build_population

pop_cfg = {
    "type": "binned_lognormals",
    "N": [1e9],
    "GMD": [150e-9],
    "GSD": [1.6],
    "aero_spec_names": [["SO4","OC"]],
    "aero_spec_fracs": [[0.2, 0.8]],
    "N_bins": 20,
    "N_sigmas": 5
    }
pop = build_population(config)
aero_spec_names = np.array([species.name for species in pop.species])
aero_spec_masses = np.array(pop.spec_masses)
num_concs = np.array(pop.num_concs)
pHs = np.random.normal(loc=4.5, scale=0.5, size=num_concs.shape[0])

simulate_parcel(
    aero_spec_names, aero_spec_masses, num_concs, pHs,
    z_start=0.0, z_end=1000.0, dt=1.0, updraft_velocity=0.5,
    S0=0.85, P0=101325.0, T0=298.0, radius_scale='log',
    gas_names=None, gas_concs=None, condensation = True, 
    cocondensation = False, aq_chemistry = None, 
    gas_chemistry = False, output_filename='trajectory.pkl')

```

### Analyze and plot results

```python
import matplotlib.pyplot as plt
data=pickle.load(open('trajectory.pkl','rb'))

plt.plot(data['S'], data['z'])
plt.xlim(1.0,)
plt.ylabel('altitude [m]')
plt.xlabel('saturation ratio')
plt.show()

spec_idx = np.where(data['particle species']=='Dwet')[0][0]
plt.plot(data['particles'][:,:,spec_idx], data['z'], '-r')
plt.xscale('log')
plt.xlabel('wet diameter [m]')
plt.ylabel('altitude [m]')
plt.show()

```

More examples are available under `examples/`.

---

## Repository structure

```
src/ld_chem/
    constants.py             # Physical constants
    gases.py                 # Trace gas representation
    particles.py             # Definition of particle species
    reactions.py             # Definition of gas and aqueous phase reactions
    run.py                   # Sets up and runs simulations
    scenario.py              # Simulation setup
    systems.py               # Call solvers and update state
    utilities.py             # Ensures mass balance during solving
    write_files.py           # Writes backup files and outputs data
    mechanisms/              # Defines gas and aqueous phase reactions
    processes/               # Stores differential equation definitions
    species_data/            # Aerosol and gas species definitions

```

## Definitions of Chemical Mechanisms

LD-Chem was designed with flexible gas- and aqueous-phase chemistry definitions. Both aqueous and gas-phase mechanisms are defined in simple data format files, allowing easy customization and extension.

### Aqueous-Phase Reactions

The default aqueous-phase reactions are defined in `src/ld_chem/mechanisms/aq_reactions.dat` with the following columns:

| Column | Description | Unit |
|--------|-------------|------|
| `reactants` | Comma-separated list of reactant species | - |
| `products` | Comma-separated list of product species | - |
| `rate` | Pre-exponential rate constant | mol/m³^(1-n)/s |
| `-dH/R` | Negative enthalpy of reaction divided by gas constant | K |
| `group` | Reaction classification/mechanism name | - |

The temperature-dependent reaction rate is calculated as:

$$k(T) = k_0 \cdot \exp\left(\frac{\Delta H}{R}\left(\frac{1}{T} - \frac{1}{298}\right)\right)$$

Default aqueous chemistry groups include:
- `sulfate` - SO₂ and sulfuric acid chemistry
- `ammonium` - Ammonia and ammonium equilibria
- `nitrate` - Nitrogen oxide chemistry
- `IEPOX` - IEPOX secondary organic aerosol reactions

### Gas-Phase Reactions

Gas-phase reactions are defined in `src/ld_chem/mechanisms/gas_reactions.dat` with the following columns:

| Column | Description | Unit |
|--------|-------------|------|
| `reactants` | Comma-separated list of reactant species | - |
| `products` | Comma-separated list of product species | - |
| `rate` | Pre-exponential rate constant | (molec/cm³)^(1-n)/s |
| `high_P_limit` | High pressure limit (Troe reactions only) | (molec/cm³)^(1-n)/s |
| `T_dependence` | Temperature dependence parameter | K or dimensionless |
| `form` | Functional form of temperature dependence | - |

Four temperature-dependent rate forms are supported:

- **`exp`**: Arrhenius-like temperature dependence 

$$k(T) = k_0 \cdot \exp\left(\frac{\Delta E}{T}\right)$$

- **`power`**: Power-law temperature dependence
  $$k(T) = k_0 \cdot \left(\frac{T}{300}\right)^n$$

- **`troe`**: Troe Three-Parameter Fall-off Region (pressure-dependent bimolecular reactions)
  Accounts for N₂/H₂O collision broadening and high-pressure limits

- **`HO2_water_enhancement`**: Water-enhanced HO₂ self-reaction specific to atmospheric conditions

  $$k = (k_1 + k_2) \cdot \left(1 + 8.4 \times 10^{-4} [H_2O] \exp(2200/T)\right)$$

### Custom Mechanisms

There are two ways to define custom aqueous- or gas-phase chemical mechanisms:

**Option 1: Edit the default mechanism files**

Modify the default mechanism files in `src/ld_chem/mechanisms/`:
1. Edit `aq_reactions.dat` and/or `gas_reactions.dat` directly
2. Add or remove reactions and assign them to group names
3. Include the group names in a list assigned to the `aq_chemistry` and/or `gas_chemistry` arguments when running a simulation

**Option 2: Create custom mechanism files and specify the path**

Create your own mechanism files and point the model to them:
1. Create new `aq_reactions.dat` and/or `gas_reactions.dat` files with your custom reactions
2. Pass the directory path to the `mechanism_data_path` argument when running a simulation
3. This allows you to maintain multiple mechanism configurations without modifying the source code. It should be noted that `aq_reactions.dat` and `gas_reactions.dat` are read using the same `mechanism_data_path`.

Example using Option 1:
```python
from ld_chem import simulate_parcel

# define aerosol properties (single organic particle)
aero_spec_names=[['OC']]
aero_spec_masses=[[1.0]]
num_concs=[1e6]
pHs=[3.0]

# Run a parcel simulation with the default sulfate mechanisms
simulate_parcel(
    aero_spec_names, aero_spec_masses, num_concs, pHs,
    aq_chemistry=['sulfate'],
    gas_chemistry=True,
    output_filename='trajectory.pkl'
)
```

Example using Option 2:
```python
from ld_chem import simulate_parcel

# define aerosol properties (single organic particle)
aero_spec_names=[['OC']]
aero_spec_masses=[[1.0]]
num_concs=[1e6]
pHs=[3.0]

# Point to custom mechanism directory
simulate_parcel(
    aero_spec_names, aero_spec_masses, num_concs, pHs,
    mechanism_data_path='/path/to/custom/mechanisms/',
    aq_chemistry=['custom_group_1'],
    gas_chemistry=True,
    output_filename='trajectory.pkl'
)
``` 

---

## License

See the `LICENSE` file in this repository.
