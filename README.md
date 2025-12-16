[![CI](https://github.com/lfierce2/multipart_archived/actions/workflows/ci.yml/badge.svg)](https://github.com/lfierce2/multipart_archived/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lfierce2/multipart_archived/branch/payton-dev/graph/badge.svg )](https://codecov.io/gh/lfierce2/multipart_archived)

# multipart

> A Python toolkit for simulating aqueous chemistry and cloud-aerosol processes in individual particles.

`multipart` is a lightweight Python library that provides a framework for simulating activation of particles into cloud droplets, aqueous chemistry, gas chemistry, and gas-particle mass transfer. The framework can be run in two modes: adiabatic parcel and LES. Adiabatic parcel simulations are driven by a user-defined constant updraft velocity. LES simulations can be run at any scale (despite the name "large eddy simulation"). They are driven by time series of position, saturation ratio, temperature, pressure, and trace gas concentrations (if available). The framework enables reproducible process-level investigations if aerosol-cloud interactions.

---

## Installation

```bash
git clone git@github.com:lfierce2/multipart_archived.git
cd part2pop
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
src/multipart/
    constants.py             # Physical constants
    gases.py                 # Trace gas representation
    particles.py             # Definition of particle species
    reactions.py             # Definition of gas and aqueous phase reactions
    run.py                   # Sets up and runs simulations
    scenario.py              # Simulation setup
    systems.py               # Call solvers and update state
    utilities.py             # Ensures mass balance during solving
    write_files.py           # Writes backup files and outputs data
    mechanisms/               # Defines gas and aqueous phase reactions
    processes/               # Stores differential equation definitions
    species_data/            # Aerosol and gas species definitions

```

---

## License

See the `LICENSE` file in this repository.
