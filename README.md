[![CI](https://github.com/pnnl/LD-Chem/actions/workflows/ci.yml/badge.svg)](https://github.com/pnnl/LD-Chem/actions/workflows/ci.yml)

# Lagrangian Droplets with Chemistry Model (LD-Chem)

LD-Chem is a Python package for simulating aqueous chemistry and cloud-aerosol
processes in individual particles. It supports adiabatic parcel simulations and
trajectory-driven simulations using time series of position, saturation ratio,
temperature, pressure, and trace gas concentrations when available.

LD-Chem extends the Lagrangian Droplets model lineage.

## Features

- Adiabatic parcel and trajectory-driven simulation modes
- Particle activation and condensational growth
- Gas-particle mass transfer
- Configurable aqueous-phase and gas-phase chemistry
- Aerosol population definitions through `part2pop`
- Numba-accelerated process calculations

## Installation
Clone the repo, create and activate the conda environment, and install LD-Chem locally:

```bash
git clone https://github.com/pnnl/LD-Chem.git
cd LD-Chem
conda env create -f environment.yml 
conda activate ld-chem
pip install -e .
```

To verify the installation, run the test suite:
```bash
python -m pytest
```

## Quick Start

This is a lightweight example of an adiabatic parcel. It is intended to demonstrate
the package interface with a simple aerosol population under idealized conditions. 

```python
import pickle

import numpy as np
from part2pop.population import build_population

from ld_chem import simulate_parcel

pop_cfg = {
    "type": "binned_lognormals",
    "N": [1e9],
    "GMD": [150e-9],
    "GSD": [1.6],
    "aero_spec_names": [["SO4", "OC"]],
    "aero_spec_fracs": [[0.2, 0.8]],
    "N_bins": 20,
    "N_sigmas": 5,
}

pop = build_population(pop_cfg)
aero_spec_names = np.array([species.name for species in pop.species])
aero_spec_masses = np.array(pop.spec_masses)
num_concs = np.array(pop.num_concs)
pHs = np.full(num_concs.shape[0], 4.5)

simulate_parcel(
    aero_spec_names,
    aero_spec_masses,
    num_concs,
    pHs,
    z_start=0.0,
    z_end=10.0,
    dt=1.0,
    updraft_velocity=0.5,
    S0=0.85,
    P0=101325.0,
    T0=298.0,
    radius_scale="log",
    condensation=True,
    cocondensation=False,
    aq_chemistry=False,
    gas_chemistry=False,
    output_filename="trajectory.pkl",
)

with open("trajectory.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
```

More complete examples are available under `examples/`.

## Repository Structure

```text
src/ld_chem/
    constants.py             # Physical constants
    gases.py                 # Trace gas representation
    particles.py             # Particle species helpers
    reactions.py             # Gas and aqueous reaction definitions
    run.py                   # Simulation entry points
    scenario.py              # Simulation setup
    systems.py               # Solvers and state updates
    utilities.py             # Runtime consistency checks
    write_files.py           # Output writers
    mechanisms/              # Default gas and aqueous reactions
    processes/               # Differential equation definitions
    species_data/            # Aerosol and gas species definitions
```

## Chemical Mechanisms

LD-Chem uses plain data files for default aqueous- and gas-phase mechanisms:

- `src/ld_chem/mechanisms/aq_reactions.dat`
- `src/ld_chem/mechanisms/gas_reactions.dat`

Custom mechanisms can be supplied by passing `mechanism_data_path` to the run
functions. The custom directory should contain the same expected mechanism file
names.

Default species data are stored in:

- `src/ld_chem/species_data/aero_data.dat`
- `src/ld_chem/species_data/gas_data.dat`

## Reproducibility and Data

This repository bundles source code, tests, examples, default mechanisms, and
small species/mechanism data files needed by the package. It does not bundle an
unpublished paper-specific dataset.

For a publication release, paper-specific input datasets and large generated
outputs should be archived separately and cited through the manuscript Data
Availability statement. Final DOI or accession values should be added to the
manuscript and release notes only after they are minted by the selected archive.

## Versioning

LD-Chem uses calendar-based release tags of the form vYYYY.N for citable
research-software snapshots. These release numbers do not imply
semantic-versioning compatibility guarantees. Changes affecting scientific
results, inputs, outputs, dependencies, or public APIs are documented in the
release notes.

## Citation

See `CITATION.cff` for software citation metadata. A final software DOI should
be added there after an official release DOI is minted.

## License

LD-Chem is distributed under the BSD-2-Clause license. See `LICENSE.txt`.
