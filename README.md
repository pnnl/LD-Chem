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

For development and testing from a clean clone:

```bash
git clone https://github.com/pnnl/LD-Chem.git
cd LD-Chem
python -m pip install -e ".[test]"
python -m pytest
```

The conda environment file can be used to create a consistent local scientific
Python environment before installing the package:

```bash
conda env create -f environment.yml
conda activate ld-chem
```

The editable pip install remains the preferred way to install LD-Chem during
development because it exercises the same package metadata used by CI.

## Quick Start

This is a lightweight smoke-test style example. It is intended to demonstrate
the package interface with a small synthetic aerosol population, not to
reproduce a full publication simulation.

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

Runnable cases are available under `cases/`.

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

cases/
    basic_examples/          # Small instructional cases
    hiscale_20160425/        # HISCALE April 25, 2016 research case scaffold
```

`cases/` contains runnable LD-Chem cases, including small basic examples and larger research workflows.

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

This repository bundles source code, tests, cases, default mechanisms, and
small species/mechanism data files needed by the package. It does not bundle an
unpublished paper-specific dataset.

For a publication release, paper-specific input datasets and large generated
outputs should be archived separately and cited through the manuscript Data
Availability statement. Final DOI or accession values should be added to the
manuscript and release notes only after they are minted by the selected archive.

## Citation

See `CITATION.cff` for software citation metadata. A final software DOI should
be added there after an official release DOI is minted.

## License

License text is not currently included in this repository. Before publication
release, add the project license file and ensure package metadata matches it.
