# Preprocessed LD-Chem inputs

This directory contains, or documents how to obtain, the preprocessed LD-Chem inputs for the HISCALE April 25, 2016 trajectory ensemble associated with Beeler et al. 2026.

Each file contains a list or array of inputs where each row or entry corresponds to one LD-Chem trajectory simulation.

## Expected files

```text
FLEXPART_trajectories.pkl
diameters.pkl
aero_spec_names.pkl
aero_spec_fracs.pkl
aero_spec_masses.pkl
number_concentrations.pkl
pHs.pkl
```

## File descriptions

### `FLEXPART_trajectories.pkl`

A list or array of trajectory dictionaries. Each entry corresponds to one LD-Chem trajectory simulation.

Each trajectory dictionary should contain time series arrays with keys such as:

```text
t    time [s]
x    longitude [degrees east], or None if unavailable
y    latitude [degrees north], or None if unavailable
z    altitude [m]
T    temperature [K]
P    pressure [Pa]
s    saturation ratio [-]
gas  gas-phase concentration time series, when available
```

### `diameters.pkl`

Particle dry diameters for each simulation.

Units: m.

### `aero_spec_names.pkl`

Aerosol species names for each simulation.

### `aero_spec_fracs.pkl`

Aerosol species mass fractions for each simulation.

### `aero_spec_masses.pkl`

Aerosol species masses for each particle in each simulation.

Units: kg.

### `number_concentrations.pkl`

Particle number concentrations for each simulation.

Units: m-3.

### `pHs.pkl`

Particle pH values for each simulation.

## Run one simulation

From this directory, this example shows how a single simulation can be run from the preprocessed inputs.

```python
from ld_chem.run import simulate_les_trajectory
from part2pop.population import build_population
import numpy as np
import pickle

simulation_number = 0

diameters = pickle.load(open("diameters.pkl", "rb"))[simulation_number]
aero_spec_names = pickle.load(open("aero_spec_names.pkl", "rb"))[simulation_number]
aero_spec_fracs = pickle.load(open("aero_spec_fracs.pkl", "rb"))[simulation_number]
num_concs = pickle.load(open("number_concentrations.pkl", "rb"))[simulation_number]
aero_spec_masses = pickle.load(open("aero_spec_masses.pkl", "rb"))[simulation_number]
pHs = pickle.load(open("pHs.pkl", "rb"))[simulation_number]
trajectory = pickle.load(open("FLEXPART_trajectories.pkl", "rb"))[simulation_number]

pop_cfg = {
    "type": "monodisperse",
    "N": num_concs,
    "D": diameters,
    "aero_spec_names": aero_spec_names,
    "aero_spec_fracs": aero_spec_fracs,
}

aerosol_population = build_population(pop_cfg)
diameters_from_masses = aerosol_population.get_particle_var("dry_diameter")
assert np.isclose(diameters_from_masses, diameters).all()

simulate_les_trajectory(
    aero_spec_names[0],
    aero_spec_masses,
    num_concs,
    pHs,
    trajectory,
    dt=5.0,
    restart_filename=f"trajectory_restart_{simulation_number:06d}.pkl",
    radius_scale="log",
    output_filename=f"trajectory_{simulation_number:06d}.pkl",
    aq_chemistry=["IEPOX", "sulfate", "nitrate", "ammonium"],
    write_every=30.0,
    condensation=True,
    gas_chemistry=True,
    print_to_screen=True,
    cocondensation=False,
    relaxation_time=24.475,
)
```

## Notes

The preprocessed inputs are provided so users can run the LD-Chem ensemble without repeating the full WRF-FLEXPART and ARM preprocessing workflow.

The preprocessing workflow is documented in the parent case directory.
