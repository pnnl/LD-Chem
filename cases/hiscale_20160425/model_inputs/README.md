# HISCALE model inputs

This directory contains committed, LD-Chem-ready input files for the HISCALE April 25, 2016 trajectory ensemble associated with Beeler et al. 2026.

These files are the recommended entry point for running the HISCALE case.

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

Each file contains a list or array where each row or entry corresponds to one LD-Chem trajectory simulation.

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

From the parent case directory:

```bash
python run_case.py --run-ensemble --max-simulations 1
```

This runs one LD-Chem simulation using the committed inputs in this directory.
