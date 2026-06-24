## Overview

This folder contains inputs for each trajectory simulation used in "Epoxide-driven secondary organic aerosol formation is modulated by aerosol-cloud cycling", Nature Communications (2026). Each file contains a list of inputs where each row corresponds to one simulation. 

## Quick Start

This is a lightweight example of a single simulation. It is intended to demonstrate
how each simulation in the ensemble is run. Parallelization or scripting is highly reccommended if running a lrge number of simulations.

```python
from ld_chem.run import simulate_les_trajectory
from part2pop.population import build_population 
import numpy as np
import pickle

# STEP 1: Select which simulation to run (row in input arrays)
simulation_number = 0

# STEP 2: Read in aerosol population and trajectory information
diameters=pickle.load(open("diameters.pkl", "rb"))[simulation_number]
aero_spec_names=pickle.load(open("aero_spec_names.pkl", "rb"))[simulation_number]
aero_spec_fracs=pickle.load(open("aero_spec_fracs.pkl", "rb"))[simulation_number]
num_concs=pickle.load(open("number_concentrations.pkl", "rb"))[simulation_number]
aero_spec_masses=pickle.load(open("aero_spec_masses.pkl", "rb"))[simulation_number]
pHs=pickle.load(open("pHs.pkl", "rb"))[simulation_number]
FLEXPART_trajectory = pickle.load(open("FLEXPART_trajectories.pkl", 'rb'))[simulation_number]

# STEP 3: Define aerosol population using part2pop to make sure masses and diameters are consistent
pop_cfg = {
    "type": "monodisperse",
    "N": num_concs,
    "D": diameters,
    "aero_spec_names": aero_spec_names,
    "aero_spec_fracs": aero_spec_fracs
}
aerosol_population = build_population(pop_cfg)
Dps_from_masses = aerosol_population.get_particle_var("dry_diameter")
assert np.isclose(Dps_from_masses, diameters).all()

# Step 4: Run the simulation
simulate_les_trajectory(
    aero_spec_names[0], aero_spec_masses, num_concs, pHs, FLEXPART_trajectory,
    dt=5.0, restart_filename=f"trajectory_restart_{str(simulation_number).zfill(6)}.pkl", radius_scale='log',
    output_filename=f"trajectory_{str(simulation_number).zfill(6)}.pkl", 
    aq_chemistry=['IEPOX','sulfate','nitrate','ammonium'],
    write_every=30.0, condensation=True, gas_chemistry=True, print_to_screen=True,
    cocondensation=False, relaxation_time=24.475)
```
