# Overview

This folder contains inputs for each trajectory simulation used in "Epoxide-driven secondary organic aerosol formation is modulated by aerosol-cloud cycling", Nature Communications (2026). Each file contains a list of inputs where each row corresponds to one simulation. 

# Quick Start

## This is a lightweight example of a single simulation from Beeler et. al. (2026). It is intended to demonstrate how each simulation in the ensemble is run. Parallelization or scripting is highly reccommended if running a lrge number of simulations.

```python
from ld_chem.run import simulate_les_trajectory
from part2pop.population import build_population 
import numpy as np
import pickle

# STEP 1: Select which simulation to run (row in input arrays)
simulation_number = 0

# STEP 2: Read in aerosol population and trajectory information
diameters=pickle.load(open("paper_input_files/diameters.pkl", "rb"))[simulation_number]
aero_spec_names=pickle.load(open("paper_input_files/aero_spec_names.pkl", "rb"))[simulation_number]
aero_spec_fracs=pickle.load(open("paper_input_files/aero_spec_fracs.pkl", "rb"))[simulation_number]
num_concs=pickle.load(open("paper_input_files/number_concentrations.pkl", "rb"))[simulation_number]
aero_spec_masses=pickle.load(open("paper_input_files/aero_spec_masses.pkl", "rb"))[simulation_number]
pHs=pickle.load(open("paper_input_files/pHs.pkl", "rb"))[simulation_number]
FLEXPART_trajectory = pickle.load(open("paper_input_files/FLEXPART_trajectories.pkl", 'rb'))[simulation_number]

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
    cocondensation=True, relaxation_time=24.475)
```

## This is an example of how to generate input files from measured data and run an LD-Chem simulation for the genreated input files.

Step 1: Run input_file_generation.py from command line and provide paths to measured data.
```shell
python3 input_file_generation.py \
    --N_particles 100 \
    --size_distribution_file ../examples/example_datasets/HISCALE_data_0425/BEASD_G1_20160425155810_R2_HISCALE_001s.txt \
    --AIMMS_file ../examples/example_datasets/HISCALE_data_0425/AIMMS20_G1_20160425155810_R2_HISCALE020h.txt \
    --SPLAT_file ../examples/example_datasets/HISCALE_data_0425/Splat_Composition_25-Apr-2016.txt \
    --AMS_file ../examples/example_datasets/HISCALE_data_0425/HiScaleAMS_G1_20160425_R0.txt \
    --FLEXPART_file ../examples/example_datasets/HISCALE_data_0425/FLEXPART_output_traj_0001.txt \
    --gas_phase_directory ../examples/example_datasets/HISCALE_data_0425/CIMS_data
    --output_directory .
```

Step 2. Load input files and run trajectory (change file paths and names as needed).

```python

from ld_chem.run import simulate_les_trajectory
import pickle

aero_spec_names = pickle.load(open("aero_spec_names.pkl", "rb"))
aero_spec_masses = pickle.load(open("aero_spec_masses.pkl", "rb"))
num_concs = pickle.load(open("number_concentrations.pkl", "rb"))
pHs = pickle.load(open("pHs.pkl", "rb"))
flexpart_trajectory = pickle.load(open("FLEXPART_trajectory.pkl", "rb"))

simulate_les_trajectory(
    aero_spec_names, aero_spec_masses, num_concs, pHs, flexpart_trajectory,
    dt=5.0, restart_filename=f"trajectory_restart_{str(0).zfill(6)}.pkl", radius_scale='log',
    output_filename=f"trajectory_{str(0).zfill(6)}.pkl", 
    aq_chemistry=None,
    write_every=30.0, condensation=True, gas_chemistry=False, print_to_screen=True,
    cocondensation=False, relaxation_time=24.475
)

```


