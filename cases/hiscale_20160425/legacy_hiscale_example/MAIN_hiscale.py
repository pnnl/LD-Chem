#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Laura Fierce and Payton Beeler

This example script demonstrates how to run LES trajectory simulations using LD-Chem. 

The first step of simulations is to define the diameters, number concentrations, 
and composition of each aerosol particle. This can be done manually within run scripts,
read from external files, or generated from other models (such as part2pop). Any method
that provides the necessary aerosol information is acceptable, as long as it can be 
converted to numpy arrays. This example will use part2pop to define the initial 
aerosol population.


"""
import ld_chem
from ld_chem.run import simulate_les_trajectory, restart_trajectory
from part2pop.population import build_population 
import numpy as np
import pickle

# STEP 1: Define the aerosol population based on HI-SCALE measurements on 4/25/2016
data_path="../example_datasets/HISCALE_data_0425/"
pop_cfg = {
  "type": "hiscale_observations",
  "N_particles": 100,
  "beasd_file": data_path+"BEASD_G1_20160425155810_R2_HISCALE_001s.txt",
  "aimms_file": data_path+"AIMMS20_G1_20160425155810_R2_HISCALE020h.txt",
  "splat_file": data_path+"Splat_Composition_25-Apr-2016.txt",
  "ams_file": data_path+"HiScaleAMS_G1_20160425_R0.txt",
  "z": 100.0,
  "dz": 100.0,
  "splat_cutoff_nm": 85,
  "splat_species": {'BC': ['soot'], 'OIN': ['Dust'], 'SO4': ['sulfate_nitrate_org'], 'NO3': ['nitrate_amine_org'],
                    'OC': ['org28', 'org30_43', 'BB_SOA', 'org_amines', 'BB', 'pyridine'], 'IEPOX_SOA': ['IEPOX_SOA']},
  "mass_thresholds": {'IEPOX_SOA': [[0.3,0.5,0.1], ['IEPOX_OS','tetrol','tetrol_olig', 'IEPOX_OH_SOA']],
                    'SO4': [[0.5,0.7,0.1], ['SO4']],
                    'NO3': [[0.5,0.7,0.1], ['NO3']],
                    'OC': [[0.5,0.7,0.1], ['OC']],
                    'BC': [[0.5,0.7,0.1], ['BC']],
                    'OIN': [[0.5,0.7,0.1], ['OIN']]},
}

aerosol_population = build_population(pop_cfg)
aero_spec_names = np.array([species.name for species in aerosol_population.species]) # Get species names
aero_spec_masses = np.array(aerosol_population.spec_masses) # Get mass of species in each particle
num_concs = np.array(aerosol_population.num_concs) # Get number concentration of each particle
pHs=np.random.normal(size=num_concs.shape[0], loc=2.28, scale=0.78) # average pH for spring IOP (https://acp.copernicus.org/articles/21/5101/2021/acp-21-5101-2021.html)

# Step 2: Read in LES trajectory data
# Trajectory data must be a dict of numpy arrays with the following keys/values:
# 't': 1D array of time (s)
# 'x': 1D array of longutde (degrees); can be None
# 'y': 1D array of latitude (degrees); can be None
# 'z': 1D array of altitude (m)
# 'T': 1D array of temperature (K) 
# 'P': 1D array of pressure (Pa)
# 's': 1D array of saturation ratio (fractional)
# 'gas': dict of 1D arrays of gas-phase species concentrations (ppb)
trajectory_data = pickle.load(open("../example_datasets/example_FLEXPART_trajectory.pkl", 'rb'))

# Step 3: Run the LES trajectory simulation
# This example runs with three aqueous chemistry mechanism groups, 
# gas chemistry enabled, and with a set entrainment rate (relaxation time >0.0).
# A higher relaxation time = slower mass exchange with background gas concentrations.
simulate_les_trajectory(
    aero_spec_names, aero_spec_masses, num_concs, pHs, trajectory_data,
    dt=5.0, restart_filename='trajectory_restart.pkl', radius_scale='log',
    output_filename='trajectory.pkl', aq_chemistry=None,
    write_every=10.0, condensation=True, gas_chemistry=False, print_to_screen=True,
    cocondensation=False, relaxation_time=25.0)

# Step 4: plot time series of saturation ratio and wet diameters
import matplotlib.pyplot as plt
data=pickle.load(open('trajectory.pkl','rb'))

plt.plot(data['times'], data['S'])
plt.ylabel('saturation ratio')
plt.xlabel('time [s]')
plt.show()

spec_idx = np.where(data['particle species']=='Dwet')[0][0]
plt.plot(data['times'], data['particles'][:,:,spec_idx], '-b')
plt.yscale('log')
plt.ylabel('wet diameter [m]')
plt.xlabel('time [s]')
plt.show()
