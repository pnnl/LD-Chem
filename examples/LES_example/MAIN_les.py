#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Laura Fierce and Payton Beeler

This example script demonstrates how to run LES trajectory simulations using multipart. 

The first step of simulations is to define the diameters, number concentrations, 
and composition of each aerosol particle. This can be done manually within run scripts,
read from external files, or generated from other models (such as part2pop). Any method
that provides the necessary aerosol information is acceptable, as long as it can be 
converted to numpy arrays. This example will use part2pop to define the initial 
aerosol population.


"""
from multipart.run import simulate_les_trajectory, restart_trajectory
from part2pop.population import build_population 
import numpy as np
import pickle

# STEP 1: Define the aerosol population
# Define an 2-mode lognormal aerosol population using part2pop
# First mode: Pure sulfate
# Second mode: Pure organics

# pop_cfg = {
#     "type": "binned_lognormals",
#     "N": [1e9, 1e9],  # number concentration of each mode (m^-3)
#     "GMD": [150e-9, 150e-9],  # geometric mean diameter of each mode (m)
#     "GSD": [1.6, 1.6],  # geometric standard deviation of each mode
#     "aero_spec_names": [["SO4"],["OC"]], # two modes of externally mixed aerosols
#     "aero_spec_fracs": [[1.0], [1.0]], # mass fraction of each species in each particle
#     "N_bins": 10,  # number of bins to discretize the population
#     "N_sigmas": 5, # D_range is +/- 5 geometric standard deviations
#     "species_modifications": {"OC": {"density": 1200}}, # modify default density of OC to 1200 kg/cm^3
#     }

pop_cfg = {
    "type": "monodisperse",
    "N": [1e9],  # number concentration of each mode (m^-3)
    "D": [100e-9],  # geometric mean diameter of each mode (m)
    "aero_spec_names": [["OC", "BC"]], # two modes of externally mixed aerosols
    "aero_spec_fracs": [[0.8, 0.2]], # mass fraction of each species in each particle
    "species_modifications": {"OC": {"density": 1200}}, # modify default density of OC to 1200 kg/cm^3
    }

aerosol_population = build_population(pop_cfg)
aero_spec_names = np.array([species.name for species in aerosol_population.species]) # Get species names
aero_spec_masses = np.array(aerosol_population.spec_masses) # Get mass of species in each particle
num_concs = np.array(aerosol_population.num_concs) # Get number concentration of each particle
pHs = np.random.normal(loc=4.5, scale=0.5, size=num_concs.shape[0]) # Random pH for each particle

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
trajectory_data = pickle.load(open('../example_datasets/example_FLEXPART_trajectory.pkl', 'rb'))

# Step 3: Run the LES trajectory simulation
# This example runs with three aqueous chemistry mechanism groups, 
# gas chemistry enabled, and with a set entrainment rate (relaxation time >0.0).
# A higher relaxation time = slower mass exchange with background gas concentrations.
simulate_les_trajectory(
    aero_spec_names, aero_spec_masses, num_concs, pHs, trajectory_data,
    dt=5.0, restart_filename='trajectory_restart.pkl', radius_scale='log',
    output_filename='trajectory.pkl', aq_chemistry=['sulfate','nitrate','IEPOX'],
    write_every=10.0, condensation=True, gas_chemistry=True, print_to_screen=True,
    cocondensation=True, relaxation_time=25.0)

# Step 4: plot time series of saturation ratio and wet diameters
import matplotlib.pyplot as plt
data=pickle.load(open('trajectory.pkl','rb'))

plt.plot(data['times'], data['S'])
plt.ylabel('saturation ratio')
plt.xlabel('time [s]')
plt.show()

spec_idx = np.where(data['particle species']=='Dwet')[0][0]
mode1_idx = np.where(data['particles'][0,:,np.where(data['particle species']=='SO4')[0][0]]>0)[0] # pick out only the sulfate particles
mode2_idx = np.where(data['particles'][0,:,np.where(data['particle species']=='OC')[0][0]]>0)[0] # pick out only the organic particles
for i, idx in enumerate(mode1_idx):
    plt.plot(data['times'], data['particles'][:,idx,spec_idx], '-r', label='pure sulfate mode' if i == 0 else "")
for i, idx in enumerate(mode2_idx):
    plt.plot(data['times'], data['particles'][:,idx,spec_idx], '-g', label='pure organics mode' if i == 0 else "")
plt.legend()
plt.yscale('log')
plt.ylabel('wet diameter [m]')
plt.xlabel('time [s]')
plt.show()

