#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Laura Fierce and Payton Beeler

This example script demonstrates how to run an adiabatic parcel simulation using LD-Chem. 

The first step of simulations is to define the diameters, number concentrations, 
and composition of each aerosol particle. This can be done manually within run scripts,
read from external files, or generated from other models (such as part2pop). Any method
that provides the necessary aerosol information is acceptable, as long as it can be 
converted to numpy arrays. This example will use part2pop to define the initial 
aerosol population.

"""
from ld_chem.run import simulate_parcel, restart_trajectory
from part2pop.population import build_population 
import numpy as np
import pickle

# STEP 1: Define the aerosol population
# Define an 2-mode lognormal aerosol population using part2pop
# First mode: Pure sulfate
# Second mode: Pure organics
pop_cfg = {
    "type": "binned_lognormals",
    "N": [1e9, 1e9],  # number concentration of each mode (m^-3)
    "GMD": [150e-9, 150e-9],  # geometric mean diameter of each mode (m)
    "GSD": [1.6, 1.6],  # geometric standard deviation of each mode
    "aero_spec_names": [["SO4"],["OC"]], # two modes of externally mixed aerosols
    "aero_spec_fracs": [[1.0], [1.0]], # mass fraction of each species in each particle
    "N_bins": 10,  # number of bins to discretize the population
    "N_sigmas": 5, # D_range is +/- 5 geometric standard deviations
    "species_modifications": {"OC": {"density": 1200}}, # modify default density of OC to 1200 kg/cm^3
    }

aerosol_population = build_population(pop_cfg)
aero_spec_names = np.array([species.name for species in aerosol_population.species]) # Get species names
aero_spec_masses = np.array(aerosol_population.spec_masses) # Get mass of species in each particle
num_concs = np.array(aerosol_population.num_concs) # Get number concentration of each particle
pHs = np.random.normal(loc=4.5, scale=0.5, size=num_concs.shape[0]) # Random pH for each particle

# Step 2: Run the adiabatic parcel simulation
simulate_parcel(
    aero_spec_names, aero_spec_masses, num_concs, pHs,
    z_start=0., z_end=1000., dt=1.0, updraft_velocity=0.5,
    S0=0.85, P0=101325, T0=298, radius_scale='log',
    restart_filename='trajectory_restart.pkl', 
    output_filename='trajectory.pkl', write_every=5.0,
    gas_names=['IEPOX', 'SO2'], gas_concs=[0.4, 1.0], condensation = True, 
    cocondensation = False, aq_chemistry = None, 
    gas_chemistry = False)

# Step 3: Plot the vertical saturation ratio profile and particle diameters
import matplotlib.pyplot as plt
data=pickle.load(open('trajectory.pkl','rb'))
plt.plot(data['S'], data['z'])
plt.xlim(1.0,)
plt.ylabel('altitude [m]')
plt.xlabel('saturation ratio')
plt.show()

spec_idx = np.where(data['particle species']=='Dwet')[0][0] # want to plot Dwet
mode1_idx = np.where(data['particles'][0,:,np.where(data['particle species']=='SO4')[0][0]]>0)[0] # pick out only the sulfate particles
mode2_idx = np.where(data['particles'][0,:,np.where(data['particle species']=='OC')[0][0]]>0)[0] # pick out only the organic particles
for i, idx in enumerate(mode1_idx):
    plt.plot(data['particles'][:,idx,spec_idx], data['z'], '-r', label='pure sulfate mode' if i == 0 else "")
for i, idx in enumerate(mode2_idx):
    plt.plot(data['particles'][:,idx,spec_idx], data['z'], '-g', label='pure organics mode' if i == 0 else "")
plt.xscale('log')
plt.xlabel('wet diameter [m]')
plt.ylabel('altitude [m]')
plt.legend()
plt.show()








