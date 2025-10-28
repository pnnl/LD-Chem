#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:06:10 2024

@author: beel083
"""
# %% 
# copy the necessary modules to the UNIT_TESTS directory
# probably need a different way to do this but I don't 
# want to mess with sys.path

import shutil, os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as font_manager

files1 = ['particles.py', 'constants.py', 'scenario.py', 'aerosol_species.py',
         'utilities.py', 'systems.py', 'driver.py', 'visualization.py', 
         'TraceGases.py', 'Reactions.py', 'write_files.py']

for file in files1:
    source = '../../multipart/'+file
    destination = os.getcwd()+'/'+file
    shutil.copy(source, destination)
    
files2 = ['UnitTests_driver.py', 'UnitTests_scenario.py', 'UnitTests_visualization.py']

for file in files2:
    source = '../'+file
    destination = os.getcwd()+'/'+file
    shutil.copy(source, destination)

directories = ['../../multipart/processes', '../../species_data', '../../mechanisms']
for directory in directories:
    source = directory
    destination = source.replace('.', '')
    destination = destination.replace('/', '')
    destination = destination.replace('multipart', '')
    if os.path.isdir(destination):
        shutil.rmtree(destination)    
    destination = os.getcwd()+'/'+destination
    shutil.copytree(source, destination)


# %% do the runs

from UnitTests_driver import simulate_sulfate_partitioning


gas_concentrations=np.logspace(-8, 12, 50)

trajectory_ensemble = simulate_sulfate_partitioning(gas_concentrations,
        t_end=30, dt=0.5, updraft_velocity=0.0,
        Ddry=100e-9, sigma=1.0, Ntot=1e6, Npart=1,
        S0=0.85, P0=101325, T0=298, pH0=12.0,
        accom=1., verbosity=50,
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=['SO2'], gas_conc=[1.0],
        radius_scale='log',solver='ode15s',
        specdata_path='species_data/', mechanism_data_path='mechanisms/',
        condensation = True, 
        collisions = False, settling = False,
        cocondensation = True, aq_chemistry = ['sulfate'], freezing = False)

# UnitTests_visualization.plot_equilibrium_fractions(trajectory_ensemble[0], 'SO2', axis='time')

# %% make the plot

axis_label_fontsize=13
axis_tick_fontsize=11
legend_fontsize=12
markersize=7
fontname = 'Helvetica'
font = font_manager.FontProperties(family=fontname, size=legend_fontsize)

fig, ax = plt.subplots(1, 1, figsize=(1.0*6.4, 1.0*4.8), constrained_layout=False)
ax.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
ax.tick_params(which="major", axis="both", length=6)
ax.tick_params(which="minor", axis="both", length=4)
ax.grid(which='major', color='grey', alpha=0.4, linewidth=1)


# plot the published data
published_data = pd.read_excel('published_data.xls', sheet_name='S&P fig 7.8') 
ax.plot(published_data['pH'][::2], published_data['SO2'][::2], 'bo', markersize=markersize)
ax.plot(published_data['pH'][::2], published_data['HSO3'][::2], 'rs', markersize=markersize)
ax.plot(published_data['pH'][::2], published_data['SO3'][::2], 'g^', markersize=markersize)


# process the data and plot
pHs = []
SO2_fraction = []
HSO3_fraction = []
SO3_fraction = []

for trajectory in trajectory_ensemble:
    particle=trajectory.parcel_states[-1].particle_population.particles[0]
    pHs.append(particle.get_pH())
    total_moles=0
    for species in ['SO2', 'HSO3', 'SO3']:
        total_moles+=particle.masses[particle.get_species_idx(species)]/particle.species[particle.get_species_idx(species)].molar_mass
    SO2_fraction.append((particle.masses[particle.get_species_idx('SO2')]/particle.species[particle.get_species_idx('SO2')].molar_mass)/total_moles)
    HSO3_fraction.append((particle.masses[particle.get_species_idx('HSO3')]/particle.species[particle.get_species_idx('HSO3')].molar_mass)/total_moles)
    SO3_fraction.append((particle.masses[particle.get_species_idx('SO3')]/particle.species[particle.get_species_idx('SO3')].molar_mass)/total_moles)
    
ax.plot(pHs, SO2_fraction, '-b', label=r'SO$_2$')
ax.plot(pHs, HSO3_fraction, '-r', linestyle='dashed', label=r'HSO$_3$')
ax.plot(pHs, SO3_fraction, '-g', linestyle='dashdot', label=r'SO$_3$')

ax.set_xlim(np.min(pHs), np.max(pHs))
ax.set_ylim(0,1)
ax.set_xlabel('Droplet pH', font=fontname, labelpad=15, fontsize=axis_label_fontsize)
ax.set_ylabel('aqueous fraction', font=fontname, labelpad=15, fontsize=axis_label_fontsize)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, prop=font)
ax.set_xlim(0, 8)
ax.text(-0.18, 1.15, 'B', font=fontname, fontsize=1.5*axis_label_fontsize, transform=ax.transAxes)

for label in ax.get_xticklabels():
    label.set_fontproperties(fontname)
    label.set_fontsize(axis_tick_fontsize)
for label in ax.get_yticklabels():
    label.set_fontproperties(fontname)
    label.set_fontsize(axis_tick_fontsize)

fig.savefig('S(IV)_partitioning.png', bbox_inches='tight', dpi=200)

output_data={'pH': pHs, 'SO2 fraction': np.array(SO2_fraction), 'HSO3 fraction': np.array(HSO3_fraction), 'SO3 fraction': np.array(SO3_fraction)}
pickle.dump(output_data, open('S4_fractions.pkl', 'wb'))

# %% make the plot

axis_label_fontsize=18
axis_tick_fontsize=16
legend_fontsize=16
markersize=11

fig, ax = plt.subplots(1, 1, figsize=(1.0*6.4, 1.0*4.8), constrained_layout=False)
ax.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
ax.tick_params(which="major", axis="both", length=6)
ax.tick_params(which="minor", axis="both", length=4)
ax.grid(which='major', color='grey', alpha=0.4, linewidth=1)

# plot the published data
published_data = pd.read_excel('published_data.xls', sheet_name='S&P fig 7.5') 
ax.plot(published_data['pH'][::2], published_data['SO2'][::2], 'bo', markersize=markersize)
ax.plot(published_data['pH'][::2], published_data['HSO3'][::2], 'rs', markersize=markersize)
ax.plot(published_data['pH'][::2], published_data['SO3'][::2], 'g^', markersize=markersize)

# process the data and plot
pHs = np.zeros(len(gas_concentrations))
gas_conc = np.zeros(len(gas_concentrations))
SO2_conc = np.zeros(len(gas_concentrations))
HSO3_conc = np.zeros(len(gas_concentrations))
SO3_conc = np.zeros(len(gas_concentrations))

for ii, (trajectory) in enumerate(trajectory_ensemble):
    particle=trajectory.parcel_states[-1].particle_population.particles[0]
    pHs[ii]=particle.get_pH()
    water_volume=particle.get_vol_tot()-particle.get_vol_dry() # m^3
    SO2_conc[ii]=(particle.masses[particle.get_species_idx('SO2')]/particle.species[particle.get_species_idx('SO2')].molar_mass)/(1000*water_volume) # mol/L
    HSO3_conc[ii]=(particle.masses[particle.get_species_idx('HSO3')]/particle.species[particle.get_species_idx('HSO3')].molar_mass)/(1000*water_volume) # mol/L
    SO3_conc[ii]=(particle.masses[particle.get_species_idx('SO3')]/particle.species[particle.get_species_idx('SO3')].molar_mass)/(1000*water_volume) # mol/L
    gas=trajectory.parcel_states[-1].TraceGas_population
    gas_conc[ii]=gas.concs[gas.get_species_idx('SO2')]
    

ax.plot(pHs, SO2_conc/gas_conc, '-b', label=r'SO$_2$')
ax.plot(pHs, HSO3_conc/gas_conc, '-r', linestyle='dashed', label=r'HSO$_3$')
ax.plot(pHs, SO3_conc/gas_conc, '-g', linestyle='dashdot', label=r'SO$_3$')

ax.set_xlim(np.min(pHs), np.max(pHs))
ax.set_ylim(1e-10, 1e-2)
ax.set_yscale('log')
ax.set_xlabel('pH', font=fontname, labelpad=15, fontsize=axis_label_fontsize)
ax.set_ylabel('aqueous concentration (M/ppb)', font=fontname, labelpad=15, fontsize=axis_label_fontsize)
ax.legend(fontsize=legend_fontsize, prop=font)
ax.set_xlim(0, 8)

for label in ax.get_xticklabels():
    label.set_fontproperties(fontname)
    label.set_fontsize(axis_tick_fontsize)
for label in ax.get_yticklabels():
    label.set_fontproperties(fontname)
    label.set_fontsize(axis_tick_fontsize)

fig.savefig('S(IV)_concentrations.png', bbox_inches='tight', dpi=200)


# %%
# delete all the modules that got moved to UNIT_TESTS/condensation 
# directory

for file in files1:
    os.remove(file)
    
for file in files2:
    os.remove(file)
    
for directory in directories:
    directory = directory.replace('.', '')
    directory = directory.replace('/', '')
    directory = directory.replace('multipart', '')
    shutil.rmtree(directory)

