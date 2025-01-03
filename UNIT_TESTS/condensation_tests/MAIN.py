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

import shutil, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

files1 = ['particles.py', 'constants.py', 'scenario.py', 'aerosol_species.py',
         'utilities.py', 'systems.py', 'driver.py', 'visualization.py', 
         'TraceGases.py', 'Reactions.py']

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
    if os.path.isdir(destination):
        shutil.rmtree(destination)    
    destination = os.getcwd()+'/'+destination
    shutil.copytree(source, destination)

from UnitTests_driver import simulate_condensation_test

# %% set up the plot

axis_label_fontsize=18
axis_tick_fontsize=16
legend_fontsize=16
markersize=11

fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = plt.subplots(2, 3, figsize=(3.0*6.4, 2.0*4.8), constrained_layout=False, sharex='col', sharey='row')
ax11.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
ax11.tick_params(which="major", axis="both", length=6)
ax11.tick_params(which="minor", axis="both", length=4)
ax11.grid(which='major', color='grey', alpha=0.4, linewidth=1)
ax21.tick_params(axis='both', which="major",labelsize=axis_tick_fontsize, pad=8, width=1)
ax21.tick_params(which="major", axis="both", length=6)
ax21.tick_params(which="minor", axis="both", length=4)
ax21.grid(which='major', color='grey', alpha=0.4, linewidth=1)
ax12.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
ax12.tick_params(which="major", axis="both", length=6)
ax12.tick_params(which="minor", axis="both", length=4)
ax12.grid(which='major', color='grey', alpha=0.4, linewidth=1)
ax22.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
ax22.tick_params(which="major", axis="both", length=6)
ax22.tick_params(which="minor", axis="both", length=4)
ax22.grid(which='major', color='grey', alpha=0.4, linewidth=1)
ax13.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
ax13.tick_params(which="major", axis="both", length=6)
ax13.tick_params(which="minor", axis="both", length=4)
ax13.grid(which='major', color='grey', alpha=0.4, linewidth=1)
ax23.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
ax23.tick_params(which="major", axis="both", length=6)
ax23.tick_params(which="minor", axis="both", length=4)
ax23.grid(which='major', color='grey', alpha=0.4, linewidth=1)
# ax14.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
# ax14.tick_params(which="major", axis="both", length=6)
# ax14.tick_params(which="minor", axis="both", length=4)
# ax14.grid(which='major', color='grey', alpha=0.4, linewidth=1)
# ax24.tick_params(axis='both', which="major",labelsize=axis_tick_fontsize, pad=8, width=1)
# ax24.tick_params(which="major", axis="both", length=6)
# ax24.tick_params(which="minor", axis="both", length=4)
# ax24.grid(which='major', color='grey', alpha=0.4, linewidth=1)

# %% changing updraft velocity

# read previously published data from files
published_data = pd.read_excel('published_data.xls', sheet_name='changing_velocity') 

for x, m in zip(np.unique(published_data['study']), ['o', 'v', 's', '^', '<']):
    ix = np.where(published_data['study'] == x)
    ax11.plot(published_data['updraft velocity'][ix[0]], published_data['max SS'][ix[0]], marker=m, mfc='w', mec='k', linewidth=0, markersize=markersize, label=x)
    ax21.plot(published_data['updraft velocity'][ix[0]], published_data['activated fraction'][ix[0]], marker=m, mfc='w', mec='k', linewidth=0, markersize=markersize)


# do the simulations
number_mode_diameter = 100e-9
sigma = 2.0
S0 = 0.85
Ntot = 1e9
Vs = np.logspace(-1, 1, 10)

print('============= updraft velocity runs:', len(Vs), 'scenarios =============')

trajectory_ensemble = simulate_condensation_test(N_scenarios=len(Vs),
        z_start=0.,z_end=1000.,dt=1.0,
        Ddry=number_mode_diameter,sigma=sigma,Ntot=Ntot, Npart=20,
        updraft_velocity=Vs,S0=S0,P0=101325,T0=298,
        pH0=7.0,accom=1., verbosity=50,
        radius_scale='lin',solver='ode15s',
        species_names=['AS'], mass_fractions=np.array([1.]),
        gas_names=None, gas_conc=None,
        specdata_path='species_data/', mechanism_data_path='mechanisms/',
        condensation = True, collisions = False, settling = False,
        cocondensation = False, chemistry = None, freezing = False)

# process the data and plot
max_S = []
activated_fraction = []
for trajectory in trajectory_ensemble:
    max_S.append(100*(trajectory.get_max_S()-1))
    activated_fraction.append(trajectory.get_activated_fraction())
    # UnitTests_visualization.plot_SS_diameters(trajectory, axis='height')

ax11.plot(Vs, max_S, '-ro')
ax21.plot(Vs, activated_fraction, '-ro')

# fix the plot
ax11.set_xscale('log')
ax11.set_yscale('log')
ax11.set_ylim(1e-2, 10)
ax11.set_xlim(0.08, 12)
ax11.set_ylabel('Maximum Supersaturation (%)', fontsize=axis_label_fontsize, labelpad=15)
ax11.text(0.04, 0.95, ' A ', fontsize=axis_label_fontsize, ha='left', va='top', transform=ax11.transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'})
ax21.set_ylim(0,1)
ax21.set_ylabel('Activated Fraction', fontsize=axis_label_fontsize, labelpad=27)
ax21.set_xlabel('Updraft Velocity (m/s)', fontsize=axis_label_fontsize, labelpad=15)
ax21.text(0.04, 0.95, ' B ', fontsize=axis_label_fontsize, ha='left', va='top', transform=ax21.transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'})

# %% changing Number concentration

# read previously published data from files
published_data = pd.read_excel('published_data.xls', sheet_name='changing_Ntot') 

for x, m in zip(np.unique(published_data['study']), ['o', 'v', 's', '^', '<']):
    ix = np.where(published_data['study'] == x)
    ax12.plot(published_data['number concentration'][ix[0]], published_data['max SS'][ix[0]], marker=m, mfc='w', mec='k', linewidth=0, markersize=markersize, label=x)
    ax22.plot(published_data['number concentration'][ix[0]], published_data['activated fraction'][ix[0]], marker=m, mfc='w', mec='k', linewidth=0, markersize=markersize)


# do the simulations
number_mode_diameter = 100e-9
sigma = 2.0
S0 = 0.85
Ntot = np.logspace(8,10,10)
Vs = 0.5

print()
print('============= number concentration runs:', len(Ntot), 'scenarios =============')

trajectory_ensemble = simulate_condensation_test(N_scenarios=len(Ntot),
        z_start=0.,z_end=1000.,dt=0.5,
        Ddry=number_mode_diameter,sigma=sigma,Ntot=Ntot, Npart=20,
        updraft_velocity=Vs,S0=S0,P0=101325,T0=298,
        pH0=7.0,accom=1., verbosity=50,
        radius_scale='lin',solver='ode15s',
        species_names=['AS'], mass_fractions=np.array([1.]),
        gas_names=None, gas_conc=None,
        specdata_path='species_data/', mechanism_data_path='mechanisms/',
        condensation = True, collisions = False, settling = False,
        cocondensation = False, chemistry = None, freezing = False)

# process the data and plot
max_S = []
activated_fraction = []
for trajectory in trajectory_ensemble:
    max_S.append(100*(trajectory.get_max_S()-1))
    activated_fraction.append(trajectory.get_activated_fraction())
    # UnitTests_visualization.plot_SS_diameters(trajectory, axis='height')

ax12.plot(np.array((Ntot))/100**3, max_S, '-ro')
ax22.plot(np.array((Ntot))/100**3, activated_fraction, '-ro')


# fix the plot
ax22.set_xscale('log')
ax12.set_xlim(80, 12000)
ax22.set_xlabel(r'Number Concentration (cm$^{-3}$)', fontsize=axis_label_fontsize, labelpad=15)

ax12.text(0.04, 0.95, ' C ', fontsize=axis_label_fontsize, ha='left', va='top', transform=ax12.transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'})
ax22.text(0.04, 0.95, ' D ', fontsize=axis_label_fontsize, ha='left', va='top', transform=ax22.transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'})
ax12.legend(loc='center', bbox_to_anchor=(0.5, 1.2), fontsize=legend_fontsize, ncol=3, frameon=False)


# %% changing number mode diameter
# read previously published data from files
published_data = pd.read_excel('published_data.xls', sheet_name='changing_Dpg') 

for x, m in zip(np.unique(published_data['study']), ['o', 'v', 's', '^', '<']):
    ix = np.where(published_data['study'] == x)
    ax13.plot(published_data['mode diameter'][ix[0]], published_data['max SS'][ix[0]], marker=m, mfc='w', mec='k', linewidth=0, markersize=markersize, label=x)
    ax23.plot(published_data['mode diameter'][ix[0]], published_data['activated fraction'][ix[0]], marker=m, mfc='w', mec='k', linewidth=0, markersize=markersize)


# do the simulations
number_mode_diameter = 2*np.logspace(-8,-6.698970004336019,10)
sigma = 2.0
S0 = 0.85
Ntot = 1e9
Vs = 0.5

print()
print('============= number mode radius:', len(number_mode_diameter), 'scenarios =============')

trajectory_ensemble = simulate_condensation_test(N_scenarios=len(number_mode_diameter),
        z_start=0.,z_end=1000.,dt=1.0,
        Ddry=number_mode_diameter,sigma=sigma,Ntot=Ntot, Npart=20,
        updraft_velocity=Vs,S0=S0,P0=101325,T0=298,
        pH0=7.0,accom=1., verbosity=50,
        radius_scale='lin',solver='ode15s',
        species_names=['AS'], mass_fractions=np.array([1.]),
        gas_names=None, gas_conc=None,
        specdata_path='species_data/', mechanism_data_path='mechanisms/',
        condensation = True, collisions = False, settling = False,
        cocondensation = False, chemistry = None, freezing = False)

# process the data and plot
max_S = []
activated_fraction = []
for trajectory in trajectory_ensemble:
    max_S.append(100*(trajectory.get_max_S()-1))
    activated_fraction.append(trajectory.get_activated_fraction())
    # UnitTests_visualization.plot_SS_diameters(trajectory, axis='height')

ax13.plot(0.5*np.array((number_mode_diameter))*1e6, max_S, '-ro')
ax23.plot(0.5*np.array((number_mode_diameter))*1e6, activated_fraction, '-ro')


# fix the plot
ax23.set_xscale('log')
ax13.set_xlim(0.008, 1.2)
ax23.set_xlabel(r'Number Mode Radius ($\mu$m)', fontsize=axis_label_fontsize, labelpad=15)

ax13.text(0.04, 0.95, ' E ', fontsize=axis_label_fontsize, ha='left', va='top', transform=ax13.transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'})
ax23.text(0.04, 0.95, ' F ', fontsize=axis_label_fontsize, ha='left', va='top', transform=ax23.transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'})

fig.savefig('condensation_tests.png', bbox_inches='tight', dpi=200)
plt.show()

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
    shutil.rmtree(directory)

shutil.rmtree('processes')
