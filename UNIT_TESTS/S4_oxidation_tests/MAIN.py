#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:06:10 2024

@author: beel083
"""
# %% 
# copy the necessary modules to the UNIT_TESTS directory
# probably need a different way to do this but I don't 
# want to mess with sys.path

import shutil, os, sys
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
    destination = destination.replace('multipart', '')
    if os.path.isdir(destination):
        shutil.rmtree(destination)    
    destination = os.getcwd()+'/'+destination
    shutil.copytree(source, destination)


from UnitTests_driver import simulate_sulfate_oxidation

# %% set up the plot

axis_label_fontsize=13
axis_tick_fontsize=11
legend_fontsize=12
markersize=7

fig, ax = plt.subplots(1, 1, figsize=(1.0*6.4, 1.0*4.8), constrained_layout=False)
ax.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
ax.tick_params(which="major", axis="both", length=6)
ax.tick_params(which="minor", axis="both", length=4)
ax.grid(which='major', color='grey', alpha=0.4, linewidth=1)

pHs=np.linspace(2, 8, 10)

# %% do the ozone runs

dSO4_dt = simulate_sulfate_oxidation(pHs,
        t_end=120.0, dt=0.5, updraft_velocity=0.0,
        Ddry=96.92e-9, sigma=1.0, Ntot=1e6, Npart=1,
        S0=0.85, P0=101325, T0=271,pH0=10.0,
        accom=1., verbosity=50,
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=['SO2', 'O3'], gas_conc=[40.0, 1.0],
        radius_scale='lin',solver='CVODE',
        specdata_path='species_data/', mechanism_data_path='mechanisms/',
        condensation = True, 
        collisions = False, settling = False,
        cocondensation = True, chemistry = ['sulfate'], freezing = False) # kg/m^3/s


published_data = pd.read_excel('published_data.xls', sheet_name='O3')
ax.plot(published_data['pH'], published_data['SO4 rate'], 'bo', markersize=markersize, label=r'O$_3$')
ax.plot(pHs, dSO4_dt*1e9*3600, '-b')


# %% do the H2O2 runs

dSO4_dt = simulate_sulfate_oxidation(pHs,
        t_end=120.0, dt=0.5, updraft_velocity=0.0,
        Ddry=96.6e-9, sigma=1.0, Ntot=1e6, Npart=1,
        S0=0.85, P0=101325, T0=271,
        accom=1., verbosity=50,
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=['SO2', 'H2O2'], gas_conc=[40.0, 0.1],
        radius_scale='lin',solver='CVODE',
        specdata_path='species_data/', mechanism_data_path='mechanisms/',
        condensation = True, 
        collisions = False, settling = False,
        cocondensation = True, chemistry = ['sulfate'], freezing = False) # kg/m^3/s

published_data = pd.read_excel('published_data.xls', sheet_name='H2O2')
ax.plot(published_data['pH'], published_data['SO4 rate'], 'rs', markersize=markersize, label=r'H$_2$O$_2$')
ax.plot(pHs, dSO4_dt*1e9*3600, '-r')

# %% do the NO2 runs

dSO4_dt = simulate_sulfate_oxidation(pHs,
        t_end=120.0, dt=0.5, updraft_velocity=0.0,
        Ddry=96.6e-9, sigma=1.0, Ntot=1e6, Npart=1,
        S0=0.85, P0=101325, T0=271,
        accom=1., verbosity=50,
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=['SO2', 'NO2'], gas_conc=[40.0, 66.0],
        radius_scale='lin',solver='CVODE',
        specdata_path='species_data/', mechanism_data_path='mechanisms/',
        condensation = True, 
        collisions = False, settling = False,
        cocondensation = True, chemistry = ['sulfate'], freezing = False) # kg/m^3/s

published_data = pd.read_excel('published_data.xls', sheet_name='NO2')
ax.plot(published_data['pH'], published_data['SO4 rate'], 'g^', markersize=markersize, label=r'NO$_2$')
ax.plot(pHs, dSO4_dt*1e9*3600, '-g')

# %% do the HNO2 runs

dSO4_dt = simulate_sulfate_oxidation(pHs,
        t_end=120.0, dt=0.5, updraft_velocity=0.0,
        Ddry=96.6e-9, sigma=1.0, Ntot=1e6, Npart=1,
        S0=0.85, P0=101325, T0=271,
        accom=1., verbosity=50,
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=['SO2', 'HNO2'], gas_conc=[40.0, 9.0],
        radius_scale='lin',solver='CVODE',
        specdata_path='species_data/', mechanism_data_path='mechanisms/',
        condensation = True, 
        collisions = False, settling = False,
        cocondensation = True, chemistry = ['sulfate'], freezing = False) # kg/m^3/s

published_data = pd.read_excel('published_data.xls', sheet_name='HONO')
ax.plot(published_data['pH'], published_data['SO4 rate'], 'v', mfc='c', mec='c', markersize=markersize, label=r'HNO$_2$')
ax.plot(pHs, dSO4_dt*1e9*3600, '-', color='c')
  

# %% do the O2+TMI runs

dSO4_dt = simulate_sulfate_oxidation(pHs,
        t_end=120.0, dt=0.5, updraft_velocity=0.0,
        Ddry=96.6e-9, sigma=1.0, Ntot=1e6, Npart=1,
        S0=0.85, P0=101325, T0=271,
        accom=1., verbosity=50,
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=['SO2', 'O2'], gas_conc=[40.0, 20.0],
        radius_scale='lin',solver='CVODE',
        specdata_path='species_data/', mechanism_data_path='mechanisms/',
        condensation = True, 
        collisions = False, settling = False,
        cocondensation = True, chemistry = ['sulfate'], freezing = False) # kg/m^3/s


published_data = pd.read_excel('published_data.xls', sheet_name='O2+TMI')
ax.plot(published_data['pH'], published_data['SO4 rate'], '<', mfc='gold', mec='gold', markersize=markersize, label=r'O$_2$+TMI')
ax.plot(pHs, dSO4_dt*1e9*3600, '-', color='gold')  

# %% fix the plot

ax.set_xlim(2, 8)
ax.legend(ncol=3, fontsize=legend_fontsize, frameon=False, loc='center', bbox_to_anchor=(0.5, 1.125))
ax.set_ylim(1e-5, 1e5)
ax.set_yscale('log')
ax.set_xlabel('Droplet pH', fontsize=axis_label_fontsize, labelpad=15)
ax.set_ylabel(r'SO$_4^{2-}$ produxtion rate ($\mu$g m$^3$ hr$^{-1}$)', fontsize=axis_label_fontsize, labelpad=15)
fig.savefig('S4_oxidation.png', bbox_inches='tight', dpi=200)


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

