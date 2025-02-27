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
    destination = destination.replace('multipart', '')
    if os.path.isdir(destination):
        shutil.rmtree(destination)    
    destination = os.getcwd()+'/'+destination
    shutil.copytree(source, destination)


# %% do the runs
import UnitTests_visualization
from UnitTests_driver import simulate_gas_partitioning

trajectory_ensemble = simulate_gas_partitioning(N_scenarios=1,
        t_end=600,dt=1.0,updraft_velocity=0.0,
        Ddry=100e-9,sigma=1.0,Ntot=1e9, Npart=1,
        S0=0.85,P0=101325,T0=298,
        accom=1., verbosity=50,
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=['SO2'], gas_conc=[10.0],
        radius_scale='log',solver='ode15s',
        specdata_path='species_data/', mechanism_data_path='mechanisms/',
        condensation = True, 
        collisions = False, settling = False,
        cocondensation = True, chemistry = None, freezing = False)

FractionPlot, ConcPlot = UnitTests_visualization.plot_equilibrium_fractions(trajectory_ensemble[0], 'SO2', axis='time')
FractionPlot.savefig('FractionPlot_SO2.png', bbox_inches='tight', dpi=200)

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

