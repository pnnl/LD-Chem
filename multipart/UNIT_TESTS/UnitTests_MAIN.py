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

files = ['particles.py', 'constants.py', 'scenario.py', 'aerosol_species.py',
         'utilities.py', 'systems.py', 'driver.py', 'visualization.py', 
         'TraceGases.py', 'Reactions.py']

for file in files:
    source = '../'+file
    destination = os.getcwd()+'/'+file
    shutil.copy(source, destination)

directories = ['../processes', '../../species_data', '../../mechanisms']
for directory in directories:
    source = directory
    destination = source.replace('.', '')
    destination = destination.replace('/', '')
    if os.path.isdir(destination):
        shutil.rmtree(destination)    
    destination = os.getcwd()+'/'+destination
    shutil.copytree(source, destination)


# %% do the runs

from UnitTests_driver import simulate_chemistry_tests
import UnitTests_visualization

trajectory_ensemble = simulate_chemistry_tests(N_scenarios=1,
        t_end=60,dt=0.1,
        Ddry=100e-9,sigma=1.0,Ntot=1e6, Npart=1,
        S0=0.85,P0=101325,T0=298,pH0=-0.6,
        accom=1., verbosity=50,
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=['SO2','O3'], gas_conc=[5.0,1.0],
        radius_scale='log',solver='CVODE',
        specdata_path='species_data/', mechanism_data_path='mechanisms/',
        condensation = True, 
        collisions = False, settling = False,
        cocondensation = True, chemistry = None, freezing = False)

UnitTests_visualization.plot_equilibrium_fractions(trajectory_ensemble[0], 'H+', axis='time')

# %%
# delete all the modules that got moved to UNIT_TESTS 
# directory

for file in files:
    os.remove(file)
    
for directory in directories:
    directory = directory.replace('.', '')
    directory = directory.replace('/', '')
    shutil.rmtree(directory)

