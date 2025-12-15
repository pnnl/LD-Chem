#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 09:53:00 2025

@author: beel083
"""

# %% import files
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.font_manager as font_manager
import os, shutil, time
from numba.typed import Dict
from numba import types
multipart_directory='../multipart/'
files = ['particles.py', 'HISCALE_data_processing.py', 'SPLAT_initialization.py', 'scenario.py',
          'TraceGases.py', 'Reactions.py', 'constants.py', 'aerosol_species.py', 'utilities.py',
          'systems.py']
directories=['processes']
for file in files:
    shutil.copy(multipart_directory+file, os.getcwd())
for directory in directories:
    shutil.copytree(multipart_directory+directory, os.getcwd()+'/'+directory)
from HISCALE_data_processing import classify, get_CD_status, Particle_MassFracs



# %% get the data

mass_thresholds={'IEPOX': [[0.3,0.5,0.1], ['IEPOX_OS','tetrol','tetrol_olig', 'IEPOX_OH_SOA']],
                'AS': [[0.5,0.7,0.1], ['SO4']],
                'AN': [[0.5,0.7,0.1], ['NO3']],
                'OC': [[0.5,0.7,0.1], ['OC']],
                'BC': [[0.5,0.7,0.1], ['BC']],
                'OIN': [[0.5,0.7,0.1], ['OIN']],
                'NH4': [[0.5,0.7,0.1], ['NH4']]}


directory = '/Users/beel083/Library/CloudStorage/OneDrive-PNNL/Desktop/multipart_archived-main/'


filename = directory + 'entrainment_tests/trajectory_0/trajectory_000493_1e1_10s.pkl'
trajectory = pickle.load(open(filename, 'rb'))

NumConcs=trajectory['particles'][:,:,np.where(trajectory['particle species']=='num conc')[0][0]]
dry_diameters=trajectory['particles'][:,:,np.where(trajectory['particle species']=='Ddry')[0][0]]
diameters=trajectory['particles'][:,:,np.where(trajectory['particle species']=='Dwet')[0][0]]
water_volume=trajectory['particles'][:,:,np.where(trajectory['particle species']=='H2O')[0][0]]/1000.0
moles_Hplus=trajectory['particles'][:,:,np.where(trajectory['particle species']=='H+')[0][0]]/1e-3
pHs=-1.0*np.log10(moles_Hplus/(1000*water_volume))
activated_fraction=trajectory['activated fraction']
altitude=trajectory['z']
temperature=trajectory['T']
pressure=trajectory['P']
S=trajectory['S']
particle_dry_MassFracs, dry_species = Particle_MassFracs(trajectory['particles'], trajectory['particle species'],
                                                              specdata_path='../species_data/')
mass_fractions = {}
masses = {}
for ptype in mass_thresholds.keys():
    mass_fractions[ptype]=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0])))
    masses[ptype]=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0])))
activations=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0])))
deactivations=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0])))
CRT=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0])))
cloud_state=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0])))
   
for ptype in mass_thresholds.keys():
    for species in mass_thresholds[ptype][1]:
        species_idx = np.where(dry_species==species)[0][0]
        mass_fractions[ptype]+=particle_dry_MassFracs[:,:,species_idx]#.reshape(-1)
        species_idx = np.where(trajectory['particle species']==species)[0][0]
        masses[ptype]+=trajectory['particles'][:,:,species_idx]#.reshape(-1)


for TimeStep in range(len(trajectory['times'])):
    if trajectory['activated fraction'][TimeStep]>0:
        cloud_droplets = get_CD_status(trajectory['particles'][TimeStep,:,np.where(trajectory['particle species']=='Dwet')[0][0]],
                                            trajectory['particles'][TimeStep,:,np.where(trajectory['particle species']=='Ddry')[0][0]],
                                            trajectory['particles'][TimeStep,:,np.where(trajectory['particle species']=='kappa')[0][0]],
                                            trajectory['particles'][TimeStep,:,np.where(trajectory['particle species']=='num conc')[0][0]],
                                            trajectory['T'][TimeStep])
        idx=np.where(cloud_droplets>0)[0]
        cloud_state[TimeStep,idx]=1
        CRT[TimeStep+1:,idx]+=1
    

for pNumber in range(len(trajectory['particles'][0])):
    switches=cloud_state[1:,pNumber]-cloud_state[:-1,pNumber]
    activation_events=np.where(switches>0)[0]
    deactivation_events=np.where(switches<0)[0]
    for TimeStep in activation_events:
        activations[TimeStep+1:,pNumber]+=1
    for TimeStep in deactivation_events:
        deactivations[TimeStep+1:,pNumber]+=1

pickle.dump(cloud_state, open('CloudState.pkl', 'wb'))
pickle.dump(CRT, open('CRT.pkl', 'wb'))
pickle.dump(activations, open('activations.pkl', 'wb'))
pickle.dump(deactivations, open('deactivations.pkl', 'wb'))
pickle.dump(NumConcs, open('NumConcs.pkl', 'wb'))
pickle.dump(mass_fractions, open('MassFracs.pkl', 'wb'))
pickle.dump(masses, open('Masses.pkl', 'wb'))
pickle.dump(dry_diameters, open('DryDiameters.pkl', 'wb'))
pickle.dump(diameters, open('Diameters.pkl', 'wb'))
pickle.dump(activated_fraction, open('ActFraction.pkl', 'wb'))
pickle.dump(altitude, open('Altitudes.pkl', 'wb'))
pickle.dump(temperature, open('Temperatures.pkl', 'wb'))
pickle.dump(pressure, open('Pressures.pkl', 'wb'))
pickle.dump(S, open('Saturation.pkl', 'wb'))
pickle.dump(pHs, open('pHs.pkl', 'wb'))


# %% remove files

for file in files:
    os.remove(file)
for directory in directories:
    shutil.rmtree(directory)
