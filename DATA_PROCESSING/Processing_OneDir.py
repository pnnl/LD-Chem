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

def get_files(*dirs):
    # Get a list of sets, each containing file names in the respective directory
    file_sets = []
    for d in dirs:
        try:
            files = {f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))}
            file_sets.append(files)
        except FileNotFoundError:
            print(f"Directory not found: {d}")
            return []
        except PermissionError:
            print(f"Permission denied: {d}")
            return []
    
    # Use set intersection to find common files
    if not file_sets:
        return []
    
    common_files = set.intersection(*file_sets)
    return sorted(common_files)

# %% get the data

mass_thresholds={'IEPOX': [[0.3,0.5,0.1], ['IEPOX_OS','tetrol','tetrol_olig', 'IEPOX_OH_SOA']],
                'AS': [[0.5,0.7,0.1], ['SO4']],
                'AN': [[0.5,0.7,0.1], ['NO3']],
                'OC': [[0.5,0.7,0.1], ['OC']],
                'BC': [[0.5,0.7,0.1], ['BC']],
                'OIN': [[0.5,0.7,0.1], ['OIN']],
                'NH4': [[0.5,0.7,0.1], ['NH4']]}


traj_dirs = ['../entrainment_tests/time_steps/1s',
             '../entrainment_tests/time_steps/3s',
             '../entrainment_tests/time_steps/5s',
            '../entrainment_tests/time_steps/10s',
             '../entrainment_tests/time_steps/15s']
output_directory = 'time_steps_processed/15s'
dir = '../entrainment_tests/time_steps/15s'

if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

trajfiles = get_files(*traj_dirs)
print(dir+'/'+trajfiles[0])
trajectory = pickle.load(open(dir+'/'+trajfiles[0], 'rb'))


mass_fractions = {}
masses = {}
for ptype in mass_thresholds.keys():
    mass_fractions[ptype]=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
    masses[ptype]=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
    activations=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
    deactivations=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
    CRT=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
    cloud_state=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
    NumConcs=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
    dry_diameters=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
    diameters=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
    pHs=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
    activated_fraction=np.zeros((len(trajectory['particles']), len(trajfiles)))
    altitude=np.zeros((len(trajectory['particles']), len(trajfiles)))
    temperature=np.zeros((len(trajectory['particles']), len(trajfiles)))
    pressure=np.zeros((len(trajectory['particles']), len(trajfiles)))
    S=np.zeros((len(trajectory['particles']), len(trajfiles)))

for FileNumber, (file) in enumerate(trajfiles):
    steptime0 = time.time()
    trajectory = pickle.load(open(dir+'/'+file, 'rb'))

    NumConcs[:,:,FileNumber]=trajectory['particles'][:,:,np.where(trajectory['particle species']=='num conc')[0][0]]
    dry_diameters[:,:,FileNumber]=trajectory['particles'][:,:,np.where(trajectory['particle species']=='Ddry')[0][0]]
    diameters[:,:,FileNumber]=trajectory['particles'][:,:,np.where(trajectory['particle species']=='Dwet')[0][0]]
    water_volume=trajectory['particles'][:,:,np.where(trajectory['particle species']=='H2O')[0][0]]/1000.0
    moles_Hplus=trajectory['particles'][:,:,np.where(trajectory['particle species']=='H+')[0][0]]/1e-3
    pHs[:,:,FileNumber]=-1.0*np.log10(moles_Hplus/(1000*water_volume))
    activated_fraction[:,FileNumber]=trajectory['activated fraction']
    altitude[:,FileNumber]=trajectory['z']
    temperature[:,FileNumber]=trajectory['T']
    pressure[:,FileNumber]=trajectory['P']
    S[:,FileNumber]=trajectory['S']
    particle_dry_MassFracs, dry_species = Particle_MassFracs(trajectory['particles'], trajectory['particle species'],
                                                                 specdata_path='../species_data/')
    
    for ptype in mass_thresholds.keys():
        for species in mass_thresholds[ptype][1]:
            species_idx = np.where(dry_species==species)[0][0]
            mass_fractions[ptype][:,:,FileNumber]+=particle_dry_MassFracs[:,:,species_idx]#.reshape(-1)
            species_idx = np.where(trajectory['particle species']==species)[0][0]
            masses[ptype][:,:,FileNumber]+=trajectory['particles'][:,:,species_idx]#.reshape(-1)
    

    for TimeStep in range(len(trajectory['times'])):
        if trajectory['activated fraction'][TimeStep]>0:
            cloud_droplets = get_CD_status(trajectory['particles'][TimeStep,:,np.where(trajectory['particle species']=='Dwet')[0][0]],
                                                trajectory['particles'][TimeStep,:,np.where(trajectory['particle species']=='Ddry')[0][0]],
                                                trajectory['particles'][TimeStep,:,np.where(trajectory['particle species']=='kappa')[0][0]],
                                                trajectory['particles'][TimeStep,:,np.where(trajectory['particle species']=='num conc')[0][0]],
                                                trajectory['T'][TimeStep])
            idx=np.where(cloud_droplets>0)[0]
            cloud_state[TimeStep,idx,FileNumber]=1
            CRT[TimeStep+1:,idx,FileNumber]+=1
        
    
    for pNumber in range(len(trajectory['particles'][0])):
        switches=cloud_state[1:,pNumber,FileNumber]-cloud_state[:-1,pNumber,FileNumber]
        activation_events=np.where(switches>0)[0]
        deactivation_events=np.where(switches<0)[0]
        for TimeStep in activation_events:
            activations[TimeStep+1:,pNumber,FileNumber]+=1
        for TimeStep in deactivation_events:
            deactivations[TimeStep+1:,pNumber,FileNumber]+=1
    
    print(str(dir+'/'+file), trajectory['particles'][:,:,np.where(trajectory['particle species']=='num conc')[0][0]].shape, str(FileNumber+1)+'/'+str(len(trajfiles))+' -- '+str(round(time.time() - steptime0, 2))+ 's/it')

pickle.dump(cloud_state, open(output_directory+'/CloudState.pkl', 'wb'))
pickle.dump(CRT, open(output_directory+'/CRT.pkl', 'wb'))
pickle.dump(activations, open(output_directory+'/activations.pkl', 'wb'))
pickle.dump(deactivations, open(output_directory+'/deactivations.pkl', 'wb'))
pickle.dump(NumConcs, open(output_directory+'/NumConcs.pkl', 'wb'))
pickle.dump(mass_fractions, open(output_directory+'/MassFracs.pkl', 'wb'))
pickle.dump(masses, open(output_directory+'/Masses.pkl', 'wb'))
pickle.dump(dry_diameters, open(output_directory+'/DryDiameters.pkl', 'wb'))
pickle.dump(diameters, open(output_directory+'/Diameters.pkl', 'wb'))
pickle.dump(activated_fraction, open(output_directory+'/ActFraction.pkl', 'wb'))
pickle.dump(altitude, open(output_directory+'/Altitudes.pkl', 'wb'))
pickle.dump(temperature, open(output_directory+'/Temperatures.pkl', 'wb'))
pickle.dump(pressure, open(output_directory+'/Pressures.pkl', 'wb'))
pickle.dump(S, open(output_directory+'/Saturation.pkl', 'wb'))
pickle.dump(pHs, open(output_directory+'/pHs.pkl', 'wb'))


# %% remove files

for file in files:
    os.remove(file)
for directory in directories:
    shutil.rmtree(directory)

