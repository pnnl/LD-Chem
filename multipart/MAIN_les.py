#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Laura Fierce and Payton Beeler
"""
import numpy as np
from driver import simulate_les_trajectories
# import visualization
import matplotlib.pyplot as plt
# from initialization import splat_setup, optimize_splat_size_distribution
import pickle, sys

# %% LES trajectory

diameters=pickle.load(open('diameters', 'rb'))
num_concs=pickle.load(open('num_concs', 'rb'))
aero_spec_names=pickle.load(open('aero_spec_names', 'rb'))
aero_spec_fracs=pickle.load(open('aero_spec_fracs', 'rb'))
pHs=pickle.load(open('pHs', 'rb'))

gas_data=pickle.load(open('gas_data', 'rb'))
les_number=pickle.load(open('trajectory_number', 'rb'))

gas_names = ['SO2', 'O3', 'H2O2', 'NO2', 'IEPOX', 'OH']

les_output_file = sys.argv[1]+'/parcel_traces_'+les_number+'.pkl'

#with open('RUN_PROGRESS.out', 'w') as f:
print('Reading', les_output_file)#, file=f)

simulate_les_trajectories(les_output_file=les_output_file, output_path=str(sys.argv[2]),
        dt=3.0,diameters=diameters,N_concs=num_concs,
        pHs=pHs, accom=1.0, verbosity=50,
        radius_scale='log',solver='ode15s',
        species_names=aero_spec_names, mass_fractions=aero_spec_fracs,
        gas_names=gas_names, gas_data=gas_data,
        specdata_path='/Users/beel083/Library/CloudStorage/OneDrive-PNNL/Desktop/multipart_archived-main/species_data/',
        mechanism_data_path='/Users/beel083/Library/CloudStorage/OneDrive-PNNL/Desktop/multipart_archived-main/mechanisms/',
        condensation = True, collisions = False, settling = False,
        cocondensation = True,  entrainment = True, freezing = False, 
        gas_chemistry = True, aq_chemistry = ['sulfate', 'nitrate'],
        relaxation_time = 24.475, write_every=60.0)


# traj=pickle.load(open('trajectory_000493.pkl', 'rb'))
# species='IEPOX'

# y = 1e9*np.sum(traj['particles'][:,:,np.where(traj['particle species']==species)[0][0]]*traj['particles'][:,:,np.where(traj['particle species']=='num conc')[0][0]], axis=1)
# y = traj['gases'][:,np.where(traj['gas species']==species)[0][0]]
# y = 1e6*traj['particles'][:,:,np.where(traj['particle species']=='Dwet')[0][0]]
# plt.plot(traj['times'], 1e6*traj['particles'][:,:,np.where(traj['particle species']==species)[0][0]])
# plt.plot(traj['times'], traj['gases'][:,np.where(traj['gas species']==species)[0][0]])
# plt.plot(traj['times'], traj['P'])

# plt.plot(traj['times'], y)

# plt.ylabel(species)
# plt.xlabel('time')
# plt.show()
# print(np.min(y),
      # np.max(y))



