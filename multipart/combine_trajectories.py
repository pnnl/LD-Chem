#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:43:58 2024

@author: beel083
"""

import os, tqdm, gc, pickle, time, sys
import visualization, HISCALE_data_processing
# from systems import ParcelState, ParcelTrajectory
# from particles import ParticlePopulation
import matplotlib.pyplot as plt
import numpy as np
# from driver import simulate_les_trajectories

data_directory='../output_LesFile_995'
combined_traj_output='../output_LesFile_995/trajectory.pkl'

N_files=10
'''
aq_order=np.zeros(0)
aero_datafile = '../species_data/aero_data.dat'
with open(aero_datafile) as data_file:
    for line in data_file:
        try:
            name_in_file,density,ions_in_solution,molar_mass,kappa = line.split()
            aq_order=np.append(aq_order, name_in_file)
        except:
            x=1
aq_order=np.append(aq_order, 'num conc')
aq_order=np.append(aq_order, 'Ddry')
aq_order=np.append(aq_order, 'Dwet')
aq_order=np.append(aq_order, 'kappa')
        
gas_order=np.zeros(0)
gas_datafile = '../species_data/gas_data.dat'
with open(gas_datafile) as data_file:
    for line in data_file:
        try:
            name_in_file,alpha,molar_mass,H0,H_exp = line.split()
            gas_order=np.append(gas_order, name_in_file)
        except:
            x=1

num_particles=0   
print()     
print('setting up the arrays...')
pbar = tqdm.tqdm(total = N_files)
for file in range(N_files):#files:
    f = open(data_directory+'/run_'+str(file+1)+'.pkl', 'rb')
    trajectory_temp = pickle.load(f)
    num_parcel_states=len(trajectory_temp[0].parcel_states)
    num_particles+=len(trajectory_temp[0].parcel_states[0].particle_population.particles)
    ts_all=trajectory_temp[0].ts
    pbar.update(1)
pbar.close()
print()

combined_trajectory={}
combined_trajectory['times']=ts_all
combined_trajectory['particles']=np.zeros((num_parcel_states, num_particles, len(aq_order)))
combined_trajectory['particle species']=aq_order
combined_trajectory['gases']=np.zeros((num_parcel_states, len(gas_order)))
combined_trajectory['gas species']=gas_order
combined_trajectory['x']=np.zeros((num_parcel_states))
combined_trajectory['y']=np.zeros((num_parcel_states))
combined_trajectory['z']=np.zeros((num_parcel_states))
combined_trajectory['S']=np.zeros((num_parcel_states))
combined_trajectory['T']=np.zeros((num_parcel_states))
combined_trajectory['P']=np.zeros((num_parcel_states))
combined_trajectory['activated fraction']=np.zeros((num_parcel_states))

print('saving values...')
pbar = tqdm.tqdm(total = N_files)
for file in range(N_files):
    f = open(data_directory+'/run_'+str(file+1)+'.pkl', 'rb')
    trajectory = pickle.load(f)
    trajectory = trajectory[0]
    
    for pState in range(num_parcel_states):
        
        combined_trajectory['x'][pState]=trajectory.parcel_states[pState].x
        combined_trajectory['y'][pState]=trajectory.parcel_states[pState].y
        combined_trajectory['z'][pState]=trajectory.parcel_states[pState].z
        combined_trajectory['S'][pState]=trajectory.parcel_states[pState].S
        combined_trajectory['T'][pState]=trajectory.parcel_states[pState].T
        combined_trajectory['P'][pState]=trajectory.parcel_states[pState].P
        combined_trajectory['activated fraction'][pState]=trajectory.parcel_states[pState].get_activated_fraction()
        
        particles = trajectory.parcel_states[pState].particle_population.particles
        num_concs = trajectory.parcel_states[pState].particle_population.num_concs
        for ii, (particle, num_conc) in enumerate(zip(trajectory.parcel_states[pState].particle_population.particles, trajectory.parcel_states[pState].particle_population.num_concs)):
            pNumber=int(file*float(len(particles))+ii)
            
            Ddry=particle.get_Ddry()
            traj_idx = np.where(aq_order=='Ddry')[0][0]
            combined_trajectory['particles'][pState, pNumber, traj_idx]=Ddry
            
            Dwet=particle.get_Dwet()
            traj_idx = np.where(aq_order=='Dwet')[0][0]
            combined_trajectory['particles'][pState, pNumber, traj_idx]=Dwet
            
            for species in aq_order:
                if species=='num conc':
                    traj_idx = np.where(aq_order==species)[0][0]
                    combined_trajectory['particles'][pState, pNumber, traj_idx]=num_conc
                elif species=='kappa':
                    traj_idx = np.where(aq_order==species)[0][0]
                    combined_trajectory['particles'][pState, pNumber, traj_idx]=particle.get_tkappa()
                else:
                    traj_idx = np.where(aq_order==species)[0][0]
                    particle_idx = particle.get_species_idx(species)
                    if particle_idx!=None:
                        combined_trajectory['particles'][pState, pNumber, traj_idx]=particle.masses[particle_idx]    
                        
        gases = trajectory.parcel_states[pState].TraceGas_population.gases
        gas_concs = trajectory.parcel_states[pState].TraceGas_population.concs
        for species in gas_order:
            traj_idx = np.where(gas_order==species)[0][0]
            subtraj_idx = trajectory.parcel_states[pState].TraceGas_population.get_species_idx(species)
            if subtraj_idx!=None:
                combined_trajectory['gases'][pState, traj_idx]=trajectory.parcel_states[pState].TraceGas_population.concs[subtraj_idx]
    
    pbar.update(1)
pbar.close()
print()
time0=time.time()
f=open(combined_traj_output, 'wb')
pickle.dump(combined_trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)
print('Writing time:', round(time.time() - time0, 2), 'seconds')
'''
time0=time.time()
f=open(combined_traj_output, 'rb')
traj=pickle.load(f)
print('Reading time:', round(time.time() - time0, 2), 'seconds')
print()

splat_file='../datasets/HISCALE_data_0425/Splat_Composition_25-Apr-2016.txt'
size_distribution_file='../datasets/HISCALE_data_0425/BEASD_G1_20160425155810_R2_HISCALE_001s.txt'
ams_file='../datasets/HISCALE_data_0425/HiScaleAMS_G1_20160425_R0.txt'
aimms_file='../datasets/HISCALE_data_0425/AIMMS20_G1_20160425155810_R2_HISCALE020h.txt'

# these NEED to be the same as in SPLAT_initialization.py
splat_species = {'BC': ['soot'],
                  'OIN': ['Dust'],
                  'AS': ['sulfate_nitrate_org'],
                  'AN': ['nitrate_amine_org'], 
                  'OC': ['org28', 'org30_43', 'BB_SOA', 'org_amines', 'BB', 'pyridine'], 
                  'IEPOX': ['IEPOX_SOA']}

mass_fractions={'IEPOX': [[0.5,0.75,0.1], ['IEPOX_OS', 'tetrol', 'tetrol_olig']],
                'AS': [[0.3,0.75,0.1], ['SO4', 'NH4']],
                'AN': [[0.5,0.75,0.1], ['NO3', 'NH4']], 
                'OC': [[0.5,0.75,0.1], ['OC']], 
                'BC': [[0.5,0.75,0.1], ['BC']],
                'OIN': [[0.5,0.75,0.1], ['OIN']]}

# fig = HISCALE_data_processing.plot_diameters(traj, axis='time')
# fig.savefig(data_directory+'/Diameters.png', dpi=200, bbox_inches='tight')

# IndMass_fig, TotalMass_fig = HISCALE_data_processing.plot_aq_species(traj, ['IEPOX_OS', 'tetrol', 'tetrol_olig'], axis='time')
# TotalMass_fig.savefig(data_directory+'/IEPOX_SOA_mass.png', dpi=200, bbox_inches='tight')

# fig=HISCALE_data_processing.plot_activated_fraction(traj)
# fig.savefig(data_directory+'/ActivatedFraction.png', dpi=200, bbox_inches='tight')

# fig = HISCALE_data_processing.initial_SizeDist(traj, size_distribution_file,
#                                           splat_species, mass_fractions, start_time=960, end_time=59160)
# fig.savefig(data_directory+'/initial_SizeDist.png', dpi=200, bbox_inches='tight')

# fig = HISCALE_data_processing.cloud_composition(traj, splat_file, splat_species, size_distribution_file,
#                       mass_fractions, resolution=60)
# fig.savefig(data_directory+'/cloud_droplet_comp.png', dpi=200, bbox_inches='tight')

# fig = HISCALE_data_processing.initial_N_MassFracs(traj, ams_file, splat_file,
#                                             splat_species, mass_fractions, start_time=960, 
#                                             end_time=59160)
# fig.savefig(data_directory+'/initial_N_MassFracs.png', dpi=200, bbox_inches='tight')

# fig=HISCALE_data_processing.ModelComposition(traj, mass_fractions, splat_species, resolution=120)
# fig.savefig(data_directory+'/ModelComposition.png', dpi=200, bbox_inches='tight')

# fig=HISCALE_data_processing.MassFraction_TimeSeries(traj, mass_fractions, 'IEPOX', splat_species, resolution=60)
# fig.savefig(data_directory+'/mass_fracs.png', dpi=200, bbox_inches='tight')

fig=HISCALE_data_processing.VerticalComposition(traj, mass_fractions, splat_species, 
                                                splat_file, aimms_file, bins=15)
fig.savefig(data_directory+'/vertical_composition.png', dpi=200, bbox_inches='tight')
plt.show()



