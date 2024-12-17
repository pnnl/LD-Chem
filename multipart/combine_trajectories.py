#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:43:58 2024

@author: beel083
"""

import pickle, os, tqdm
import visualization
from systems import ParcelState, ParcelTrajectory
from particles import ParticlePopulation
import matplotlib.pyplot as plt
# from driver import simulate_les_trajectories

data_directory='../output_run2'
combined_traj_output='../output_run1/trajectory.pkl'

files=os.listdir(data_directory)

trajectories_all = []
pbar = tqdm.tqdm(total = len(files))
for file in files:
    f=open(data_directory+'/'+file, 'rb')
    trajectory = pickle.load(f)
    trajectories_all.append(trajectory[0])
    pbar.update(1)
pbar.close()

num_parcel_states=len(trajectories_all[0].parcel_states)
ts_all=trajectories_all[0].ts
parcel_states_all=[]

# start loop through parcel_states
for ii in range(num_parcel_states):
    
    trajectory = trajectories_all[0]
    
    particles = trajectory.parcel_states[ii].particle_population.particles
    num_concs = trajectory.parcel_states[ii].particle_population.num_concs
    ids = trajectory.parcel_states[ii].particle_population.ids
    
    for jj in range(1, len(trajectories_all)):
        trajectory = trajectories_all[jj]
    
        for particle, num_conc, ID in zip(trajectory.parcel_states[ii].particle_population.particles, trajectory.parcel_states[ii].particle_population.num_concs, trajectory.parcel_states[ii].particle_population.ids):
            particles.append(particle)
            num_concs.append(num_conc)
            ids.append(ID)
    
    
    particle_population=ParticlePopulation(particles=particles, num_concs=num_concs, ids=ids)
    combined_parcel_state=ParcelState(x=trajectory.parcel_states[ii].x,
                                      y=trajectory.parcel_states[ii].y,
                                      z=trajectory.parcel_states[ii].z,
                                      u=trajectory.parcel_states[ii].u,
                                      v=trajectory.parcel_states[ii].v,
                                      w=trajectory.parcel_states[ii].w,
                                      S=trajectory.parcel_states[ii].S,
                                      T=trajectory.parcel_states[ii].T,
                                      P=trajectory.parcel_states[ii].P,
                                      particle_population=particle_population,
                                      TraceGas_population=trajectory.parcel_states[ii].TraceGas_population)  
    
    parcel_states_all.append(combined_parcel_state)
    


combined_trajectory=ParcelTrajectory(ts=ts_all, parcel_states=parcel_states_all)

f=open(combined_traj_output, 'wb')
pickle.dump(combined_trajectory, f)


f=open(combined_traj_output, 'rb')
traj=pickle.load(f)
visualization.plot_diameters(traj, axis='time')
fig=visualization.plot_trajectory_values(traj, resolution=30)
plt.show()