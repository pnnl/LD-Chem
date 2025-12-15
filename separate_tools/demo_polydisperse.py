#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 08:21:45 2024

@author: fier887
"""
import scenario
import systems
import driver
import numpy as np
import copy
from dataclasses import replace
import visualization

monodisperse_population = scenario.make_monodisperse_population(100e-9, 1e9, ['Na','Cl'], [0.5, 0.5])

polydisperse_population = scenario.make_polydisperse_population(
    np.array([80e-9,100e-9]), [1e9,2e9], ['Na','Cl'],
    np.array([[0.5, 0.5],[0.2, 0.8]]))

# polydisperse_population = scenario.make_polydisperse_population(
#     np.array([80e-9]), [1e9], ['Na','Cl'],
#     np.array([[0.5, 0.5]]))


dt = 0.1
radius_scale = 'log'
force_cvode = False
sigma = 1.
accom = 0.1
verbosity = 50 # quite



parcel_scenario = scenario.create_parcel_scenario(
    aerosol_population = polydisperse_population,
    # Ddry=100e-9,sigma=1.0,Ntot=1e6,Npart=1,
    updraft_velocity=0.5,
    S0=0.99,P0=101325,T0=298,z_start=0.0,z_end=1000,
    # species_names=['NaaCl'],mass_fractions=np.array([1.]),
    dt=dt, specdata_path='../species_data/')


processes = systems.Processes(
    condensation = True, 
    collisions = False, 
    settling = False,
    cocondensation = False, 
    chemistry = False, 
    freezing = False)


for (one_trajectory_settings, start_time, end_time
      ) in zip(parcel_scenario.trajectories_settings,parcel_scenario.start_times,parcel_scenario.end_times):        
    ParcelState_0 = driver.get_initial_parcel(one_trajectory_settings, start_time)        
    Ntimes = int((end_time - start_time)/dt + 1)        
    t_eval = np.linspace(start_time, end_time, Ntimes)  
    parcel_states = [copy.deepcopy(ParcelState_0)]
    for (t1,t2) in zip(t_eval[:-1],t_eval[1:]):
        ParcelState_Next = systems.update_state(
            ParcelState_0, processes, dt,
            solver='ode15s',
            radius_scale=radius_scale,
            sigma=sigma, accom=accom, verbosity=verbosity)
        parcel_states.append(copy.deepcopy(ParcelState_Next))
        ParcelState_0=replace(ParcelState_Next)
    parcel_trajectory = systems.ParcelTrajectory(ts=t_eval, parcel_states=parcel_states)

visualization.plot_parcel_trajectories(parcel_trajectory)