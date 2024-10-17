#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:32:34 2024

@author: beel083
"""
import numpy as np
from UnitTests_scenario import create_constant_parcel
from systems import Processes
from driver import get_initial_parcel
import copy, tqdm, time, sys
from systems import update_state, ParcelTrajectory
from dataclasses import replace


# this module simulates the same scenario as Seinfeld and Pandis
# figure 7.21 and 7.22 (closed system) 
def simulate_chemistry_tests(N_scenarios=1,
        t_end=600,dt=1.,
        Ddry=100e-9,sigma=1.0,Ntot=1e6, Npart=1,
        S0=-0.15,P0=101325,T0=298,pH0=7.0,
        accom=1., verbosity=50,
        radius_scale='lin',solver='CVODE',
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=None, gas_conc=None,
        specdata_path='../species_data/', mechanism_data_path='../mechanisms/',
        condensation = True, collisions = False, settling = False,
        cocondensation = False, chemistry = None, freezing = False):
        
    trajectory_ensemble = []

    scenario = create_constant_parcel(
                aerosol_population = None,
                Ddry=Ddry,sigma=sigma,Ntot=Ntot,Npart=Npart,updraft_velocity=0.0,
                S0=S0,P0=P0,T0=T0,pH0=pH0,t_end=t_end,
                species_names=species_names,mass_fractions=mass_fractions,
                gas_names=gas_names, gas_conc=gas_conc,
                dt=dt, specdata_path=specdata_path, mechanism_data_path=mechanism_data_path,
                chemistry=chemistry, cocondensation=cocondensation) 
    
    # scale the number concentration so that there is 1e-6 m^3 water per m^3 air
    aerosol_population = scenario.trajectories_settings[0].population0
    wL = 0
    for ii,(particle,num_conc) in enumerate(zip(aerosol_population.particles,aerosol_population.num_concs)):
        water_mass = particle.masses[particle.idx_h2o]
        wL += num_conc*water_mass/particle.get_rho_w() # m^3 water per m^3 air
    mult = 1e-6/wL
    for i in range(len(scenario.trajectories_settings[0].population0.num_concs)):
        scenario.trajectories_settings[0].population0.num_concs[i] *= mult
    
    processes = Processes(
        condensation = condensation, 
        collisions = collisions, 
        settling = settling,
        cocondensation = cocondensation, 
        chemistry = chemistry, 
        freezing = freezing)  
    
    print()
    for (one_trajectory_settings, start_time, end_time
          ) in zip(scenario.trajectories_settings,scenario.start_times,scenario.end_times):        
        
        runtime0 = time.time()
        print('Running trajectory', str(i+1)+',', Npart,'particles...')        
        ParcelState_0 = get_initial_parcel(one_trajectory_settings, start_time)
        Ntimes = int((end_time - start_time)/dt)        
        t_eval = np.linspace(start_time, end_time, Ntimes) 
        parcel_states = [copy.deepcopy(ParcelState_0)]
        pbar = tqdm.tqdm(total = len(t_eval))
        for (t1,t2) in zip(t_eval[:-1],t_eval[1:]):   
            ParcelState_Next = update_state(
                ParcelState_0, processes, dt, 
                radius_scale=radius_scale,solver=solver,
                sigma=sigma, accom=accom, verbosity=verbosity,
                mechanism_data_path=mechanism_data_path)
            parcel_states.append(copy.deepcopy(ParcelState_Next))
            ParcelState_0=replace(ParcelState_Next)            
    
            # print()
            # print('here')
            # print()
            # sys.exit()
            
            pbar.update(1)

        pbar.close()
        print('Solving time:', round(time.time() - runtime0, 2), 'seconds')
        print()
        
        parcel_trajectory = ParcelTrajectory(ts=t_eval, parcel_states=parcel_states)
        trajectory_ensemble.append(parcel_trajectory)
        
    return trajectory_ensemble