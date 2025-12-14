#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:32:34 2024

@author: beel083
"""
import numpy as np
from UnitTests_scenario import create_constant_parcel
from scenario import create_parcel_scenario
from systems import Processes
from driver import get_initial_parcel
import copy, tqdm, time, sys, os, shutil
from systems import update_state, ParcelTrajectory
from dataclasses import replace
from scenario import make_AqReactions, make_GasReactions
from processes import aqueous_chemistry as AC
from processes import wall_losses
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
from scipy.integrate import ode
from Reactions import AqueousReactions
from numba.typed import Dict
from numba import types
from write_files import write_original, overwrite


# this module simulates the same scenarios as previously
# published values for similar adiabatic parcel models
def simulate_condensation_test(N_scenarios=1,
        z_start=0.,z_end=1000.,dt=1.,
        Ddry=100e-9,sigma=1.0,Ntot=1e6, Npart=1,
        updraft_velocity=0.5,S0=-0.15,P0=101325,T0=298,
        pH0=7.0,accom=1., verbosity=50,
        radius_scale='lin',solver='CVODE',
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        output_path=None,
        gas_names=None, gas_conc=None,
        specdata_path='../species_data/', mechanism_data_path='../mechanisms/',
        condensation = True, collisions = False, settling = False,
        cocondensation = False, aq_chemistry = None, freezing = False,
        gas_chemistry = False, entrainment = False, relaxation_time = None, 
        write_every=1.0):
        
    if output_path:
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)
    else:
        output_path=os.getcwd()
    
    
    for i in range(N_scenarios):
        
        try:
            V=updraft_velocity[i]
        except:
            V=updraft_velocity
        try:
            N_tot=Ntot[i]
        except:
            N_tot=Ntot
        try:
            D_dry=Ddry[i]
        except:
            D_dry=Ddry
         
        scenario = create_parcel_scenario(
                Ddry=D_dry,sigma=sigma,Ntot=N_tot,Npart=Npart,
                updraft_velocity=V,
                S0=S0,P0=P0,T0=T0,pH0=pH0,z_start=z_start,z_end=z_end,
                species_names=species_names,mass_fractions= mass_fractions,
                gas_names=gas_names, gas_conc=gas_conc,
                dt=dt, specdata_path=specdata_path,
                mechanism_data_path=mechanism_data_path,
                aq_chemistry=aq_chemistry, gas_chemistry=gas_chemistry,
                cocondensation=cocondensation)             
        
        if gas_chemistry:
            gas_reactions = make_GasReactions(mechanism_data_path=mechanism_data_path)
        else:
            gas_reactions = None
        
        if aq_chemistry:
            aq_reactions = make_AqReactions(aq_chemistry=aq_chemistry, mechanism_data_path=mechanism_data_path)
        else:
            aq_reactions = None
    
        processes = Processes(
            condensation = condensation, 
            collisions = collisions, 
            settling = settling,
            cocondensation = cocondensation, 
            aq_chemistry = aq_chemistry, 
            gas_chemistry=gas_chemistry,
            freezing = freezing,
            entrainment = entrainment)   
        
        print()
        for (one_trajectory_settings, start_time, end_time
              ) in zip(scenario.trajectories_settings,scenario.start_times,scenario.end_times):        
            
            runtime0 = time.time()
            print('Running trajectory', str(i+1)+',', Npart,'particles...')        
            ParcelState_0 = get_initial_parcel(one_trajectory_settings, start_time)
            Ntimes = int((end_time - start_time)/dt)        
            t_eval = np.linspace(start_time, end_time, Ntimes) 
            breaker=False
            output_filename=output_path+'/trajectory_'+str(i)+'.pkl'
            write_original(t_eval[0], ParcelState_0, output_filename, specdata_path=specdata_path)
            last_written=t_eval[0]
            
            pbar = tqdm.tqdm(total = len(t_eval))
            for (t1,t2) in zip(t_eval[:-1],t_eval[1:]): 
                
                # steptime0=time.time()
                
                ParcelState_Next = update_state(t1, t2,
                    ParcelState_0, processes, dt,
                    radius_scale=radius_scale,solver=solver,
                    accom=accom, verbosity=verbosity,
                    mechanism_data_path=mechanism_data_path,
                    aq_reactions=aq_reactions, gas_reactions=gas_reactions,
                    rtol=1e-4, atol=1e-8) # 1e-7, 1e-14
                
                # adjust the number concentration based on the new temperature and pressure
                Ns=np.array(ParcelState_0.particle_population.num_concs)
                Ns*=((ParcelState_Next.P*ParcelState_0.T)/(ParcelState_0.P*ParcelState_Next.T))
                ParcelState_Next.particle_population.num_concs=list(Ns)      
                
                # print timestep and time for timestep
                # counter+=1
                #with open('RUN_PROGRESS.out', 'a') as f:
                # print(str(counter)+'/'+str(len(t_eval))+' -- '+str(round(time.time() - steptime0, 2))+ 's/it')#, file=f)

                # check for NaNs
                total_mass = []
                for particle, num_conc in zip(ParcelState_Next.particle_population.particles, ParcelState_Next.particle_population.num_concs):
                    total_mass.append(num_conc*np.sum(particle.masses))
                
                #with open('RUN_PROGRESS.out', 'a') as f:
                # print(str(ParcelState_Next.S)+' '+str(1e9*np.sum(np.array(total_mass))))#, file=f)
                
                # utilities.water_mole_balance(original_ParcelState, ParcelState_Next)
                # print('')#, file=f)
                   
                # kill the program if there is a NaN
                if np.isnan(np.sum(total_mass)):
                    print('ERROR (NaNs)')
                    sys.exit()
                
                # stop the simulation early once we reach max supersaturation
                if ParcelState_Next.S < ParcelState_0.S:
                    overwrite(t2, ParcelState_0, output_filename, specdata_path=specdata_path)
                    breaker = True
                    
                # update parcel state
                ParcelState_0=ParcelState_Next
                pbar.update(1)
                
                # write backup files
                if t2-last_written>=write_every:
                    overwrite(t2, ParcelState_0, output_filename, specdata_path=specdata_path)
                    last_written=t2
                    # if breaker:
                    #     break            
    
            pbar.close()
            if breaker:
                print('Early stopping at t = '+str(t2)+' s')
            print('Solving time:', round(time.time() - runtime0, 2), 'seconds')            
            # print('Maximum saturation ratio:', parcel_trajectory.get_max_S())
            # print('Average cloud droplet diameter:', np.round(2.0*1e6*parcel_trajectory.get_avg_droplet_radius(),4), 'micron')
            # print('Activated fraction:', str(np.round(100*parcel_trajectory.get_activated_fraction(),3))+'%')
            
            # parcel_trajectory = ParcelTrajectory(ts=t_eval, parcel_states=parcel_states)
            # trajectory_ensemble.append(parcel_trajectory)
            
            
    
    return


# this module simulates partitioning between gas and
# aqueous phases and claculates the equilibrium fractions
# in gas and aqueous phases
def simulate_gas_partitioning(N_scenarios=1,
        t_end=600,dt=1.,updraft_velocity=0.0,
        Ddry=100e-9,sigma=1.0,Ntot=1e6, Npart=1,
        S0=-0.15,P0=101325,T0=298,pH0=7.0,
        accom=1., verbosity=50,
        radius_scale='lin',solver='CVODE',
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=None, gas_conc=None,
        specdata_path='../species_data/', mechanism_data_path='../mechanisms/',
        condensation = True, collisions = False, settling = False,
        cocondensation = False, aq_chemistry = None, freezing = False,
        gas_chemistry=False, entrainment=False, write_every=1.0, relaxation_time=None):
        
    
    # trajectory_ensemble = []   
    
    scenario = create_constant_parcel(
                aerosol_population = None,
                Ddry=Ddry,sigma=sigma,Ntot=Ntot,Npart=Npart,updraft_velocity=updraft_velocity,
                S0=S0,P0=P0,T0=T0,pH0=pH0,t_end=t_end,
                species_names=species_names,mass_fractions=mass_fractions,
                gas_names=gas_names, gas_conc=gas_conc,
                dt=dt, specdata_path=specdata_path, mechanism_data_path=mechanism_data_path,
                aq_chemistry=aq_chemistry, cocondensation=cocondensation)     
    
    if gas_chemistry:
        gas_reactions = make_GasReactions(mechanism_data_path=mechanism_data_path)
    else:
        gas_reactions = None
    
    if aq_chemistry:
        aq_reactions = make_AqReactions(aq_chemistry=aq_chemistry, mechanism_data_path=mechanism_data_path)
    else:
        aq_reactions = None
    
    processes = Processes(
        condensation = condensation, 
        collisions = collisions, 
        settling = settling,
        cocondensation = cocondensation, 
        aq_chemistry = aq_chemistry, 
        gas_chemistry=gas_chemistry,
        freezing = freezing,
        entrainment = entrainment)   
    
    print()
    for (one_trajectory_settings, start_time, end_time
          ) in zip(scenario.trajectories_settings,scenario.start_times,scenario.end_times):        
        
        runtime0 = time.time()
        print('Running trajectory,', Npart,'particles...')        
        ParcelState_0 = get_initial_parcel(one_trajectory_settings, start_time)
        
        Ntimes = int((end_time - start_time)/dt)        
        t_eval = np.linspace(start_time, end_time, Ntimes) 
        # parcel_states = [copy.deepcopy(ParcelState_0)]
        pbar = tqdm.tqdm(total = len(t_eval))
        for (t1,t2) in zip(t_eval[:-1],t_eval[1:]):  
            ParcelState_Next = update_state(t1, t2,
                ParcelState_0, processes, dt, 
                radius_scale=radius_scale,solver=solver,
                accom=accom, verbosity=verbosity,
                mechanism_data_path=mechanism_data_path,
                aq_reactions=aq_reactions)
            # parcel_states.append(copy.deepcopy(ParcelState_Next))
            ParcelState_0=replace(ParcelState_Next)            
    
            # print()
            # print('here')
            # print()
            # sys.exit()
            
            pbar.update(1)

        pbar.close()
        print('Solving time:', round(time.time() - runtime0, 2), 'seconds')            
        # parcel_trajectory = ParcelTrajectory(ts=t_eval, parcel_states=parcel_states)
        # print('Maximum saturation ratio:', parcel_trajectory.get_max_S())
        # print('Average cloud droplet diameter:', np.round(2.0*1e6*parcel_trajectory.get_avg_droplet_radius(),4), 'micron')
        # print('Activated fraction:', str(np.round(100*parcel_trajectory.get_activated_fraction(),3))+'%')
        
        
        # parcel_trajectory = ParcelTrajectory(ts=t_eval, parcel_states=parcel_states)
        # trajectory_ensemble.append(parcel_trajectory)
    
    return ParcelState_Next




# this module simulates the same scenario as Seinfeld and Pandis
# figures 7.5 and 7.8 
def simulate_sulfate_partitioning(gas_concs,
        t_end=600,dt=1.,updraft_velocity=0.0,
        Ddry=100e-9,sigma=1.0,Ntot=1e6, Npart=1,
        S0=-0.15,P0=101325,T0=298, pH0=7.0,
        accom=1., verbosity=50,
        radius_scale='lin',solver='CVODE',
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=None, gas_conc=None,
        specdata_path='../species_data/', mechanism_data_path='../mechanisms/',
        condensation = True, collisions = False, settling = False,
        cocondensation = False, chemistry = None, freezing = False):
        
    trajectory_ensemble = []   
    print('Running trajectories...')
    pbar = tqdm.tqdm(total = len(gas_concs))        
    for SO2_conc in gas_concs:
        
        scenario = create_constant_parcel(
                    aerosol_population = None,
                    Ddry=Ddry,sigma=sigma,Ntot=Ntot,Npart=Npart,updraft_velocity=updraft_velocity,
                    S0=S0,P0=P0,T0=T0,pH0=pH0,t_end=t_end,
                    species_names=species_names,mass_fractions=mass_fractions,
                    gas_names=gas_names, gas_conc=gas_conc,
                    dt=dt, specdata_path=specdata_path, mechanism_data_path=mechanism_data_path,
                    chemistry=chemistry, cocondensation=cocondensation)     
        
        # change the SO2 gas concentration (changes the final pH)
        idx = scenario.trajectories_settings[0].gas0.get_species_idx('SO2')
        scenario.trajectories_settings[0].gas0.concs[idx]=SO2_conc

        if chemistry:
            aq_reactions = make_AqReactions(chemistry=chemistry, mechanism_data_path=mechanism_data_path)

            # remove the sulfate oxidation reactions (only doing the equilibrium reactions)
            reactions=aq_reactions.reactions[:-5]
            ids=aq_reactions.ids[:-5]
            aq_reactions=AqueousReactions(reactions=reactions, ids=ids)            
       
        else:
            aq_reactions = None
            
        
            
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
        
        for (one_trajectory_settings, start_time, end_time
              ) in zip(scenario.trajectories_settings,scenario.start_times,scenario.end_times):        
            
            runtime0 = time.time()
            
            ParcelState_0 = get_initial_parcel(one_trajectory_settings, start_time)
            
            Ntimes = int((end_time - start_time)/dt)        
            t_eval = np.linspace(start_time, end_time, Ntimes) 
            parcel_states = [copy.deepcopy(ParcelState_0)]
            
            for (t1,t2) in zip(t_eval[:-1],t_eval[1:]):  
                ParcelState_Next = update_state(t1, t2,
                    ParcelState_0, processes, dt, 
                    radius_scale=radius_scale,solver=solver,
                    accom=accom, verbosity=verbosity,
                    mechanism_data_path=mechanism_data_path,
                    aq_reactions=aq_reactions)
                parcel_states.append(copy.deepcopy(ParcelState_Next))
                ParcelState_0=replace(ParcelState_Next)            
            
            parcel_trajectory = ParcelTrajectory(ts=t_eval, parcel_states=parcel_states)
            trajectory_ensemble.append(parcel_trajectory)
       
        pbar.update(1)

    pbar.close()
    
    return trajectory_ensemble


def simulate_sulfate_oxidation(pHs,
        t_end=600,dt=1.,updraft_velocity=0.0,
        Ddry=100e-9,sigma=1.0,Ntot=1e6, Npart=1,
        S0=-0.15,P0=101325,T0=298,pH0=7.0,
        accom=1., verbosity=50,
        radius_scale='lin',solver='CVODE',
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=None, gas_conc=None,
        specdata_path='../species_data/', mechanism_data_path='../mechanisms/',
        condensation = True, collisions = False, settling = False,
        cocondensation = False, chemistry = None, freezing = False):
        
    # print('Running trajectories...')
    # pbar = tqdm.tqdm(total = len(gas_concs))  
    output = np.zeros(len(pHs))      
    for jj,(pH) in enumerate(pHs):       
        
        scenario = create_constant_parcel(
                    aerosol_population = None,
                    Ddry=Ddry,sigma=sigma,Ntot=Ntot,Npart=Npart,updraft_velocity=updraft_velocity,
                    S0=S0,P0=P0,T0=T0,pH0=pH,t_end=t_end,
                    species_names=species_names,mass_fractions=mass_fractions,
                    gas_names=gas_names, gas_conc=gas_conc,
                    dt=dt, specdata_path=specdata_path, mechanism_data_path=mechanism_data_path,
                    chemistry=chemistry, cocondensation=cocondensation) 
        
        if chemistry:
            aq_reactions = make_AqReactions(chemistry=chemistry, mechanism_data_path=mechanism_data_path)
        else:
            aq_reactions = None            
        
        # scale the number concentration so that there is 1e-6 m^3 water per m^3 air
        aerosol_population = scenario.trajectories_settings[0].population0
        AWC = 0
        for ii,(particle,num_conc) in enumerate(zip(aerosol_population.particles,aerosol_population.num_concs)):
            water_mass = particle.masses[particle.idx_h2o]
            AWC += 1e9*num_conc*water_mass # ug water per m^3 air
        
        mult = 300.0/AWC
        for i in range(len(scenario.trajectories_settings[0].population0.num_concs)):
            scenario.trajectories_settings[0].population0.num_concs[i] *= mult        
        
        # bring S4 species to equilibrium
        for reaction in aq_reactions.reactions:
            if reaction.reactants==['SO2'] and reaction.products==['HSO3', 'H+']:
                kf1 = reaction.rate0
            elif reaction.reactants==['HSO3', 'H+'] and reaction.products==['SO2']:
                kr1 = reaction.rate0
            elif reaction.reactants==['HSO3'] and reaction.products==['SO3', 'H+']:
                kf2=reaction.rate0
            elif reaction.reactants==['SO3', 'H+'] and reaction.products==['HSO3']:
                kr2=reaction.rate0
        
        particle = scenario.trajectories_settings[0].population0.particles[0]
        water_volume = particle.get_vol_tot()-particle.get_vol_dry()
        idx=particle.get_species_idx('H+')
        Hplus_conc = (particle.masses[idx]/particle.species[idx].molar_mass)/water_volume # mol/m^3
        Keq1 = kf1/kr1 # mol/m^3
        Keq2 = kf2/kr2 # mol/m^3
        GasPopulation = scenario.trajectories_settings[0].gas0
        H0 = GasPopulation.gases[GasPopulation.get_species_idx('SO2')].get_Heff(T0)
        Heff = H0*(1+(Keq1/Hplus_conc)+((Keq1*Keq2)/(Hplus_conc*Hplus_conc)))
        P_SO2 = 1e-9*GasPopulation.concs[GasPopulation.get_species_idx('SO2')]*P0
        S6_conc = Heff*P_SO2
        x_SO2 = np.power(1+(Keq1/Hplus_conc)+((Keq1*Keq2)/(Hplus_conc*Hplus_conc)), -1.0)
        x_HSO3 = np.power(1+(Hplus_conc/Keq1)+(Keq2/Hplus_conc), -1.0)
        x_SO3 = np.power(1+(Hplus_conc/Keq2)+((Hplus_conc*Hplus_conc)/(Keq1*Keq2)), -1.0)
        SO2_conc = x_SO2*S6_conc
        HSO3_conc = x_HSO3*S6_conc
        SO3_conc = x_SO3*S6_conc
        particle.masses[particle.get_species_idx('SO2')]=water_volume*particle.species[particle.get_species_idx('SO2')].molar_mass*SO2_conc
        particle.masses[particle.get_species_idx('HSO3')]=water_volume*particle.species[particle.get_species_idx('HSO3')].molar_mass*HSO3_conc
        particle.masses[particle.get_species_idx('SO3')]=water_volume*particle.species[particle.get_species_idx('SO3')].molar_mass*SO3_conc
        scenario.trajectories_settings[0].population0.particles[0] = particle        
        
        # set up the original gas concentrations
        TraceGas_population = scenario.trajectories_settings[0].gas0
        if TraceGas_population:
            Cgas_0 = []
            gas_names = []
            gas_molar_masses = []
            gas_alphas = []
            gas_Heffs = []
            for gas, gas_ppb in zip(TraceGas_population.gases, TraceGas_population.concs):
                gas_names.append(gas.name)               
                Cgas_0.append((gas_ppb*1e-9*P0)/(8.314*T0)) # mol/m^3
                gas_molar_masses.append(gas.molar_mass)
                gas_alphas.append(gas.alpha)
                gas_Heffs.append(gas.get_Heff(T0))
        
        # # set up the initial aqueous concentrations
        particle = scenario.trajectories_settings[0].population0.particles[0]
        water_volume_0 = particle.get_vol_tot()-particle.get_vol_dry() # m^3
        Caq_0 = np.empty(0)
        aq_names = Dict.empty(key_type=types.unicode_type, value_type=types.int32)
        for ii, (species) in enumerate(particle.species):
            aq_names[species.name]=ii
            Caq_0=np.append(Caq_0, (particle.masses[particle.get_species_idx(species.name)]/species.molar_mass)/water_volume_0) # mol/m^3
        
        # get the sulfate production rates from each oxidant
        dCaq_dt_all = np.zeros(len(Caq_0))
        dCaq_dt_all = AC.H2O2_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T0)
        dCaq_dt_all = AC.O3_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T0)
        dCaq_dt_all = AC.NO2_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T0)
        dCaq_dt_all = AC.HNO2_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T0)
        if 'O2' in gas_names:
            dCaq_dt_all = AC.O2_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T0)
        
        idx=np.where(np.array((aq_names))=='SO4')[0][0]
        num_conc = scenario.trajectories_settings[0].population0.num_concs[0]
        molar_mass=particle.species[idx].molar_mass
        dSO4_dt = dCaq_dt_all[idx]*water_volume_0*num_conc*molar_mass # kg/m^3/s
        output[jj]=dSO4_dt # kg/m^3/s
        
    return output


def simulate_IEPOX_chemistry(mu_star, N_scenarios=1,
        t_end=600,dt=1.,updraft_velocity=0.0,
        Ddry=100e-9,sigma=1.0,Ntot=1e6, Npart=1,
        S0=-0.15,P0=101325,T0=298,pH0=7.0,
        accom=1., verbosity=50,
        radius_scale='lin',solver='CVODE',
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=None, gas_conc=None,
        specdata_path='../species_data/', mechanism_data_path='../mechanisms/',
        condensation = True, collisions = False, settling = False,
        cocondensation = False, aq_chemistry = None, freezing = False,
        gas_chemistry=False, entrainment=False, write_every=1.0, relaxation_time=None):
    
    trajectory_ensemble = [] 

      
    scenario = create_constant_parcel(
                aerosol_population = None,
                Ddry=Ddry,sigma=sigma,Ntot=Ntot,Npart=Npart,updraft_velocity=updraft_velocity,
                S0=S0,P0=P0,T0=T0,pH0=pH0,t_end=t_end,
                species_names=species_names,mass_fractions=mass_fractions,
                gas_names=gas_names, gas_conc=gas_conc,
                dt=dt, specdata_path=specdata_path, mechanism_data_path=mechanism_data_path,
                chemistry=chemistry, cocondensation=cocondensation) 

    print(scenario)
    
    '''
    if chemistry:
        aq_reactions = make_AqReactions(chemistry=chemistry, mechanism_data_path=mechanism_data_path)
    else:
        aq_reactions = None

    processes = Processes(
        condensation = condensation, 
        collisions = collisions, 
        settling = settling,
        cocondensation = cocondensation, 
        chemistry = chemistry, 
        freezing = freezing)    
    
    if chemistry:
        if 'sulfate' in chemistry:
            particle_population=scenario.trajectories_settings[0].population0.particles
            for ii, (particle) in enumerate(particle_population):
                water_volume=particle.get_vol_tot()-particle.get_vol_dry()
                SO4_conc=(particle.masses[particle.get_species_idx('SO4')]/particle.species[particle.get_species_idx('SO4')].molar_mass)/water_volume
                Hplus_conc=(particle.masses[particle.get_species_idx('H+')]/particle.species[particle.get_species_idx('H+')].molar_mass)/water_volume
                HSO4_conc=(SO4_conc*Hplus_conc)/0.01
                H2SO4_conc=(HSO4_conc*Hplus_conc)/1000.0
                particle.masses[particle.get_species_idx('HSO4')]=HSO4_conc*particle.species[particle.get_species_idx('HSO4')].molar_mass*water_volume
                particle.masses[particle.get_species_idx('H2SO4')]=H2SO4_conc*particle.species[particle.get_species_idx('H2SO4')].molar_mass*water_volume
    
    print()    
    for (one_trajectory_settings, start_time, end_time
          ) in zip(scenario.trajectories_settings,scenario.start_times,scenario.end_times):        
        
        # runtime0 = time.time()
        print('Running trajectory,', Npart,'particles...')        
        ParcelState_0 = get_initial_parcel(one_trajectory_settings, start_time)
        Ntimes = int((end_time - start_time)/dt)        
        t_eval = np.linspace(start_time, end_time, Ntimes) 
        parcel_states = [copy.deepcopy(ParcelState_0)]
        pbar = tqdm.tqdm(total = len(t_eval))
        for (t1,t2) in zip(t_eval[:-1],t_eval[1:]):   
            ParcelState_Next = update_state(t1, t2,
                ParcelState_0, processes, dt, 
                radius_scale=radius_scale,solver=solver,
                accom=accom, verbosity=verbosity,
                mechanism_data_path=mechanism_data_path,
                aq_reactions=aq_reactions)
            
            # do the chamber wall losses
            Dps = np.zeros(len(ParcelState_Next.particle_population.particles))
            Ns_0 = np.zeros(len(ParcelState_Next.particle_population.particles))
            densities = np.zeros(len(ParcelState_Next.particle_population.particles))
            for ii,(particle,num_conc) in enumerate(zip(ParcelState_Next.particle_population.particles, ParcelState_Next.particle_population.num_concs)):
                Dps[ii] = particle.get_Dwet()
                Ns_0[ii] = num_conc
                densities[ii] = particle.get_trho()
            
            rhs = lambda t, Ns: wall_losses.particle_wall_loss(Ns, Dps, densities, ParcelState_Next.T, mu_star)
            if solver == 'CVODE': 
                prob = Explicit_Problem(rhs, Ns_0)
                sim = CVode(prob)
                sim.atol=1.0e-10
                sim.rtol=1.0e-10
                sim.verbosity=verbosity
                output=sim.simulate(dt)
                Ns_next=output[1][-1] # 1/m^3
            elif solver == 'ode15s':
                ode15s = ode(rhs).set_integrator('lsoda', method='bdf', 
                                                  rtol=1E-10, atol=1E-10, nsteps=5000)
                ode15s.set_initial_value(Ns_0, 0.0)
                Ns_next = ode15s.integrate(ode15s.t+dt)  # mol/m^3
            
            # update the particle number concentrations
            for ii in range(len(ParcelState_Next.particle_population.num_concs)):
                ParcelState_Next.particle_population.num_concs[ii]=Ns_next[ii]

            parcel_states.append(copy.deepcopy(ParcelState_Next))
            ParcelState_0=replace(ParcelState_Next)            

            pbar.update(1)

        pbar.close()
        print()
        parcel_trajectory = ParcelTrajectory(ts=t_eval, parcel_states=parcel_states)
        trajectory_ensemble.append(parcel_trajectory)
    '''
    return trajectory_ensemble










