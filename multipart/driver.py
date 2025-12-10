#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver for the Multiscale Particle-based (MultiPart) microphysics model

@author: fier887
"""
# from dataclasses import dataclass
from systems import ParcelState
from systems import Processes
from scenario import create_scenario_from_DNS, create_parcel_scenario, create_hysplit_scenario, create_les_scenario, create_pichamber_scenario
# from typing import Tuple
from typing import Callable
import numpy as np
from dataclasses import replace
import copy, time, utilities
import matplotlib.pyplot as plt
import sys, pickle
from processes import air_thermo
from scenario import make_AqReactions, make_GasReactions
from write_files import write_original, overwrite
import scipy.optimize as opt

from systems import update_state, air_from_les#, ParcelTrajectory, TrajectoryEnsemble, TrajectoryInteractions


# @dataclass
# class Simulation:
#     scenario: Scenario
#     t_eval: Tuple(float, ...)
#     processes: Processes
#     parcel_states: Tuple(ParcelState, ...) | None = None
    
#     def initialize(self):
#         # update parcel_states = [ParcelState]
#         pass
    
#     def run(self):
#         # use state_updater append parcel_states with new ParcelState at each t_eval
        
#         pass
        
# t_start=0.;t_end=3600.;dt=1.;
# case_num=2;dns_dir='/Users/fier887/Downloads/New_cases (7-27-22)/';
# Ddry=100e-9; Nper=1e6; species_names=['NaCl']; mass_fractions=np.array([1.]);
# specdata_path='../species_data/';
# condensation = True; collisions = False; settling = False;
# cocondensation = False; chemistry = False; freezing = False

# here is some text
'''
def simulate_dns_trajectories(
        t_start=0.,t_end=3600.,dt=1.,this_many=None,
        case_num=2,dns_dir='/Users/fier887/Downloads/New_cases (7-27-22)/',
        Ddry=100e-9, Nper=1e6, species_names=['NaCl'], mass_fractions=np.array([1.]),
        specdata_path='../species_data/', 
        radius_scale='lin',force_cvode=False,
        accom=1., verbosity=50,
        condensation = True, collisions = False, settling = False,
        cocondensation = False, chemistry = False, freezing = False,
        fluctuations = False):
    
    scenario = create_scenario_from_DNS(
            case_num=case_num,dns_dir=dns_dir,
            Ddry=Ddry, Nper=Nper, species_names=species_names, mass_fractions= mass_fractions,
            dt=None, specdata_path=specdata_path, this_many=this_many)
    processes = Processes(
        condensation = condensation, 
        collisions = collisions, 
        settling = settling,
        cocondensation = cocondensation, 
        chemistry = chemistry, 
        freezing = freezing, fluctuations = fluctuations)    
    parcel_trajectories = []
    for (one_trajectory_settings, start_time, end_time
          ) in zip(scenario.trajectories_settings,scenario.start_times,scenario.end_times):
        ParcelState_0 = get_initial_parcel(one_trajectory_settings, start_time)
        Ntimes = int((t_end - t_start)/dt + 1)
        t_eval = np.linspace(start_time, end_time, Ntimes)
        parcel_states = [ParcelState_0]
        plt.plot(t_eval,one_trajectory_settings.S_fun(t_eval)); plt.show()
        for (t1,t2) in zip(t_eval[:-1],t_eval[1:]):
            ParcelState_Next = update_state(
                ParcelState_0, processes, dt, 
                accom=accom, verbosity=verbosity,
                radius_scale=radius_scale,force_cvode=force_cvode)
            parcel_states.append(ParcelState_Next)
            ParcelState_0=ParcelState_Next
        parcel_trajectory = ParcelTrajectory(ts=t_eval, parcel_states=parcel_states)
        
        parcel_trajectories.append(parcel_trajectory)
    trajectory_interactions = TrajectoryInteractions() # does nothing right now
    trajectory_ensemble = TrajectoryEnsemble(
        parcel_trajectories=parcel_trajectories, 
        trajectory_interactions=trajectory_interactions)
    return trajectory_ensemble
'''

'''
def simulate_hysplit_trajectories(hysplit_tdump_file=None,
        scenario_numbers='all',dt=1.0,Ddry=100e-9,sigma=1.0,Ntot=1e6, Npart=1,
        pH0=7.0, accom=1., verbosity=50,
        radius_scale='lin',solver='CVODE',
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=None, gas_conc=None,
        specdata_path='../species_data/', mechanism_data_path='../mechanisms/',
        condensation = True, collisions = False, settling = False,
        cocondensation = False, chemistry = False, freezing = False):
    
    if not hysplit_tdump_file:
        print('WARNING: No HYSPLIT file specified!')
        sys.exit()
    
    scenario = create_hysplit_scenario(hysplit_tdump_file,
                scenario_numbers=scenario_numbers, Ddry=Ddry,
                sigma=sigma,Ntot=Ntot,Npart=Npart,
                pH0=pH0,species_names=species_names,mass_fractions=mass_fractions,
                gas_names=gas_names, gas_conc=gas_conc, 
                dt=dt, specdata_path=specdata_path,
                mechanism_data_path=mechanism_data_path,
                chemistry=chemistry, cocondensation=cocondensation)    

    processes = Processes(
        condensation = condensation, 
        collisions = collisions, 
        settling = settling,
        cocondensation = cocondensation, 
        chemistry = chemistry, 
        freezing = freezing) 
    
    trajectory_ensemble = []
    traj=0

    
    for (one_trajectory_settings, start_time, end_time
          ) in zip(scenario.trajectories_settings,scenario.start_times,scenario.end_times):
        
        runtime0 = time.time()
        print('Running trajectory', str(traj+1)+',', Npart,'particles...')
        ParcelState_0 = get_initial_parcel(one_trajectory_settings, start_time)  
        t_start=one_trajectory_settings.t_data[0]
        t_end=one_trajectory_settings.t_data[-1]
        Ntimes = int((t_end - t_start)/dt + 1)
        t_eval = np.linspace(t_start, t_end, Ntimes)
        parcel_states = [ParcelState_0]        
        
        pbar = tqdm.tqdm(total = len(t_eval))
        for (t1,t2) in zip(t_eval[:-1],t_eval[1:]):
            # print(t1, t2)
            
            ParcelState_0.z = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.z_data)
            ParcelState_0.P = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.P_data)
            ParcelState_0.S = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.S_data)
            ParcelState_0.T = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.T_data)
            
            ParcelState_Next = update_state(t1, t2,
                ParcelState_0, processes, dt, 
                radius_scale=radius_scale,solver=solver,
                sigma=sigma, accom=accom, verbosity=verbosity)
            
            parcel_states.append(ParcelState_Next)
            ParcelState_0=ParcelState_Next
            
            pbar.update(1)
        pbar.close()
        print('Solving time:', round(time.time() - runtime0, 2), 'seconds')            
        parcel_trajectory = ParcelTrajectory(ts=t_eval, parcel_states=parcel_states)
        print('Maximum saturation ratio:', parcel_trajectory.get_max_S())
        print('Average cloud droplet diameter:', np.round(2.0*1e6*parcel_trajectory.get_avg_droplet_radius(),4), 'micron')
        print('Activated fraction:', str(np.round(100*parcel_trajectory.get_activated_fraction(),3))+'%')
        print()
        
        traj+=1
        parcel_trajectory = ParcelTrajectory(ts=t_eval, parcel_states=parcel_states)
        trajectory_ensemble.append(parcel_trajectory)
    
    return trajectory_ensemble
'''

def simulate_les_trajectories(les_output_file=None, output_path=None,
        dt=1.0,diameters=np.array([100e-9]),N_concs=np.array([1e6]),
        pHs=np.array([7.0]), accom=1., verbosity=50,
        radius_scale='lin',solver='CVODE',
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=None, gas_data=None,
        specdata_path='../species_data/', mechanism_data_path='../mechanisms/',
        condensation = True, collisions = False, settling = False,
        cocondensation = False, aq_chemistry = False, freezing = False,
        entrainment = False, relaxation_time=None, gas_chemistry = False, write_every=60.):
    
    #sys.stdout = open('output.log', 'w')

    if not les_output_file:
        with open('RUN_PROGRESS.out', 'a') as f:
            print('WARNING: No LES file specified!', file=f)
        sys.exit()
    
    LES_gases = copy.deepcopy(gas_names)
    scenario = create_les_scenario(les_output_file, 
                            diameters=diameters,N_concs=N_concs,
                            pHs=pHs,species_names=species_names,
                            mass_fractions=mass_fractions,
                            gas_names=gas_names, gas_data=gas_data, 
                            dt=dt, specdata_path=specdata_path,
                            mechanism_data_path=mechanism_data_path,
                            aq_chemistry=aq_chemistry, gas_chemistry=gas_chemistry,
                            cocondensation=cocondensation)    

    if gas_chemistry:
        gas_reactions = make_GasReactions(mechanism_data_path=mechanism_data_path)
    else:
        gas_reactions = None
    
    if aq_chemistry:
        aq_reactions = make_AqReactions(chemistry=aq_chemistry, mechanism_data_path=mechanism_data_path)
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
    
    # equilibrate the S(VI) and nitrate species
    if aq_chemistry:
        if 'sulfate' in aq_chemistry:
            particle_population=scenario.trajectories_settings[0].population0.particles
            for ii, (particle) in enumerate(particle_population):
                water_volume=particle.get_vol_tot()-particle.get_vol_dry()
                SO4_conc=(particle.masses[particle.get_species_idx('SO4')]/particle.species[particle.get_species_idx('SO4')].molar_mass)/water_volume
                Hplus_conc=(particle.masses[particle.get_species_idx('H+')]/particle.species[particle.get_species_idx('H+')].molar_mass)/water_volume
                HSO4_conc=(SO4_conc*Hplus_conc)/0.01
                H2SO4_conc=(HSO4_conc*Hplus_conc)/1000.0
                particle.masses[particle.get_species_idx('HSO4')]=HSO4_conc*particle.species[particle.get_species_idx('HSO4')].molar_mass*water_volume
                particle.masses[particle.get_species_idx('H2SO4')]=H2SO4_conc*particle.species[particle.get_species_idx('H2SO4')].molar_mass*water_volume
        if 'nitrate' in aq_chemistry:
            particle_population=scenario.trajectories_settings[0].population0.particles
            for ii, (particle) in enumerate(particle_population):
                water_volume=particle.get_vol_tot()-particle.get_vol_dry()
                NO3_conc=(particle.masses[particle.get_species_idx('NO3')]/particle.species[particle.get_species_idx('NO3')].molar_mass)/water_volume
                Hplus_conc=(particle.masses[particle.get_species_idx('H+')]/particle.species[particle.get_species_idx('H+')].molar_mass)/water_volume
                HNO3_conc=(NO3_conc*Hplus_conc)/15.625
                particle.masses[particle.get_species_idx('HNO3')]=HNO3_conc*particle.species[particle.get_species_idx('HNO3')].molar_mass*water_volume    
    
    # equilibrate the co-condensing species
    # if scenario.trajectories_settings[0].gas0 and processes.cocondensation:
    #     for ii, (particle) in enumerate(particle_population):
    #         water_volume=particle.get_vol_tot()-particle.get_vol_dry()
    #         T0 = np.interp(0.0, scenario.trajectories_settings[0].t_data, scenario.trajectories_settings[0].T_data)
    #         P0 = np.interp(0.0, scenario.trajectories_settings[0].t_data, scenario.trajectories_settings[0].P_data)
    #         for gas, conc in zip(scenario.trajectories_settings[0].gas0.gases, scenario.trajectories_settings[0].gas0.concs):
    #             if gas.molar_mass > 0:
    #                 Cx = 1e-9*conc*P0*gas.get_Heff(T0) # mil/m^3
    #                 Mx = gas.molar_mass*Cx*water_volume
    #                 particle.masses[particle.get_species_idx(gas.name)]=Mx
    
    
    # particle=scenario.trajectories_settings[0].population0.particles[0]
    # for ii, (species) in enumerate(particle.species):
    #     print(species.name, particle.masses[ii])
    # sys.exit()

    for (one_trajectory_settings, start_time, end_time
          ) in zip(scenario.trajectories_settings,scenario.start_times,scenario.end_times):
        
        runtime0 = time.time()
        ParcelState_0 = get_initial_parcel(one_trajectory_settings, start_time)
        t_start=one_trajectory_settings.t_data[0]
        t_end=one_trajectory_settings.t_data[-1]
        Ntimes = int((t_end - t_start)/dt + 1)
        t_eval = np.linspace(t_start, t_end, Ntimes)
        last_written=t_start
        restart_filename=output_path+'/trajectory_'+les_output_file[-10:-4]+'_RESTART.pkl'
        output_filename=output_path+'/trajectory_'+les_output_file[-10:-4]+'.pkl'
        status_filename=output_path+'/trajectory_'+les_output_file[-10:-4]+'_STATUS'
        write_original(t_start, ParcelState_0, output_filename, specdata_path=specdata_path)
        f = open(status_filename, 'w')
        f.write('in progress')
        f.close()
        
        #with open('RUN_PROGRESS.out', 'a') as f:
        print('')#, file=f)
        print('Running trajectory', output_filename[-10:-4]+',', len(N_concs),'particles...')#, file=f)
        
        counter=0
        for (t1,t2) in zip(t_eval[:-1],t_eval[1:]):
        #for (t1,t2) in zip(t_eval[:199],t_eval[1:200]):
            steptime0 = time.time()
            
            # original_ParcelState = copy.deepcopy(ParcelState_0) # keep this for mole balance at end of time step
            
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
            
            # check that the moles of water in the system is consistent
            # (needs to happen before mass transfer with background)
            #if processes.condensation:
            #    utilities.water_mole_balance(original_ParcelState, ParcelState_Next)            
            
            # get new air state from LES
            ParcelState_Next=air_from_les(ParcelState_Next, processes, t2, one_trajectory_settings, 
                                          relaxation_time, dt, solver, gas_data, LES_gases, 
                                          rtol=1e-4, atol=1e-8)
            
            # print timestep and time for timestep
            counter+=1
            #with open('RUN_PROGRESS.out', 'a') as f:
            print(str(counter)+'/'+str(len(t_eval))+' -- '+str(round(time.time() - steptime0, 2))+ 's/it')#, file=f)

            
            # check for NaNs
            total_mass = []
            for particle, num_conc in zip(ParcelState_Next.particle_population.particles, ParcelState_Next.particle_population.num_concs):
                total_mass.append(num_conc*np.sum(particle.masses))
            
            #with open('RUN_PROGRESS.out', 'a') as f:
            print(str(ParcelState_Next.S)+' '+str(1e9*np.sum(np.array(total_mass))))#, file=f)
            
            # utilities.water_mole_balance(original_ParcelState, ParcelState_Next)
            print('')#, file=f)
                    
            # kill the program if there is a NaN
            if np.isnan(np.sum(total_mass)):
                print('ERROR')
                f = open(status_filename, 'w')
                f.write('killed (NaNs)')
                f.close()
                sys.exit()
            
            # update parcel state
            ParcelState_0=ParcelState_Next
            
            # write backup files
            if t2-last_written>=write_every:
                f = open(status_filename, 'w')
                f.write('in progress')
                f.close()
                overwrite(t2, ParcelState_0, output_filename, specdata_path=specdata_path)
                ParcelState_dict = {'time': t2, 'parcel state': ParcelState_Next, 'dt':dt, 'accom':accom, 'verbosity':verbosity, 'radius_scale':radius_scale, 'solver':solver, 'specdata_path':specdata_path, 'mechanism_data_path':mechanism_data_path, 'processes':processes, 'write_every':write_every,'one_trajectory_settings':one_trajectory_settings, 'aq_reactions':aq_reactions, 'gas_data':gas_data, 'relaxation_time':relaxation_time}
                pickle.dump(ParcelState_dict, open(restart_filename, 'wb'))
                last_written=t2
            
            #pbar.update(1)
        #pbar.close()
        
        #with open('RUN_PROGRESS.out', 'a') as f:
        print('')#, file=f)
        print('Solving time:', round(time.time() - runtime0, 2), 'seconds')#, file=f)
        # print('Maximum saturation ratio:', parcel_trajectory.get_max_S())#, file=f)
        # print('Average cloud droplet diameter:', np.round(2.0*1e6*parcel_trajectory.get_avg_droplet_radius(),4), 'micron')#, file=f)
        # print('Activated fraction:', str(np.round(100*parcel_trajectory.get_activated_fraction(),3))+'%')#, file=f)
        print('')#, file=f)
        
        f = open(status_filename, 'w')
        f.write('complete')
        f.close()
        
    return
    
    
def restart_les_trajectories(output_path=None, ParcelState_file=None, trajectory_file=None):
    
    #sys.stdout = open('output.log', 'w')
    data = pickle.load(open(output_path+'/'+ParcelState_file, 'rb'))
    ParcelState_0 = data['parcel state']
    one_trajectory_settings = data['one_trajectory_settings']
    t_start=data['time']
    t_end=one_trajectory_settings.t_data[-1]
    dt=data['dt']
    accom=data['accom']
    verbosity=data['verbosity']
    radius_scale=data['radius_scale']
    solver=data['solver']
    specdata_path=data['specdata_path']
    mechanism_data_path=data['mechanism_data_path']
    processes=data['processes']
    write_every=data['write_every']
    aq_reactions=data['aq_reactions']
    gas_data=data['gas_data']
    relaxation_time=data['relaxation_time']
    
    trajectory_ensemble = []
    runtime0 = time.time()
    #with open('RUN_PROGRESS.out', 'w') as f:
    print('Restarting trajectory', trajectory_file[-10:-4]+',', len(ParcelState_0.particle_population.particles),'particles...')#, file=f)
    print('')#, file=f)
    
    Ntimes = int((t_end - t_start)/dt + 1)
    t_eval = np.linspace(t_start, t_end, Ntimes)
    parcel_states = [ParcelState_0]
    parcel_ts = [t_eval[0]]
    last_written=t_start
    restart_filename=output_path+'/trajectory_'+trajectory_file[-10:-4]+'_RESTART.pkl'
    output_filename=output_path+'/trajectory_'+trajectory_file[-10:-4]+'.pkl'
    status_filename=output_path+'/trajectory_'+trajectory_file[-10:-4]+'_STATUS'
    
    #pbar = tqdm.tqdm(total = len(t_eval))
    counter=0
    for (t1,t2) in zip(t_eval[:-1],t_eval[1:]):
        steptime0 = time.time()
                                
        original_ParcelState = copy.deepcopy(ParcelState_0) # keep this for mole balance at end of time step
            
        ParcelState_Next = update_state(t1, t2,
            ParcelState_0, processes, dt,
            radius_scale=radius_scale,solver=solver,
            accom=accom, verbosity=verbosity,
            mechanism_data_path=mechanism_data_path,
            aq_reactions=aq_reactions, rtol=1e-6, atol=1e-12)
            
        # adjust the number concentration based on the new temperature and pressure
        Ns=np.array(ParcelState_0.particle_population.num_concs)
        Ns*=((ParcelState_Next.P*ParcelState_0.T)/(ParcelState_0.P*ParcelState_Next.T))
        ParcelState_Next.particle_population.num_concs=list(Ns)
            
        # check that the moles of water in the system is consistent
        # (needs to happen before mass transfer with background)
        if processes.condensation:
            utilities.water_mole_balance(original_ParcelState, ParcelState_Next)
            
        # get new air state from LES
        ParcelState_Next=air_from_les(ParcelState_Next, processes, t2, one_trajectory_settings, relaxation_time, dt, solver, gas_data, rtol=1e-4, atol=1e-8)
        
        # print timestep and time for timestep
        counter+=1
        
        #with open('RUN_PROGRESS.out', 'a') as f:
        print(str(counter)+'/'+str(len(t_eval))+' -- '+str(round(time.time() - steptime0, 2))+ 's/it')#, file=f)
        
        # check for NaNs
        total_mass = []
        for particle, num_conc in zip(ParcelState_Next.particle_population.particles, ParcelState_Next.particle_population.num_concs):
            idx_OS=particle.get_species_idx('IEPOX_OS')
            idx_tet=particle.get_species_idx('tetrol')
            idx_olig=particle.get_species_idx('tetrol_olig')
            total_mass.append(num_conc*np.sum(particle.masses))

        #with open('RUN_PROGRESS.out', 'a') as f:
        print(str(ParcelState_Next.S)+' '+str(1e9*np.sum(np.array(total_mass))))#, file=f)
        print('')#, file=f)
                
        # kill the program if there is a NaN
        if np.isnan(np.sum(total_mass)):
            print('ERROR')
            f = open(status_filename, 'w')
            f.write('killed (NaNs)')
            f.close()
            sys.exit()
        
        # update parcel state
        ParcelState_0=ParcelState_Next
        
        # write backup files
        if t2-last_written>=write_every:
            f = open(status_filename, 'w')
            f.write('in progress')
            f.close()

            overwrite(t2, ParcelState_0, output_filename, specdata_path=specdata_path)
            ParcelState_dict = {'time': t2, 'parcel state': ParcelState_Next, 'dt':dt, 'accom':accom, 'verbosity':verbosity, 'radius_scale':radius_scale, 'solver':solver, 'specdata_path':specdata_path, 'mechanism_data_path':mechanism_data_path, 'processes':processes, 'write_every':write_every,'one_trajectory_settings':one_trajectory_settings, 'aq_reactions':aq_reactions, 'gas_data':gas_data}
            pickle.dump(ParcelState_dict, open(restart_filename, 'wb'))
            last_written=t2
        
        #pbar.update(1)
    #pbar.close()
    
    #with open('RUN_PROGRESS.out', 'a') as f:
    print('')#, file=f)
    print('Solving time:', round(time.time() - runtime0, 2), 'seconds')#, file=f)
    # print('Maximum saturation ratio:', parcel_trajectory.get_max_S())#, file=f)
    # print('Average cloud droplet diameter:', np.round(2.0*1e6*parcel_trajectory.get_avg_droplet_radius(),4), 'micron')#, file=f)
    # print('Activated fraction:', str(np.round(100*parcel_trajectory.get_activated_fraction(),3))+'%')#, file=f)
    print('')#, file=f)
        
    f = open(status_filename, 'w')
    f.write('complete')
    f.close()
    
    return


'''
def simulate_parcel_trajectories(N_scenarios=1,
        z_start=0.,z_end=1000.,dt=1.,
        Ddry=100e-9,sigma=1.0,Ntot=1e6, Npart=1,
        updraft_velocity=0.5,S0=-0.15,P0=101325,T0=298,
        pH0=7.0, accom=1., verbosity=50,
        radius_scale='lin',solver='CVODE',
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=None, gas_conc=None,
        specdata_path='../species_data/', mechanism_data_path='../mechanisms/',
        condensation = True, collisions = False, settling = False,
        cocondensation = False, chemistry = False, freezing = False):
    """
    Parameters
    ----------
    z_start : float, optional
        Altitude where simulation starts. The default is 0.
    z_end : float, optional
        Altitude where simulation ends. The default is 1000.
    dt : float, optional
        Timestep for integration. The default is 1.
    Ddry : float, optional
        Mode diameter of lognormal distribution in m. If input 
        value for sigma is 1.0, then Ddry will be the diameter
        of all particles (monodisperse case). The default is 100e-9.
# FIX THIS:
    sigma : float, optional
        geometric standard deviation of lognormal size distribution. 
        If sigma = 1.0 all particles will be the same size (monodisperse
        case). If sigma > 1.0 particles will have diverse sizes 
        (polydisperse case). The default is 1.0.
    Ntot : float, optional
        Total number concentration of particles in m^-3. 
        The default is 1e6.
    Npart : int, optional
        Number of super-particles in the simulation. If sigma = 1.0, 
        Npart is equal to the default value of 1.
    updraft_velocity : float, optional
        Updraft velocity of parcel in m/s. The default is 0.5.
    S0 : float, optional
        Initial supersaturation in parcel. The default is -0.15.
    P0 : float, optional
        Initial pressure in parcel in Pa. The default is 101325.
    T0 : float, optional
        Initial temperature in parcel in K. The default is 298.
    species_names : list, optional
        List of strings describing what particles are made of. 
        The default is ['NaCl'].
    mass_fractions : list, optional
        List of floats describing the mass fraction of each element in
        species_names. Must add up to 1.0. The default is np.array([1.]).
    specdata_path : string, optional
        Path to location of aero_data.dat file. The default is 
        '../species_data/'.
    condensation : bool, optional
        Whether condensation of water is considered in simulation. 
        The default is True.
    collisions : bool, optional
        Whether coagulation is considered in simulation. 
        The default is False.
    settling : bool, optional
        Whether gravitational settling is considered in model. 
        The default is False.
    cocondensation : bool, optional
        Whether co-condensation of semivolatile species is considered
        in simulation. The default is False.
    chemistry : bool, optional
        Whether gas and aqueous chemistry is considered in 
        simulation. The default is False.
    freezing : bool, optional
        Whether ice nucleation is considered in simulation. 
        The default is False.

    Returns
    -------
    parcel_trajectory : ParcelTrajectory
        Time series of parcel state during simulation.
    """
    trajectory_ensemble = []
    
    for i in range(N_scenarios):
        try: 
            z0=z_start[i]
        except:
            z0=z_start
        try:
            z_f=z_end[i]
        except:
            z_f=z_end
        try:
            timestep=dt[i]
        except:
            timestep=dt
        try:
            D_dry=Ddry[i]
        except:
            D_dry=Ddry
        try:
            s=sigma[i]
        except:
            s=sigma
        try:
            N_tot=Ntot[i]
        except:
            N_tot=Ntot 
        try:
            N_part=Npart[i]
        except:
            N_part=Npart
        try:
            V=updraft_velocity[i]
        except:
            V=updraft_velocity
        try:
            S_0=S0[i]
        except:
            S_0=S0
        try:
            T_0=T0[i]
        except:
            T_0=T0
        try:
            P_0=P0[i]
        except:
            P_0=P0
        try:
            pH_0=pH0[i]
        except:
            pH_0=pH0
            
        end_time=z_end/updraft_velocity
                
        scenario = create_parcel_scenario(
                Ddry=D_dry,sigma=s,Ntot=N_tot,Npart=N_part,
                updraft_velocity=V,
                S0=S_0,P0=P_0,T0=T_0,pH0=pH_0,z_start=z0,z_end=z_f,
                species_names=species_names,mass_fractions= mass_fractions,
                gas_names=gas_names, gas_conc=gas_conc,
                dt=timestep, specdata_path=specdata_path,
                mechanism_data_path=mechanism_data_path,
                chemistry=chemistry, cocondensation=cocondensation)  
        
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
            print('Running trajectory', str(i+1)+',', N_part,'particles...')
            ParcelState_0 = get_initial_parcel(one_trajectory_settings, start_time)            
            Ntimes = int((end_time - start_time)/dt + 1)        
            t_eval = np.linspace(start_time, end_time, Ntimes)  
            parcel_states = [copy.deepcopy(ParcelState_0)]
            
            pbar = tqdm.tqdm(total = len(t_eval))
            for (t1,t2) in zip(t_eval[:-1],t_eval[1:]):
                ParcelState_Next = update_state(t1, t2,
                    ParcelState_0, processes, dt, 
                    radius_scale=radius_scale,solver=solver,
                    sigma=sigma, accom=accom, verbosity=verbosity)
                parcel_states.append(copy.deepcopy(ParcelState_Next))
                ParcelState_0=replace(ParcelState_Next)
                pbar.update(1)
            pbar.close()
            print('Solving time:', round(time.time() - runtime0, 2), 'seconds')            
            parcel_trajectory = ParcelTrajectory(ts=t_eval, parcel_states=parcel_states)
            print('Maximum saturation ratio:', parcel_trajectory.get_max_S())
            print('Average cloud droplet diameter:', np.round(2.0*1e6*parcel_trajectory.get_avg_droplet_radius(),4), 'micron')
            print('Activated fraction:', str(np.round(100*parcel_trajectory.get_activated_fraction(),3))+'%')
            print()
            trajectory_ensemble.append(parcel_trajectory)
        
    return trajectory_ensemble


def simulate_PiChamber_trajectories(output_path=None,
        dt=1.0,diameters=np.array([100e-9]),N_concs=np.array([1e6]),
        pHs=np.array([7.0]), accom=1., verbosity=50,
        radius_scale='lin',solver='CVODE',
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        gas_names=None, gas_concentrations=np.array([1.0]),
        specdata_path='../species_data/', mechanism_data_path='../mechanisms/',
        trajectory_path='../datasets/', run_number=0,
        condensation = True, collisions = False, settling = False,
        cocondensation = False, chemistry = False, freezing = False, write_every=60.):
    
    scenario = create_pichamber_scenario(trajectory_path=trajectory_path, run_number=run_number,
                            diameters=diameters,N_concs=N_concs,
                            pHs=pHs,species_names=species_names,
                            mass_fractions=mass_fractions,
                            gas_names=gas_names, gas_concentrations=gas_concentrations, 
                            dt=dt, specdata_path=specdata_path,
                            mechanism_data_path=mechanism_data_path,
                            chemistry=chemistry, cocondensation=cocondensation)
    
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
    
    # equilibrate the S(VI) species
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
    
    # particle=scenario.trajectories_settings[0].population0.particles[0]
    # for ii, (species) in enumerate(particle.species):
    #     print(species.name, particle.masses[ii])
    # sys.exit()
    
    trajectory_ensemble = []
    traj=0
    
    for (one_trajectory_settings, start_time, end_time
          ) in zip(scenario.trajectories_settings,scenario.start_times,scenario.end_times):
        
        runtime0 = time.time()
        ParcelState_0 = get_initial_parcel(one_trajectory_settings, start_time)
        t_start=one_trajectory_settings.t_data[0]
        t_end=one_trajectory_settings.t_data[-1]
        Ntimes = int((t_end - t_start)/dt + 1)
        t_eval = np.linspace(t_start, t_end, Ntimes)
        parcel_states = [ParcelState_0]
        parcel_ts = [t_eval[0]]
        last_written=t_start
        restart_filename=output_path+'/trajectory_'+str(run_number).zfill(6)+'_RESTART.pkl'
        output_filename=output_path+'/trajectory_'+str(run_number).zfill(6)+'.pkl'
        status_filename=output_path+'/trajectory_'+str(run_number).zfill(6)+'_STATUS'
        write_original(ParcelTrajectory(ts=parcel_ts, parcel_states=parcel_states), output_filename, specdata_path=specdata_path)

        print()
        print('Running trajectory', str(run_number).zfill(6)+',', len(N_concs),'particles...')
        
        counter=0        
        #pbar = tqdm.tqdm(total = len(t_eval))
        for (t1,t2) in zip(t_eval[:-1],t_eval[1:]):
            steptime0 = time.time()
            
            ParcelState_Next = update_state(t1, t2,
                ParcelState_0, processes, dt,
                radius_scale=radius_scale,solver=solver,
                accom=accom, verbosity=verbosity,
                mechanism_data_path=mechanism_data_path,
                aq_reactions=aq_reactions, rtol=1e-4, atol=1e-8)
                
            ParcelState_Next.z = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.z_data)
            ParcelState_Next.x = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.x_data)
            ParcelState_Next.y = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.y_data)
            ParcelState_Next.P = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.P_data)
            ParcelState_Next.S = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.S_data)
            ParcelState_Next.T = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.T_data)
                    
            # update the gas concentrations
            # if cocondensation and ParcelState_Next.TraceGas_population:
            #     new_gas_conc = []
            #     for gas in ParcelState_Next.TraceGas_population.gases:
            #         if ParcelState_Next.z < np.min(gas_data[gas.name]['alt']):
            #             f = lambda x, a, b: a*x**b
            #             params, covariance = opt.curve_fit(f, gas_data[gas.name]['alt'][:2], gas_data[gas.name]['ppb'][:2], p0=[1, 0.1])
            #             new_gas_conc.append(f(ParcelState_Next.z, params[0], params[1]))
            #         elif ParcelState_Next.z > np.max(gas_data[gas.name]['alt']):
            #             f = lambda x, a, b: a*x**b
            #             params, covariance = opt.curve_fit(f, gas_data[gas.name]['alt'][-2:], gas_data[gas.name]['ppb'][-2:], p0=[1, 0.1])
            #             new_gas_conc.append(f(ParcelState_Next.z, params[0], params[1]))
            #         else:
            #             new_gas_conc.append(np.interp(ParcelState_Next.z, xp=gas_data[gas.name]['alt'], fp=gas_data[gas.name]['ppb']))
            #     ParcelState_Next.TraceGas_population.concs=new_gas_conc
            
            # for gas, old_conc, new_conc in zip(ParcelState_0.TraceGas_population.gases, ParcelState_0.TraceGas_population.concs, ParcelState_Next.TraceGas_population.concs):
            #     print(gas.name, old_conc, new_conc)
            
                       
            # print timestep and time for timestep
            counter+=1
            print(str(counter)+'/'+str(len(t_eval))+' -- '+str(round(time.time() - steptime0, 2))+ 's/it', flush=True)
            
            # check for NaNs
            total_mass = []
            for particle, num_conc in zip(ParcelState_Next.particle_population.particles, ParcelState_Next.particle_population.num_concs):
                total_mass.append(num_conc*np.sum(particle.masses))
            print(ParcelState_Next.S, 1e9*np.sum(np.array(total_mass)), ParcelState_Next.TraceGas_population.concs[0])
            print()
                    
            # kill the program if there is a NaN
            if np.isnan(np.sum(total_mass)):
                print('ERROR')
                f = open(status_filename, 'w')
                f.write('killed (NaNs)')
                f.close()
                sys.exit()
            
            # update parcel state
            parcel_states.append(ParcelState_Next)
            parcel_ts.append(t2)
            ParcelState_0=ParcelState_Next
            
            # write backup files
            if t2-last_written>=write_every:
                f = open(status_filename, 'w')
                f.write('in progress')
                f.close()
                overwrite(ParcelTrajectory(ts=parcel_ts, parcel_states=parcel_states), output_filename, specdata_path=specdata_path)
                ParcelState_dict = {'time': t2, 'parcel state': ParcelState_Next, 'dt':dt, 'accom':accom, 'verbosity':verbosity, 'radius_scale':radius_scale, 'solver':solver, 'specdata_path':specdata_path, 'mechanism_data_path':mechanism_data_path, 'processes':processes, 'write_every':write_every,'one_trajectory_settings':one_trajectory_settings, 'aq_reactions':aq_reactions}
                pickle.dump(ParcelState_dict, open(restart_filename, 'wb'))
                last_written=t2
            
            #pbar.update(1)
        #pbar.close()
        print()
        print('Solving time:', round(time.time() - runtime0, 2), 'seconds')
        parcel_trajectory = ParcelTrajectory(ts=t_eval, parcel_states=parcel_states)
        print('Maximum saturation ratio:', parcel_trajectory.get_max_S())
        print('Average cloud droplet diameter:', np.round(2.0*1e6*parcel_trajectory.get_avg_droplet_radius(),4), 'micron')
        print('Activated fraction:', str(np.round(100*parcel_trajectory.get_activated_fraction(),3))+'%')
        print()
        
        traj+=1
        parcel_trajectory = ParcelTrajectory(ts=t_eval, parcel_states=parcel_states)
        trajectory_ensemble.append(parcel_trajectory)
        f = open(status_filename, 'w')
        f.write('complete')
        f.close()
    
    return trajectory_ensemble
'''
  



def get_initial_parcel(one_trajectory_settings, start_time=None):
    
    if one_trajectory_settings.x_data is not None:
        x0 = np.interp(start_time,one_trajectory_settings.t_data, one_trajectory_settings.x_data)
    else:
        x0 = one_trajectory_settings.x0
    if one_trajectory_settings.y_data is not None:
        y0 = np.interp(start_time,one_trajectory_settings.t_data, one_trajectory_settings.y_data)
    else:
        y0 = one_trajectory_settings.y0
    if one_trajectory_settings.z_data is not None:
        z0 = np.interp(start_time,one_trajectory_settings.t_data, one_trajectory_settings.z_data)
    else:
        z0 = one_trajectory_settings.z0
        
    if one_trajectory_settings.u_data is not None:
        u0 = np.interp(start_time,one_trajectory_settings.t_data, one_trajectory_settings.u_data)
    else:
        u0 = one_trajectory_settings.u0
    if one_trajectory_settings.v_data is not None:
        v0 = np.interp(start_time,one_trajectory_settings.t_data, one_trajectory_settings.v_data)
    else:
        v0 = one_trajectory_settings.v0
    if one_trajectory_settings.w_data is not None:
        w0 = np.interp(start_time,one_trajectory_settings.t_data, one_trajectory_settings.w_data)
    else:
        w0 = one_trajectory_settings.w0

    if one_trajectory_settings.S_data is not None:
        S0 = np.interp(start_time,one_trajectory_settings.t_data, one_trajectory_settings.S_data)
    else:
        S0 = one_trajectory_settings.S0
    if one_trajectory_settings.P_data is not None:
        P0 = np.interp(start_time,one_trajectory_settings.t_data, one_trajectory_settings.P_data)
    else:
        P0 = one_trajectory_settings.P0
    if one_trajectory_settings.T_data is not None:
        T0 = np.interp(start_time,one_trajectory_settings.t_data, one_trajectory_settings.T_data)
    else:
        T0 = one_trajectory_settings.T0
    
    population0 = one_trajectory_settings.population0
    gas0 = one_trajectory_settings.gas0
    
    return ParcelState(
        x=x0, y=y0, z=z0, u=u0, v=v0, w=w0, 
        S=S0, P=P0, T=T0, particle_population=population0,
        TraceGas_population=gas0)
    
