#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver for the Multiscale Particle-based (MultiPart) microphysics model

@author: fier887
"""
# from dataclasses import dataclass
from systems import ParcelState
from systems import Processes
from scenario import create_scenario_from_DNS, create_parcel_scenario#, create_constant_parcel_scenario
# from typing import Tuple
from typing import Callable
import numpy as np
from dataclasses import replace
import copy, tqdm, time
import matplotlib.pyplot as plt

from systems import update_state, ParcelTrajectory, TrajectoryEnsemble, TrajectoryInteractions

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


def simulate_parcel_trajectories(N_scenarios=1,
        z_start=0.,z_end=1000.,dt=1.,
        Ddry=100e-9,sigma=1.0,Ntot=1e6, Npart=1,
        updraft_velocity=0.5,S0=-0.15,P0=101325,T0=298,
        accom=1., verbosity=50,
        radius_scale='lin',solver='CVODE',
        species_names=['NaCl'], mass_fractions=np.array([1.]),
        specdata_path='../species_data/', condensation = True, 
        collisions = False, settling = False,
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
            
        end_time=z_end/updraft_velocity
                
        scenario = create_parcel_scenario(
                Ddry=D_dry,sigma=s,Ntot=N_tot,Npart=N_part,
                updraft_velocity=V,
                S0=S_0,P0=P_0,T0=T_0,z_start=z0,z_end=z_f,
                species_names=species_names,mass_fractions= mass_fractions,
                dt=timestep, specdata_path=specdata_path)  
        
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
                ParcelState_Next = update_state(
                    ParcelState_0, processes, dt, 
                    radius_scale=radius_scale,solver=solver,
                    sigma=sigma, accom=accom, verbosity=verbosity)
                parcel_states.append(copy.deepcopy(ParcelState_Next))
                ParcelState_0=replace(ParcelState_Next)
                pbar.update(1)
            pbar.close()
            print('Solving time:', round(time.time() - runtime0, 2), 'seconds')
            print()
                
            parcel_trajectory = ParcelTrajectory(ts=t_eval, parcel_states=parcel_states)
            trajectory_ensemble.append(parcel_trajectory)
        
    return trajectory_ensemble

    
def get_initial_parcel(one_trajectory_settings, start_time=None):
    
    # if x_fun is defined, x0 is ignored
    if isinstance(one_trajectory_settings.x_fun, Callable):
        x0 = one_trajectory_settings.x_fun(start_time)
    else:
        x0 = one_trajectory_settings.x0
    if isinstance(one_trajectory_settings.y_fun, Callable):
        y0 = one_trajectory_settings.y_fun(start_time)
    else:
        y0 = one_trajectory_settings.y0
    if isinstance(one_trajectory_settings.z_fun, Callable):
        z0 = one_trajectory_settings.z_fun(start_time)
    else:
        z0 = one_trajectory_settings.z0
        
    if isinstance(one_trajectory_settings.u_fun, Callable):
        u0 = one_trajectory_settings.u_fun(start_time)
    else:
        u0 = one_trajectory_settings.u0
    if isinstance(one_trajectory_settings.v_fun, Callable):
        v0 = one_trajectory_settings.v_fun(start_time)
    else:
        v0 = one_trajectory_settings.v0
    if isinstance(one_trajectory_settings.w_fun, Callable):
        w0 = one_trajectory_settings.w_fun(start_time)
    else:
        w0 = one_trajectory_settings.w0

    if isinstance(one_trajectory_settings.S_fun, Callable):
        S0 = one_trajectory_settings.S_fun(start_time)
    else:
        S0 = one_trajectory_settings.S0
    if isinstance(one_trajectory_settings.P_fun, Callable):
        P0 = one_trajectory_settings.P_fun(start_time)
    else:
        P0 = one_trajectory_settings.P0
    if isinstance(one_trajectory_settings.T_fun, Callable):
        T0 = one_trajectory_settings.T_fun(start_time)
    else:
        T0 = one_trajectory_settings.T0
    
    population0 = one_trajectory_settings.population0
    gas0 = one_trajectory_settings.gas0
    
    return ParcelState(
        x=x0, y=y0, z=z0, u=u0, v=v0, w=w0, 
        S=S0, P=P0, T=T0, particle_population=population0,
        TraceGas_population=gas0)
    
