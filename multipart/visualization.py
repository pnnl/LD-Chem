#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce and Payton Beeler
"""

import matplotlib.pyplot as plt
import numpy as np

def this_is_a_test():
    print('10:44 am')
    pass

def plot_SS_diameters(trajectory, axis='height'):
    
    # fix this -- make these into time series
    d_drys = []
    d_wets = []
    z = []
    S = []
    P = []
    T = []
    t = []
    for i in range(0, len(trajectory.parcel_states)):
        particle_population = trajectory.parcel_states[i].particle_population
        S.append(trajectory.parcel_states[i].S)
        T.append(trajectory.parcel_states[i].T)
        P.append(trajectory.parcel_states[i].P)
        z.append(trajectory.parcel_states[i].w*trajectory.ts[i])
        t.append(trajectory.ts[i])
        
        temp_dry = []
        temp_wet = []
        for ii,(particle,num_conc) in enumerate(zip(particle_population.particles,particle_population.num_concs)):
            d_dry = particle.get_Ddry()
            d_wet = particle.get_Dwet()
            temp_wet.append(d_wet)
            temp_dry.append(d_dry)
        d_drys.append(temp_dry)
        d_wets.append(temp_wet)
    
    if axis == 'height':
        plt.plot(np.array((d_wets))*1e6, z, '-b')
        plt.xscale('log')
        plt.xlabel('wet diameter (micron)')
        plt.ylabel('altitude (m)')
        plt.show()
        
        plt.plot(np.array((S))-1, z, '-r')
        plt.xlabel('supersaturation')
        plt.ylabel('altitude (m)')
        plt.xlim(0,)
        plt.show()
        
        # plt.plot(np.array((d_wets))/np.array((d_drys)), z, '-g')
        # plt.xscale('log')
        # plt.xlabel('wet diameter/dry diameter')
        # plt.ylabel('altitude (m)')
        # plt.show()
    
    elif axis == 'time':
        plt.plot(np.array((t))/60, np.array((d_wets))*1e6, '-b')
        plt.ylabel('wet diameter (micron)')
        plt.xlabel('time (min)')
        plt.show()
        
        plt.plot(np.array((t))/60, np.array((S))-1, '-r')
        plt.ylabel('supersaturation')
        plt.xlabel('time (min)')
        plt.xlim(0,)
        plt.show()
        
        plt.plot(np.array((t))/60, np.array((d_wets))/np.array((d_drys)), '-g')
        plt.ylabel('wet diameter/dry diameter')
        plt.xlabel('time (min)')
        plt.show()
    
    # print()
    # print(np.max(SS))
    # print()
    # print(np.min(np.array((d_wets))/np.array((d_drys))))
    
    return

def plot_aq_species(trajectory, species, axis='height'):
    
    # fix this -- make these into time series
    total_mass = []
    z = []
    t = []
    aq_masses = []
    total_mass = []
    
    idx_aq = -1
    for i in range(len(trajectory.parcel_states[0].particle_population.particles[0].species)):
        name = trajectory.parcel_states[0].particle_population.particles[0].species[i].name
        if name == species:
            idx_aq = i
    if idx_aq == -1:
        print('PLOTTING WARNING:', species, 'is not tracked in this simulation!')
        return
        
    for i in range(0, len(trajectory.parcel_states)):
        z.append(trajectory.parcel_states[i].w*trajectory.ts[i])
        t.append(trajectory.ts[i])
        
        particle_population = trajectory.parcel_states[i].particle_population
        temp_perpart = []
        temp_total = 0
        for ii,(particle,num_conc) in enumerate(zip(particle_population.particles,particle_population.num_concs)):
            temp_perpart.append(particle.masses[idx_aq])
            temp_total += num_conc*particle.masses[idx_aq]
        aq_masses.append(temp_perpart)
        total_mass.append(temp_total)
    
    if axis == 'height':
        plt.plot(np.array((aq_masses))*1e9, z, '-b')
        plt.xscale('log')
        plt.xlabel(str(species)+r' mass ($\mu$g)')
        plt.ylabel('altitude (m)')
        plt.show()
        
        plt.plot(np.array((total_mass)), z, '-r')
        plt.xlabel(str(species)+r' mass concentration ($\mu$g/m$^3$)')
        plt.ylabel('altitude (m)')
        plt.xlim(0,)
        plt.show()
    
    elif axis == 'time':
        plt.plot(np.array((t))/60, np.array((aq_masses))*1e9, '-b')
        plt.ylabel(str(species)+r' mass ($\mu$g)')
        plt.xlabel('time (min)')
        plt.show()
        
        plt.plot(np.array((t))/60, np.array((total_mass)), '-r')
        plt.ylabel(str(species)+r' mass concentration ($\mu$g/m$^3$)')
        plt.xlabel('time (min)')
        plt.xlim(0,)
        plt.show()
    
    return


def plot_gas_species(trajectory, species, axis='height'):
    
    # fix this -- make these into time series
    z = []
    t = []
    gas_conc = []
    
    idx_gas = -1
    for i in range(len(trajectory.parcel_states[0].TraceGas_population.gases)):
        name = trajectory.parcel_states[0].TraceGas_population.gases[i].name
        if name == species:
            idx_gas = i
    if idx_gas == -1:
        print('PLOTTING WARNING:', species, 'is not tracked in this simulation!')
        return
    
    for i in range(0, len(trajectory.parcel_states)):
        z.append(trajectory.parcel_states[i].w*trajectory.ts[i])
        t.append(trajectory.ts[i])
        gas_conc.append(trajectory.parcel_states[i].TraceGas_population.concs[idx_gas])
    
    if axis == 'height':
        plt.plot(gas_conc, z, '-b')
        plt.xscale('log')
        plt.xlabel(str(species)+r' gas concentration (ppb)')
        plt.ylabel('altitude (m)')
        plt.show()
    
    elif axis == 'time':
        plt.plot(np.array((t))/60, gas_conc, '-b')
        plt.ylabel(str(species)+r' gas concentration (ppb)')
        plt.xlabel('time (min)')
        plt.show()
    
    return


def plot_DNS_trajectories(trajectory_ensemble):
    
    # fix this -- make these into time series
    d_drys = []
    d_wets = []
    z = []
    SS = []
    P = []
    T = []
    for i in range(0, len(trajectory_ensemble.parcel_trajectories)):
        particle_population = trajectory_ensemble.parcel_trajectories.parcel_states[i].particle_population
        SS.append(trajectory_ensemble.parcel_trajectories.parcel_states[i].S)
        T.append(trajectory_ensemble.parcel_trajectories.parcel_states[i].T)
        P.append(trajectory_ensemble.parcel_trajectories.parcel_states[i].P)
        z.append(trajectory_ensemble.parcel_trajectories.parcel_states[i].w*trajectory_ensemble.ts[i])
        
        temp_dry = []
        temp_wet = []
        for ii,(particle,num_conc) in enumerate(zip(particle_population.particles,particle_population.num_concs)):
            d_dry = particle.get_Ddry()
            d_wet = particle.get_Dwet()
            temp_wet.append(d_wet)
            temp_dry.append(d_dry)
        d_drys.append(temp_dry)
        d_wets.append(temp_wet)
    
    plt.plot(np.array((d_wets))*1e6, z, '-b')
    plt.xscale('log')
    plt.show()
    
    plt.plot(SS, z, '-r')
    plt.xlim(0,)
    plt.show()
    
    print()
    print(np.max(SS))

