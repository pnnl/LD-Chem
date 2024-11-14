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

def plot_diameters(trajectory, axis='height'):
    
    # fix this -- make these into time series
    d_drys = []
    d_wets = []
    z = []
    S = []
    t = []
    for i in range(0, len(trajectory.parcel_states)):
        particle_population = trajectory.parcel_states[i].particle_population
        S.append(trajectory.parcel_states[i].S)
        z.append(trajectory.parcel_states[i].z)
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
        fig, ax = plt.subplots(1, 1)
        ax2=ax.twiny()
        ax2.spines['bottom'].set_color('blue')
        ax2.spines['top'].set_color('red')
        ax.tick_params(axis='x', which="both",color='blue', labelcolor='blue')
        ax2.tick_params(axis='x', which="both",color='red', labelcolor='red')
        ax.plot(np.array((d_wets))*1e6, z, '-b')
        ax2.plot(S, z, '-r')
        ax.set_xlabel(r'wet diameter ($\mu$m)', color='blue')
        ax.set_xscale('log')
        ax2.set_xlabel('saturation ratio', color='red')
        ax2.set_xlim(1.0,)
        ax.set_ylabel('altitude (m)')

    
    elif axis == 'time':
        fig, ax = plt.subplots(1, 1)
        ax2=ax.twinx()
        ax2.spines['left'].set_color('blue')
        ax2.spines['right'].set_color('red')
        ax.tick_params(axis='y', which="both",color='blue', labelcolor='blue')
        ax2.tick_params(axis='y', which="both",color='red', labelcolor='red')
        ax.plot(np.array((t))/60, np.array((d_wets))*1e6, '-b')
        ax.set_yscale('log')
        ax2.plot(np.array((t))/60, S, '-r')
        ax.set_ylabel(r'wet diameter ($\mu$m)', color='blue')
        ax2.set_ylabel('saturation ratio', color='red')
        ax.set_xlabel('time (min)')

    
    return fig


def plot_trajectory_values(trajectory):
    
    z = []
    S = []
    t = []
    F_activated = []
    for i in range(0, len(trajectory.parcel_states)):
        S.append(trajectory.parcel_states[i].S)
        z.append(trajectory.parcel_states[i].z)
        F_activated.append(trajectory.parcel_states[i].get_activated_fraction())
        t.append(trajectory.ts[i])
      
    fig, ax = plt.subplots(1, 1)
    ax2=ax.twinx()
    ax2.spines['left'].set_color('blue')
    ax2.spines['right'].set_color('red')
    ax.tick_params(axis='y', which="both",color='blue', labelcolor='blue')
    ax2.tick_params(axis='y', which="both",color='red', labelcolor='red')
    ax.plot(np.array((t))/60, S, '-b')
    ax2.plot(np.array((t))/60, F_activated, '-r')
    ax2.set_ylim(0,)
    ax2.set_ylabel('activated fraction', color='red')
    ax.set_ylabel('saturation ratio', color='blue')
    ax.set_xlabel('time (min)')
    
    return fig
    

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

