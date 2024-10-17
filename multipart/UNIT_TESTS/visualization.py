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

def plot_parcel_trajectory(trajectory, axis='height'):
    
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
        
        plt.plot(np.array((d_wets))/np.array((d_drys)), z, '-g')
        plt.xscale('log')
        plt.xlabel('wet diameter/dry diameter')
        plt.ylabel('altitude (m)')
        plt.show()
    
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

