#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce and Payton Beeler
"""

import matplotlib.pyplot as plt
import numpy as np
# from SPLAT_initialization import read_FIMS
import sys, tqdm

R = 8.314 # m^3 Pa/mol K

def this_is_a_test():
    print('10:44 am')
    pass

'''
def plot_size_distribution(trajectory,  size_distribution_file, start_time=None, 
                           end_time=None, axis='height'):
    
    Dp_lowers, Dp_uppers, N_measured, N_error = read_FIMS(size_distribution_file, 
                                                  start_time, end_time, 0.0, 
                                                  0.0) # diameters in nm and N in #/cm^3
    
    histograms_wet=[]
    histograms_dry=[]
    S=[]
    z=[]
    t=[]
    ymax=np.ceil(np.max(np.log10(np.array((Dp_lowers)))))+1    
    for i in range(0, len(trajectory.parcel_states)):
        particle_population = trajectory.parcel_states[i].particle_population
        S.append(trajectory.parcel_states[i].S)
        z.append(trajectory.parcel_states[i].z)
        t.append(trajectory.ts[i])
        
        temp_dry = []
        temp_wet = []
        temp_N = []
        for ii,(particle,num_conc) in enumerate(zip(particle_population.particles,particle_population.num_concs)):
            d_dry = particle.get_Ddry()
            d_wet = particle.get_Dwet()
            temp_wet.append(1e9*d_wet)
            temp_dry.append(1e9*d_dry)
            temp_N.append(num_conc/100**3)
        hist_temp=np.histogram(temp_dry, bins=Dp_uppers, weights=temp_N)
        histograms_dry.append(hist_temp[0])        
        hist_temp=np.histogram(temp_wet, bins=np.logspace(np.log10(np.min(Dp_lowers)), ymax, len(Dp_uppers)), weights=temp_N)
        histograms_wet.append(hist_temp[0])
    
    if axis == 'height':
        fig, ax = plt.subplots(1, 1)
        # ax2=ax.twiny()
        # ax2.spines['bottom'].set_color('blue')
        # ax2.spines['top'].set_color('red')
        # ax.tick_params(axis='x', which="both",color='blue', labelcolor='blue')
        # ax2.tick_params(axis='x', which="both",color='red', labelcolor='red')
        # ax.plot(np.array((d_wets))*1e6, z, '-b')
        # ax2.plot(S, z, '-r')
        # ax.set_xlabel(r'wet diameter ($\mu$m)', color='blue')
        # ax.set_xscale('log')
        # ax2.set_xlabel('saturation ratio', color='red')
        # ax2.set_xlim(1.0,)
        # ax.set_ylabel('altitude (m)')

    
    elif axis == 'time':
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(2.0*6.4, 1.0*4.8), sharey=True)
        vmin=0
        vmax=100*np.ceil(np.max(np.array((histograms_dry))/100))
        
        cmap = plt.get_cmap('Greys')
        colors = cmap(np.arange(cmap.N))
        ax1.pcolormesh(np.array((t))/60, Dp_uppers[:-1]/1000, 
                            np.transpose(np.array((histograms_dry))), 
                            cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(len(N_measured)-1):
            c = int(len(colors)*((N_measured[i]-vmin)/(vmax-vmin)))
            if c == len(colors):
                c -= 1
            ax1.plot(0, Dp_uppers[i]/1000, 'o', mfc = colors[c], mec = 'k')
            
        im = ax2.pcolormesh(np.array((t))/60, np.logspace(np.log10(np.min(Dp_lowers)), ymax, len(Dp_uppers))[:-1]/1000, 
                            np.transpose(np.array((histograms_wet))), 
                            cmap=cmap, vmin=vmin, vmax=vmax)
        cax = ax2.inset_axes([1.05, 0.0, 0.05, 1.0], transform=ax2.transAxes)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.set_title(r'$N$ (cm$^{-3}$)', pad=20)
        ax1.set_yscale('log')
        ax1.set_xlim(-1,)
        ax2.set_xlim(-1,)
        ax1.set_ylim(np.min(Dp_lowers)/1000, (10**ymax)/1000)
        ax1.set_ylabel(r'diameter ($\mu$m)')
        ax1.set_xlabel('time (min)')
        ax2.set_xlabel('time (min)')
        ax1.set_title('dry diameter')
        ax2.set_title('wet diameter')
        
    
    return fig
'''   

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
        
        fig, ax = plt.subplots(1, 1)
        ax2=ax.twinx()
        ax2.spines['left'].set_color('blue')
        ax2.spines['right'].set_color('red')
        ax.tick_params(axis='y', which="both",color='blue', labelcolor='blue')
        ax2.tick_params(axis='y', which="both",color='red', labelcolor='red')
        ax.plot(np.array((t))/60, np.array((d_drys))/np.array((d_drys[0])), '-b')
        # ax.set_yscale('log')
        ax2.plot(np.array((t))/60, S, '-r')
        ax.set_ylabel(r'dry diameter/initial diameter', color='blue')
        ax2.set_ylabel('saturation ratio', color='red')
        ax.set_xlabel('time (min)')

    plt.show()    

    return fig


def plot_trajectory_values(trajectory, resolution=60):
    
    dt=trajectory.ts[1]-trajectory.ts[0]
    didx=int(resolution/dt)
    z = []
    S = []
    t = []
    F_activated = []
    print('Plotting activated fraction...')
    pbar = tqdm.tqdm(total = len(trajectory.parcel_states[::didx]))
    for i in range(0, len(trajectory.parcel_states), didx):
        S.append(trajectory.parcel_states[i].S)
        z.append(trajectory.parcel_states[i].z)
        F_activated.append(trajectory.parcel_states[i].get_activated_fraction())
        t.append(trajectory.ts[i])
        pbar.update(1)
    pbar.close()
        
      
    fig, ax = plt.subplots(1, 1)
    ax2=ax.twinx()
    ax2.spines['left'].set_color('blue')
    ax2.spines['right'].set_color('red')
    ax.tick_params(axis='y', which="both",color='blue', labelcolor='blue')
    ax2.tick_params(axis='y', which="both",color='red', labelcolor='red')
    ax.plot(np.array((t))/60, S, '-b')
    ax2.plot(np.array((t))/60, F_activated, '-r')
    ax2.set_ylim(0,1)
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
    
    # idx_aq = -1
    # for i in range(len(trajectory.parcel_states[0].particle_population.particles[0].species)):
    #     name = trajectory.parcel_states[0].particle_population.particles[0].species[i].name
    #     if name == species:
    #         idx_aq = i
    # if idx_aq == -1:
    #     print('PLOTTING WARNING:', species, 'is not tracked in this simulation!')
    #     return
        
    for i in range(0, len(trajectory.parcel_states)):
        z.append(trajectory.parcel_states[i].z)
        t.append(trajectory.ts[i])
        
        particle_population = trajectory.parcel_states[i].particle_population
        temp_perpart = []
        temp_total = 0
        for ii,(particle,num_conc) in enumerate(zip(particle_population.particles,particle_population.num_concs)):
            temp_perpart.append(particle.masses[particle.get_species_idx(species)])
            temp_total += num_conc*particle.masses[particle.get_species_idx(species)]
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
        
        plt.plot(np.array((t))/60, np.array((aq_masses))/np.array((aq_masses[0])), '-r')
        plt.ylabel(str(species)+r' mass ratio')
        plt.xlabel('time (min)')
        plt.show()
        
        plt.plot(np.array((t))/60, np.array((total_mass))*1e9, '-g')
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
    
    return
    

def plot_aqueous_fraction(trajectory, species, axis='height', resolution=60):
    
    z = []
    t = []
    aq_concentrations = []
    gas_concentrations = []
    
    idx_aq = -1
    for i in range(len(trajectory.parcel_states[0].particle_population.particles[0].species)):
        name = trajectory.parcel_states[0].particle_population.particles[0].species[i].name
        if name == species:
            idx_aq = i
    idx_gas = -1
    for i in range(len(trajectory.parcel_states[0].TraceGas_population.gases)):
        name = trajectory.parcel_states[0].TraceGas_population.gases[i].name
        if name == species:
            idx_gas = i
    if idx_aq == -1:
        print('PLOTTING WARNING: aqueous', species, 'is not tracked in this simulation!')
        return
    elif idx_gas == -1:
        print('PLOTTING WARNING:', species, 'gas is not tracked in this simulation!')
        return
    
    total_aq_timeseries = [] # ppb
    total_gas_timeseries = [] # ppb
    wL_timeseries = []
    #Fa_eq_timeseries = []
    dt=trajectory.ts[1]-trajectory.ts[0]
    didx=int(resolution/dt)
    print('Plotting aqueous fraction...')
    pbar = tqdm.tqdm(total = len(trajectory.parcel_states[::didx]))
    for i in range(0, len(trajectory.parcel_states), didx):
        z.append(trajectory.parcel_states[i].w*trajectory.ts[i])
        t.append(trajectory.ts[i])
        
        particle_population = trajectory.parcel_states[i].particle_population
        aq_mole_conc = 0
        wL = 0
        for ii,(particle,num_conc) in enumerate(zip(particle_population.particles,particle_population.num_concs)):
            aq_mole_conc += (particle.masses[idx_aq]/particle.species[idx_aq].molar_mass)*num_conc # mol/m^3                
            wL += num_conc*particle.masses[particle.idx_h2o]/particle.get_rho_w() # m^3 water per m^3 air
                
        wL_timeseries.append(wL)
        T = trajectory.parcel_states[i].T
        P = trajectory.parcel_states[i].P
        #H_eff = trajectory.parcel_states[i].TraceGas_population.gases[idx_gas].get_Heff(T) # mol/m^3*Pa
        #Fa_eq_timeseries.append((H_eff*R*T*wL)/(1+H_eff*R*T*wL))
        total_aq_timeseries.append(1e9*aq_mole_conc*(R*T/P)) # ppb in aeous phase
        total_gas_timeseries.append(trajectory.parcel_states[i].TraceGas_population.concs[idx_gas])
        pbar.update(1)
    pbar.close()
    
    #Fa_eq_timeseries = np.array((Fa_eq_timeseries))
    total_aq_timeseries = np.array((total_aq_timeseries))
    total_gas_timeseries = np.array((total_gas_timeseries))
    wL_timeseries = np.array((wL_timeseries))
    
    # if axis == 'height':
    #     fig1, ax1 = plt.subplots(1, 1)
    #     ax1.plot(total_aq_timeseries/(total_aq_timeseries+total_gas_timeseries), np.array((z)), '-b', label='model aqueous fraction', linewidth=2)
    #     ax1.plot(total_gas_timeseries/(total_aq_timeseries+total_gas_timeseries), np.array((z)), '-r', label='model gas fraction', linewidth=2)
    #     ax1.plot(1-Fa_eq_timeseries, np.array((z)), '--k', linewidth=2)
    #     ax1.plot(Fa_eq_timeseries, np.array((z)), '--k', linewidth=2)
    #     ax1.set_xlabel(str(species)+' mole fraction')
    #     ax1.legend(loc='center', ncol=2, bbox_to_anchor=(0.5, 1.1))
    #     ax1.set_ylabel('altitude (m)')
    #     ymin = np.floor(np.log10(np.min(Fa_eq_timeseries)))-1
    #     ax1.set_xscale('log')
    #     # plt.text(t[-1]/60, 1.15*Fa_eq_timeseries[-1], 'equilibrium aqueous fraction', ha='right', va='bottom')
    #     # plt.text(t[-1]/60, 1.15*(1-Fa_eq_timeseries[-1]), 'equilibrium gas fraction', ha='right', va='bottom')
    #     ax1.set_xlim(10**ymin, 5)        
        
    #     fig2, ax2 = plt.subplots(1, 1)
    #     ax2.plot(total_gas_timeseries, np.array((z)), '-r', label='gas', linewidth=2)
    #     ax2.plot(total_aq_timeseries, np.array((z)), '-b', label='aqueous', linewidth=2)
    #     ax2.plot(total_aq_timeseries+total_gas_timeseries, np.array((z)), '--k', label='total', linewidth=2)
    #     ax2.set_xlabel(str(species)+' concentration (ppb)')
    #     ax2.set_ylabel('Altitude (m)')
    #     ax2.set_xscale('log')
    #     ax2.legend(loc='center', ncol=3, bbox_to_anchor=(0.5, 1.1))
    
    if axis == 'time':
        fig1, ax1 = plt.subplots(1, 1)
        ax2 = ax1.twinx()
        ax2.spines['left'].set_color('blue')
        ax2.spines['right'].set_color('red')
        ax1.tick_params(axis='y', which="both",color='blue', labelcolor='blue')
        ax2.tick_params(axis='y', which="both",color='red', labelcolor='red')
        
        ax1.plot(np.array((t))/60, total_aq_timeseries/(total_aq_timeseries+total_gas_timeseries), '--b', label='aqueous fraction', linewidth=2)
        ax1.plot(np.array((t))/60, total_gas_timeseries/(total_aq_timeseries+total_gas_timeseries), '-b', label='gas fraction', linewidth=2)
        ax1.set_ylabel(str(species)+' fraction')
        ax1.legend(loc='center', ncol=2, bbox_to_anchor=(0.5, 1.1))
        ax1.set_xlabel('time (min)')
        
        ax2.plot(np.array((t))/60, wL_timeseries, '-r', linewidth=2)
        
        ax1.set_ylim(0,1)
        ax2.set_ylim(0,)
        
    return fig1
    

