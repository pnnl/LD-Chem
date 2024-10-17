#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:38:04 2024

@author: beel083
"""

import matplotlib.pyplot as plt
import numpy as np
import sys

R = 8.314 # m^3 Pa/mol K

def plot_equilibrium_fractions(trajectory, species, axis='height'):
    
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
    Fa_eq_timeseries = []
    for i in range(0, len(trajectory.parcel_states)):
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
        H_eff = trajectory.parcel_states[i].TraceGas_population.gases[idx_gas].get_Heff(T) # mol/m^3*Pa
        Fa_eq_timeseries.append((H_eff*R*T*wL)/(1+H_eff*R*T*wL))
        total_aq_timeseries.append(1e9*aq_mole_conc*(R*T/P)) # ppb in aeous phase
        total_gas_timeseries.append(trajectory.parcel_states[i].TraceGas_population.concs[idx_gas])
    
    Fa_eq_timeseries = np. array((Fa_eq_timeseries))
    total_aq_timeseries = np. array((total_aq_timeseries))
    total_gas_timeseries = np. array((total_gas_timeseries))
    
    if axis == 'height':
        plt.plot(total_aq_timeseries/(total_aq_timeseries+total_gas_timeseries), np.array((z)), '-b', label='model aqueous fraction', linewidth=2)
        plt.plot(total_gas_timeseries/(total_aq_timeseries+total_gas_timeseries), np.array((z)), '-r', label='model gas fraction', linewidth=2)
        plt.plot(1-Fa_eq_timeseries, np.array((z)), '--k', linewidth=2)
        plt.plot(Fa_eq_timeseries, np.array((z)), '--k', linewidth=2)
        plt.xlabel(str(species)+' mole fraction')
        plt.legend(loc='center', ncol=2, bbox_to_anchor=(0.5, 1.1))
        plt.ylabel('altitude (m)')
        ymin = np.floor(np.log10(np.min(Fa_eq_timeseries)))-1
        plt.xscale('log')
        # plt.text(t[-1]/60, 1.15*Fa_eq_timeseries[-1], 'equilibrium aqueous fraction', ha='right', va='bottom')
        # plt.text(t[-1]/60, 1.15*(1-Fa_eq_timeseries[-1]), 'equilibrium gas fraction', ha='right', va='bottom')
        plt.xlim(10**ymin, 5)
        plt.show()
        
        
        plt.plot(total_gas_timeseries, np.array((z)), '-r', label='gas', linewidth=2)
        plt.plot(total_aq_timeseries, np.array((z)), '-b', label='aqueous', linewidth=2)
        plt.plot(total_aq_timeseries+total_gas_timeseries, np.array((z)), '--k', label='total', linewidth=2)
        plt.xlabel(str(species)+' concentration (ppb)')
        plt.ylabel('Altitude (m)')
        plt.xscale('log')
        plt.legend(loc='center', ncol=3, bbox_to_anchor=(0.5, 1.1))
        plt.show()
    
    elif axis == 'time':
        plt.plot(np.array((t))/60, total_aq_timeseries/(total_aq_timeseries+total_gas_timeseries), '-b', label='model aqueous fraction', linewidth=2)
        plt.plot(np.array((t))/60, total_gas_timeseries/(total_aq_timeseries+total_gas_timeseries), '-r', label='model gas fraction', linewidth=2)
        plt.plot(np.array((t))/60, 1-Fa_eq_timeseries, '--k', linewidth=2)
        plt.plot(np.array((t))/60, Fa_eq_timeseries, '--k', linewidth=2)
        plt.ylabel(str(species)+' mole fraction')
        plt.legend(loc='center', ncol=2, bbox_to_anchor=(0.5, 1.1))
        plt.xlabel('time (min)')
        ymin = np.floor(np.log10(np.min(Fa_eq_timeseries)))-1
        plt.yscale('log')
        plt.text(t[-1]/60, 1.15*Fa_eq_timeseries[-1], 'equilibrium aqueous fraction', ha='right', va='bottom')
        plt.text(t[-1]/60, 1.15*(1-Fa_eq_timeseries[-1]), 'equilibrium gas fraction', ha='right', va='bottom')
        plt.ylim(10**ymin, 5)
        plt.xlim(0,)
        plt.show()
        
        
        plt.plot(np.array((t))/60, total_gas_timeseries, '-r', label='gas', linewidth=2)
        plt.plot(np.array((t))/60, total_aq_timeseries, '-b', label='aqueous', linewidth=2)
        plt.plot(np.array((t))/60, total_aq_timeseries+total_gas_timeseries, '--k', label='total', linewidth=2)
        plt.ylabel(str(species)+' concentration (ppb)')
        plt.xlabel('time (min)')
        plt.yscale('log')
        plt.xlim(0,)
        plt.legend(loc='center', ncol=3, bbox_to_anchor=(0.5, 1.1))
        plt.show()
