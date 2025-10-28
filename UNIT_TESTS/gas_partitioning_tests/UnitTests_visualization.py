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

def equilibrium_fractions(ParcelState, species):
    
    idx_aq = ParcelState.particle_population.particles[0].get_species_idx(species)
    idx_gas = ParcelState.TraceGas_population.get_species_idx(species)

    if not type(idx_aq)==type(1):
        print('PLOTTING WARNING: aqueous', species, 'is not tracked in this simulation!')
        sys.exit()
    elif not type(idx_gas)==type(1):
        print('PLOTTING WARNING:', species, 'gas is not tracked in this simulation!')
        sys.exit()

    particle_population = ParcelState.particle_population
    aq_mole_conc = 0
    wL = 0
    for ii,(particle,num_conc) in enumerate(zip(particle_population.particles,particle_population.num_concs)):
        aq_mole_conc += (particle.masses[idx_aq]/particle.species[idx_aq].molar_mass)*num_conc # mol/m^3                
        wL += num_conc*particle.masses[particle.idx_h2o]/particle.get_rho_w() # m^3 water per m^3 air
            
    T = ParcelState.T
    P = ParcelState.P
    H_eff = ParcelState.TraceGas_population.gases[idx_gas].get_Heff(T) # mol/m^3*Pa
    Fa_eq=(H_eff*R*T*wL)/(1+H_eff*R*T*wL)
    total_aq=1e9*aq_mole_conc*(R*T/P) # ppb in aeous phase
    total_gas=ParcelState.TraceGas_population.concs[idx_gas]
    
    aqueous = {'equilibrium': Fa_eq, 'model': total_aq/(total_aq+total_gas)}
    gas = {'equilibrium': 1-Fa_eq, 'model': total_gas/(total_aq+total_gas)}
    
    return aqueous, gas

def plot_sulfate_concentrations(trajectory, axis='height'):
    
    z = []
    t = []    
    names = []
    for i in range(len(trajectory.parcel_states[0].particle_population.particles[0].species)):
        name = trajectory.parcel_states[0].particle_population.particles[0].species[i].name
        names.append(name)
    if 'SO2' not in names or 'HSO3' not in names or 'SO3' not in names:
        print('PLOTTING WARNING: not all aqueous sulfur species are tracked in this simulation!')
        return

    SO2_timeseries = [] # mol/L
    HSO3_timeseries = [] # mol/L
    SO3_timeseries = [] # mol/L
    SO2_gas_timeseries = []
    pH_timeseries = []
    for i in range(0, len(trajectory.parcel_states)):
        z.append(trajectory.parcel_states[i].w*trajectory.ts[i])
        t.append(trajectory.ts[i])
        particle_population = trajectory.parcel_states[i].particle_population
        moles_SO2=0
        moles_HSO3=0
        moles_SO3=0
        total_water_volume=0
        pH_temp=[]
        for ii,(particle,num_conc) in enumerate(zip(particle_population.particles,particle_population.num_concs)):
            total_water_volume+=100*(particle.get_vol_tot()-particle.get_vol_dry()) # L
            moles_SO2+=particle.masses[particle.get_species_idx('SO2')]/particle.species[particle.get_species_idx('SO2')].molar_mass
            moles_HSO3+=particle.masses[particle.get_species_idx('HSO3')]/particle.species[particle.get_species_idx('HSO3')].molar_mass
            moles_SO3+=particle.masses[particle.get_species_idx('SO3')]/particle.species[particle.get_species_idx('SO3')].molar_mass
            pH_temp.append(-1.0*np.log10((particle.masses[particle.get_species_idx('H+')]/particle.species[particle.get_species_idx('H+')].molar_mass)/(100*(particle.get_vol_tot()-particle.get_vol_dry()))))
        SO2_timeseries.append(moles_SO2/total_water_volume)
        HSO3_timeseries.append(moles_HSO3/total_water_volume)
        SO3_timeseries.append(moles_SO3/total_water_volume)
        pH_timeseries.append(pH_temp)
        SO2_gas_timeseries.append(trajectory.parcel_states[i].TraceGas_population.concs[trajectory.parcel_states[i].TraceGas_population.get_species_idx('SO2')])
        
    SO2_timeseries=np.array((SO2_timeseries))
    HSO3_timeseries=np.array((HSO3_timeseries))
    SO3_timeseries=np.array((SO3_timeseries))
    
    if axis == 'height':
        plt.plot(SO2_timeseries/(SO2_timeseries+HSO3_timeseries+SO3_timeseries), np.array((z)), '-b', label=r'SO$_2$', linewidth=2)
        plt.plot(HSO3_timeseries/(SO2_timeseries+HSO3_timeseries+SO3_timeseries), np.array((z)), '-r', label=r'HSO$_3^-$', linewidth=2)
        plt.plot(SO3_timeseries/(SO2_timeseries+HSO3_timeseries+SO3_timeseries), np.array((z)), '-g', label=r'SO$_3^{2-}$', linewidth=2)
        plt.plot(SO2_timeseries+HSO3_timeseries+SO3_timeseries, np.array((z)), '--k', label='total', linewidth=2)
        plt.xlabel('aqueous mole fraction')
        plt.legend(loc='center', ncol=4, bbox_to_anchor=(0.5, 1.1))
        plt.ylabel('altitude (m)')
        plt.show()
        
        plt.plot(SO2_gas_timeseries, np.array((z)), '-r', label='gas', linewidth=2)
        # plt.plot(total_aq_timeseries, np.array((z)), '-b', label='aqueous', linewidth=2)
        # plt.plot(total_aq_timeseries+total_gas_timeseries, np.array((z)), '--k', label='total', linewidth=2)
        # plt.xlabel(str(species)+' concentration (ppb)')
        # plt.ylabel('Altitude (m)')
        # plt.xscale('log')
        # plt.legend(loc='center', ncol=3, bbox_to_anchor=(0.5, 1.1))
        plt.show()
    
    elif axis == 'time':
        plt.plot(np.array((t))/60, SO2_timeseries/(SO2_timeseries+HSO3_timeseries+SO3_timeseries), '-b', label='SO$_2$', linewidth=2)
        plt.plot(np.array((t))/60, HSO3_timeseries/(SO2_timeseries+HSO3_timeseries+SO3_timeseries), '-r', label=r'HSO$_3^-$', linewidth=2)
        plt.plot(np.array((t))/60, SO3_timeseries/(SO2_timeseries+HSO3_timeseries+SO3_timeseries), '-g', label=r'SO$_3^{2-}$', linewidth=2)
        plt.ylabel('aqueous mole fraction')
        plt.legend(loc='center', ncol=3, bbox_to_anchor=(0.5, 1.1))
        plt.xlabel('time (min)')
        # ymin = np.floor(np.log10(np.min(Fa_eq_timeseries)))-1
        # plt.yscale('log')
        # plt.text(t[-1]/60, 1.15*Fa_eq_timeseries[-1], 'equilibrium aqueous fraction', ha='right', va='bottom')
        # plt.text(t[-1]/60, 1.15*(1-Fa_eq_timeseries[-1]), 'equilibrium gas fraction', ha='right', va='bottom')
        # plt.ylim(10**ymin, 5)
        # plt.xlim(0,)
        plt.show()
        
        
        plt.plot(np.array((t))/60, pH_timeseries, '-r', label='gas', linewidth=2)
        # plt.plot(np.array((t))/60, total_aq_timeseries, '-b', label='aqueous', linewidth=2)
        # plt.plot(np.array((t))/60, total_aq_timeseries+total_gas_timeseries, '--k', label='total', linewidth=2)
        # plt.ylabel(str(species)+' concentration (ppb)')
        # plt.xlabel('time (min)')
        # plt.yscale('log')
        # plt.xlim(0,)
        # plt.legend(loc='center', ncol=3, bbox_to_anchor=(0.5, 1.1))
        plt.show()

    
    return


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

def plot_pH(trajectory, axis='height'):
    
    z=[]
    t=[]
    pH_timeseries = []
    for i in range(0, len(trajectory.parcel_states)):
        z.append(trajectory.parcel_states[i].w*trajectory.ts[i])
        t.append(trajectory.ts[i])
        particle_population = trajectory.parcel_states[i].particle_population
        pH_temp=[]
        for ii,(particle,num_conc) in enumerate(zip(particle_population.particles,particle_population.num_concs)):
            pH_temp.append(particle.get_pH())
        pH_timeseries.append(pH_temp)
        
    pH_timeseries=np.array((pH_timeseries))
    
    if axis == 'height':

        plt.plot(pH_timeseries, np.array((z)), '-r', linewidth=2)
        plt.xlabel('pH')
        plt.ylabel('Altitude (m)')
        plt.show()
    
    elif axis == 'time':
        
        plt.plot(np.array((t))/60, pH_timeseries, '-r', linewidth=2)
        plt.ylabel('pH')
        plt.xlabel('time (min)')
        plt.xlim(0,)
        plt.show()
    
    return