#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:38:52 2024

@author: fier887
"""
import numpy as np
import sys
from processes import water_uptake
from processes.air_thermo import H2O_gas_conc

def get_number(string_val):
    if string_val.endswith('\n'):
        string_val = string_val[:-2]
        
    if '×' in string_val:
        idx = string_val.find('×')
        front_part = float(string_val[:idx])
        back_part = string_val[(idx+1):][2:]
        if back_part.startswith('−'):
            exponent = -float(back_part[1:])
        else:
            exponent = float(back_part)
        number = front_part*10.**exponent
    else:
        number = float(string_val)
    return number


def check_gas_condensation(ParcelState_0, ParcelState_Next, gas_feedback):

    for ii, (name, dc_dt) in enumerate(zip(gas_feedback.names, gas_feedback.dc_dts)):
        Cpart_0 = 0.0
        for particle, num_conc in zip(ParcelState_0.particle_population.particles,ParcelState_0.particle_population.num_concs):
            idx = particle.get_species_idx(name)
            mols=particle.masses[idx]/particle.species[idx].molar_mass # mol
            Cpart_0 += 1e9*mols*num_conc*((8.314*ParcelState_0.T)/ParcelState_0.P) # ppb
            
        Cpart_next = 0.0
        for particle, num_conc in zip(ParcelState_Next.particle_population.particles,ParcelState_Next.particle_population.num_concs):
            idx = particle.get_species_idx(name)
            mols=particle.masses[idx]/particle.species[idx].molar_mass # mol
            Cpart_next += 1e9*mols*num_conc*((8.314*ParcelState_Next.T)/ParcelState_Next.P) # ppb

        check = np.isclose(-1.0*dc_dt, Cpart_next-Cpart_0)
        if not check:
            gas_feedback.dc_dts[ii]=-1.0*(Cpart_next-Cpart_0)
    
    return gas_feedback


def check_water_condensation(ParcelState_0, ParcelState_Next, dwc_dt):
    
    masses_0=np.zeros(len(ParcelState_0.particle_population.particles))
    for ii, (particle, num_conc) in enumerate(zip(ParcelState_0.particle_population.particles,ParcelState_0.particle_population.num_concs)):
        masses_0[ii]=particle.masses[particle.idx_h2o]*num_conc # kg water / m^3 air
    
    masses_next=np.zeros(len(ParcelState_0.particle_population.particles))
    for ii, (particle, num_conc) in enumerate(zip(ParcelState_Next.particle_population.particles,ParcelState_Next.particle_population.num_concs)):
        masses_next[ii]=particle.masses[particle.idx_h2o]*num_conc # kg water / m^3 air
    
    check = np.isclose(dwc_dt, np.sum(masses_next-masses_0))
    if not check:
        print('ERROR: Moles of condensing water does not match the change in particles!')
        sys.exit()
   
    return
    
def water_mole_balance(ParcelState_0, ParcelState_Next):
    
    Cgas_0 = 1e9*((ParcelState_0.S*water_uptake.es(ParcelState_0.T - 273.15))/ParcelState_0.P) # ppb
    Cpart_0 = 0
    for particle, num_conc in zip(ParcelState_0.particle_population.particles,ParcelState_0.particle_population.num_concs):
        idx = particle.idx_h2o
        mols=particle.masses[idx]/particle.species[idx].molar_mass # mol
        Cpart_0 += 1e9*mols*num_conc*((8.314*ParcelState_0.T)/ParcelState_0.P) # ppb
    
    Cgas_next = 1e9*((ParcelState_Next.S*water_uptake.es(ParcelState_Next.T - 273.15))/ParcelState_Next.P) # ppb
    Cpart_next = 0
    for particle, num_conc in zip(ParcelState_Next.particle_population.particles,ParcelState_Next.particle_population.num_concs):
        idx = particle.idx_h2o
        mols=particle.masses[idx]/particle.species[idx].molar_mass # mol
        Cpart_next += 1e9*mols*num_conc*((8.314*ParcelState_Next.T)/ParcelState_Next.P) # ppb
    
    check = np.isclose(-1.0*(Cgas_next-Cgas_0), Cpart_next-Cpart_0)  
    if not check:
        print('ERROR: Saturation ratio adjustment does not match the change in particles!')
    
    return
 
def check_mass_balance(original_population, new_population):
    
    Mass_0 = 0.0
    for particle, num_conc in zip(original_population.particles, original_population.num_concs): 
        Mass_0+=np.sum(particle.masses)
        
    Mass_new = 0.0
    for particle, num_conc in zip(new_population.particles, new_population.num_concs):
        Mass_new+=np.sum(particle.masses)
    
    # print(Mass_0, Mass_new, abs((Mass_0-Mass_new)/Mass_0), abs(Mass_0-Mass_new))
    # print()
    # import sys
    # sys.exit()
    
    check = np.isclose(Mass_0, Mass_new, rtol=1e-3, atol=1e-8)
    if not check:
        print()
        print(Mass_0, Mass_new)
        print()
        print('ERROR: Mass not conserved in aqueous chemistry module!')
        sys.exit()
        
    return
    
def check_gas_chemistry(ParcelState_0, ParcelState_Next):

    Mass_0 = 0.0
    for gas, conc in zip(ParcelState_0.TraceGas_population.gases,
        ParcelState_0.TraceGas_population.concs):
        Mass_0 += 1e-9*conc*gas.molar_mass # kg/m^3
    Mass_0 += 18e-3*H2O_gas_conc(ParcelState_0.S, ParcelState_0.T, ParcelState_0.P)

    Mass_next = 0.0
    for gas, conc in zip(ParcelState_Next.TraceGas_population.gases, ParcelState_Next.TraceGas_population.concs):
        Mass_next += 1e-9*conc*gas.molar_mass # kg/m^3
    Mass_next += 18e-3*H2O_gas_conc(ParcelState_Next.S, ParcelState_Next.T, ParcelState_Next.P)

    check = np.isclose(Mass_0, Mass_next, rtol=1e-3, atol=1e-8)
    if not check:
        print()
        print(Mass_0, Mass_next)
        print()
        print('ERROR: Mass not conserved in gas chemistry module!')
        sys.exit()
    
    return
        
    
    
    
    


