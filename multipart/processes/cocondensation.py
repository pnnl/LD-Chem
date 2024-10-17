#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:33:11 2024

@author: beel083
"""
import numpy as np
import sys

R = 8.314 # m^3*Pa/mol*K

def dCaq_dt(Caq, Cgas, radius, T, Heff, alpha, molec_mass):
    Dg = (1/100**2)*1.9*np.power(molec_mass, (-2/3)) # m^2/s
    w = np.sqrt((8*R*T)/(np.pi*molec_mass)) # thermal velocity, m/s
    kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alpha)), -1.0) # mass uptake, 1/s
    dC_dt_aq = kmt*(Cgas - (Caq/(Heff*R*T))) # mol/m^3*s
    return dC_dt_aq
    
def cocondensation_wrapper(t, Caq, Cgas, radius, T, Heff, alpha, molec_mass):
    dC_dt_aq = dCaq_dt(Caq, Cgas, radius, T, Heff, alpha, molec_mass)
    return dC_dt_aq
    

# def handle_negative_event(aerosol_population, gas, gas_conc, T, P):
#     print()
#     print()
#     print(gas.name)
#     Ctot = (gas_conc*1e-9*P)/(R*T) # mol/m^3
    
#     wL = 0
#     mass_aq = 0
#     for ii,(particle,num_conc) in enumerate(zip(aerosol_population.particles,aerosol_population.num_concs)):
#         water_mass = particle.masses[particle.idx_h2o]
#         wL += num_conc*water_mass/particle.get_rho_w() # m^3 water per m^3 air
#         mass_aq += particle.masses[particle.get_species_idx(gas.name)]*num_conc
    
#     Caq = (R*T*mass_aq)/(gas.molar_mass*P)
#     aq_fraction = (gas.get_Heff(T)*R*T*wL)/(1+gas.get_Heff(T)*R*T*wL)
#     print(1-aq_fraction, aq_fraction)
#     mult = aq_fraction/(Caq*1e9/gas_conc)
#     new_feedback = (1-aq_fraction)*gas_conc - gas_conc
    
    
#     check 
#     mass_aq = 0
#     for ii,(particle,num_conc) in enumerate(zip(aerosol_population.particles,aerosol_population.num_concs)):
#         mass_aq += particle.masses[particle.get_species_idx(gas.name)]*num_conc
    
#     Caq = (R*T*mass_aq)/(gas.molar_mass*P)
    
#     print(Caq*1e9/gas_conc)
#     print()
#     sys.exit()
    
#     return mult*particle.masses[particle.get_species_idx(gas.name)], new_feedback
    