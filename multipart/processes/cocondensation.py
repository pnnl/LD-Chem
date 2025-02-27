#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:33:11 2024

@author: beel083
"""
import numpy as np
import sys
import numba as nb

R = 8.314 # m^3*Pa/mol*K


@nb.njit()
def dCaq_dt(Caq_all, aq_names, Cgas_all, gas_names, gas_molec_masses, gas_alphas, 
            gas_Heffs, T, S, radius, l_org, inorg_radius, water_volume):
    
    dCaq_dt_all=np.zeros(len(Caq_all))    
        
    for ii, (Cgas, GasName, molar_mass, alpha, Heff) in enumerate(zip(Cgas_all, gas_names, gas_molec_masses, gas_alphas, gas_Heffs)):
        
        if molar_mass>0.0:
            if GasName == 'IEPOX':
                
                dIEPOX_dt = IEPOX_condensation(Caq_all, aq_names, Cgas_all, gas_names, 
                                                radius, T, S, l_org, inorg_radius, 
                                                water_volume, molar_mass, alpha) # mol/m^3*s
                
                for ii, (aq_name) in enumerate(aq_names):
                    if aq_name == 'IEPOX':
                        idx = ii
                dCaq_dt_all[idx] = dIEPOX_dt
            
            else:
                Dg = (1/100**2)*1.9*np.power(molar_mass, (-2/3)) # m^2/s
                w = np.sqrt((8*R*T)/(np.pi*molar_mass)) # thermal velocity, m/s
                kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alpha)), -1.0) # mass uptake, 1/s
                for ii, (aq_name) in enumerate(aq_names):
                    if aq_name == GasName:
                        idx = ii
                Caq = Caq_all[idx]
                dCaq_dt_all[idx] = kmt*(Cgas - (Caq/(Heff*R*T))) # mol/m^3*s
      
    return dCaq_dt_all


@nb.njit()
def IEPOX_condensation(Caq_all, aq_names, Cgas_all, gas_names, radius, T, S, 
                       l_org, inorganic_radius, water_volume, molar_mass, alpha):
    
    dIEPOX_dt = 0
    
    for ii, (name) in enumerate(gas_names):
        if name == 'IEPOX':
            IEPOX_gas_conc = Cgas_all[ii]
        
    for ii, (name) in enumerate(aq_names):
        if name == 'H2O':
            H2O_conc = 0.001*Caq_all[ii] # mol/L
        elif name == 'H+':
            Hplus_conc = 0.001*Caq_all[ii] # mol/L
        elif name == 'HSO4':
            HSO4_conc = 0.001*Caq_all[ii] # mol/L
        elif name == 'NH4':
            NH4_conc = 0.001*Caq_all[ii] # mol/L
        elif name == 'SO4':
            SO4_conc = 0.001*Caq_all[ii] # mol/L
    
    if 'HSO4' not in aq_names:
        HSO4_conc = 0
    if 'NH4' not in aq_names:
        NH4_conc = 0
    if 'SO4' not in aq_names:
        SO4_conc = 0
    
    kaqs = [1.8e-4, 2.62e-6, 6.2e-8, 1.91e-4]
    kaq = kaqs[0]*Hplus_conc*H2O_conc + kaqs[1]*HSO4_conc*H2O_conc + kaqs[2]*NH4_conc*H2O_conc + kaqs[3]*Hplus_conc*SO4_conc # 1/s
    
    Haq=3.0e2*(1000/101325) # mol/m^3 Pa, AS: 3.0e4*(1000/101325)
    
    V = (4.0/3.0)*np.pi*radius**3 # m^3
    Ap = 4.0*np.pi*radius**2 # m^2
    w = np.sqrt((8*R*T)/(np.pi*molar_mass)) # thermal velocity, m/s
    Dg = (1/100**2)*1.9*np.power(molar_mass, (-2/3)) # m^2/s
    Gamma_aq_inv = (4*V*R*T*Haq*kaq)/(Ap*w) # unitless
    
    Horg=6.0E2*(1000/101325) # mol/m^3 Pa, AS: 2.0e3*(1000/101325)
    
    Eta_org=6.92448e9*np.exp(-2.48362e1*S) # Pa*s (fit of table S3 in "Effect of the Aerosol-Phase State on Secondary Organic Aerosol Formation from the Reactive Uptake of Isoprene-Derived Epoxydiols (IEPOX)")
    Dorg=(1.380649E-23*T)/(6*np.pi*1e-10*Eta_org)
    Gamma_org_inv = ((w*l_org)/(4*R*T*Horg*Dorg))*(radius/inorganic_radius)
    
    Gamma_IEPOX = np.power(((w*radius)/(4.0*Dg))+(1/alpha)+Gamma_aq_inv+Gamma_org_inv, -1.0)
    dIEPOX_dt = (Gamma_IEPOX/4)*Ap*w*IEPOX_gas_conc # mol/s
            
    return dIEPOX_dt/water_volume # mol/m^s*s


# def cocondensation_wrapper(t, Caq_all, Cgas_all, radius, T, TraceGas_population):
#     dC_dt_aq = dCaq_dt(Caq_all, Cgas_all, radius, T, TraceGas_population)
#     return dC_dt_aq
    

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
    