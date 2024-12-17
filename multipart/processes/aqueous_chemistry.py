#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 07:38:15 2024

@author: beel083
"""
import numba as nb
import numpy as np
import sys
# from cocondensation import dCaq_dt as temp

R = 8.314 # m^3*Pa/mol*K

@nb.njit()
def dCaq_dt(Caq_0, reactants_all, products_all, rates, aq_names, T):
    
    dCaq_dt_all = np.zeros(len(Caq_0))
    
    for ii in range(len(rates)):
        
        # this is weird, but it works on constance
        reactants=reactants_all[ii].split()
        products=products_all[ii].split()
        rate=rates[ii]
        
        # some reactions have pH-dependence
        if reactants==['S(IV)','O3']:
            dCaq_dt_all=O3_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T)
        elif reactants==['S(IV)','H2O2']:
            dCaq_dt_all=H2O2_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T)
        elif reactants==['S(IV)','NO2']:
            dCaq_dt_all=NO2_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T)
        elif reactants==['S(IV)','HNO2']:
            dCaq_dt_all=HNO2_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T)
        elif reactants==['S(IV)','O2']:
            dCaq_dt_all=O2_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T)
        elif reactants==['IEPOX','H2O']:
            dCaq_dt_all=IEPOX_SOA_chemistry(Caq_0, dCaq_dt_all, aq_names, T)
        else:
            dCaq = rate
            for reactant in reactants:
                for jj, (name) in enumerate(aq_names):
                    if name == reactant:
                        idx = jj
                dCaq *= Caq_0[idx] # mol/L/s
                
            for reactant in reactants:
                for jj, (name) in enumerate(aq_names):
                    if name == reactant:
                        idx = jj
                dCaq_dt_all[idx]-=dCaq # mol/m^3/s
            for product in products:
                for jj, (name) in enumerate(aq_names):
                    if name == product:
                        idx = jj
                dCaq_dt_all[idx]+=dCaq # mol/m^3/s

    return dCaq_dt_all # mol/m^3/s


@nb.njit()
def O3_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T):
    
    for ii, (name) in enumerate(aq_names):
        if name == 'SO2':
            SO2_conc=Caq_0[ii]
            SO2_idx = ii
        elif name == 'HSO3':
            HSO3_conc=Caq_0[ii]
            HSO3_idx = ii
        elif name == 'SO3':
            SO3_conc=Caq_0[ii]
            SO3_idx = ii
        elif name == 'O3':
            O3_conc=Caq_0[ii]
            O3_idx = ii
        elif name == 'H2SO4':
            H2SO4_idx = ii
        elif name == 'HSO4':
            HSO4_idx = ii
        elif name == 'SO4':
            SO4_idx = ii
        elif name == 'H+':
            Hplus_conc=Caq_0[ii]

    k1=4.9E1
    k2=6.2E2*np.exp(-5530*((1/T)-(1/298)))
    k3=4.0E6*np.exp(-5280*((1/T)-(1/298)))
    dCaq_dt_all[SO2_idx]-=k1*SO2_conc*O3_conc
    dCaq_dt_all[HSO3_idx]-=k2*HSO3_conc*O3_conc
    dCaq_dt_all[SO3_idx]-=k3*SO3_conc*O3_conc
    dCaq_dt_all[O3_idx]-=k1*SO2_conc*O3_conc
    dCaq_dt_all[O3_idx]-=k2*HSO3_conc*O3_conc
    dCaq_dt_all[O3_idx]-=k3*SO3_conc*O3_conc
    
    Keq1 = 1000.0*1000
    Keq2 = 1E-2*1000
    x_H2SO4 = np.power(1+(Keq1/Hplus_conc)+((Keq1*Keq2)/(Hplus_conc*Hplus_conc)), -1.0)
    x_HSO4 = np.power(1+(Hplus_conc/Keq1)+(Keq2/Hplus_conc), -1.0)
    x_SO4 = np.power(1+(Hplus_conc/Keq2)+((Hplus_conc*Hplus_conc)/(Keq1*Keq2)), -1.0)
    
    dS6_dt=k1*SO2_conc*O3_conc+k2*HSO3_conc*O3_conc+k3*SO3_conc*O3_conc
    dCaq_dt_all[H2SO4_idx]+=x_H2SO4*dS6_dt
    dCaq_dt_all[HSO4_idx]+=x_HSO4*dS6_dt
    dCaq_dt_all[SO4_idx]+=x_SO4*dS6_dt
    
    return dCaq_dt_all

@nb.njit()
def H2O2_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T):

    for ii, (name) in enumerate(aq_names):
        if name == 'HSO3':
            HSO3_conc=Caq_0[ii]
            HSO3_idx = ii
        elif name == 'H2O2':
            H2O2_conc=Caq_0[ii]
            H2O2_idx = ii
        elif name == 'H2SO4':
            H2SO4_idx = ii
        elif name == 'HSO4':
            HSO4_idx = ii
        elif name == 'SO4':
            SO4_idx = ii
        elif name == 'H+':
            Hplus_conc=Caq_0[ii]

    k1 = 1.45E2*np.exp(-4430*((1/T)-(1/298)))
    k2 = 0.013
    dCaq_dt_all[HSO3_idx]-=(k1*Hplus_conc*HSO3_conc*H2O2_conc)/(1+k2*Hplus_conc)
    dCaq_dt_all[H2O2_idx]-=(k1*Hplus_conc*HSO3_conc*H2O2_conc)/(1+k2*Hplus_conc)    
    
    Keq1 = 1000*1000
    Keq2 = 1.0E-2*1000
    x_H2SO4 = np.power(1+(Keq1/Hplus_conc)+((Keq1*Keq2)/(Hplus_conc*Hplus_conc)), -1.0)
    x_HSO4 = np.power(1+(Hplus_conc/Keq1)+(Keq2/Hplus_conc), -1.0)
    x_SO4 = np.power(1+(Hplus_conc/Keq2)+((Hplus_conc*Hplus_conc)/(Keq1*Keq2)), -1.0)
    
    dS6_dt=(k1*Hplus_conc*HSO3_conc*H2O2_conc)/(1+k2*Hplus_conc)
    dCaq_dt_all[H2SO4_idx]+=x_H2SO4*dS6_dt
    dCaq_dt_all[HSO4_idx]+=x_HSO4*dS6_dt
    dCaq_dt_all[SO4_idx]+=x_SO4*dS6_dt
    
    return dCaq_dt_all

@nb.njit()
def NO2_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T):
    
    for ii, (name) in enumerate(aq_names):
        if name == 'SO2':
            SO2_conc=Caq_0[ii]
            SO2_idx = ii
        elif name == 'HSO3':
            HSO3_conc=Caq_0[ii]
            HSO3_idx = ii
        elif name == 'SO3':
            SO3_conc=Caq_0[ii]
            SO3_idx = ii
        elif name == 'NO2':
            NO2_conc=Caq_0[ii]
            NO2_idx = ii
        elif name == 'H2SO4':
            H2SO4_idx = ii
        elif name == 'HSO4':
            HSO4_idx = ii
        elif name == 'SO4':
            SO4_idx = ii
        elif name == 'H+':
            Hplus_conc=Caq_0[ii]

    k1=1.24E4
    dCaq_dt_all[SO2_idx]-=k1*SO2_conc*NO2_conc
    dCaq_dt_all[HSO3_idx]-=k1*HSO3_conc*NO2_conc
    dCaq_dt_all[SO3_idx]-=k1*SO3_conc*NO2_conc
    dCaq_dt_all[NO2_idx]-=k1*NO2_conc*(SO2_conc+HSO3_conc+SO3_conc)
    
    Keq1 = 1000*1000
    Keq2 = 1.0E-2*1000
    x_H2SO4 = np.power(1+(Keq1/Hplus_conc)+((Keq1*Keq2)/(Hplus_conc*Hplus_conc)), -1.0)
    x_HSO4 = np.power(1+(Hplus_conc/Keq1)+(Keq2/Hplus_conc), -1.0)
    x_SO4 = np.power(1+(Hplus_conc/Keq2)+((Hplus_conc*Hplus_conc)/(Keq1*Keq2)), -1.0)
    
    dS6_dt=k1*NO2_conc*(SO2_conc+HSO3_conc+SO3_conc)
    dCaq_dt_all[H2SO4_idx]+=x_H2SO4*dS6_dt
    dCaq_dt_all[HSO4_idx]+=x_HSO4*dS6_dt
    dCaq_dt_all[SO4_idx]+=x_SO4*dS6_dt
    
    return dCaq_dt_all

@nb.njit()
def HNO2_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T):
    
    for ii, (name) in enumerate(aq_names):
        if name == 'SO2':
            SO2_conc=Caq_0[ii]
            SO2_idx = ii
        elif name == 'HSO3':
            HSO3_conc=Caq_0[ii]
            HSO3_idx = ii
        elif name == 'SO3':
            SO3_conc=Caq_0[ii]
            SO3_idx = ii
        elif name == 'HNO2':
            HNO2_conc=Caq_0[ii]
            HNO2_idx = ii
        elif name == 'H2SO4':
            H2SO4_idx = ii
        elif name == 'HSO4':
            HSO4_idx = ii
        elif name == 'SO4':
            SO4_idx = ii
        elif name == 'H+':
            Hplus_conc=Caq_0[ii]

    k1 = 2.0e-7
    dCaq_dt_all[SO2_idx]-=1000*k1*SO2_conc*HNO2_conc
    dCaq_dt_all[HSO3_idx]-=1000*k1*HSO3_conc*HNO2_conc
    dCaq_dt_all[SO3_idx]-=1000*k1*SO3_conc*HNO2_conc
    dCaq_dt_all[HNO2_idx]-=1000*k1*HNO2_conc*(SO2_conc+HSO3_conc+SO3_conc)
    
    Keq1 = 1000*1000
    Keq2 = 1.0E-2*1000
    x_H2SO4 = np.power(1+(Keq1/Hplus_conc)+((Keq1*Keq2)/(Hplus_conc*Hplus_conc)), -1.0)
    x_HSO4 = np.power(1+(Hplus_conc/Keq1)+(Keq2/Hplus_conc), -1.0)
    x_SO4 = np.power(1+(Hplus_conc/Keq2)+((Hplus_conc*Hplus_conc)/(Keq1*Keq2)), -1.0)
    
    dS6_dt=k1*HNO2_conc*(SO2_conc+HSO3_conc+SO3_conc)
    dCaq_dt_all[H2SO4_idx]+=1000*x_H2SO4*dS6_dt
    dCaq_dt_all[HSO4_idx]+=1000*x_HSO4*dS6_dt
    dCaq_dt_all[SO4_idx]+=1000*x_SO4*dS6_dt
    
    return dCaq_dt_all

@nb.njit()
def O2_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T):
    
    for ii, (name) in enumerate(aq_names):
        if name == 'SO2':
            SO2_conc=0.001*Caq_0[ii]
            SO2_idx = ii
        elif name == 'HSO3':
            HSO3_conc=0.001*Caq_0[ii]
            HSO3_idx = ii
        elif name == 'SO3':
            SO3_conc=0.001*Caq_0[ii]
            SO3_idx = ii
        elif name == 'H2SO4':
            H2SO4_idx = ii
        elif name == 'HSO4':
            HSO4_idx = ii
        elif name == 'SO4':
            SO4_idx = ii
        elif name == 'H+':
            Hplus_conc=0.001*Caq_0[ii]
            
    pH = -1.0*np.log10(Hplus_conc)
    
    Mn_conc = 2550.592266*1e-6 # M
    if pH > 2.5:
        Fe_conc = 1e-6*1.0204e3*np.exp(-6.6*(pH-2.5))
    else:
        Fe_conc = 1e-6*1.0204e3
    if pH <= 4.2:
        k1 = 8.72e7*(Hplus_conc**(-0.74))*Mn_conc*Fe_conc
    else:
        k1 = 7.51e13*(Hplus_conc**(0.67))*Mn_conc*Fe_conc
    
    dCaq_dt_all[SO2_idx]-=1000*k1*SO2_conc
    dCaq_dt_all[HSO3_idx]-=1000*k1*HSO3_conc
    dCaq_dt_all[SO3_idx]-=1000*k1*SO3_conc
    
    Keq1 = 1000*1000
    Keq2 = 1.0E-2*1000
    x_H2SO4 = np.power(1+(Keq1/Hplus_conc)+((Keq1*Keq2)/(Hplus_conc*Hplus_conc)), -1.0)
    x_HSO4 = np.power(1+(Hplus_conc/Keq1)+(Keq2/Hplus_conc), -1.0)
    x_SO4 = np.power(1+(Hplus_conc/Keq2)+((Hplus_conc*Hplus_conc)/(Keq1*Keq2)), -1.0)
    
    dS6_dt=k1*(SO2_conc+HSO3_conc+SO3_conc)
    dCaq_dt_all[H2SO4_idx]+=1000*x_H2SO4*dS6_dt
    dCaq_dt_all[HSO4_idx]+=1000*x_HSO4*dS6_dt
    dCaq_dt_all[SO4_idx]+=1000*x_SO4*dS6_dt

    return dCaq_dt_all

@nb.njit()
def IEPOX_SOA_chemistry(Caq_0, dCaq_dt_all, aq_names, T):
    
    HSO4_conc = 0
    NH4_conc = 0
    SO4_conc = 0
    for ii, (name) in enumerate(aq_names):
        if name == 'H2O':
            H2O_conc=0.001*Caq_0[ii]
            H2O_idx = ii
        elif name == 'H+':
            Hplus_conc=0.001*Caq_0[ii]
        elif name == 'HSO4':
            HSO4_conc=0.001*Caq_0[ii]
            HSO4_idx = ii
        elif name == 'SO4':
            SO4_conc=0.001*Caq_0[ii]
            SO4_idx = ii
        elif name == 'NH4':
            NH4_conc=0.001*Caq_0[ii]
            NH4_idx = ii
        elif name == 'IEPOX':
            IEPOX_conc=Caq_0[ii]
            IEPOX_idx = ii
        elif name == 'IEPOX_OS':
            IEPOX_OS_idx = ii
        elif name == 'tetrol':
            tetrol_conc = Caq_0[ii]
            tetrol_idx = ii
        elif name == 'tetrol_olig':
            tetrol_olig_idx = ii
            
    
    kaqs = [1.8e-4, 2.62e-6, 6.2e-8, 1.91e-4]
    kaq = kaqs[0]*Hplus_conc*H2O_conc + kaqs[1]*HSO4_conc*H2O_conc + kaqs[2]*NH4_conc*H2O_conc + kaqs[3]*Hplus_conc*SO4_conc # 1/s
        
    tau_olig=24 # AS: 12, ABS: 1.5
    BETA=0.35 # AS: 0.35, ABS: 0.6
    
    dCaq_dt_all[IEPOX_idx] -= kaq*IEPOX_conc # mol/m^3*s
    dCaq_dt_all[IEPOX_OS_idx] += BETA*kaq*IEPOX_conc # mol/m^3*s
    dCaq_dt_all[tetrol_idx] += (1-BETA)*kaq*IEPOX_conc # mol/m^3*s
    dCaq_dt_all[tetrol_olig_idx] += (1/(tau_olig*3600))*tetrol_conc # mol/m^3*s
    dCaq_dt_all[tetrol_idx] -= (1/(tau_olig*3600))*tetrol_conc # mol/m^3*s
    
    # dCaq_dt_all[particle.get_species_idx('H+')] -= (kaqs[0]*Hplus_conc+kaqs[3]*Hplus_conc*SO4_conc)*H2O_conc*Caq_0[particle.get_species_idx('IEPOX')]
    dCaq_dt_all[H2O_idx] -= (kaqs[0]*Hplus_conc*H2O_conc+kaqs[1]*HSO4_conc*H2O_conc+kaqs[2]*NH4_conc*H2O_conc)*IEPOX_conc
    if HSO4_conc > 0:
        dCaq_dt_all[HSO4_idx] -= kaqs[1]*HSO4_conc*H2O_conc*IEPOX_conc
    if NH4_conc > 0:
        dCaq_dt_all[NH4_idx] -= kaqs[2]*NH4_conc*H2O_conc*IEPOX_conc
    if SO4_conc > 0:
        dCaq_dt_all[SO4_idx] -= kaqs[3]*Hplus_conc*SO4_conc*IEPOX_conc
    
    return dCaq_dt_all



    
    
    
    