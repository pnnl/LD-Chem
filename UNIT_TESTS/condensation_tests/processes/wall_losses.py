#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:20:49 2024

@author: beel083
"""
import numpy as np
import constants as c

def particle_wall_loss(Ns_0, Dps, densities, T, mu_star):
        
    Ke = 9.0e9 # N*m^2/C^2
    mu = 1.8e-5 # kg/m*s, dynamic viscosity of air
    nu = 1.562e-5 # m^2/s
    rho_air = 1.31246 # kg/m^3
    
    # These values are for the chamber used in
    # "Observationally Constrained Modeling of the Reactive Uptake of
    # Isoprene-Derived Epoxydiols under Elevated Relative Humidity and
    # Varying Acidity of Seed Aerosol Conditions"
    V = 1 # m^3
    Av = 4 # m^2
    Au = 1 # m^2
    Ad = 1 # m^2
    A = 6 # m^2
    E = 10*100 # V/m
            
    charge_fractions = np.zeros((4, len(Dps)))
    deposition_rate = np.zeros((4, len(Dps)))
    
    prefix = np.sqrt((Ke*c.e**2)/(np.pi*Dps*c.kb*T))
    
    # fraction of charges in each size bin
    for i in [0, 1, 2, 3]:
        charge_fractions[i] = prefix*np.exp((-Ke*i**2*c.e**2)/(Dps*c.kb*T))
        mult = np.sum(charge_fractions[i])
        charge_fractions[i] *= 1/mult
          
    D_brownian = (c.R*T)/(6*np.pi*mu*Dps*c.Na) # Brownian diffusion coefficient, m^2/s
    
    r_plus = (Dps*mu_star)/(2*nu)
    Sc = nu/D_brownian
    a = 0.5*np.log((10.92*np.power(Sc, -1/3)+r_plus)/(np.power(Sc, -1)+7.669e-4*np.power(r_plus, 3)))\
        + np.sqrt(3)*np.arctan((2*r_plus-10.92*np.power(Sc, -1/3))/(np.sqrt(3)*10.92*np.power(Sc, -1/3)))
    b = 0.5*np.log((np.power(10.92*np.power(Sc, -1/3)+r_plus, 3))/(np.power(Sc, -1)+7.669e-4*np.power(r_plus, 3)))\
        + np.sqrt(3)*np.arctan((2*r_plus-10.92*np.power(Sc, -1/3))/(np.sqrt(3)*10.92*np.power(Sc, -1/3)))
    I = 3.64*np.power(Sc, 2/3)*(a-b)+39
    k_dv = mu_star/I[0] # m/s
    V_settling = (2*(densities-rho_air)*9.8*Dps**2)/(9*mu) # m/s    
    
    k_du = V_settling/(1-np.exp((-V_settling*I)/(mu_star))) # m/s
    k_dd = V_settling/(np.exp((V_settling*I)/mu_star)-1) # m/s
    k_uncharged = (k_dv*Av+k_du*Au+k_dd*Ad)/V # 1/s    
    deposition_rate[0] = k_uncharged
    
    alpha = 1.142
    beta = 0.558
    gamma = -0.999
    Kn = (2*66.4e-9)/Dps
    slip_correction = 1 + Kn*(alpha+beta*np.exp(gamma/Kn)) # Cunningham slip correction based on mobility diameter    
    for i in [1, 2, 3]:
        deposition_rate[i] = (A*i*c.e*slip_correction*E)/(V*3*np.pi*mu*Dps) # 1/s
    
    wall_losses = np.sum(charge_fractions*deposition_rate*Ns_0, axis=0) #1/m^3*s
    
    return -1*wall_losses