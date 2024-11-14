#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 07:38:15 2024

@author: beel083
"""
import numpy as np
import sys

R = 8.314 # m^3*Pa/mol*K

def dCaq_dt(Caq_0, particle, aq_reactions, T):
    
    dCaq_dts = []
    for species,mass in zip(particle.species, particle.masses):
        dCaq_dts.append(0)
        
    for reaction in aq_reactions.reactions:
        rate=reaction.get_rate(T) # (M^(1-n)/s)
        dCaq_dt = rate
        for reactant in reaction.reactants:
            dCaq_dt *= Caq_0[particle.get_species_idx(reactant)] # mol/L/s
        
        for reactant in reaction.reactants:
            dCaq_dts[particle.get_species_idx(reactant)]+=-1.0*dCaq_dt # mol/L/s
            
        for product in reaction.products:
            dCaq_dts[particle.get_species_idx(product)]+=dCaq_dt # mol/L/s

    return 1000*dCaq_dts # mol/m^3/s