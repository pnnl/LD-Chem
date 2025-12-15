#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 07:38:15 2024

@author: beel083
"""
import numba as nb
import numpy as np
import sys
import multipart.constants as c

@nb.njit()
def dCgas_dt(Cgas_0, reactants_all, products_all, rates, gas_names, T, P):
    
    dCgas_dt_all = np.zeros(len(Cgas_0))
    
    for ii in range(len(rates)):
        
        reactants=reactants_all[ii].split()
        products=products_all[ii].split()
        rate=rates[ii]
        
        dCgas = rate
        for reactant in reactants:
            idx = 10000
            for jj, (name) in enumerate(gas_names):
                if name == reactant:
                    idx = jj
            dCgas *= Cgas_0[idx] # mol/L/s
            
        for reactant in reactants:
            idx = 10000
            for jj, (name) in enumerate(gas_names):
                if name == reactant:
                    idx = jj
            dCgas_dt_all[idx]-=dCgas # mol/m^3/s

        for product in products:
            for jj, (name) in enumerate(gas_names):
                if name == product:
                    idx = jj
            dCgas_dt_all[idx]+=dCgas # mol/m^3/s
        
    return dCgas_dt_all # mol/m^3/s






    
    
    
    
