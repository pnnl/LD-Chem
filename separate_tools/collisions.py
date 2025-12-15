#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""
import numpy as np
import numba as nb
from numba.pycc import CC
auxcc = CC("collisions")
auxcc.verbose = True

@nb.njit()
@auxcc.export("dstate_dt", "f8(f8)")
def dstate_dt(particle_state_vars):
    pass
    
@nb.njit()
@auxcc.export("dxdt_sdm", "f8(f8,f8)")
def dxdt_sdm(Ns,rs):
    # note: need to reformulate this to get rate of change of aerosol species (not simply dr/dt)
    vols = np.pi*4./3.*rs**3.
    idx=np.argsort(rs)
    
    dNs_dt = np.zeros(Ns.shape)
    dVs_dt = np.zeros(vols.shape)
    for ii in idx:
        Ni = Ns[ii]
        vol_i = vols[ii]
        for jj in idx[ii:]:
            Nj = Ns[jj]
            vol_j = vols[jj]
            
            K_ij = summation_kernel(vol_i,vol_j)
            dNdt_ij = K_ij*Ni*Nj
            
            # vol_i <= vol_j
            dNs_dt[ii] -= dNdt_ij
            dVs_dt[jj] += vol_i*dNdt_ij
    
    drs_dt = (dVs_dt * 3./4. * np.pi - rs**3. * dNs_dt)/(3. * rs**2. * Ns)
    
    return drs_dt, dNs_dt

@nb.njit()
@auxcc.export("summation_kernel", "f8(f8,f8)")
def summation_kernel(vol_i,vol_j):
    # NOT a physical kernel, but has an anayltical solution for testing
    K_ij = vol_i + vol_j
    return K_ij
    