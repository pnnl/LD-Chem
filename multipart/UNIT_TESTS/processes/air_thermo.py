#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 18:03:39 2024

@author: Laura Fierce

"""

import numba as nb
import numpy as np
from numba.pycc import CC

# import pyrcel.constants as c
import constants as c

## Define double DTYPE
DTYPE = np.float64

PI = 3.14159265358979323846264338328
N_STATE_VARS = c.N_STATE_VARS

# AOT/numba stuff
auxcc = CC("parcel_aux_numba")
auxcc.verbose = True


@nb.njit()
@auxcc.export("es", "f8(f8)")
def es(T):
    """See :func:`pyrcel.thermo.es` for full documentation"""
    return 611.2 * np.exp(17.67 * T / (T + 243.5))

# or shoudl the combined ode be constructed elsehwere?
## RHS Derivative callback function
@nb.njit()
@auxcc.export("onlyparcel_ode_sys", "f8[:](f8[:], f8, i4, f8[:], f8[:], f8, f8[:], f8)")
def dstate_dt(x, V, dwc_dt, dwi_dt):
    """Calculates the instantaneous time-derivative of the parcel model system.

    Given a current state vector `y` of the parcel model, computes the tendency
    of each term including thermodynamic (pressure, temperature, etc) and aerosol
    terms. The basic aerosol properties used in the model must be passed along
    with the state vector (i.e. if being used as the callback function in an ODE
    solver).

    Parameters
    ----------
    x : array_like
        Current state of the parcel model system,
            * x[0] = altitude, m
            * x[1] = Pressure, Pa
            * x[2] = temperature, K
            * x[3] = water vapor mass mixing ratio, kg/kg
            * x[4] = parcel saturation ratio 
    t : float
        Current simulation time, in seconds.
    drdts : array_like
        Array recording the rate of change of aerosol droplets, m/s
    rs : array_like
        Array recording the dropplet radius, m
    Ns : array_like
        Array recording aerosol number concentrations, 1/(m**3).
    V : float
        Updraft velocity, m/s.

    Returns
    -------
    dxddt : array_like
        Array of shape (``nr``+7, ) containing the evaluated parcel model
        instaneous derivative.

    Notes
    -----
    This function is implemented using numba; it does not need to be just-in-
    time compiled in order ot function correctly, but it is set up ahead of time
    so that the internal loop over each bin growth term is parallelized.

    """
    # todo: make a decision on RH vs s vs S (and be consistent!)
    #       How about rh (in fraction), which is the same as saturation ratio, S (right?)
    T, P, S, wv = x 
    
    pv_sat, rho_air, rho_air_dry = compute_thermo_props(T, P, S)
    
    ## Begin computing tendencies
    dP_dt = -1.0 * rho_air * c.g * V
    
    # No freezing implemented yet
    dwi_dt = 0.0

    ## MASS BALANCE CONSTRAINT
    dwv_dt = -1.0 * (dwc_dt + dwi_dt)
    
    ## ADIABATIC COOLING
    dT_dt = -c.g * V / c.Cp - c.L * dwv_dt / c.Cp
    dz_dt = V

    """ Alternative methods for calculation supersaturation tendency
    # Used eq 12.28 from Pruppacher and Klett in stead of (9) from Nenes et al, 2001
    #cdef double S_a, S_b, S_c, dS_dt
    #cdef double S_b_old, S_c_old, dS_dt_old
    #S_a = (S+1.0)

    ## NENES (2001)
    #S_b_old = dT_dt*wv_sat*(17.67*243.5)/((243.5+(Tv-273.15))**2.)
    #S_c_old = (rho_air*g*V)*(wv_sat/P)*((0.622*L)/(Cp*Tv) - 1.0)
    #dS_dt_old = (1./wv_sat)*(dwv_dt - S_a*(S_b_old-S_c_old))

    ## PRUPPACHER (PK 1997)
    #S_b = dT_dt*0.622*L/(Rd*T**2.)
    #S_c = g*V/(Rd*T)
    #dS_dt = P*dwv_dt/(0.622*es(T-273.15)) - S_a*(S_b + S_c)

    ## SEINFELD (SP 1998)
    #S_b = L*Mw*dT_dt/(R*T**2.)
    #S_c = V*g*Ma/(R*T)
    #dS_dt = dwv_dt*(Ma*P)/(Mw*es(T-273.15)) - S_a*(S_b + S_c)
    """

    ## GHAN (2011)
    alpha = (c.g * c.Mw * c.L) / (c.Cp * c.R * (T**2))
    alpha -= (c.g * c.Ma) / (c.R * T)
    gamma = (P * c.Ma) / (c.Mw * pv_sat)
    gamma += (c.Mw * c.L * c.L) / (c.Cp * c.R * T * T)
    dS_dt = alpha * V - gamma * dwc_dt

    # x = np.empty(shape=(nr+N_STATE_VARS), dtype='d')
    dxdt = np.empty_like(x)
    # dxdt[0] = dz_dt
    dxdt[0] = dT_dt
    dxdt[1] = dP_dt
    dxdt[2] = dS_dt
    dxdt[3] = dwv_dt
    
    return dxdt

def dstate_dt_wrapper(t, x, V, dwc_dt, dwi_dt):
    dxdt = dstate_dt(x, V, dwc_dt, dwi_dt)
    return dxdt


# replace functions up there later
@nb.njit()
def dS_dt(dwv_dt, V, T, P, pv_sat):
    ## GHAN (2011)
    alpha = (c.g * c.Mw * c.L) / (c.Cp * c.R * (T**2))
    alpha -= (c.g * c.Ma) / (c.R * T)
    gamma = (P * c.Ma) / (c.Mw * pv_sat)
    gamma += (c.Mw * c.L * c.L) / (c.Cp * c.R * T * T)
    return alpha * V + gamma * dwv_dt

@nb.njit()    
def dT_dt(dwv_dt, V):
    return -c.g * V / c.Cp - c.L * dwv_dt / c.Cp

@nb.njit()
def dP_dt(V, rho_air):
    return -1.0 * rho_air * c.g * V

@nb.njit()
def compute_thermo_props(T, P, S):
    T_c = T - 273.15  # convert temperature to Celsius
    pv_sat = es(T_c)  # saturation vapor pressure
    # wv_sat = wv / (S + 1.0)  # saturation mixing ratio
    wv = S_to_wv(S,T,P)
    
    Tv = (1.0 + 0.61 * wv) * T
    e = S * pv_sat  # water vapor pressure
    
    ## Compute air densities from current state
    rho_air = P / c.Rd / Tv
    #: TODO - port to parcel.py
    rho_air_dry = (P - e) / c.Rd / T
    return pv_sat, rho_air, rho_air_dry

@nb.njit()
# water vapor mixing ratio to supersaturation
def wv_to_S(wv,T,P):
    # mixing ratio in mols h2o/mol air to saturation ratio 
    p_hPa = 1000.
    # p_hPa = pressure/100.
    T_C = T-273.15 #Convert to degrees C
    #Saturation vapor pressure (T)
    e_sat = 6.1094*np.exp(17.625*T_C/(243.04+T_C)) #(hPa)
    #Actual Vapor Pressure
    e = wv*p_hPa/(0.622+wv) # (hPa)
    #Supersaturation
    S = e/e_sat
    return S

@nb.njit()
def S_to_wv(S,T,P):
    # saturation ratio to mixing ratio [mols h2o/mol air]
    
    p_hPa = P/100. # convert Pa to hPa
    T_C = T-273.15 # Convert from kelvin to degrees C
    
    #Saturation vapor pressure (T)
    e_sat = 6.1094*np.exp(17.625*T_C/(243.04+T_C)) #(hPa)
    
    # Vapor pressure from saturation ratio
    e = S*e_sat
    wv = e*0.622/(p_hPa - e_sat) 
    
    return wv
