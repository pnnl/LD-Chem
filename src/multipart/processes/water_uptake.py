#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Water uptake functions

based on numba implementation in Pyrcel: https://github.com/darothen/pyrcel/blob/master/pyrcel/_parcel_aux_numba.py

@author: Laura Fierce and Payton Beeler (adapted code by Daniel Rothenberg)
"""

import numba as nb
import numpy as np
from numba.pycc import CC
from .air_thermo import compute_thermo_props
import multipart.constants as c

# AOT/numba stuff
auxcc = CC("water_uptake")
auxcc.verbose = True


## Auxiliary, single-value calculations with GIL released for derivative
## calculations
@nb.njit()
@auxcc.export("sigma_w", "f8(f8)")
def sigma_w(T):
    """See :func:`pyrcel.thermo.sigma_w` for full documentation"""
    return 0.0761 - (1.55e-4) * (T - 273.15)

@nb.njit()
@auxcc.export("ka", "f8(f8, f8, f8)")
def ka(T, r, rho):
    """See :func:`pyrcel.thermo.ka` for full documentation"""
    ka_cont = 1e-3 * (4.39 + 0.071 * T)
    denom = 1.0 + (ka_cont / (c.at * r * rho * c.Cp)) * np.sqrt(
        (2 * np.pi * c.Ma) / (c.R * T)
    )
    return ka_cont / denom

@nb.njit()
@auxcc.export("dv", "f8(f8, f8, f8, f8)")
def dv(T, r, P, accom):
    """See :func:`pyrcel.thermo.dv` for full documentation"""
    P_atm = P * 1.01325e-5  # Pa -> atm
    dv_cont = 1e-4 * (0.211 / P_atm) * ((T / 273.0) ** 1.94)
    denom = 1.0 + (dv_cont / (accom * r)) * np.sqrt((2 * np.pi * c.Mw) / (c.R * T))
    return dv_cont / denom

@nb.njit()
@auxcc.export("es", "f8(f8)")
def es(T):
    """See :func:`pyrcel.thermo.es` for full documentation"""
    return 611.2 * np.exp(17.67 * T / (T + 243.5))

@nb.njit()
@auxcc.export("seq", "f8(f8, f8, f8)")
def Seq(r, r_dry, T, kappa):
    """ Saturation ratio over the aqueous droplet. From pyrcel. """
    # A = (2.0 * c.Mw * sigma_w(T)) / (c.R * T * c.rho_w * r)
    # B = 1.0
    # if kappa > 0.0:
    #     B = (r**3 - (r_dry**3)) / (r**3 - (r_dry**3) * (1.0 - kappa))
    a_w = np.power(1.0+kappa*(np.power(r_dry,3)/(np.power(r, 3)-np.power(r_dry, 3))), -1)    
    Seq = a_w*np.exp((2.0*sigma_w(T)*c.Mw)/(c.R*T*c.rho_w*r))
    return Seq #np.exp(A) * B # - 1.0

## RHS Derivative callback function
@nb.njit()
@auxcc.export("dr_dt", "f8(f8, f8, f8, f8, f8, f8, f8, f8, f8)")
def dr_dt(r_i, r_dry_i, kappa_i, P, T, S, accom):
# oneparticle_ode_sys(x, t, r_dry_i, N_i, kappa_i, P, T, s, wv, accom=1.):#, add_Seq=False):
    """Calculates the instantaneous time-derivative of the parcel model system.

    Given a current state vector `y` of the parcel model, computes the tendency
    of each term including thermodynamic (pressure, temperature, etc) and aerosol
    terms. The basic aerosol properties used in the model must be passed along
    with the state vector (i.e. if being used as the callback function in an ODE
    solver).

    Parameters
    ----------
    x : array_like
        Current state of the single particle system,
            x[0] = particle wet radius, m
        # if add_Seq==True:
        #     x[1] = equilibrium saturation ratio at particle surface
    t : float
        Current simulation time, in seconds.
    r_dry_i : float
        Current particle dry radius, m.
    kappa_i : afloat
        Current aerosol hygroscopicity.
    T : float
        Current temperature of the air parcel, K
    S : float
        Current saturation ratio of the air parcel
    wv : float
        Current water vapor mixing ratio of the air parcel, kg/kg
    accom : float, optional (default=:const:`constants.ac`)
        Condensation coefficient.
    # add_Seq : logical
    #     If true, add Seq to the particle ode system

    Returns
    -------
    dxdt : array_like
        Array of shape (1, ) if add_Seq = False (default)
        
        or 
        
        Array of shape (2, ) if add_Seq = True
    
    Notes
    -----
    This function is implemented using numba; it does not need to be just-in-
    time compiled in order ot function correctly, but it is set up ahead of time
    so that the internal loop over each bin growth term is parallelized.

    """
    pv_sat, rho_air, rho_air_dry = compute_thermo_props(T, P, S)
    dv_r = dv(T, r_i, P, accom)
    ka_r = ka(T, r_i, rho_air)
    
    ## Condensation coefficient
    G_a = (c.rho_w * c.R * T) / (pv_sat * dv_r * c.Mw)
    G_b = (c.L * c.rho_w * ((c.L * c.Mw / (c.R * T)) - 1.0)) / (ka_r * T)
    G = 1.0 / (G_a + G_b)

    ## Difference between ambient and particle equilibrium supersaturation
    Seq_i = Seq(r_i, r_dry_i, T, kappa_i)
    delta_S = S - Seq_i
    
    return (G / r_i) * delta_S ## Size tendencies


## RHS Derivative callback function
@nb.njit(parallel=True)
@auxcc.export("dlnr_dt", "f8(f8, f8, f8, f8, f8, f8, f8, f8)")
def dlnr_dt(lnr_i, r_dry_i, kappa_i, P, T, S, accom=1.0):
# oneparticle_ode_sys(x, t, r_dry_i, N_i, kappa_i, P, T, s, wv, accom=1.):#, add_Seq=False):
    """Calculates the instantaneous time-derivative of the parcel model system.

    Given a current state vector `y` of the parcel model, computes the tendency
    of each term including thermodynamic (pressure, temperature, etc) and aerosol
    terms. The basic aerosol properties used in the model must be passed along
    with the state vector (i.e. if being used as the callback function in an ODE
    solver).

    Parameters
    ----------
    x : array_like
        Current state of the single particle system,
            x[0] = particle wet radius, m
        # if add_Seq==True:
        #     x[1] = equilibrium saturation ratio at particle surface
    t : float
        Current simulation time, in seconds.
    r_dry_i : float
        Current particle dry radius, m.
    kappa_i : afloat
        Current aerosol hygroscopicity.
    T : float
        Current temperature of the air parcel, K
    S : float
        Current saturation ratio of the air parcel
    wv : float
        Current water vapor mixing ratio of the air parcel, kg/kg
    accom : float, optional (default=:const:`constants.ac`)
        Condensation coefficient.
    # add_Seq : logical
    #     If true, add Seq to the particle ode system

    Returns
    -------
    dxdt : array_like
        Array of shape (1, ) if add_Seq = False (default)
        
        or 
        
        Array of shape (2, ) if add_Seq = True
    
    Notes
    -----
    This function is implemented using numba; it does not need to be just-in-
    time compiled in order ot function correctly, but it is set up ahead of time
    so that the internal loop over each bin growth term is parallelized.

    """
    # r_i = x[0]
    r_i = np.exp(lnr_i)

    # pv_sat, rho_air = compute_thermo_props(T, P, wv, RH)
    pv_sat, rho_air, rho_air_dry = compute_thermo_props(T, P, S)
    dv_r = dv(T, r_i, P, accom)
    ka_r = ka(T, r_i, rho_air)
    
    ## Condensation coefficient
    G_a = (c.rho_w * c.R * T) / (pv_sat * dv_r * c.Mw)
    G_b = (c.L * c.rho_w * ((c.L * c.Mw / (c.R * T)) - 1.0)) / (ka_r * T)
    G = 1.0 / (G_a + G_b)

    ## Difference between ambient and particle equilibrium supersaturation
    Seq_i = Seq(r_i, r_dry_i, T, kappa_i)
    delta_S = S - Seq_i

    ## return dlnr/dt = 1/r * dr/dt    
    dr_dt = ((G / r_i) * delta_S)

    return dr_dt/r_i
