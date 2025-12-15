#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Water uptake functions

based on numba implementation in Pyrcel: https://github.com/darothen/pyrcel/blob/master/pyrcel/_parcel_aux_numba.py

@author: Laura Fierce (adapted code by Daniel Rothenberg)
"""

import numba as nb
import numpy as np
from numba.pycc import CC

import pyrcel.constants as c

## Define double DTYPE
DTYPE = np.float64

PI = 3.14159265358979323846264338328
N_STATE_VARS = c.N_STATE_VARS

# AOT/numba stuff
auxcc = CC("parcel_aux_numba")
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
        (2 * PI * c.Ma) / (c.R * T)
    )
    return ka_cont / denom


@nb.njit()
@auxcc.export("dv", "f8(f8, f8, f8, f8)")
def dv(T, r, P, accom):
    """See :func:`pyrcel.thermo.dv` for full documentation"""
    P_atm = P * 1.01325e-5  # Pa -> atm
    dv_cont = 1e-4 * (0.211 / P_atm) * ((T / 273.0) ** 1.94)
    denom = 1.0 + (dv_cont / (accom * r)) * np.sqrt((2 * PI * c.Mw) / (c.R * T))
    return dv_cont / denom


@nb.njit()
@auxcc.export("es", "f8(f8)")
def es(T):
    """See :func:`pyrcel.thermo.es` for full documentation"""
    return 611.2 * np.exp(17.67 * T / (T + 243.5))


@nb.njit()
@auxcc.export("seq", "f8(f8, f8, f8)")
def seq(r, r_dry, T, kappa):
    """See :func:`pyrcel.thermo.Seq` for full documentation."""
    A = (2.0 * c.Mw * sigma_w(T)) / (c.R * T * c.rho_w * r)
    B = 1.0
    if kappa > 0.0:
        B = (r**3 - (r_dry**3)) / (r**3 - (r_dry**3) * (1.0 - kappa))
    return np.exp(A) * B - 1.0


@nb.njit()
@auxcc.export("dseq_dr", "f8(f8, f8, f8)")
def dseq_dr(r, r_dry, T, kappa):
    """See :func:`pyrcel.thermo.Seq` for full documentation."""
    
    A = (2.0 * c.Mw * sigma_w(T)) / (c.R * T * c.rho_w * r)
    B = (r ** 3 - (r_dry ** 3)) / (r ** 3 - (r_dry ** 3) * (1.0 - kappa))
    
    dAdr = - (2.0 * c.Mw * sigma_w(T)) * (c.R * T * c.rho_w * r**2)
    dBdr = 3*r**2*(r ** 3 - (r_dry ** 3) * (1.0 - kappa))**(-1) +  (r ** 3 - (r_dry ** 3))*(- (r ** 3 - (r_dry ** 3) * (1.0 - kappa)))**(-2)*3*r**2
    
    dsdr = dAdr*np.exp(A) * B + np.exp(A) * dBdr    
    return dseq_dr



## RHS Derivative callback function
@nb.njit(parallel=True)
@auxcc.export("parcel_ode_sys", "f8[:](f8[:], f8, i4, f8[:], f8[:], f8, f8[:], f8)")
def parcel_ode_sys(y, t, nr, r_drys, Nis, V, kappas, accom):
    """Calculates the instantaneous time-derivative of the parcel model system.

    Given a current state vector `y` of the parcel model, computes the tendency
    of each term including thermodynamic (pressure, temperature, etc) and aerosol
    terms. The basic aerosol properties used in the model must be passed along
    with the state vector (i.e. if being used as the callback function in an ODE
    solver).

    Parameters
    ----------
    y : array_like
        Current state of the parcel model system,
            * y[0] = altitude, m
            * y[1] = Pressure, Pa
            * y[2] = temperature, K
            * y[3] = water vapor mass mixing ratio, kg/kg
            * y[4] = cloud liquid water mass mixing ratio, kg/kg
            * y[5] = cloud ice water mass mixing ratio, kg/kg
            * y[6] = parcel supersaturation
            * y[7:] = aerosol bin sizes (radii), m
    t : float
        Current simulation time, in seconds.
    nr : Integer
        Number of aerosol radii being tracked.
    r_drys : array_like
        Array recording original aerosol dry radii, m.
    Nis : array_like
        Array recording aerosol number concentrations, 1/(m**3).
    V : float
        Updraft velocity, m/s.
    kappas : array_like
        Array recording aerosol hygroscopicities.
    accom : float, optional (default=:const:`constants.ac`)
        Condensation coefficient.

    Returns
    -------
    x : array_like
        Array of shape (``nr``+7, ) containing the evaluated parcel model
        instaneous derivative.

    Notes
    -----
    This function is implemented using numba; it does not need to be just-in-
    time compiled in order ot function correctly, but it is set up ahead of time
    so that the internal loop over each bin growth term is parallelized.

    """
    z = y[0]
    P = y[1]
    T = y[2]
    wv = y[3]
    wc = y[4]
    wi = y[5]
    s = y[6]
    rs = y[N_STATE_VARS:]

    T_c = T - 273.15  # convert temperature to Celsius
    pv_sat = es(T_c)  # saturation vapor pressure
    wv_sat = wv / (s + 1.0)  # saturation mixing ratio
    Tv = (1.0 + 0.61 * wv) * T
    e = (1.0 + s) * pv_sat  # water vapor pressure
    
    ## Compute air densities from current state
    rho_air = P / c.Rd / Tv
    #: TODO - port to parcel.py
    rho_air_dry = (P - e) / c.Rd / T

    ## Begin computing tendencies
    dP_dt = -1.0 * rho_air * c.g * V
    dwc_dt = 0.0
    # drs_dt = np.empty(shape=(nr), dtype=DTYPE)
    drs_dt = np.empty_like(rs)

    for i in nb.prange(nr):
        r = rs[i]
        r_dry = r_drys[i]
        kappa = kappas[i]
        Ni = Nis[i]

        ## Non-continuum diffusivity/thermal conductivity of air near
        ## near particle
        dv_r = dv(T, r, P, accom)
        ka_r = ka(T, r, rho_air)

        ## Condensation coefficient
        G_a = (c.rho_w * c.R * T) / (pv_sat * dv_r * c.Mw)
        G_b = (c.L * c.rho_w * ((c.L * c.Mw / (c.R * T)) - 1.0)) / (ka_r * T)
        G = 1.0 / (G_a + G_b)

        ## Difference between ambient and particle equilibrium supersaturation
        seq_r = seq(r, r_dry, T, kappa)
        delta_s = s - seq_r
        
        ## Size and liquid water tendencies
        dr_dt = (G / r) * delta_s
        dwc_dt += (
            Ni * r * r * dr_dt
        )  # Contribution to liq. water tendency due to growth
        drs_dt[i] = dr_dt

    dwc_dt *= 4.0 * PI * c.rho_w / rho_air_dry  # Hydrated aerosol size -> water mass
    # use rho_air_dry for mixing ratio definition consistency
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
    x = np.empty_like(y)
    x[0] = dz_dt
    x[1] = dP_dt
    x[2] = dT_dt
    x[3] = dwv_dt
    x[4] = dwc_dt
    x[5] = dwi_dt
    x[6] = dS_dt
    x[N_STATE_VARS:] = drs_dt[:]

    return x

## RHS Derivative callback function
@nb.njit(parallel=True)
@auxcc.export("oneparticle_ode_sys", "f8[:](f8[:], f8, f8, f8, f8, f8, f8, f8, f8, f8)")
def oneparticle_ode_sys(x, t, r_dry_i, N_i, kappa_i, P, T, s, wv, accom=1.):#, add_Seq=False):
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
    N_i : float
        Aerosol number concentration, 1/(m**3).
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
    r_i = x[0]
    
    T_c = T - 273.15  # convert temperature to Celsius
    pv_sat = es(T_c)  # saturation vapor pressure
    # wv_sat = wv / (S + 1.0)  # saturation mixing ratio
    Tv = (1.0 + 0.61 * wv) * T
    e = (1.0 + s) * pv_sat  # water vapor pressure

    ## Compute air densities from current state
    rho_air = P / c.Rd / Tv
    #: TODO - port to parcel.py
    rho_air_dry = (P - e) / c.Rd / T
    
    ## Begin computing tendencies
    # dP_dt = -1.0 * rho_air * c.g * V
    # dwc_dt = 0.0
    # drs_dt = np.empty(shape=(nr), dtype=DTYPE)
    # drs_dt = np.empty_like(rs)
    
    # for i in nb.prange(nr):
    #     r = rs[i]
    #     r_dry = r_drys[i]
    #     kappa = kappas[i]
    #     Ni = Nis[i]

    ## Non-continuum diffusivity/thermal conductivity of air near
    ## near particle
    dv_r = dv(T, r_i, P, accom)
    ka_r = ka(T, r_i, rho_air)
    
    ## Condensation coefficient
    G_a = (c.rho_w * c.R * T) / (pv_sat * dv_r * c.Mw)
    G_b = (c.L * c.rho_w * ((c.L * c.Mw / (c.R * T)) - 1.0)) / (ka_r * T)
    G = 1.0 / (G_a + G_b)

    ## Difference between ambient and particle equilibrium supersaturation
    seq_i = seq(r_i, r_dry_i, T, kappa_i)
    # if add_Seq:
    #     Seq_r = x[1]
    delta_s = s - seq_i

    ## Size and liquid water tendencies
    dr_dt = (G / r_i) * delta_s
    
    
    # dseq_dt = dseq_dr(r_i, r_dry_i, T, kappa_i) * dr_dt
    # dwc_dt += (
    #     Ni * r * r * dr_dt
    # )  # Contribution to liq. water tendency due to growth
    
    # dwc_dt *= 4.0 * PI * c.rho_w / rho_air_dry  # Hydrated aerosol size -> water mass
    # # use rho_air_dry for mixing ratio definition consistency
    # # No freezing implemented yet
    # dwi_dt = 0.0

    # ## MASS BALANCE CONSTRAINT
    # dwv_dt = -1.0 * (dwc_dt + dwi_dt)

    # ## ADIABATIC COOLING
    # dT_dt = -c.g * V / c.Cp - c.L * dwv_dt / c.Cp

    # dz_dt = V

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
    # alpha = (c.g * c.Mw * c.L) / (c.Cp * c.R * (T**2))
    # alpha -= (c.g * c.Ma) / (c.R * T)
    # gamma = (P * c.Ma) / (c.Mw * pv_sat)
    # gamma += (c.Mw * c.L * c.L) / (c.Cp * c.R * T * T)
    # dS_dt = alpha * V - gamma * dwc_dt

    # x = np.empty(shape=(nr+N_STATE_VARS), dtype='d')
    # x = np.empty_like(y)
    # x[0] = dz_dt
    # x[1] = dP_dt
    # x[2] = dT_dt
    # x[3] = dwv_dt
    # x[4] = dwc_dt
    # x[5] = dwi_dt
    # x[6] = dS_dt
    # x[N_STATE_VARS:] = drs_dt[:]
    # dxdt = np.empty(shape=2,dtype='d')
    # dxdt[0] = dr_dt
    # dxdt[1] = dseq_dt
    dxdt = np.array([dr_dt])
    return dxdt


## RHS Derivative callback function
@nb.njit(parallel=True)
@auxcc.export("onlyparcel_ode_sys", "f8[:](f8[:], f8, i4, f8[:], f8[:], f8, f8[:], f8)")
def onlyparcel_ode_sys(x, t, drdts, rs,  Ns, V):
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
            * y[0] = altitude, m
            * y[1] = Pressure, Pa
            * y[2] = temperature, K
            * y[3] = water vapor mass mixing ratio, kg/kg
            * y[4] = parcel saturation ratio 
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
    z = x[0]
    P = x[1]
    T = x[2]
    wv = x[3]
    s = x[4]
    T_c = T - 273.15  # convert temperature to Celsius
    pv_sat = es(T_c)  # saturation vapor pressure
    wv_sat = wv / (s + 1.0)  # saturation mixing ratio
    Tv = (1.0 + 0.61 * wv) * T
    e = (1.0 + s) * pv_sat  # water vapor pressure

    ## Compute air densities from current state
    rho_air = P / c.Rd / Tv
    #: TODO - port to parcel.py
    rho_air_dry = (P - e) / c.Rd / T

    ## Begin computing tendencies
    dP_dt = -1.0 * rho_air * c.g * V
    dwc_dt = 0.0
    
    nr = len(rs)
    for i in nb.prange(nr):
        ri = rs[i]
        Ni = Ns[i]
        dri_dt = drdts[i]
        dwc_dt += (
            Ni * ri * ri * dri_dt
        )  # Contribution to liq. water tendency due to growth
    
    dwc_dt *= 4.0 * PI * c.rho_w / rho_air_dry  # Hydrated aerosol size -> water mass
    # use rho_air_dry for mixing ratio definition consistency
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
    ds_dt = alpha * V - gamma * dwc_dt

    # x = np.empty(shape=(nr+N_STATE_VARS), dtype='d')
    dxdt = np.empty_like(x)
    dxdt[0] = dz_dt
    dxdt[1] = dP_dt
    dxdt[2] = dT_dt
    dxdt[3] = dwv_dt
    dxdt[4] = ds_dt
    
    return dxdt

def wv_to_s(wv,T,P):
    # mixing ratio in mols h2o/mol air
    p_hPa = 1000.
    # p_hPa = pressure/100.
    T_C = T-273.15 #Convert to degrees C
    #Saturation vapor pressure (T)
    e_sat = 6.1094*np.exp(17.625*T_C/(243.04+T_C)) #(hPa)
    #Actual Vapor Pressure
    e = wv*p_hPa/(0.622+wv) # (hPa)
    #Supersaturation
    s = e/e_sat - 1.
    return s


def s_to_wv(s,T,P):
    # mixing ratio in mols h2o/mol air
    p_hPa = 1000.
    # p_hPa = pressure/100.
    T_C = T-273.15 #Convert to degrees C
    #Saturation vapor pressure (T)
    e_sat = 6.1094*np.exp(17.625*T_C/(243.04+T_C)) #(hPa)
    
    e = (s + 1.)*e_sat
    wv = e*0.622/(p_hPa - e)
    return wv
