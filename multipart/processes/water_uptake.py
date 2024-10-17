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
from scipy.optimize import fsolve
import sys

# import scipy.optimize as opt
from scipy.constants import R

from processes.air_thermo import compute_thermo_props
# rework this -- don't want to include pyrcel as a dependency
import constants as c

## Define double DTYPE
DTYPE = np.float64

PI = 3.14159265358979323846264338328
N_STATE_VARS = c.N_STATE_VARS

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
def Seq(r, r_dry, T, kappa):
    """ Saturation ratio over the aqueous droplet. From pyrcel. """
    # A = (2.0 * c.Mw * sigma_w(T)) / (c.R * T * c.rho_w * r)
    # B = 1.0
    # if kappa > 0.0:
    #     B = (r**3 - (r_dry**3)) / (r**3 - (r_dry**3) * (1.0 - kappa))
    a_w = np.power(1.0+kappa*(np.power(r_dry,3)/(np.power(r, 3)-np.power(r_dry, 3))), -1)    
    Seq = a_w*np.exp((2.0*sigma_w(T)*c.Mw)/(R*T*c.rho_w*r))
    return Seq #np.exp(A) * B # - 1.0

@nb.njit()
@auxcc.export("dSeq_dr", "f8(f8, f8, f8)")
def dSeq_dr(r, r_dry, T, kappa):
    """Partial deriviative of Seq with respect to radius."""
    
    A = (2.0 * c.Mw * sigma_w(T)) / (c.R * T * c.rho_w * r)
    B = (r ** 3 - (r_dry ** 3)) / (r ** 3 - (r_dry ** 3) * (1.0 - kappa))
    
    dAdr = - (2.0 * c.Mw * sigma_w(T)) * (c.R * T * c.rho_w * r**2)
    dBdr = 3*r**2*(r ** 3 - (r_dry ** 3) * (1.0 - kappa))**(-1) +  (r ** 3 - (r_dry ** 3))*(- (r ** 3 - (r_dry ** 3) * (1.0 - kappa)))**(-2)*3*r**2
    
    return dAdr*np.exp(A) * B + np.exp(A) * dBdr #double check this


@nb.njit()
@auxcc.export("dh2o_dt", "f8(f8, f8, f8, f8, f8, f8, f8, f8, f8)")
# should send in Ddry, kappa, etc. (not Particle) --> no non-numba funs in the ODE solver
def dh2o_dt(m_h2o, r_dry_i, kappa_i, P, T, s, wv, accom=1.,rho_w=1000.):
    # r_dry_i, N_i, kappa_i = get this from the Particle functions
    # TODAY -- A --> make r_dry_i, D_dry_i, kappa_i, etc. all functions of Particle
    r_i = get_ri(r_dry_i, m_h2o, rho_w=rho_w)
    return dh2o_dr(r_i,rho_w=rho_w)*dr_dt(r_i, r_dry_i, kappa_i, P, T, s, wv, accom=accom)

# make this part of Particle 
@nb.njit()
@auxcc.export("get_ri", "f8(f8, f8)")
def get_ri(r_dry_i, m_h2o, rho_w=1000.):
    Vdry = np.pi/6.*(r_dry_i*2.)**3
    Vh2o = m_h2o/rho_w
    V = Vdry + Vh2o
    Di = (V*6./np.pi)**(1./3.)
    r_i = Di/2.
    return r_i 
              
@nb.njit()
@auxcc.export("dh2o_dr", "f8(f8, f8, f8)")
def dh2o_dr(r_i, rho_w=1000.):
    dvol_dr = np.pi*(2.*r_i)**2.
    dmh2o_dr = dvol_dr*rho_w   
    return dmh2o_dr

@nb.njit()
@auxcc.export("dh2o_dlnr", "f8(f8, f8, f8)")
def dh2o_dlnr(lnr_i, rho_w=1000.):
    r_i = np.log(lnr_i)
    dlnr_dr = 1./r_i 
    dmh2o_dlnr = dh2o_dr(r_i, rho_w=rho_w)/dlnr_dr(r_i)
    return dmh2o_dlnr

# def update_particle(particle, r0, r_next, idx_h2o=-1, rho_w=1000.):
#     Dwet_next = 2.*r_next
#     Dwet_0 = 2.*r0
#     dmh2o = np.pi/6.*(Dwet_next**3 - Dwet_0**3)*rho_w
#     particle.masses[particle.idx_h2o] += dmh2o
#     return particle

def update_particle(particle, r_next, rho_w=1000.):
    Dwet_next = 2.*r_next
    m_h2o = np.pi/6*(Dwet_next**3.-particle.get_Ddry()**3.)*rho_w
    # dmh2o = np.pi/6.*(Dwet_next**3 - Dwet_0**3)*rho_w
    particle.masses[particle.idx_h2o] = m_h2o #+= dmh2o
    return particle
    

## RHS Derivative callback function
@nb.njit()
@auxcc.export("dr_dt", "f8(f8, f8, f8, f8, f8, f8, f8, f8, f8)")
def dr_dt(r_i, r_dry_i, kappa_i, P, T, S, wv, accom=1.):
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
    # if add_Seq:
    #     Seq_r = x[1]
    # s = (RH-100.)/100.
    delta_S = S - Seq_i
    
    # print()
    # print("=============================== HERE ===================================")
    # print(r_i*1e9, r_dry_i*1e9, kappa_i, S, Seq_i)
    # print("========================================================================")
    # print()
    
    
    return (G / r_i) * delta_S ## Size tendencies


## RHS Derivative callback function
@nb.njit(parallel=True)
@auxcc.export("dlnr_dt", "f8(f8, f8, f8, f8, f8, f8, f8, f8)")
# @nb.njit(error_model='numpy')
def dlnr_dt(lnr_i, r_dry_i, kappa_i, P, T, S, wv, accom=1.0):
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
    # if add_Seq:
    #     Seq_r = x[1]
    delta_S = S - Seq_i

    ## return dlnr/dt = 1/r * dr/dt
    
    dr_dt = ((G / r_i) * delta_S)
    
    return dr_dt/r_i

def water_uptake_wrapper(t, r_i, r_dry_i, kappa_i, P, T, S, wv, accom, radius_scale):
    if radius_scale == 'log':
        drdt = dlnr_dt(r_i, r_dry_i, kappa_i, P, T, S, wv, accom=accom)
    elif radius_scale == 'lin':
        drdt = dr_dt(r_i, r_dry_i, kappa_i, P, T, S, wv, accom=accom)
    
    return drdt
    


def equilibrate_water(aerosol_population,S0,T0,P0,pH0):
    """
    Adds water to particles until they are in equilibrium with the
    surrounding air. Should only be used during run initialization.

    Parameters
    ----------
    aerosol_population : ParticlePopulation
        Describes what each particle is made of.
    S0 : float
        Ambient saturation ratio.
    T0 : float
        Ambient temperature.
    P0 : float
        Ambient pressure.
    pH0 : float
        Aerosol pH.

    Returns
    -------
    ParticlePopulation
        Particle population with correct water mass.
    """       
    mass_water = lambda radius, dry_radius: (4.0*np.pi/3.0)*c.rho_w*(radius**3-dry_radius**3)
    for ii,(particle,num_conc) in enumerate(zip(aerosol_population.particles,aerosol_population.num_concs)):
        species_names = []
        for i in range(len(particle.species)):
            species_names.append(particle.species[i].name)
            if particle.species[i].name == 'H2O':
                particle.idx_h2o = i

        r_dry = particle.get_Ddry()/2.
        kappa = particle.get_tkappa()
        f = lambda r: Seq(r, r_dry, T0, kappa) - S0
        r = fsolve(f, r_dry)
        particle.masses[particle.idx_h2o]=mass_water(r, r_dry) 
        
        if 'H+' in species_names:
            water_volume = 1000*(particle.get_vol_tot() - particle.get_vol_dry()) # L
            Hplus_conc = 10**(-1.0*pH0) # mol/L
            particle.masses[particle.get_species_idx('H+')]=(water_volume*Hplus_conc)/particle.species[particle.get_species_idx('H+')].molar_mass        
    
    return aerosol_population


# =============================================================================
# these might belong elsewhere
# =============================================================================


# particle properties 

# =============================================================================
# These functions all read in "Particle", should they be in the file with the Particle class?
# =============================================================================

# def equilibrate_h2o(Particle,RH,T):
#     masses = Particle.masses
#     AeroSpecs = Particle.species
#     idx_h2o, = np.where([AeroSpec.name.upper()=='H2O' for AeroSpec in AeroSpecs])
#     idx_h2o = idx_h2o[0]
#     Ddry = get_Ddry(Particle)
#     tkappa = get_tkappa(Particle)
#     sigma_h2o=get_sigma_h2o(T)
#     rho_h2o=AeroSpecs[idx_h2o].density
#     MW_h2o=AeroSpecs[idx_h2o].molecular_mass
#     Dwet = get_Dwet(Ddry, tkappa, RH, T, sigma_h2o=sigma_h2o, rho_h2o=rho_h2o, MW_h2o=MW_h2o)
#     mass_h2o = rho_h2o*np.pi/6.*(Dwet**3. - Ddry**3.)
#     Particle.masses[idx_h2o] = masses
#     return Particle

# def get_Dwet(Particle):
#     masses = Particle.masses
#     AeroSpecs = Particle.species
#     tot_vol = 0.
#     for kk,(AeroSpec,aero_mass) in enumerate(zip(AeroSpecs,masses)):
#         tot_vol += aero_mass/AeroSpec.density
#     Dwet = (tot_vol*6./np.pi)**(1./3.)
#     return Dwet

# def get_Ddry(Particle):
#     masses = Particle.masses
#     AeroSpecs = Particle.species
#     tot_vol = 0.
#     for kk,(AeroSpec,aero_mass) in enumerate(zip(AeroSpecs,masses)):
#         if AeroSpec.name.upper() != 'H2O':
#             tot_vol += aero_mass/AeroSpec.density
#     Ddry = (tot_vol*6./np.pi)**(1./3.)
#     return Ddry
    
# def get_tkappa(Particle):
#     masses = Particle.masses
#     AeroSpecs = Particle.species
#     tot_vol = 0.
#     tot_volKap = 0.
#     for kk,(AeroSpec,aero_mass) in enumerate(zip(AeroSpecs,masses)):
#         if AeroSpec.name.upper() != 'H2O':
#             tot_vol += aero_mass/AeroSpec.density
#             tot_volKap += AeroSpec.kappa*aero_mass/AeroSpec.density
#     effective_kappa = tot_volKap/tot_vol
#     return effective_kappa
    
# def get_critical_supersaturation(Particle, T, return_D_crit=False, phase_partitioning=False):
#     masses = Particle.masses
#     AeroSpecs = Particle.species
#     idx_h2o, = np.where([AeroSpec.name.upper()=='H2O' for AeroSpec in AeroSpecs])
#     Ddry=get_Ddry(Particle)
#     tkappa=get_tkappa(Particle)
#     sigma_h2o=get_sigma_h2o(T)
#     rho_h2o=AeroSpecs[idx_h2o].density
#     MW_h2o=AeroSpecs[idx_h2o].molecular_mass
#     if return_D_crit:
#         return compute_critical_supersaturation(Ddry,tkappa,T,sigma_h2o=sigma_h2o,MW_h2o=MW_h2o,rho_h2o=rho_h2o)
#     else:
#         return compute_critical_supersaturation(Ddry,tkappa,T,sigma_h2o=sigma_h2o,MW_h2o=MW_h2o,rho_h2o=rho_h2o)
    
# # supersaturation functions, these might stay here
# def compute_critical_supersaturation(
#         Ddry, tkappa, T, return_D_crit=False,
#         sigma_h2o=71.97/1000.,MW_h2o=18./1000.,rho_h2o=1000.):
#     A = 4.*sigma_h2o*MW_h2o/(R*T*rho_h2o);
    
#     if tkappa>0.2 and not return_D_crit:
#         s_critical = (np.exp((4.*A**3./(27.*Ddry**3.*tkappa))**(0.5))-1.)*100.
#     else:
#         f = lambda x: compute_Sc_funsixdeg(x,A,tkappa,Ddry)
#         soln = opt.root(f,Ddry*10);
#         x = soln.x[0]
#         D_critical = x
#         s_critical = (((x**3.0-Ddry**3.0)/(x**3-Ddry**3*(1.0-tkappa))*np.exp(A/x)) - 1.)*100.
    
#     if return_D_crit:
#         return s_critical,D_critical
#     else:
#         return s_critical
    
# def compute_Sc_funsixdeg(diam,A,tkappa,dry_diam):
#     c6=1.0;
#     c4=-(3.0*(dry_diam**3)*tkappa/A); 
#     c3=-(2.0-tkappa)*(dry_diam**3); 
#     c0=(dry_diam**6.0)*(1.0-tkappa);
    
#     z = c6*(diam**6.0) + c4*(diam**4.0) + c3*(diam**3.0) + c0;
#     return z

# # another get_Dwet
# def get_Dwet(Ddry, kappa, RH, T, sigma_h2o=0.072, rho_h2o=1000., MW_h2o=18e-3):
#     if RH>0. and kappa>0.:
#         A = 4*sigma_h2o*MW_h2o/(R*T*rho_h2o)
#         zero_this = lambda gf: RH/np.exp(A/(Ddry*gf))-(gf**3.-1.)/(gf**3.-(1.-kappa))
#         return Ddry*opt.brentq(zero_this,1.,10000000.)
#     else:
#         return Ddry

# def get_sigma_h2o(T):
#     return 0.0761 - 1.55e-4 * (T - 273.15)
