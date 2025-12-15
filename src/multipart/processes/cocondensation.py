#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:33:11 2024

@author: beel083
"""
import numpy as np
import numba as nb
from dataclasses import dataclass
from scipy.integrate import ode

R = 8.314 # m^3*Pa/mol*K

@dataclass
class GasFeedback:
    names: Tuple[GasSpecies, ...]
    dc_dts: Tuple[float, ...]


@nb.njit()
def dCaq_dt(X, radii, water_volumes, num_concs, molar_mass, alpha, Heff, T):
    '''
    Parameters
    ----------
    X : array of float64
        Array of cancentrations of a given condensable gaseous species. The first element
        X[0] is the gaseous concentration in mol/m^3. The remaining elements (X[1:]) are 
        the aqueous concentrations in each particle in mol/m^3.
    radii : array of float64
        Wet radius of each particle in nm.
    water_volumes : array of float64
        Array of water volume in each particle in m^3.
    num_concs : array of float64
        Number concentration of particles in 1/m^3.
    molar_mass : float64
        Molar mass of the condensing species in kg.
    alpha : float64
        Condensation coefficient of the condensing species (unitless).
    Heff : float64
        Effective Henry's Law coefficient of condensign species in mol/m^3*Pa.
    T : float64
        Temperature in Kelvin.

    Returns
    -------
    dX_dt : array of float64
        Change in gas and aqueous concentrations in mol/m^3. The first element (dX_dt[0])
        corresponds to the change in gas concentration. The remaining elements (dX_dt[1:])
        correspond to the change in aqueous concentration in each particle. 

    '''
    dX_dt = np.zeros(X.shape)
    Cgas = X[0] # mol/m^3
    Caq = X[1:] # mol/m^3
    Dg = (1/100**2)*1.9*np.power(molar_mass, (-2/3)) # m^2/s
    w = np.sqrt((8*R*T)/(np.pi*molar_mass)) # thermal velocity, m/s
    kmt = np.power((radii**2/(3*Dg))+((4.0*radii)/(3.0*w*alpha)), -1.0) # mass uptake, 1/s
    dX_dt[1:] = kmt*(Cgas - (Caq/(Heff*R*T))) # mol / m^3 water *s
    dX_dt[0] = (-1.0*np.sum(dX_dt[1:]*water_volumes*num_concs))#/Cgas # mol / m^3 air    
    
    return dX_dt


@nb.njit()
def IEPOX_condensation(X, H2O_concs, Hplus_concs, HSO4_concs, NH4_concs,
                       SO4_concs, radii, T, S, l_orgs, inorganic_radii, 
                       num_concs, water_volumes, molar_mass, alpha):

    dX_dt = np.zeros(X.shape)
    Cgas = X[0]
    
    kaqs = [1.8e-4, 2.62e-6, 6.2e-8, 1.91e-4]
    kaq = kaqs[0]*Hplus_concs*H2O_concs + kaqs[1]*HSO4_concs*H2O_concs + kaqs[2]*NH4_concs*H2O_concs + kaqs[3]*Hplus_concs*SO4_concs # 1/s
    Haq=3.0e4*(1000/101325) # mol/m^3 Pa
    
    V = (4.0/3.0)*np.pi*radii**3 # m^3
    Ap = 4.0*np.pi*radii**2 # m^2
    w = np.sqrt((8*R*T)/(np.pi*molar_mass)) # thermal velocity, m/s
    Dg = (1/100**2)*1.9*np.power(molar_mass, (-2/3)) # m^2/s
    Gamma_aq_inv = (4*V*R*T*Haq*kaq)/(Ap*w) # unitless
    Horg=1.0e3*(1000/101325) # mol/m^3 Pa
    
    Eta_org=6.92448e9*np.exp(-2.48362e1*S) # Pa*s (fit of table S3 in "Effect of the Aerosol-Phase State on Secondary Organic Aerosol Formation from the Reactive Uptake of Isoprene-Derived Epoxydiols (IEPOX)")
    Dorg=(1.380649E-23*T)/(6*np.pi*1e-10*Eta_org)
    Gamma_org_inv = ((w*l_orgs)/(4*R*T*Horg*Dorg))*(radii/inorganic_radii)
    
    Gamma_IEPOX = np.power(((w*radii)/(4.0*Dg))+(1/alpha)+Gamma_aq_inv+Gamma_org_inv, -1.0)
    dIEPOX_dt = (Gamma_IEPOX/4)*Ap*w*Cgas # mol/s
    dX_dt[1:] = dIEPOX_dt/water_volumes # mol / m^3 water *s
    dX_dt[0] = (-1.0*np.sum(dX_dt[1:]*water_volumes*num_concs))#/Cgas # mol / m^3 air   
      
    return dX_dt # mol/m^3*s

@nb.njit()
def dCaq_dt_diffusion_limited(X, radii, water_volumes, num_concs, molar_mass, alpha, Heff, T, P, Dl_0):
    
    dX_dt = np.zeros(X.shape)
    Cgas = X[0] # mol/m^3
    Cpart = X[1:]
    Dg = (1/100**2)*1.9*np.power(molar_mass, (-2/3)) # m^2/s
    beta = beta_FS(radii, T, P, alpha)
    
    Dl = Dl_0*(T/298)*(water_viscosity(298)/water_viscosity(T)) # m^2/s
    kl = Dl/radii # m/s
    Kg = ((radii/(Dg*beta))+((Heff*R*T)/kl))**(-1)
    dX_dt[1:] = (3.0/radii)*Kg*(Cgas-(Cpart/(Heff*R*T))) # mol / m^3 water *s
    
    # well-mixed droplet
    #dX_dt[1:] = ((3.0*Dg)/radii**2)*beta*(Cgas-(Cpart/(Heff*R*T))) # mol / m^3 water *s
    
    # zero concentration at surface
    #dX_dt[1:] = ((3.0*Dg)/radii**2)*beta*(Cgas-(Cpart/(Heff*R*T))) # mol / m^3 water *s
    
    dX_dt[0] = (-1.0*np.sum(dX_dt[1:]*water_volumes*num_concs)) # mol / m^3 air
    
    return dX_dt
    
   
@nb.njit()
def beta_FS(r, T, P, alpha, kB=1.380649e-23, d_air=0.365e-9):
     """
     Calculate the Fuchs–Sutugin correction factor beta_FS.
    
     Parameters
     ----------
     r : float
         Droplet radius in meters.
     T : float, optional
         Temperature in K (default = 298.15 K).
     P : float, optional
         Pressure in Pa (default = 101325 Pa).
     alpha : float, optional
         Mass accommodation coefficient (0 < alpha <= 1, default = 1.0).
    
     Returns
     -------
     beta : float
         Fuchs–Sutugin correction factor.
     """
     lambda_air = (kB * T) / (np.sqrt(2) * np.pi * d_air**2 * P) # mean free path, m
     Kn = lambda_air / r # Knudsen number
     beta = (1 + Kn) / (1 + (4/(3*alpha) + 0.377)*Kn + (4/(3*alpha))*Kn**2) # Fuchs–Sutugin correction

     return beta 

@nb.njit()
def water_viscosity(T):
    """Viscosity of water (Pa·s) using the VFT correlation, 273–373 K.
    Korson, L., Drost-Hansen, W., & Millero, F. J. (1969). Viscosity of Water at Various Temperatures. J. Phys. Chem., 73 (1), 34–39. DOI: 10.1021/j100721a006. """
    A, B, C = 2.414e-5, 247.8, 140.0
    return A * np.exp(B / (T - C))


def cocondensation_solver(population0, gas_population, P, T, S,
                          dt=1.0, solver='CVODE', verbosity=50, 
                          atol=1e-10, rtol=1e-10):
    population_next=population0.clone_detached()
    gas_feedback=GasFeedback(names=[], dc_dts=[])
    radii = 0.5*population0.get_particle_var('wet_diameter')
    dry_radii = 0.5*population0.get_particle_var('dry_diameter')
    water_volumes=(4.0/3.0)*np.pi*(radii**3-dry_radii**3)
    for gas, gas_ppb in zip(gas_population.gases, gas_population.concs):
        
        if gas.H0 > 0.0 and gas.alpha > 0.0:
            # initial condition
            X0 = np.array([(gas_ppb*1e-9*P)/(8.314*T)]) # mol/m^3
            spec_idx = population0.get_species_idx(gas.name)
            particle_concs = (population0.spec_masses[:,spec_idx]/population0.species[spec_idx].molar_mass)/water_volumes # mol/m^3
            X0 = np.append(X0, particle_concs)

            # define function
            if gas.name == 'IEPOX':
                H2O_concs=np.zeros(len(population0.num_concs))
                Hplus_concs=np.zeros(H2O_concs.shape)
                HSO4_concs=np.zeros(H2O_concs.shape)
                NH4_concs=np.zeros(H2O_concs.shape)
                SO4_concs=np.zeros(H2O_concs.shape)
                # l_orgs=np.zeros(H2O_concs.shape)
                # inorganic_radii=np.zeros(H2O_concs.shape)
                if population0.get_species_idx("H2O") is not None:
                    idx = population0.get_species_idx("H2O")
                    H2O_concs=(population0.spec_masses[:,idx]/population0.species[idx].molar_mass)/water_volumes
                if population0.get_species_idx("H+") is not None:
                    idx = population0.get_species_idx("H+")
                    Hplus_concs=(population0.spec_masses[:,idx]/population0.species[idx].molar_mass)/water_volumes
                if population0.get_species_idx("HSO4") is not None:
                    idx = population0.get_species_idx("HSO4")
                    HSO4_concs=(population0.spec_masses[:,idx]/population0.species[idx].molar_mass)/water_volumes
                if population0.get_species_idx("NH4") is not None:
                    idx = population0.get_species_idx("NH4")
                    NH4_concs=(population0.spec_masses[:,idx]/population0.species[idx].molar_mass)/water_volumes
                if population0.get_species_idx("SO4") is not None:
                    idx = population0.get_species_idx("SO4")
                    SO4_concs=(population0.spec_masses[:,idx]/population0.species[idx].molar_mass)/water_volumes
                
                V_Org=np.zeros(H2O_concs.shape)
                for species in ['IEPOX_OS', 'tetrol', 'tetrol_olig', 'IEPOX_OH_SOA']:
                    idx = population0.get_species_idx(species)
                    if idx is not None:
                        V_Org += population0.spec_masses[:,idx]/population0.species[idx].density
                
                V_NonOrg = ((4.0/3.0)*np.pi*dry_radii**3) - V_Org
                h2o_NonOrg_radius = ((3*(water_volumes+V_NonOrg))/(4.0*np.pi))**(1.0/3.0)
                inorganic_radii = ((3*V_NonOrg)/(4.0*np.pi))**(1.0/3.0)
                l_orgs = radii-h2o_NonOrg_radius # m
                
                rhs = lambda t, X: IEPOX_condensation(
                    X, H2O_concs, Hplus_concs, HSO4_concs, NH4_concs, 
                    SO4_concs, radii, T, S, l_orgs, inorganic_radii, 
                    population0.num_concs, water_volumes, gas.molar_mass, gas.alpha)
            
            # these are super soluble and fully dissociate, so 
            # assume that mass transfer is diffusion limited
            elif gas.name in ['HNO3', 'H2SO4']:
                if gas.name == 'HNO3':
                    Dl0 = 1.25e-9 # m^2/s (liquid phase diffusion coefficient reference value at 298 K, Newman et.al. 1973)
                elif gas.name == 'H2SO4':
                    Dl0 = 0.5e-10 # m^2/s (liquid phase diffusion coefficient reference value at 298 K, Leaist et.al. 1984)
                rhs = lambda t, X: dCaq_dt_diffusion_limited(
                    X, radii, water_volumes, population0.num_concs, gas.molar_mass, 
                    gas.alpha, gas.get_Heff(T), T, P, Dl0)
            else:
                rhs = lambda t, X: dCaq_dt(
                    X, radii, water_volumes, population0.num_concs, 
                    gas.molar_mass, gas.alpha, gas.get_Heff(T), T)        
    
            # solve
            ode15s = ode(rhs).set_integrator('lsoda', method='bdf', rtol=rtol, atol=atol, nsteps=5000)
            ode15s.set_initial_value(X0, 0.0)
            X_next = ode15s.integrate(ode15s.t+dt)  # mol/m^3

            # update the particles
            population_next.spec_masses[:,spec_idx]=X_next[1:]*water_volumes*population_next.species[spec_idx].molar_mass # kg
            
            # add to the feedbacks
            gas_feedback.names.append(gas.name)
            gas_feedback.dc_dts.append(1e9*(X_next[0]-X0[0])*((8.314*T)/P)) # ppb
    
    # keep data type consistent
    gas_feedback.dc_dts=np.array(gas_feedback.dc_dts)

    return population_next, gas_feedback    
