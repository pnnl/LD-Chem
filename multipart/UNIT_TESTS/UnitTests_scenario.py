""" scenario - types and functions supporting the creation of single
simulation scenarios.

@author: Laura Fierce
"""


from dataclasses import dataclass
import numpy as np
# from scipy.interpolate import interp1d
from particles import ParticlePopulation
from TraceGases import TraceGasPopulation
# from particles import make_particle
from scenario import get_aero_spec_fracs, make_polydisperse_population, make_monodisperse_population, lognormal_distribution
from scenario import make_TraceGas_population, make_AqReactions
from processes.water_uptake import equilibrate_water
# from aerosol_species import retrieve_one_species
from typing import Tuple, Callable, Optional
import sys
from scipy.special import erfinv
# from systems import Processes
# import matplotlib.pyplot as plt

@dataclass
class TrajectorySettings: # settings driving on trajectory simulation (trajectories can interact)
    population0: ParticlePopulation
    gas0: TraceGasPopulation
    
    x0: Optional[float]
    y0: Optional[float]
    z0: Optional[float]
    
    u0: Optional[float]
    v0: Optional[float]
    w0: Optional[float]
    
    S0: Optional[float]
    P0: Optional[float]
    T0: Optional[float]
    
    x_fun: Optional[Callable[[float],float]] = None
    y_fun: Optional[Callable[[float],float]] = None
    z_fun: Optional[Callable[[float],float]] = None
    
    u_fun: Optional[Callable[[float],float]] = None
    v_fun: Optional[Callable[[float],float]] = None
    w_fun: Optional[Callable[[float],float]] = None
    
    S_fun: Optional[Callable[[float],float]] = None
    P_fun: Optional[Callable[[float],float]] = None
    T_fun: Optional[Callable[[float],float]] = None    

@dataclass
class Scenario:
    # settings needed to simulate ensemble of trajectories
    trajectories_settings: Tuple[TrajectorySettings, ...]
    start_times: Tuple[float, ...]
    end_times: Tuple[float, ...]
    dt: float
  

def create_constant_parcel(
        aerosol_population = None, TraceGas_population=None,
            Ddry=100e-9,sigma=1.0,Ntot=1e6,Npart=1,updraft_velocity=0.0,
            S0=-0.15,P0=101325,T0=298,pH0=7.0,t_end=600.0,
            species_names=['NaCl'],mass_fractions=np.array([1.]),
            gas_names=None, gas_conc=None,
            dt=1.0, specdata_path='../species_data/', mechanism_data_path='../mechamisms/',
            chemistry=None, cocondensation=False):
    
    if Npart > 1 and sigma == 1.0:
        print('WARNING: Sigma = 1.0 and Npart > 1! Setting Npart to 1 to speed up calculations.')
        Npart = 1
    
    if cocondensation:
        TraceGas_population = make_TraceGas_population(gas_names, gas_conc, specdata_path=specdata_path)
    else:
        TraceGas_population=None
    
    if chemistry:
        aq_reactions = make_AqReactions(chemistry=chemistry, mechanism_data_path=mechanism_data_path)
    else:
        aq_reactions = None
    
    if aerosol_population == None:
        
        aero_spec_names, aero_spec_fracs = get_aero_spec_fracs(
            molecule_names=species_names, molecule_mass_fracs=mass_fractions,
            specdata_path=specdata_path)
        if Npart > 1:
            aero_spec_fracs_copy = np.zeros((Npart, len(aero_spec_fracs)))
            aero_spec_fracs_copy[:] = aero_spec_fracs
            aero_spec_fracs = aero_spec_fracs_copy
        
        if np.iterable(Ddry):
            aerosol_population = make_polydisperse_population(Ddry, Ntot, aero_spec_names, aero_spec_fracs, specdata_path=specdata_path,surface_tension=0.072, aq_reactions=aq_reactions, gases=TraceGas_population)
        elif sigma==1.0:
            aerosol_population = make_monodisperse_population(Ddry, Ntot, aero_spec_names, aero_spec_fracs, specdata_path=specdata_path, aq_reactions=aq_reactions, gases=TraceGas_population)        
        else:
            Dpg = Ddry*np.exp(np.log(sigma)*np.log(sigma))
            Dmin = Dpg*sigma**(-np.sqrt(2)*erfinv(0.999)) 
            Dmax = Dpg*sigma**(np.sqrt(2)*erfinv(0.999)) 
            model_Dps = np.logspace(np.log10(Dmin), np.log10(Dmax), Npart) # nm        
            model_Ns = lognormal_distribution(model_Dps, Ntot, Dpg, sigma) # m^-3 
            mult = Ntot/np.sum(model_Ns)
            model_Ns *= mult 
            aerosol_population = make_polydisperse_population(model_Dps, model_Ns, aero_spec_names, aero_spec_fracs, specdata_path=specdata_path, aq_reactions=aq_reactions, gases=TraceGas_population)        

        aerosol_population = equilibrate_water(aerosol_population,S0,T0,P0,pH0)     
    
    trajectories_settings = [TrajectorySettings(
            x0=None,y0=None,z0=0.0,
            u0=None,v0=None,w0=updraft_velocity,
            S0=S0, T0=T0, P0=P0,
            w_fun=lambda t: updraft_velocity,
            population0=aerosol_population,
            gas0=TraceGas_population)]
    start_times=[0.0]
    end_times=[t_end]
    
    return Scenario(
        trajectories_settings=trajectories_settings,
        start_times=start_times, end_times=end_times, dt=dt)



