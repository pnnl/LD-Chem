#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:39:14 2024

@author: beel083
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import sys

R = 8.314

@dataclass(frozen=True)
class GasSpecies:
    """GasSpecies: the definition of a gaseous species in terms of species-
    specific parameters (no state information)"""
    name: str          # name of the species
    alpha: float
    molar_mass: float
    H0: float
    H_exp: float
    
    def get_Heff(self, T):
        # returns Henry's Law coefficient at given temperature in mol/m^3*Pa
        return (1000/101325)*self.H0*np.exp(self.H_exp*((1/T)-(1/298))) 

    
# @dataclass(frozen=True)
# class AqueousSpecies:
#     """AqSpecies: the definition of a species that dissolves into
#     aqueous phase but does not remain in aerosol phase in terms of species-
#     specific parameters (no state information)"""
#     name: str          # name of the species
#     molar_mass: float

@dataclass
class TraceGasPopulation:
    """TraceGasPopulation: the definition of gas composition
    in terms of the amounts of different constituent species """
    gases: Tuple[GasSpecies, ...]
    concs: Tuple[float, ...]
    ids: Tuple[int, ...]
    
    def get_species_idx(self, species):
        idx = None
        for ii in range(len(self.gases)):
            if self.gases[ii].name == species:
                return ii
        return idx
        
    def clone_detached(self):
        """Return a copy that shares immutable data but has detached numeric arrays."""
        return TraceGasPopulation(
            gases=self.gases,                    # shared: typically immutable metadata
            concs=self.concs.copy(),             # detached: new NumPy array
            ids=tuple(self.ids),                 # safe shallow copy
        )
    
# @dataclass
# class AqueousPopulation:
#     """AqueousPopulation: the definition of species which can dissolve
#     in water in terms of the concentration (mol/L) of different constituent 
#     species """
#     species: Tuple[AqueousSpecies, ...]
#     concs: Tuple[float, ...]
#     ids: Tuple[int, ...]
    

def retrieve_gas_species(name, specdata_path='../species_data/'):
    gas_datafile = specdata_path + 'gas_data.dat'
    with open(gas_datafile) as data_file:
        for line in data_file:
            if name == line.split()[0]:
                name_in_file,alpha,molar_mass,H0,H_exp = line.split()
    
    return GasSpecies(
        name=name,
        alpha=float(alpha),
        molar_mass=float(molar_mass.replace('d','e')),
        H0=float(H0.replace('d','e')),
        H_exp=float(H_exp.replace('d','e')))


def equilibrate_gases(aerosol_population,TraceGas_population,T,P):
    
    for gas, gas_conc in zip(TraceGas_population.gases, TraceGas_population.concs):
        if gas.name != 'IEPOX':
            Caq_x = gas_conc*1e-9*P*gas.get_Heff(T)  # mol/m^3
            for i,(particle,num_conc) in enumerate(zip(aerosol_population.particles,aerosol_population.num_concs)):
                water_volume = particle.get_vol_tot()-particle.get_vol_dry() # m^3            
                idx = particle.get_species_idx(gas.name)
                if idx and particle.species[idx].density==0:
                    particle.masses[idx]=water_volume*Caq_x*particle.species[idx].molar_mass
    return aerosol_population
    

# def retrieve_aq_species(name, specdata_path='../species_data/'):
#     aq_datafile = specdata_path + 'aq_species_data.dat'
#     with open(aq_datafile) as data_file:
#         for line in data_file:
#             if line.upper().startswith(name.upper()):
#                 name_in_file,molar_mass = line.split()
#     return AqueousSpecies(
#         name=name,
#         molar_mass=float(molar_mass.replace('d','e')))


# def get_equilibrium_reactions(name, mechanism_data_path='../mechanisms/'):
#     eq_datafile = mechanism_data_path + 'equilibrium_data.dat'
#     print()
#     with open(eq_datafile) as data_file:
#         for line in data_file:
#             if line.upper().startswith(name.upper()):
#                 name_in_file,products,Keq_temp,Keq_exp_temp = line.split()
#                 products=products.split(',')
#                 Keq = []
#                 Keq_exp = []
#                 for ii in range(len(products)):
#                     Keq.append(float(Keq_temp.split(',')[ii]))
#                     Keq_exp.append(float(Keq_exp_temp.split(',')[ii]))
#     print(products, Keq, Keq_exp)
#     print()
#     sys.exit()
    
#     return 1
