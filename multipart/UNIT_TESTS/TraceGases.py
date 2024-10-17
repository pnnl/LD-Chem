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


@dataclass
class TraceGasPopulation:
    """Particle: the definition of an individual aerosol particle
    in terms of the amounts of different constituent species """
    gases: Tuple[GasSpecies, ...]
    concs: Tuple[float, ...]
    ids: Tuple[int, ...]
    

def retrieve_gas_species(name, specdata_path='../species_data/'):
    aero_datafile = specdata_path + 'gas_data.dat'
    with open(aero_datafile) as data_file:
        for line in data_file:
            if line.upper().startswith(name.upper()):
                name_in_file,alpha,molar_mass,H0,H_exp = line.split()
    return GasSpecies(
        name=name,
        alpha=float(alpha),
        molar_mass=float(molar_mass.replace('d','e')),
        H0=float(H0.replace('d','e')),
        H_exp=float(H_exp.replace('d','e')))


def get_equilibrium_reactions(name, mechanism_data_path='../mechanisms/'):
    eq_datafile = mechanism_data_path + 'equilibrium_data.dat'
    print()
    with open(eq_datafile) as data_file:
        for line in data_file:
            if line.upper().startswith(name.upper()):
                name_in_file,products,Keq_temp,Keq_exp_temp = line.split()
                products=products.split(',')
                Keq = []
                Keq_exp = []
                for ii in range(len(products)):
                    Keq.append(float(Keq_temp.split(',')[ii]))
                    Keq_exp.append(float(Keq_exp_temp.split(',')[ii]))
    print(products, Keq, Keq_exp)
    print()
    sys.exit()
    
    return 1