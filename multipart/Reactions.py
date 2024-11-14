#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:58:24 2024

@author: beel083
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import sys

@dataclass(frozen=True)
class Reaction:
    """Reaction: the definition of an aqueous or gas 
    phase reaction. Aqueous reaction rates have units
    of M^(1-n)/s and gas phase reaction rates have units
    of (molec/cm^3)^(1-n)/s (no state information)."""
    reactants: list          # name of the species
    products: list
    rate0: float
    neg_dH_R: float
    
    def get_rate(self, T):
        # returns rate at given temperature
        return self.rate0*np.exp(self.neg_dH_R*((1/T)-(1/298)))  # (M^(1-n)/s)

# @dataclass(frozen=True)
# class AqueousSpecies:
#     """AqSpecies: the definition of a species that dissolves into
#     aqueous phase but does not remain in aerosol phase in terms of species-
#     specific parameters (no state information)"""
#     name: str          # name of the species
#     molar_mass: float
#     equilibria: Tuple[int, ...]
    
@dataclass
class AqueousReactions:
    """EquilibriumReactions: the definition of which aqueous reactions
    are accounted for in the model"""
    reactions: Tuple[Reaction, ...]
    ids: Tuple[int, ...]
    
# @dataclass
# class AqueousPopulation:
#     """AqueousPopulation: the definition of the aqueous species tracked
#     in the model"""
#     species: Tuple[AqueousSpecies, ...]
#     concentrations: Tuple[float, ...]
#     equilibria: Tuple[int, ...]
#     ids: Tuple[int, ...]

# def retrieve_eq_reactions(name, mechanism_data_path='../species_data/'):
#     reaction_datafile = mechanism_data_path + 'equilibrium_data.dat'
#     with open(reaction_datafile) as data_file:
#         for line in data_file:
#             if line.upper().startswith(name.upper()):
#                 gas,products,Keq0,Keq_exp = line.split()
#                 products=products.split(',')
#                 Keq0=np.array((Keq0.split(',')), dtype='float64')
#                 Keq_exp=np.array((Keq_exp.split(',')), dtype='float64')
#     return EqReaction(
#         gas=gas,
#         products=tuple(products),
#         Keq0=tuple(Keq0),
#         Keq_exp=tuple(Keq_exp))
    
# def retrieve_aq_species(name, eq_reactions, specdata_path='../species_data/'):
#     aq_datafile = specdata_path + 'aq_species_data.dat'
#     aero_datafile = specdata_path + 'aero_data.dat'
#     breaker=0
#     with open(aq_datafile) as data_file:
#         for line in data_file:
#             if line.upper().startswith(name.upper()):
#                 name_in_file,molar_mass = line.split()
#                 breaker=1
#     if breaker==0:
#         with open(aero_datafile) as data_file:
#             for line in data_file:
#                 if line.upper().startswith(name.upper()):
#                     name_in_file,density,ions_in_solution,molar_mass,kappa = line.split()
    
#     breaker=0
#     reactions=[]
#     for ii in range(len(eq_reactions.reactions)):
#         if name in eq_reactions.reactions[ii].products:
#             reactions.append(eq_reactions.ids[ii])
#             breaker=1
#     if breaker==0:
#         reactions=None    
    
#     return AqueousSpecies(
#         name=name,
#         molar_mass=float(molar_mass.replace('d','e')),
#         equilibria=tuple(reactions))