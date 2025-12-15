""" scenario - types and functions supporting the creation of aerosol particles
and particle populations

@author: Laura Fierce
"""


# Multi-scale particle-based microphysics model (multipart)
import sys, warnings
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Callable

warnings.filterwarnings('ignore')

@dataclass(frozen=True)
class AerosolSpecies:
    """AerosolSpecies: the definition of an aerosol species in terms of species-
    specific parameters (no state information)"""
    name: str          # name of the species
    density: float
    kappa: float
    molar_mass: float
    surface_tension: float

def retrieve_one_species(name, specdata_path='species_data/',surface_tension=0.072):
    aero_datafile = specdata_path + 'aero_data.dat'
    with open(aero_datafile) as data_file:
        for line in data_file:
            if line.split()[0]==name:
                name_in_file,density,ions_in_solution,molar_mass,kappa = line.split()
    
    return AerosolSpecies(
        name=name_in_file,
        density=float(density),
        kappa=float(kappa),
        molar_mass=float(molar_mass.replace('d','e')),
        surface_tension=surface_tension)


    
    
    
    
    
