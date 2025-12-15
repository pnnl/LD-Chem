#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:39:08 2024

@author: fier887
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class AerosolSpecies:
    """AerosolSpecies: the definition of an aerosol species in terms of species-
specific parameters (no state information)"""
    name: str          # name of the species
    density: float
    kappa: float
    molar_mass: float
    surface_tension: Optional[float] = 0.072
    
    # other parameters controlling phase partitioning
    # refractive_index: float

def retrieve_one_species(name, specdata_path='../species_data/',surface_tension=0.072):
    aero_datafile = specdata_path + 'aero_data.dat'
    with open(aero_datafile) as data_file:
        for line in data_file:
            if line.upper().startswith(name.upper()):
                name_in_file,density,ions_in_solution,molar_mass,kappa = line.split()
    return AerosolSpecies(
        name=name,
        density=float(density),
        kappa=float(kappa),
        molar_mass=float(molar_mass.replace('d','e')))

# move this into optical properties?
# def retrieve_RIs(
#         name,
#         specdata_path='../species_data/',
#         wavelengths=np.array([550e-9]),spec_params={}):
#     if name.upper() == 'H2O':
#         ri_h2o_filename = specdata_path + 'ri_water.csv'
#         # ri_h2o_filename = '../species_data/ri_water.csv'
#         wavelength_list = []
#         n_list = []
#         k_list = []
#         with open(ri_h2o_filename) as data_file:
#             for line in data_file:
#                 if not 'Wavelength' in line:
#                     split_output = line.split(',')
#                     wavelength_list.append(1e-6*get_number(split_output[0]))
#                     n_list.append(get_number(split_output[3]))
#                     k_list.append(get_number(split_output[4]))
#         n_vals = np.interp(wavelengths, np.array(wavelength_list), np.array(n_list))
#         k_vals = np.interp(wavelengths, np.array(wavelength_list), np.array(k_list))
#     else:
#         RI_params = get_RI_params(name)
#         if name in spec_params.keys():
#             for RI_var in spec_params[name].keys():
#                 RI_params[RI_var] = spec_params[name][RI_var]
#         n_vals = RI_params['n_550']*(wavelengths/(550e-9))**(-RI_params['alpha_n'])
#         k_vals = RI_params['k_550']*(wavelengths/(550e-9))**(-RI_params['alpha_k'])
#     refractive_indices = n_vals + 1j*k_vals
#     data_type = np.dtype(float,metadata={'units':'m','description':'wavelength of light'})
#     wavelengths = np.array(wavelengths,dtype=data_type)
#     data_type = np.dtype(complex,metadata={'units':None,'description':'complex refractive index of aerosol species'})
#     refractive_indices = np.array(refractive_indices,dtype=data_type)
    
# def get_RI_params(self,name):
#     if name.upper() in ['SO4','NH4','NO3','NA','CL','MSA']:
#         k_550 = 0.
#         n_550 = 1.55
#         alpha_n = 0.044 
#         alpha_k = 0.
        
#         # based on wavelength dependence of NaNO3 (only inorganic salt at RH=0%)
#         # data from here: https://eodg.atm.ox.ac.uk/ARIA/data?Salts/Sodium_Nitrate/10%25_(Cotterell_et_al._2017)/NaNO3_10_Cotterell_2017.ri
#         # underlying data:
#         #   Reference: Cotterell, M.I., Willoughby, R.E., Bzdek, B.R., Orr-Ewing, A.J. and Reid, J.P., A Complete Parameterization of the Relative Humidity and Wavelength Dependence of the Refractive Index of Hygroscopic Inorganic Aerosol Particles.
#         #   DOI: 10.5194/acp-17-9837-2017
#     elif name.upper() == 'BC':
#         k_550 = 0.74
#         n_550 = 1.82
#         alpha_n = 0.
#         alpha_k = 0.
#     elif name.upper() == 'OIN':
#         k_550 = 0.006
#         n_550 = 1.68
#         alpha_n = 0.
#         alpha_k = 0.
#     else: # organics
#         k_550 = 0.
#         n_550 = 1.45
#         alpha_n = 0.
#         alpha_k = 0.
    
#     RI_param_dict = {'n_550':n_550, 'k_550':k_550, 'alpha_n':alpha_n, 'alpha_k':alpha_k}
#     return RI_param_dict
    
