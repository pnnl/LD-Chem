# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 09:42:53 2024

@author: Nahin Ferdousi
"""

"For each species, obtain fitting parameters for bulk mole fraction based on molarity"

import numpy as np
import constants_pypartition as cp
import droplet_properties as drp
from ST_Fitting import molarity_vals, ST_data #for testing


def get_gammax_SL(org_species):

    if org_species == 'glutaric acid':
        gam_max = 1045.94
        
    return gam_max


def get_Ki_SL(org_species):
    
    if org_species == 'glutaric acid':
        Ki = 5e-7
        
    return Ki



def compute_xib(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet):

    def xib_func_glutaric(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet):
        molarity_drop = drp.compute_molarity_drop(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
        slope3 = 0.002555835763021986
        param2 = 0.01585504760537587
    

    
        xi_bulk = slope3*np.log(molarity_drop) + param2

  
        xi_bulk[molarity_drop == 0] = 0

        return xi_bulk

    if org_species == 'glutaric acid':
        xib = xib_func_glutaric(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
    else: 
        raise ValueError("Unsupported species")
        
    return xib
