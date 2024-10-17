# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:02:52 2024

@author: Nahin Ferdousi
"""

"ST fits for species"

"For each species, obtain fitting parameters for surface tension and bulk mole fraction"

import numpy as np
import constants_pypartition as cp
from droplet_properties import compute_molarity_drop
from ST_Fitting import molarity_vals, ST_data
def compute_ST(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet):

    def ST_func_glutaric(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet):
        
        molarity_drop = compute_molarity_drop(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
        M_thresh = 0.022550919889558464
        ST_dilute = 0.068146
        slope = -0.004105193967600265
    
        b = ST_dilute - slope*np.log(M_thresh)
    
        ST_drop =  slope*np.log(molarity_drop) + b 

  
        ST_drop[molarity_drop<=M_thresh] = ST_dilute

        return ST_drop

    if org_species == 'glutaric acid':
        ST_drop = ST_func_glutaric(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
    else: 
        raise ValueError("Unsupported species")
        
    return ST_drop


#result = compute_ST('glutaric acid', molarity_vals)

#print('result', result)







    
    