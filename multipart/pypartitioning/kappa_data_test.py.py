# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:40:59 2024

@author: Nahin Ferdousi
"""


import numpy as np
import constants_pypartition as cp
from bulkmolefrac_solver import bulk_molfrac_func
from ST_fits_database import compute_ST
import droplet_properties as drp
import partitioning as prt
import kappa_partitioning as kprt
import kappa_bulk_fitting as kbf
import unittest



def compute_kappa_test(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet, molarity_drop, RH, T):
    kappa_org_b = kbf.compute_kappa_org_b(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet, molarity_drop, RH, T)
    kappa_inorg = kprt.get_inorg_kappa(inorg_species)
    
    f_org_b, f_org_s, f_inorg = prt.compute_frac_part(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet, molarity_drop)
    
    kappa_drop_test = f_org_b*kappa_org_b + f_org_s*0 + f_inorg * kappa_inorg
    
    return kappa_drop_test



class TestKappa(unittest.TestCase):
    
    def test_kappa_drop(self):
        total_mass_frac_org = 0.5
        org_species = ['glutaric acid']
        inorg_species = ['AS'
                         ]
        
        D_dry = kbf.D_dry
        D_wet = kbf.D_wet
 
        molarity_drop = drp.compute_molarity_drop(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
        
        RH = kbf.RH_data
        T = 297
        
        
        kappa_eff = kbf.compute_effective_kappa(D_dry, D_wet, org_species, molarity_drop, RH, T)
        kappa_drop_test = compute_kappa_test(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet, molarity_drop, RH, T)
        
        self.assertEqual (kappa_eff, kappa_drop_test)
        
if __name__ == '__main__':
    unittest.main()
    
        
