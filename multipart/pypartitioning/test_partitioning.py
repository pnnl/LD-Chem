# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:22:12 2024

@author: Nahin Ferdousi
"""

import partitioning as prt
import droplet_properties as drp
import unittest


class TestOrganicVolume(unittest.TestCase):
    """Unit test for partitioning.py where organic total = organic bulk + organic surface"""
    
    def test_total_vol(self):
        total_mass_frac_org = 0.5
        org_species = ['glutaric acid']
        inorg_species = ['AS'
                         ]
        D_dry = 100*(10**-9) #m
        D_wet = 350 * (10**-9) #m
        
        molarity_drop = drp.compute_molarity_drop(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
        total = drp.compute_total_vol(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
        vol_org_b, vol_org_s = prt.compute_vol_bulk (total_mass_frac_org, org_species, inorg_species, D_dry, D_wet, molarity_drop)
        
        self.assertEqual (total, vol_org_b+vol_org_s)
        
if __name__ == '__main__':
    unittest.main()