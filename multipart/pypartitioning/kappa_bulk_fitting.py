# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:18:35 2024

@author: Nahin Ferdousi
"""

import numpy as np
import constants_pypartition as cp
from bulkmolefrac_solver import bulk_molfrac_func
from ST_fits_database import compute_ST
import droplet_properties as drp
import partitioning as prt
import kappa_partitioning as kprt
import unittest

data_path_HTDMA = r'C:\Users\Nahin Ferdousi\OneDrive\Desktop\Partitioning Model Data\HTDMA_Data.csv'

data_HTDMA = np.loadtxt(data_path_HTDMA,skiprows=1,delimiter=',',usecols=None)
D_dry = data_HTDMA[:,0]*1e-9 # in meters
D_wet = data_HTDMA[:,1]*1e-9 # in meters
total_fraction_2MGA = data_HTDMA[:,2]
RH_data = data_HTDMA[:,3]
GF_data = data_HTDMA[:,4]
T = 297

print("Ddry", D_dry)
print("Dwet", D_wet)
print("mass frac 2MGA", total_fraction_2MGA)
print('RH', RH_data)
print('GF', GF_data)



def compute_effective_kappa(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet, RH, T):
    ST_drop = compute_ST(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
    
    A = 4 * ST_drop * cp.Mw/(cp.R * T * cp.rho_w)
    
    GF = D_wet/D_dry
    
    lhs = RH/(np.exp(A/(D_dry*GF)))
    
    kappa_eff = (((GF**3)-1)/(lhs))-(GF**3)+1
    return kappa_eff



def compute_kappa_org_b(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet, molarity_drop, RH, T):
    f_org_b, f_org_s, f_inorg = prt.compute_frac_part(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
    kappa_eff = compute_effective_kappa(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet, RH, T)
    
    kappa_org_s = 0
    kappa_inorg = kprt.get_inorg_kappa(inorg_species)
    
    kappa_org_b = (kappa_eff - (f_inorg*kappa_inorg - f_org_s*kappa_org_s))/f_org_b
    
    
    return kappa_org_b

"Compute and get average of kappa_org_b"


molarity_drop = drp.compute_molarity_drop(total_fraction_2MGA, 'glutaric acid', 'AS', D_dry, D_wet)
bulk_kappa_b = compute_kappa_org_b(total_fraction_2MGA, 'glutaric acid', 'AS', D_dry, D_wet, molarity_drop, RH_data, T)

kappa_org_b_av = np.mean(bulk_kappa_b)
