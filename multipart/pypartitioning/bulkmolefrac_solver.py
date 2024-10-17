# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:40:02 2024

@author: Nahin Ferdousi
"""

import constants_pypartition as cp
from ST_fits_database import compute_ST
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import droplet_properties as drp
from ST_Fitting import molarity_vals, ST_data




def get_gammax_SL(org_species):

    if org_species == 'glutaric acid':
        gam_max = 1045.94
        
    return gam_max


def get_Ki_SL(org_species):
    
    if org_species == 'glutaric acid':
        Ki = 5e-7
        
    return Ki

def bulk_molfrac_func(molarity_vals, org_species, ST_data):
    org_species = 'glutaric acid'
    
    gam_max = get_gammax_SL(org_species)
    Ki = get_Ki_SL(org_species)
    
    xib_org_drop = (np.exp((cp.ST_w-ST_data)/(cp.R*cp.T*gam_max))-1)/Ki
    
    
    return xib_org_drop 


def fit_molarity_mol_frac(molarity_vals, slope3, param2):

  xi_bulk_fit = slope3*np.log(molarity_vals) + param2

  #M_thresh_CMC = 40
  #xi_b_CMC = 0.025

  #mask = molarity_vals >= M_thresh_CMC
  #xi_bulk_fit[mask] = xi_b_CMC

  return xi_bulk_fit


plt.figure(2)
xib_org_drop = bulk_molfrac_func(molarity_vals, 'glutaric acid', ST_data)

plt.scatter(molarity_vals, xib_org_drop)



fit_molarity = opt.curve_fit(fit_molarity_mol_frac, molarity_vals, xib_org_drop)
slope3 = fit_molarity [0][0]
param2 = fit_molarity [0][1]

print('slope3', slope3)
print('param2', param2)

molarity_test_xvals = np.linspace(0,40,100)
plt.plot(molarity_test_xvals, fit_molarity_mol_frac(molarity_test_xvals, slope3, param2), linestyle = '-.', color = 'green', label='Calculated Bulk Mole Fraction (S-L) Fit')

plt.scatter(molarity_vals, xib_org_drop, marker = 's', color = 'orange', label = 'Calculated Bulk Mole Fraction (S-L) from Data')



plt.title('Fig 1b. Calculated xib vs. 2-MGA Droplet Concentration')
plt.xlabel('Organic Molarity (mol/L)')
plt.ylabel('Calculated xib')
plt.xlim(0,40)

plt.legend()








