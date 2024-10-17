# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 10:19:44 2024

@author: Nahin Ferdousi
"""

import constants_pypartition as cp
from ST_fits_database import compute_ST
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import droplet_properties as drp



def get_gammax_Top(org_species):

    if org_species == 'glutaric acid':
        gam_max = 5e-7
        
    return gam_max


def get_Ki_Top(org_species):
    
    if org_species == 'glutaric acid':
        Ki = 1045.94
        
    return Ki


def compute_xib_top(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet):
  mol_org, mol_inorg, mol_w = drp.compute_total_mols(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
  area = drp.compute_area(D_wet)
  Gam_top = get_gammax_Top(org_species)
  Ki_top = get_Ki_Top(org_species)
  
    
  a = (area*Gam_top*Ki_top)-(mol_w*Ki_top)-((mol_org)*Ki_top)

  b = ((mol_org)*Ki_top)-(mol_org)-mol_w-(area*Gam_top*Ki_top)

  c = (mol_org)

  xib_top = -b - np.sqrt(b**2 - 4*a*c) / (2*a)
  return xib_top



def compute_bulk_molfrac_top(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet):
    xib_top = compute_xib_top(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)

    mol_org, mol_inorg, mol_w = drp.compute_total_mols(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
    mol_org_b_top = (xib_top*mol_w)/(1-xib_top)

    #print('num', xib_top*mol_w)
    #print('den', (1-xib_top))

    #print('mol_org', mol_org)
    #print('mol_org_b_top', mol_org_b_top)
    n_org_frac_top = mol_org_b_top/mol_org
    
    return mol_org_b_top, n_org_frac_top

def compute_org_mol_surf(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet):
  mol_org, mol_inorg, mol_w = drp.compute_total_mols(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
  mol_org_b_top, n_org_frac_top = compute_bulk_molfrac_top(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
  mol_org_surf = (1-n_org_frac_top)*mol_org
  return mol_org_surf


