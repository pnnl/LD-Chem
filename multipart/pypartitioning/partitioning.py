# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 19:52:54 2024

@author: Nahin Ferdousi
"""

"""Calculate the amount of organic partitioned between the bulk and the surface based on surface tension

Assume the following:
    1. Inorganic and water enters the bulk (vol_inorg_tot = vol_inorg_b and vol_w_tot = vol_w_b)
    2. Calculate ST based on droplet molarity and S-L solver
    
    3. Calculate bulk mole fraction based on (2)
    
    4. Use mole fraction to calculate moles organic in bulk where: xib = mol_org/(mol_org + mol_inorg + mol_w)
    
    5. Calculate moles ---> volume 
    
    6. Perform unit test
    
"""


import numpy as np
import constants_pypartition as cp
from bulkmolefrac_solver import bulk_molfrac_func
from ST_fits_database import compute_ST
from bulkmolefrac_database import compute_xib
import droplet_properties as drp



    
def compute_mol_bulk(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet):
    
    molarity_drop = drp.compute_molarity_drop(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
    xib_org_drop  = compute_xib(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
    mol_org_tot, mol_inorg_tot, mol_w_tot = drp.compute_total_mols(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
    mol_org_b = (xib_org_drop*mol_w_tot)/(1-xib_org_drop)

    
    for i in range(len(mol_org_tot)):
        if mol_org_b[i] > mol_org_tot[i]:
            mol_org_b[i] = mol_org_tot[i]

    n_org_frac = mol_org_b/mol_org_tot
    mol_org_s = mol_org_tot - mol_org_b
    return mol_org_b, n_org_frac



def compute_mol_surf(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet):
    mol_org_b, n_org_frac = compute_mol_bulk(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
    mol_org_tot, mol_inorg_tot, mol_w_tot = drp.compute_total_mols(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)   

    mol_org_s = (1-n_org_frac)*mol_org_tot
    
    return mol_org_s


def compute_vol_bulk (total_mass_frac_org, org_species, inorg_species, D_dry, D_wet):
    rho_org = drp.get_org_densities(org_species)
    MW_org = drp.get_org_MW(org_species)
    MV_org = MW_org/rho_org
    
    rho_inorg = drp.get_inorg_densities(inorg_species)
    MW_inorg = drp.get_inorg_MW(inorg_species)    
    MV_inorg = MW_inorg/rho_inorg
    
    mol_org_b, n_org_frac = compute_mol_bulk(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
    mol_org_s = compute_mol_surf(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
    
    vol_org_b = mol_org_b * MV_org #mol*(m3/mol)
    vol_org_s = mol_org_s* MV_inorg #mol*(m3/mol)
    
    return vol_org_b, vol_org_s

def compute_frac_part(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet):
    vol_org_b, vol_org_s = compute_vol_bulk (total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
    vol_w_tot, vol_org_tot, vol_inorg_tot = drp.compute_total_vol(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
    
    v_dry = vol_org_tot + vol_inorg_tot 
    
    f_org_b = vol_org_b / v_dry
    f_org_s = vol_org_s / v_dry
    
    f_inorg = vol_inorg_tot / v_dry 
    
    
    return f_org_b, f_org_s, f_inorg


    
        