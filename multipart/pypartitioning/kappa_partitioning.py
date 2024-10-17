# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:40:37 2024

@author: Nahin Ferdousi
"""
import numpy as np
import constants_pypartition as cp
from bulkmolefrac_solver import bulk_molfrac_func
import partitioning as prt
from ST_fits_database import compute_ST
import droplet_properties as drp
import partitioning as prt


def get_inorg_kappa(inorg_species):

    kappa_inorgs = []
    for inorg in inorg_species:
        if inorg == 'AS':
            kappa_inorgs.append(0.61) #kg/mol

    kappa_inorg = np.array(kappa_inorgs)
    return kappa_inorg



def get_org_kappa_b(org_species):

    kappa_orgs = []
    for org in org_species:
        if org == 'glutaric acid':
            kappa_orgs.append(0.78) #kg/mol #From kappa_bulk_fitting

    kappa_org = np.array(kappa_orgs)
    return kappa_org



def compute_kappa(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet, molarity_drop):
    kappa_org_b = get_org_kappa_b(org_species)
    kappa_inorg = get_inorg_kappa(inorg_species)
    
    f_org_b, f_org_s, f_inorg = prt.compute_frac_part(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet, molarity_drop)
    
    kappa_drop = f_org_b*kappa_org_b + f_org_s*0 + f_inorg * kappa_inorg
    
    return kappa_drop



def compute_kappa_Kohler(total_mass_frac_org, org_species, inorg_species):
    vol_frac_org_t= drp.compute_total_volfrac(total_mass_frac_org, org_species, inorg_species)
    kappa_org_b = get_org_kappa_b(org_species)
    kappa_inorg = get_inorg_kappa(inorg_species)
    
    
    kappa_k = (vol_frac_org_t * kappa_org_b) + (1-vol_frac_org_t)*kappa_inorg
    
    return kappa_k
    
