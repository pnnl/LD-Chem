# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:41:57 2024

@author: Nahin Ferdousi
"""

import os
import numpy as np
import constants_pypartition as cp



"Define properties of compounds"

def get_org_densities(org_species):

    
    rho_orgs = []
    for org in org_species:
        if org == 'glutaric acid':
            rho_orgs.append(1330) #kg/m3

    rho_org = np.array(rho_orgs)
    return rho_org

def get_inorg_densities(inorg_species):

    rho_inorgs = []
    for inorg in inorg_species:
        if inorg == 'AS':
            rho_inorgs.append(1770) #kg/m3

    rho_inorg = np.array(rho_inorgs)
    return rho_inorg


def get_org_MW(org_species):

    MW_orgs = []
    for org in org_species:
        if org == 'glutaric acid':
            MW_orgs.append(0.13214) #kg/mol

    MW_org = np.array(MW_orgs)
    return MW_org


def get_inorg_MW(inorg_species):

    MW_inorgs = []
    for inorg in inorg_species:
        if inorg == 'AS':
            MW_inorgs.append(0.146142) #kg/mol

    MW_inorg = np.array(MW_inorgs)
    return MW_inorg



"Compute droplet related properties"
def compute_total_volfrac(total_mass_frac_org, org_species, inorg_species):
  mass_org =(20.*10.**-6.)*total_mass_frac_org #in kg
  mass_inorg = (20.*10.**-6.)* (1.-total_mass_frac_org) #in g
  
  rho_org = get_org_densities(org_species)
  rho_inorg = get_inorg_densities(inorg_species)
  
  vol_org = mass_org/rho_org #in m^3
  vol_inorg = mass_inorg/rho_inorg #in m^3

  total_volfrac_org = vol_org/(vol_org+vol_inorg)
  
  return total_volfrac_org


def compute_total_vol(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet):
    V_dry = np.pi*(4./3.)*(D_dry/2.)**3 #in m^3
    V_wet = np.pi*(4./3.)*(D_wet/2.)**3 #in m^3
    
    vol_w_tot = V_wet - V_dry
    
    total_volfrac_org  = compute_total_volfrac(total_mass_frac_org, org_species, inorg_species)
    vol_org_tot = total_volfrac_org * V_dry
    vol_inorg_tot = (1-total_volfrac_org) * V_dry
    
    return vol_w_tot, vol_org_tot, vol_inorg_tot



def compute_total_mols(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet):
    
    vol_w_tot, vol_org_tot, vol_inorg_tot = compute_total_vol(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
    
    rho_org = get_org_densities(org_species)
    MW_org = get_org_MW(org_species)
    rho_inorg = get_inorg_densities(inorg_species)
    MW_inorg = get_inorg_MW(inorg_species)
    
    mol_org_tot = (vol_org_tot*rho_org)/MW_org #mols
    mol_inorg_tot = (vol_inorg_tot*rho_inorg)/MW_inorg #mols
    
    mol_w_tot = (vol_w_tot*cp.rho_w)/cp.Mw #mols
    
    return mol_org_tot, mol_inorg_tot, mol_w_tot




def compute_molarity_drop(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet):
    
    vol_w_tot, vol_org_tot, _ = compute_total_vol(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
    mol_org_tot, mol_inorg_tot, mol_w_tot = compute_total_mols(total_mass_frac_org, org_species, inorg_species, D_dry, D_wet)
 
    
    vol_w_tot_L = vol_w_tot*1000 #Liters
    molarity_drop = mol_org_tot/vol_w_tot_L #mol/L
    
    return molarity_drop


def compute_area(D_wet):
  area = np.pi*(D_wet/2.)**2
  return area

    



    


