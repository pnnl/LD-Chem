# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:24:15 2024

@author: Nahin Ferdousi
"""


"Generate Kohler Curves"



import kappa_partitioning as kp
import constants_pypartition as cp
import droplet_properties as drp
import partitioning as prt
import ST_fits_database as stf
from ST_fits_database import compute_ST
import numpy as np




D_dry_kohler = np.array([100*10**-9])
D_wet_kohler_log = np.logspace(np.log10(110*10**-9),-5,100)*1e9
D_wet_kohler = D_wet_kohler_log*1e-9



org_species = ['glutaric acid']

inorg_species = ['AS']
total_mass_frac_org = 0.5





def compute_vol_frac_Kohler(total_mass_frac_org, org_species, inorg_species):

  mass_frac_org = np.array(total_mass_frac_org)
  vol_frac_org = drp.compute_total_volfrac(total_mass_frac_org, org_species, inorg_species)

  #print('vol_frac_org', vol_frac_org)
  return vol_frac_org

def SSeq_Kohler(total_mass_frac_org, org_species, inorg_species, D_dry_kohler, D_wet_kohler):


  ST_w = 0.072 #N/m
  kappa_org = 0.06 #Effective kappa of 2-MGA
  kappa_AS = 0.61
  vol_frac_org = compute_vol_frac_Kohler(total_mass_frac_org, org_species, inorg_species)

  kappa_Kohler = (vol_frac_org*kappa_org) + (1-vol_frac_org)*kappa_AS

  A_kohler = (D_wet_kohler**3 - D_dry_kohler**3)/(D_wet_kohler**3 - ((D_dry_kohler**3)*(1-kappa_Kohler)))


  B_kohler = 4*ST_w*cp.Mw/(cp.R*cp.T*cp.rho_w)
  ss_k_calc = A_kohler * np.exp(B_kohler/D_wet_kohler)

  SS_k =((ss_k_calc-1))*100

  return SS_k


def SSeq_CMC(total_mass_frac_org, org_species, inorg_species, D_dry_kohler, D_wet_kohler):


  ST_CMC = 0.040 #N/m
  kappa_org = 0.06 #Effective kappa of 2-MGA
  kappa_AS = 0.61
  vol_frac_org = compute_vol_frac_Kohler(total_mass_frac_org, org_species, inorg_species)
  kappa_Kohler = (vol_frac_org*kappa_org) + (1-vol_frac_org)*kappa_AS

  A_CMC = (D_wet_kohler**3 - D_dry_kohler**3)/(D_wet_kohler**3 - ((D_dry_kohler**3)*(1-kappa_Kohler)))


  B_CMC = 4*ST_CMC*cp.Mw/(cp.R*cp.T*cp.rho_w)
  ss_cmc_calc = A_CMC * np.exp(B_CMC/D_wet_kohler)

  SS_CMC =  ((ss_cmc_calc-1))*100
  return SS_CMC




def kappa_partition_Kohler(total_mass_frac_org, org_species, inorg_species, D_dry_kohler, D_wet_kohler, molarity_drop):
  mass_frac_org = np.array(total_mass_frac_org)
  

  kappa_AS = 0.61
  kappa_org_b = kp.get_org_kappa_b(org_species)
  kappa_org_s = 0

  kappa_part_kohler = kp.compute_kappa(total_mass_frac_org, org_species, inorg_species, D_dry_kohler, D_wet_kohler, molarity_drop)

  #print('kappa', kappa_part_kohler)
  return kappa_part_kohler



def ST_part_Kohler(D_wet_kohler, D_dry_kohler, mass_frac_org):
  mass_frac_org = np.array(mass_frac_org)
  ST_part = compute_ST(total_mass_frac_org, org_species, inorg_species, D_dry_kohler, D_wet_kohler)

  return ST_part

def SSeq_part(mass_frac_org, org_species, inorg_species, D_dry_kohler, D_wet_kohler):

  """This is for partitioning"""
  mass_frac_org = np.array(mass_frac_org)
  ST_part = ST_part_Kohler(D_wet_kohler, D_dry_kohler, mass_frac_org)
  f_org_b, f_org_s, f_inorg = prt.compute_frac_part(total_mass_frac_org, org_species, inorg_species, D_dry_kohler, D_wet_kohler)
  
  kappa_inorg = kp.get_inorg_kappa(inorg_species)
  kappa_org_b = kp.get_org_kappa_b(org_species)
  kappa_org_s = 0
  
  kappa_part = f_org_b*kappa_org_b + f_org_s*kappa_org_s + f_inorg*kappa_inorg


  #print('kappa', kappa_part)


  A_part = (D_wet_kohler**3 - D_dry_kohler**3)/(D_wet_kohler**3 - ((D_dry_kohler**3)*(1-kappa_part)))


  B_part = 4*ST_part*cp.M_w/(cp.R*cp.T*cp.rho_w)


  ss_part_calc = A_part * np.exp(B_part/D_wet_kohler)

  SS_part =  ((ss_part_calc-1))*100
  return SS_part

ss_eq_k_arr = []
ss_eq_CMC_arr = []
ss_eq_part_arr = []
kappa_part_arr = []
ST_part_arr = []

for d in D_wet_kohler:

  ss_eq_k = SSeq_Kohler(total_mass_frac_org, org_species, inorg_species, D_dry_kohler, d)
  ss_eq_k_arr.append(ss_eq_k)

  ss_eq_CMC = SSeq_CMC(total_mass_frac_org, org_species, inorg_species, D_dry_kohler, d)
  ss_eq_CMC_arr.append(ss_eq_CMC)


  ss_eq_part = SSeq_part(total_mass_frac_org, D_dry_kohler, d)
  ss_eq_part_arr.append(ss_eq_part)


  ST_part = ST_part_Kohler(d, D_dry_kohler, total_mass_frac_org)
  ST_part_arr.append(ST_part)

  kappa_part = kappa_partition_Kohler(D_dry_kohler, d, total_mass_frac_org)
  kappa_part_arr.append(kappa_part)

mask = ~np.isnan(ss_eq_part)  & ~np.isnan(kappa_part)

ss_eq_part = ss_eq_part = np.where(mask, ss_eq_part_arr, np.nan)
#print(ss_eq_part)
ss_eq_part_plot = np.array(ss_eq_part)

#print('ss eq', ss_eq_part_plot)

ST_part_arr_plot = np.array(ST_part_arr)

#print('ST', ST_part_arr_plot)
kappa_part = np.where(mask, kappa_part_arr, np.nan)
kappa_part_arr_plot =  np.array(kappa_part)
