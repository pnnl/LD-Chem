#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Laura Fierce and Payton Beeler
"""
import time, visualization
import numpy as np
from driver import simulate_dns_trajectories, simulate_parcel_trajectories, simulate_hysplit_trajectories, simulate_les_trajectories
import visualization
import matplotlib.pyplot as plt
from initialization import splat_setup

# %% LES trajectory

diameters=np.array([30e-9, 50e-9, 100e-9, 150e-9, 200e-9, 250e-9])
N_concs=np.array([1000, 1000, 1000, 1000, 1000, 1000])
pHs=np.array([7.0, 7.0, 7.0, 7.0, 7.0, 7.0])
species_names=[['AS'],
               ['AS'],
               ['AS'],
               ['AS'],
               ['AS'],
               ['AS']]
mass_fractions=[[1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0]]
'''
splat_species = {'black carbon': ['soot'],
               'sulfate rich': ['sulfate_nitrate_org'],
               'nitrate rich': ['nitrate_amine_org'], 
               'organics': ['org28', 'org30_43', 'BB_SOA', 'org_amines', 'BB', 'pyridine'], 
               'IEPOX': ['IEPOX_SOA'],
               'dust': ['Dust']}

diameters=splat_setup(Npart=10, 
                      size_distribution_file='../datasets/HISCALE_data_0425/BEASD_G1_20160425155810_R2_HISCALE_001s.txt',
                      splat_file='../datasets/HISCALE_data_0425/Splat_Composition_25-Apr-2016.txt',
                      splat_species=splat_species,
                      ams_file='../datasets/HISCALE_data_0425/HiScaleAMS_G1_20160425_R0.txt',
                      start_time=960, end_time=59160, 
                      cloud_flag=0, CVI_flag=0)
'''


les_output_file = '../datasets/parcel_traces_se/parcel_traces_000000.pkl'
trajectory_ensemble = simulate_les_trajectories(les_output_file=les_output_file,
        dt=3.0,diameters=diameters,N_concs=N_concs,
        pHs=pHs, accom=1., verbosity=50,
        radius_scale='lin',solver='CVODE',
        species_names=species_names, mass_fractions=mass_fractions,
        gas_names=None, gas_conc=None,
        specdata_path='../species_data/', mechanism_data_path='../mechanisms/',
        condensation = True, collisions = False, settling = False,
        cocondensation = False, chemistry = False, freezing = False)

figure = visualization.plot_diameters(trajectory_ensemble[0], axis='time')
figure.savefig('../Fig1.png', bbox_inches='tight', dpi=400)
plt.show()

figure = visualization.plot_trajectory_values(trajectory_ensemble[0])
figure.savefig('../Fig2.png', bbox_inches='tight', dpi=400)
plt.show()

# %% HYPLIT trajectory

# hysplit_tdump_file = '../datasets/HYSPLIT/HISCALE_ensemble_trajectory_04252016.txt'
# trajectory_ensemble = simulate_hysplit_trajectories(hysplit_tdump_file=hysplit_tdump_file,
#         scenario_numbers=[0], dt=3.0,Ddry=100e-9,sigma=1.6,Ntot=1e6, Npart=3,
#         pH0=7.0, accom=1., verbosity=50,
#         radius_scale='lin',solver='CVODE',
#         species_names=['NaCl'], mass_fractions=np.array([1.]),
#         gas_names=None, gas_conc=None,
#         specdata_path='../species_data/', mechanism_data_path='../mechanisms/',
#         condensation = True, collisions = False, settling = False,
#         cocondensation = False, chemistry = False, freezing = False)

# figure = visualization.plot_diameters(trajectory_ensemble[0], axis='time')
# plt.show()

# figure = visualization.plot_trajectory_values(trajectory_ensemble[0])  
# plt.show()



# %% parcel trajectory
# print()
# start_time = time.time()
# N_scenarios = 1
# Nparts = [10]

# trajectory_ensemble = simulate_parcel_trajectories(N_scenarios=N_scenarios,
#         z_start=0.,z_end=2000.,dt=1.0,
#         radius_scale='lin',solver='ode15s',
#         Ddry=100e-9,sigma=2.0,Ntot=1e7, Npart=Nparts,
#         updraft_velocity=0.5,S0=0.85,
#         pH0=7.0,P0=101325,T0=298,
#         accom=1., verbosity=50,
#         species_names=['NaCl'], mass_fractions=np.array([1.0]),
#         gas_names=None, gas_conc=None,
#         specdata_path='../species_data/', mechanism_data_path='../mechanisms/',
#         condensation = True, collisions = False, settling = False,
#         cocondensation = False, chemistry = None, freezing = False)

# figure = visualization.plot_diameters(trajectory_ensemble[0], axis='height')
# plt.show()




# %% DNS trajectory

# trajectory_ensemble = simulate_dns_trajectories(
#         t_start=0.,t_end=3600.,dt=1.,this_many=10,
#         accom=1., verbosity=50,
#         radius_scale='log',force_cvode=False,
#         case_num=2,dns_dir='/Users/fier887/Downloads/New_cases (7-27-22)/',
#         Ddry=100e-9, Nper=1e6, species_names=['NaCl'], mass_fractions=np.array([1.]),
#         specdata_path='../species_data/',
#         condensation = True, collisions = False, settling = False,
#         cocondensation = False, chemistry = False, freezing = False)
# visualization.plot_dns_trajectories(trajectory_ensemble)


# trajectory_ensemble = simulate_dns_trajectories(
#         t_start=0.,t_end=3600.,dt=1.,this_many=10,
#         accom=1., verbosity=50,
#         radius_scale='lin',force_cvode=False,
#         case_num=2,dns_dir='/Users/fier887/Downloads/New_cases (7-27-22)/',
#         Ddry=100e-9, Nper=1e6, species_names=['NaCl'], mass_fractions=np.array([1.]),
#         specdata_path='../species_data/',
#         condensation = True, collisions = False, settling = False,
#         cocondensation = False, chemistry = False, freezing = False)




