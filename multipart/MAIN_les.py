#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Laura Fierce and Payton Beeler
"""
import numpy as np
from driver import simulate_les_trajectories
# import visualization
# import matplotlib.pyplot as plt
# from initialization import splat_setup, optimize_splat_size_distribution
import pickle, sys

# %% LES trajectory
diameters=pickle.load(open('diameters', 'rb'))
num_concs=pickle.load(open('num_concs', 'rb'))
aero_spec_names=pickle.load(open('aero_spec_names', 'rb'))
aero_spec_fracs=pickle.load(open('aero_spec_fracs', 'rb'))
pHs=pickle.load(open('pHs', 'rb'))

gas_data=pickle.load(open('gas_data', 'rb'))
les_number=pickle.load(open('trajectory_number', 'rb'))

gas_names = ['SO2', 'O3', 'H2O2', 'NO2', 'IEPOX']
les_output_file = sys.argv[1]+'/parcel_traces_'+les_number+'.pkl'

with open('RUN_PROGRESS.out', 'w') as f:
    print('Reading', les_output_file, file=f)

trajectory = simulate_les_trajectories(les_output_file=les_output_file, output_path=str(sys.argv[2]),
        dt=6.0,diameters=diameters,N_concs=num_concs,
        pHs=pHs, accom=1e-2, verbosity=50,
        radius_scale='log',solver='ode15s',
        species_names=aero_spec_names, mass_fractions=aero_spec_fracs,
        gas_names=gas_names, gas_data=gas_data,
        specdata_path='/rcfs/projects/partikkel/multipart/species_data/',
        mechanism_data_path='/rcfs/projects/partikkel/multipart/mechanisms/',
        condensation = True, collisions = False, settling = False,
        cocondensation = True, entrainment = False, freezing = False,
        chemistry = ['IEPOX', 'sulfate'], write_every=60)
    

#pickle.dump(trajectory, open('../'+output_directory+'/'+output_file+'.pkl','wb'))



'''

# visualization.plot_aq_species(trajectory_ensemble[0], 'IEPOX', axis='time')

# visualization.plot_diameters(trajectory_ensemble[0], axis='time')




# import HISCALE_data_processing

trajectory_ensemble=pickle.load(open('../output_run1/run_1.pkl','rb'))
# visualization.plot_aq_species(trajectory_ensemble[0], 'SO4', axis='time')
visualization.plot_diameters(trajectory_ensemble, axis='time')

# splat_file='../datasets/HISCALE_data_0425/Splat_Composition_25-Apr-2016.txt'
# size_distribution_file='../datasets/HISCALE_data_0425/BEASD_G1_20160425155810_R2_HISCALE_001s.txt'
# ams_file='../datasets/HISCALE_data_0425/HiScaleAMS_G1_20160425_R0.txt'


# fig = HISCALE_data_processing.initial_SizeDist(trajectory_ensemble, size_distribution_file,
#                                           splat_species, mass_fractions, start_time=960, end_time=59160)
# fig.savefig('../OUTPUT/figures/initial_SizeDist_1traj_1000p.png', dpi=200, bbox_inches='tight')


# fig = HISCALE_data_processing.cloud_composition(trajectory_ensemble, splat_file, splat_species, size_distribution_file,
#                                                 mass_fractions, resolution=60)
# fig.savefig('../OUTPUT/figures/CDRcomp_1traj_1000p.png', dpi=200, bbox_inches='tight')


# fig = HISCALE_data_processing.initial_N_MassFracs(trajectory_ensemble[0], ams_file, splat_file,
#                                             splat_species, mass_fractions, start_time=960, 
#                                             end_time=59160)
# fig.savefig('../OUTPUT/figures/initial_N_MassFracs.png', dpi=200, bbox_inches='tight')

# fig=HISCALE_data_processing.ModelComposition(trajectory_ensemble[0], mass_fractions, splat_species,
#                                               resolution=60)
# plt.show()
# fig.savefig('../OUTPUT/figures/ModelComposition.png', dpi=200, bbox_inches='tight')


# fig=visualization.plot_trajectory_values(trajectory_ensemble[0])
# fig.savefig('../OUTPUT/figures/TrajectoryValues.png', dpi=200, bbox_inches='tight')



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
'''



