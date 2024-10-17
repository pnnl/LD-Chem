#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Laura Fierce and Payton Beeler
"""
import time, visualization
import numpy as np
from driver import simulate_dns_trajectories, simulate_parcel_trajectories
import visualization


print()
start_time = time.time()
N_scenarios = 1
Nparts = [1]


trajectory_ensemble = simulate_parcel_trajectories(N_scenarios=N_scenarios,
        z_start=0.,z_end=3000.,dt=2.0,
        radius_scale='log',solver='ode15s',
        Ddry=100e-9,sigma=2.0,Ntot=1e6, Npart=Nparts,
        updraft_velocity=0.5,S0=0.85,P0=101325,T0=298,
        accom=1., verbosity=50,
        case_num=2,dns_dir='/Users/fier887/Downloads/New_cases (7-27-22)/',
        Ddry=100e-9, Nper=1e6, species_names=['NaCl'], mass_fractions=np.array([1.]),
        specdata_path='../species_data/',
        condensation = True, collisions = False, settling = False,
        cocondensation = False, chemistry = False, freezing = False)
visualization.plot_parcel_trajectories(trajectory_ensemble)


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

elapsed_time = time.time() - start_time
print(' ')
print('====================================================')
print()
print('Total solving time:', round(elapsed_time, 2), 'seconds')
print(' ')
print('====================================================')



visualization.plot_parcel_trajectories(trajectory_ensemble[0])
