#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:06:10 2024

@author: beel083
"""
# %% 
# copy the necessary modules to the UNIT_TESTS directory
# probably need a different way to do this but I don't 
# want to mess with sys.path

import shutil, os, pickle, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.simplefilter('ignore')

files1 = ['particles.py', 'constants.py', 'scenario.py', 'aerosol_species.py',
          'utilities.py', 'systems.py', 'driver.py', 'visualization.py', 
          'TraceGases.py', 'Reactions.py', 'write_files.py']

for file in files1:
    source = '../../multipart/'+file
    destination = os.getcwd()+'/'+file
    shutil.copy(source, destination)
    
files2 = ['UnitTests_driver.py', 'UnitTests_scenario.py', 'UnitTests_visualization.py']

for file in files2:
    source = '../'+file
    destination = os.getcwd()+'/'+file
    shutil.copy(source, destination)

directories = ['../../multipart/processes', '../../species_data', '../../mechanisms']
for directory in directories:
    source = directory
    destination = source.replace('.', '')
    destination = destination.replace('/', '')
    destination = destination.replace('multipart', '')
    if os.path.isdir(destination):
        shutil.rmtree(destination)    
    destination = os.getcwd()+'/'+destination
    shutil.copytree(source, destination)

from driver import simulate_les_trajectories

# %% set up the plot

axis_label_fontsize=18
axis_tick_fontsize=16
legend_fontsize=16
markersize=11
output_data={}

fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = plt.subplots(2, 3, figsize=(3.0*6.4, 2.0*4.8), constrained_layout=False, sharex='col', sharey='row')
ax11.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
ax11.tick_params(which="major", axis="both", length=6)
ax11.tick_params(which="minor", axis="both", length=4)
ax11.grid(which='major', color='grey', alpha=0.4, linewidth=1)
ax21.tick_params(axis='both', which="major",labelsize=axis_tick_fontsize, pad=8, width=1)
ax21.tick_params(which="major", axis="both", length=6)
ax21.tick_params(which="minor", axis="both", length=4)
ax21.grid(which='major', color='grey', alpha=0.4, linewidth=1)
ax12.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
ax12.tick_params(which="major", axis="both", length=6)
ax12.tick_params(which="minor", axis="both", length=4)
ax12.grid(which='major', color='grey', alpha=0.4, linewidth=1)
ax22.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
ax22.tick_params(which="major", axis="both", length=6)
ax22.tick_params(which="minor", axis="both", length=4)
ax22.grid(which='major', color='grey', alpha=0.4, linewidth=1)
ax13.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
ax13.tick_params(which="major", axis="both", length=6)
ax13.tick_params(which="minor", axis="both", length=4)
ax13.grid(which='major', color='grey', alpha=0.4, linewidth=1)
ax23.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
ax23.tick_params(which="major", axis="both", length=6)
ax23.tick_params(which="minor", axis="both", length=4)
ax23.grid(which='major', color='grey', alpha=0.4, linewidth=1)
# ax14.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
# ax14.tick_params(which="major", axis="both", length=6)
# ax14.tick_params(which="minor", axis="both", length=4)
# ax14.grid(which='major', color='grey', alpha=0.4, linewidth=1)
# ax24.tick_params(axis='both', which="major",labelsize=axis_tick_fontsize, pad=8, width=1)
# ax24.tick_params(which="major", axis="both", length=6)
# ax24.tick_params(which="minor", axis="both", length=4)
# ax24.grid(which='major', color='grey', alpha=0.4, linewidth=1)


# %% changing updraft velocity

Vs = np.logspace(-1.0, 1.0, 10) # 0.1 - 5.0 m/s
multipart_actfracs=np.zeros(len(Vs))
multipart_Smaxes=np.zeros(len(Vs))

print()
print('changing velocity...')
for ii, (V) in enumerate(Vs):
    
    print(V)
    
    diameters=np.loadtxt('pyrcel_changing_velocity/V_'+str(V)+'_diameters.txt')
    num_concs=np.loadtxt('pyrcel_changing_velocity/V_'+str(V)+'_num_concs.txt')
    pHs=np.repeat(7.0, diameters.shape)
    
    gas_data=None
    gas_names=None
    
    aero_names_temp=np.loadtxt('pyrcel_changing_velocity/V_'+str(V)+'_aero_spec_names.txt', dtype='str')
    aero_fracs_temp=np.loadtxt('pyrcel_changing_velocity/V_'+str(V)+'_aero_spec_fracs.txt')
    aero_spec_names=[]
    aero_spec_fracs=[]
    for kk in range(len(aero_fracs_temp)):
        aero_spec_names.append([aero_names_temp[kk]])
        aero_spec_fracs.append([aero_fracs_temp[kk]])
    
    temp=np.loadtxt('pyrcel_changing_velocity/V_'+str(V)+'_trajectory.txt', dtype='str')
    data_dict={}
    for kk in range(len(temp[0])):
        data_dict[temp[0, kk]]=np.array(temp[1:,kk], dtype='float64')
    pickle.dump(data_dict, open('trajectory_temp.pkl', 'wb'))
    
    les_output_file = os.getcwd()+'/trajectory_temp.pkl' #'../datasets/parcel_traces_se/parcel_traces_000000.pkl'  

    print('Reading', les_output_file)

    trajectory = simulate_les_trajectories(les_output_file=les_output_file, output_path=os.getcwd(),
            dt=5.0,diameters=diameters,N_concs=num_concs,
            pHs=pHs, accom=1., verbosity=50,
            radius_scale='log',solver='ode15s',
            species_names=aero_spec_names, mass_fractions=aero_spec_fracs,
            gas_names=gas_names, gas_data=gas_data,
            specdata_path='../../species_data/', mechanism_data_path='../../mechanisms/',
            condensation = True, collisions = False, settling = False,
            cocondensation = False, chemistry = None, freezing = False, write_every=60)
    
    multipart_Smaxes[ii]=trajectory[0].get_max_S()
    multipart_actfracs[ii]=trajectory[0].get_activated_fraction()


pyrcel_Vs, pyrcel_actfracs, pyrcel_Smax = np.loadtxt('pyrcel_changing_velocity/pyrcel_results.txt', 
                                                     delimiter=' ', skiprows=1, unpack=True)
ax11.plot(pyrcel_Vs, 100*pyrcel_Smax, 'wo', mec='k', markersize=markersize, zorder=0)
ax21.plot(pyrcel_Vs, pyrcel_actfracs, 'wo', mec='k', markersize=markersize, zorder=0)

pickle.dump({'V': Vs, 'Smax': multipart_Smaxes-1, 'ActFracs': multipart_actfracs}, open('MultiPart_Vs.pkl', 'wb'))

ax11.plot(Vs, 100*(multipart_Smaxes-1), '-r')
ax21.plot(Vs, multipart_actfracs, '-r')

# fix the plot
ax11.set_xscale('log')
ax11.set_yscale('log')
ax11.set_ylim(1e-2, 10)
ax11.set_xlim(0.08, 12)
ax11.set_ylabel('Maximum Supersaturation (%)', fontsize=axis_label_fontsize, labelpad=15)
ax11.text(0.04, 0.95, ' A ', fontsize=axis_label_fontsize, ha='left', va='top', transform=ax11.transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'})
ax21.set_ylim(0,1)
ax21.set_ylabel('Activated Fraction', fontsize=axis_label_fontsize, labelpad=27)
ax21.set_xlabel('Updraft Velocity (m/s)', fontsize=axis_label_fontsize, labelpad=15)
ax21.text(0.04, 0.95, ' B ', fontsize=axis_label_fontsize, ha='left', va='top', transform=ax21.transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'})

# %% changing number concentration

Ntots = np.logspace(2, 4, 10) # 1/cm^3
multipart_actfracs=np.zeros(len(Ntots))
multipart_Smaxes=np.zeros(len(Ntots))

print()
print('changing Ntot...')
for ii, (Ntot) in enumerate(Ntots):
    
    print(ii, Ntot)
    
    diameters=np.loadtxt('pyrcel_changing_Ntot/Ntot_'+str(Ntot)+'_diameters.txt')
    num_concs=np.loadtxt('pyrcel_changing_Ntot/Ntot_'+str(Ntot)+'_num_concs.txt')
    pHs=np.repeat(7.0, diameters.shape)
    
    gas_data=None
    gas_names=None
    
    aero_names_temp=np.loadtxt('pyrcel_changing_Ntot/Ntot_'+str(Ntot)+'_aero_spec_names.txt', dtype='str')
    aero_fracs_temp=np.loadtxt('pyrcel_changing_Ntot/Ntot_'+str(Ntot)+'_aero_spec_fracs.txt')
    aero_spec_names=[]
    aero_spec_fracs=[]
    for kk in range(len(aero_fracs_temp)):
        aero_spec_names.append([aero_names_temp[kk]])
        aero_spec_fracs.append([aero_fracs_temp[kk]])
    
    temp=np.loadtxt('pyrcel_changing_Ntot/Ntot_'+str(Ntot)+'_trajectory.txt', dtype='str')
    data_dict={}
    for kk in range(len(temp[0])):
        data_dict[temp[0, kk]]=np.array(temp[1:,kk], dtype='float64')
    pickle.dump(data_dict, open('trajectory_temp.pkl', 'wb'))
    
    les_output_file = os.getcwd()+'/trajectory_temp.pkl' #'../datasets/parcel_traces_se/parcel_traces_000000.pkl'  

    print('Reading', les_output_file)

    trajectory = simulate_les_trajectories(les_output_file=les_output_file, output_path=os.getcwd(),
            dt=5.0,diameters=diameters,N_concs=num_concs,
            pHs=pHs, accom=1., verbosity=50,
            radius_scale='log',solver='ode15s',
            species_names=aero_spec_names, mass_fractions=aero_spec_fracs,
            gas_names=gas_names, gas_data=gas_data,
            specdata_path='../../species_data/', mechanism_data_path='../../mechanisms/',
            condensation = True, collisions = False, settling = False,
            cocondensation = False, chemistry = None, freezing = False, write_every=60)
    
    multipart_Smaxes[ii]=trajectory[0].get_max_S()
    multipart_actfracs[ii]=trajectory[0].get_activated_fraction()


pyrcel_Ntots, pyrcel_actfracs, pyrcel_Smax = np.loadtxt('pyrcel_changing_Ntot/pyrcel_results.txt', 
                                                     delimiter=' ', skiprows=1, unpack=True)
ax12.plot(pyrcel_Ntots, 100*pyrcel_Smax, 'wo', mec='k', markersize=markersize, zorder=0)
ax22.plot(pyrcel_Ntots, pyrcel_actfracs, 'wo', mec='k', markersize=markersize, zorder=0)

pickle.dump({'Ntot': Ntots, 'Smax': multipart_Smaxes-1, 'ActFracs': multipart_actfracs}, open('MultiPart_Ntots.pkl', 'wb'))

ax12.plot(Ntots, 100*(multipart_Smaxes-1), '-r')
ax22.plot(Ntots, multipart_actfracs, '-r')

# fix the plot
ax22.set_xscale('log')
ax12.set_xlim(80, 12000)
ax22.set_xlabel(r'Number Concentration (cm$^{-3}$)', fontsize=axis_label_fontsize, labelpad=15)

ax12.text(0.04, 0.95, ' C ', fontsize=axis_label_fontsize, ha='left', va='top', transform=ax12.transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'})
ax22.text(0.04, 0.95, ' D ', fontsize=axis_label_fontsize, ha='left', va='top', transform=ax22.transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'})
ax12.legend(loc='center', bbox_to_anchor=(0.5, 1.2), fontsize=legend_fontsize, ncol=3, frameon=False)

# %% changing mean radius

mean_radii = np.logspace(-8,-6.698970004336019,10) # m
multipart_actfracs=np.zeros(len(mean_radii))
multipart_Smaxes=np.zeros(len(mean_radii))

print()
print('changing radii...')
for ii, (mean_radius) in enumerate(mean_radii):
    
    print(ii, mean_radius)
    
    diameters=np.loadtxt('pyrcel_changing_radius/radius_'+str(mean_radius*1e6)+'_diameters.txt')
    num_concs=np.loadtxt('pyrcel_changing_radius/radius_'+str(mean_radius*1e6)+'_num_concs.txt')
    pHs=np.repeat(7.0, diameters.shape)
    
    gas_data=None
    gas_names=None
    
    aero_names_temp=np.loadtxt('pyrcel_changing_radius/radius_'+str(mean_radius*1e6)+'_aero_spec_names.txt', dtype='str')
    aero_fracs_temp=np.loadtxt('pyrcel_changing_radius/radius_'+str(mean_radius*1e6)+'_aero_spec_fracs.txt')
    aero_spec_names=[]
    aero_spec_fracs=[]
    for kk in range(len(aero_fracs_temp)):
        aero_spec_names.append([aero_names_temp[kk]])
        aero_spec_fracs.append([aero_fracs_temp[kk]])
    
    temp=np.loadtxt('pyrcel_changing_radius/radius_'+str(mean_radius*1e6)+'_trajectory.txt', dtype='str')
    data_dict={}
    for kk in range(len(temp[0])):
        data_dict[temp[0, kk]]=np.array(temp[1:,kk], dtype='float64')
    pickle.dump(data_dict, open('trajectory_temp.pkl', 'wb'))
    
    les_output_file = os.getcwd()+'/trajectory_temp.pkl' #'../datasets/parcel_traces_se/parcel_traces_000000.pkl'  

    print('Reading', les_output_file)

    trajectory = simulate_les_trajectories(les_output_file=les_output_file, output_path=os.getcwd(),
            dt=5.0,diameters=diameters,N_concs=num_concs,
            pHs=pHs, accom=1., verbosity=50,
            radius_scale='log',solver='ode15s',
            species_names=aero_spec_names, mass_fractions=aero_spec_fracs,
            gas_names=gas_names, gas_data=gas_data,
            specdata_path='../../species_data/', mechanism_data_path='../../mechanisms/',
            condensation = True, collisions = False, settling = False,
            cocondensation = False, chemistry = None, freezing = False, write_every=60)
    
    multipart_Smaxes[ii]=trajectory[0].get_max_S()
    multipart_actfracs[ii]=trajectory[0].get_activated_fraction()


pyrcel_radii, pyrcel_actfracs, pyrcel_Smax = np.loadtxt('pyrcel_changing_radius/pyrcel_results.txt', 
                                                     delimiter=' ', skiprows=1, unpack=True)

ax13.plot(pyrcel_radii*1e6, 100*pyrcel_Smax, 'wo', mec='k', markersize=markersize, zorder=0)
ax23.plot(pyrcel_radii*1e6, pyrcel_actfracs, 'wo', mec='k', markersize=markersize, zorder=0)

pickle.dump({'radii': mean_radii, 'Smax': multipart_Smaxes-1, 'ActFracs': multipart_actfracs}, open('MultiPart_radii.pkl', 'wb'))

ax13.plot(mean_radii*1e6, 100*(multipart_Smaxes-1), '-r')
ax23.plot(mean_radii*1e6, multipart_actfracs, '-r')

# fix the plot
ax23.set_xscale('log')
ax13.set_xlim(0.008, 1.2)
ax23.set_xlabel(r'Number Mode Radius ($\mu$m)', fontsize=axis_label_fontsize, labelpad=15)

ax13.text(0.04, 0.95, ' E ', fontsize=axis_label_fontsize, ha='left', va='top', transform=ax13.transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'})
ax23.text(0.04, 0.95, ' F ', fontsize=axis_label_fontsize, ha='left', va='top', transform=ax23.transAxes, bbox={'facecolor': 'w', 'edgecolor': 'k'})

fig.savefig('TEST.png', bbox_inches='tight', dpi=200)
plt.show()

# %%
# delete all the modules that got moved to UNIT_TESTS/condensation 
# directory

for file in files1:
    os.remove(file)
    
for file in files2:
    os.remove(file)
    
for directory in directories:
    directory = directory.replace('.', '')
    directory = directory.replace('/', '')
    directory = directory.replace('multipart', '')
    shutil.rmtree(directory)

