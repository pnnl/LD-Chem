#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:45:10 2024

@author: beel083
"""
# %% 
# copy the necessary modules to the UNIT_TESTS directory
# probably need a different way to do this but I don't 
# want to mess with sys.path

import shutil, os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.integrate import trapz
from numba.typed import Dict
from numba import types
import matplotlib.font_manager as font_manager

files1 = ['particles.py', 'constants.py', 'scenario.py', 'aerosol_species.py',
         'utilities.py', 'systems.py', 'driver.py', 'visualization.py', 
         'TraceGases.py', 'Reactions.py', 'SPLAT_initialization.py', 
         'write_files.py']

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


from UnitTests_driver import simulate_IEPOX_chemistry
# from SPLAT_initialization import Nmodal_lognormal
from processes import water_uptake
from particles import make_particle, ParticlePopulation
from scenario import get_aero_spec_fracs
import constants as c

def dry_SizeDist(Dwets, species, S0, T0, P0, pH0):
    Ddrys = []
    for Dwet in Dwets:
        f = lambda d: get_wet_diameter(d, species, S0, T0, P0, pH0) - Dwet
        d_dry = fsolve(f, 0.5*Dwet)[0]
        Ddrys.append(d_dry)
    return np.array((Ddrys))

def get_wet_diameter(Dp, species, S0, T0, P0, pH0):
    aero_spec_names, aero_spec_fracs = get_aero_spec_fracs(molecule_names=[species], molecule_mass_fracs=np.array([1.]),specdata_path='species_data/')
    
    if 'H+' not in aero_spec_names:
        aero_spec_names.append('H+')
        aero_spec_fracs=np.append(aero_spec_fracs, 0.0)
    if 'OH-' not in aero_spec_names:
        aero_spec_names.append('OH-')
        aero_spec_fracs=np.append(aero_spec_fracs, 0.0)
    
    OneParticle = make_particle(Dp, aero_spec_names, aero_spec_fracs, specdata_path='species_data/')
    population0 = ParticlePopulation(particles=[OneParticle], num_concs=[1.0], ids=[0])
    population0 = water_uptake.equilibrate_water(population0, S0, T0, P0, pH0)
    return population0.particles[0].get_Dwet()


# %% do the the AS runs

mu_star = 1.975306581501756

# AS
Npart = 30
S0 = 0.51
T0 = 297
P0 = 101325
pH0 = 0.35

# fit the measured initial AS size distribution
published_data = pd.read_excel('published_data.xls', sheet_name='SizeDists_AS')               
measured_Ddrys = dry_SizeDist(np.array((published_data['Dp (t=0)']))*1e-9, 'AS', S0, T0, P0, pH0)

Ddrys = np.array((measured_Ddrys))
Ns = np.array((published_data['N (t=0)']*100**3))
idx = np.where(Ns<0)
Ns[idx[0]]=0

# 7200.0


# simulate_IEPOX_chemistry(mu_star,
#         t_end=7200.0, dt=5.0, updraft_velocity=0.0,
#         Ddry=Ddrys, Ntot=Ns, Npart=len(Ddrys),
#         S0=S0, P0=P0, T0=T0,pH0=pH0,
#         accom=1., verbosity=50,
#         species_names=['AS'], mass_fractions=np.array([1.0]),
#         gas_names=['IEPOX'], gas_conc=[500.0],
#         radius_scale='lin',solver='ode15s',
#         specdata_path='species_data/', mechanism_data_path='mechanisms/',
#         condensation = True, 
#         collisions = False, settling = False, gas_chemistry=False, entrainment=False,
#         cocondensation = True, aq_chemistry = ['IEPOX'], freezing = False,
#         relaxation_time=None, output_path='AS_runs',
#         write_every=30.0) # kg/m^3/s




#%% make the plots

axis_label_fontsize=12
axis_tick_fontsize=11
legend_fontsize=10
markersize=7
fontname = 'Helvetica'
font = font_manager.FontProperties(family=fontname, size=legend_fontsize)

MeanDp_fig, (sd,nmean,comp) = plt.subplots(1, 3, figsize=(3.0*6.4, 1.0*4.8), constrained_layout=True)
sd.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
sd.tick_params(which="major", axis="both", length=6)
sd.tick_params(which="minor", axis="both", length=4)
sd.grid(which='major', color='grey', alpha=0.4, linewidth=1)
nmean.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
nmean.tick_params(which="major", axis="both", length=6)
nmean.tick_params(which="minor", axis="both", length=4)
nmean.grid(which='major', axis='y', color='grey', alpha=0.4, linewidth=1)
comp.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
comp.tick_params(which="major", axis="both", length=6)
comp.tick_params(which="minor", axis="both", length=4)
comp.grid(which='major', axis='y', color='grey', alpha=0.4, linewidth=1)

#Gamma_fig, (IEPOX,resistors,pH) = plt.subplots(3, 1, figsize=(1.0*6.4, 3.0*4.8), constrained_layout=True, sharex=True)
#IEPOX.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
#IEPOX.tick_params(which="major", axis="both", length=6)
#IEPOX.tick_params(which="minor", axis="both", length=4)
##IEPOX.grid(which='major', color='grey', alpha=0.4, linewidth=1)
#resistors.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
#resistors.tick_params(which="major", axis="both", length=6)
#resistors.tick_params(which="minor", axis="both", length=4)
##resistors.grid(which='major', color='grey', alpha=0.4, linewidth=1)
#pH.tick_params(axis='both', which="both",labelsize=axis_tick_fontsize, pad=8, width=1)
#pH.tick_params(which="major", axis="both", length=6)
#pH.tick_params(which="minor", axis="both", length=4)
#pH.grid(which='major', color='grey', alpha=0.4, linewidth=1)


#%% do the 3 panel plot

# update the size distribution plots
trajectory = pickle.load(open('AS_runs/trajectory.pkl', 'rb'))
model_Dps = trajectory['particles'][:,:,np.where(trajectory['particle species']=='Dwet')[0][0]]
model_Ns = trajectory['particles'][:,:,np.where(trajectory['particle species']=='num conc')[0][0]]

published_data = pd.read_excel('published_data.xls', sheet_name='SizeDists_AS') 
sd.plot(published_data['Dp (t=0)'], published_data['N (t=0)']/np.max(published_data['N (t=0)']), '-', color='grey', label='t = 0 min (measured)')
sd.plot(published_data['Dp (t=120)'], published_data['N (t=120)']/np.max(published_data['N (t=120)']), '-', color='r', label='t = 120 min (measured)')

# sd.plot(model_Dps[-1]*1e9, model_Ns[-1]/100**3, 'ro', label = 't = 120 min (modeled)')
sd.plot(model_Dps[-1]*1e9, model_Ns[-1]/np.max(model_Ns[-1]), 'ro', label = 't = 120 min (modeled)')
sd.set_xscale('log')
sd.set_ylabel(r'dN/dlogdp (cm$^{-3}$)', font=fontname, fontsize=axis_label_fontsize, labelpad=15)
sd.set_ylabel(r'dN/dlogdp (cm$^{-3}$)', font=fontname, fontsize=axis_label_fontsize, labelpad=15)
sd.set_xlabel('Diameter (nm)', font=fontname, fontsize=axis_label_fontsize, labelpad=15)
sd.set_xlim(10, 1000)
sd.set_ylim(0,1.1)
sd.legend(loc='center', ncol=2, bbox_to_anchor=(0.5, 1.1), frameon=False, prop=font)
sd.text(-0.2, 1.15, 'A', transform=sd.transAxes, font=fontname, fontsize=1.5*axis_label_fontsize)
    
# tetrol_mass = 0
# tetrol_olig_mass = 0
# IEPOX_OS_mass = 0
# total_mass = 0
# Ns = []
# pHs = []
# for ii,(particle, num_conc) in enumerate(zip(AS_trajectory_ensemble[0].parcel_states[-1].particle_population.particles, AS_trajectory_ensemble[0].parcel_states[-1].particle_population.num_concs)):
#     tetrol_mass+=particle.masses[particle.get_species_idx('tetrol')]*num_conc
#     tetrol_olig_mass+=particle.masses[particle.get_species_idx('tetrol_olig')]*num_conc
#     IEPOX_OS_mass+=particle.masses[particle.get_species_idx('IEPOX_OS')]*num_conc
#     total_mass+=(particle.masses[particle.get_species_idx('tetrol')]+particle.masses[particle.get_species_idx('tetrol_olig')]+particle.masses[particle.get_species_idx('IEPOX_OS')])*num_conc    
#     pHs.append(particle.get_pH())
#     Ns.append(num_conc)
# print()
# print('AS runs: avg pH =', np.average(pHs, weights=Ns))
# print()  

# number_mean_diameters = np.zeros(len(AS_trajectory_ensemble[0].parcel_states))
# model_time = np.zeros(len(AS_trajectory_ensemble[0].parcel_states))
# for ii in range(len(AS_trajectory_ensemble[0].parcel_states)):
#     model_time[ii]=AS_trajectory_ensemble[0].ts[ii]
#     temp_Dps = np.zeros(len(AS_trajectory_ensemble[0].parcel_states[ii].particle_population.particles))
#     temp_Ns = np.zeros(len(AS_trajectory_ensemble[0].parcel_states[ii].particle_population.particles))
#     for jj,(particle, num_conc) in enumerate(zip(AS_trajectory_ensemble[0].parcel_states[ii].particle_population.particles, AS_trajectory_ensemble[0].parcel_states[ii].particle_population.num_concs)):
#         temp_Dps[jj]=particle.get_Dwet()
#         temp_Ns[jj]=num_conc
#     number_mean_diameters[ii]=np.average(temp_Dps, weights=temp_Ns)
 

number_mean_diameters = np.average(model_Dps, weights=model_Ns, axis=1)  
published_data = pd.read_excel('published_data.xls', sheet_name='number_mean_AS') 
nmean.plot(published_data['time'], published_data['Dp'], 'ko', label='measured')
nmean.plot(trajectory['times']/60, number_mean_diameters*1e9, '-r', label='modeled')
nmean.set_xlim(-10, 130)
nmean.set_xticks(np.arange(0, 140, 20))
nmean.set_ylim(80, 140)
nmean.set_ylabel('Number Mean Diameter (nm)', font=fontname, fontsize=axis_label_fontsize, labelpad=15)
nmean.set_xlabel('Time (minutes)', font=fontname, fontsize=axis_label_fontsize, labelpad=15)
nmean.legend(loc='center', ncol=2, bbox_to_anchor=(0.5, 1.1), frameon=False, prop=font)
nmean.text(-0.16, 1.15, 'B', transform=nmean.transAxes, font=fontname, fontsize=1.5*axis_label_fontsize)
    
comp.bar([0], [0.5], bottom=[0], color='k', edgecolor='k', label='non-volatile')
comp.bar([0], [0.5], bottom=[0.5], color='w', edgecolor='k', label='semi-volatile')
comp.text(0, 0.25, '50%', ha='center', va='center', fontsize=legend_fontsize, color='w')
comp.text(0, 0.75, '50%', ha='center', va='center', fontsize=legend_fontsize, color='k')
comp.bar([-10], [0.25], color='none', edgecolor='none', label=' ')

IEPOX_OS_mass = np.sum(trajectory['particles'][-1,:,np.where(trajectory['particle species']=='IEPOX_OS')[0][0]]*trajectory['particles'][-1,:,np.where(trajectory['particle species']=='num conc')[0][0]])
tetrol_mass = np.sum(trajectory['particles'][-1,:,np.where(trajectory['particle species']=='tetrol')[0][0]]*trajectory['particles'][-1,:,np.where(trajectory['particle species']=='num conc')[0][0]])
olig_mass = np.sum(trajectory['particles'][-1,:,np.where(trajectory['particle species']=='tetrol_olig')[0][0]]*trajectory['particles'][-1,:,np.where(trajectory['particle species']=='num conc')[0][0]])
total_mass = IEPOX_OS_mass+tetrol_mass+olig_mass

bottom = 0
comp.bar([1], [IEPOX_OS_mass/total_mass], bottom=[bottom], color='b', edgecolor='k', label='IEPOX OS')
comp.text(1, bottom + 0.5*(IEPOX_OS_mass/total_mass), str(int(100*IEPOX_OS_mass/total_mass))+'%', ha='center', va='center', fontsize=legend_fontsize, color='w')
bottom += IEPOX_OS_mass/total_mass
comp.bar([1], [olig_mass/total_mass], bottom=[bottom], color='darkorange', edgecolor='k', label='tetrol oligomer')
comp.text(1.6, bottom + 0.5*(olig_mass/total_mass), str(int(100*olig_mass/total_mass))+'%', ha='center', va='center', fontsize=legend_fontsize, color='k')
bottom += olig_mass/total_mass
comp.bar([1], [tetrol_mass/total_mass], bottom=[bottom], color='w', edgecolor='k', label='tetrol')
comp.text(1, bottom + 0.5*(tetrol_mass/total_mass), str(100-int(100*IEPOX_OS_mass/total_mass)-int(100*olig_mass/total_mass))+'%', ha='center', va='center', fontsize=legend_fontsize, color='k')
comp.set_ylabel('SOA Mass Fraction', font=fontname, fontsize=axis_label_fontsize, labelpad=15)
comp.set_xticks([0, 1])
comp.set_xticklabels(['measured', 'modeled'])
comp.set_xlim(-0.5, 1.5)
comp.legend(loc='center', ncol=2, bbox_to_anchor=(0.5, 1.15), frameon=False, prop=font)
comp.text(-0.16, 1.15, 'C', transform=comp.transAxes, font=fontname, fontsize=1.5*axis_label_fontsize)
    
MeanDp_fig.savefig('SizeDists.png', dpi=200, bbox_inches='tight')
plt.show()


# SO4_idx=np.where(trajectory['particle species']=='SO4')[0][0]
# HSO4_idx=np.where(trajectory['particle species']=='HSO4')[0][0]
# H2SO4_idx=np.where(trajectory['particle species']=='H2SO4')[0][0]
# NH4_idx=np.where(trajectory['particle species']=='H2SO4')[0][0]
# print(np.sum(trajectory['particles'][0,:,SO4_idx]+trajectory['particles'][0,:,NH4_idx]))
# print(np.sum(trajectory['particles'][-1,:,SO4_idx]+trajectory['particles'][-1,:,NH4_idx]))
# print(model_Dps[-1]-model_Dps[0])

# plt.plot(trajectory['times'], trajectory['particles'][:,:,NH4_idx],'-b')
# plt.plot(trajectory['times'], trajectory['particles'][:,:,SO4_idx],'-g')
# plt.plot(trajectory['times'], trajectory['particles'][:,:,np.where(trajectory['particle species']=='H2SO4')[0][0]],'-r')

# plt.yscale('log')
# plt.show()


# STOP HERE

'''
# %% do the ABS runs

mu_star = 0.45161100602630655

# ABS
S0 = 0.58
T0 = 297
P0 = 101325
pH0 = -0.9

# fit the measured initial AS size distribution
published_data = pd.read_excel('published_data.xls', sheet_name='SizeDists_ABS')               
measured_Ddrys = dry_SizeDist(np.array((published_data['Dp (t=0)']))*1e-9, 'ABS', S0, T0, P0, pH0)
idx = np.where(abs(measured_Ddrys) > 0)
measured_Ddrys = measured_Ddrys[idx[0]]

Ddrys = np.array((measured_Ddrys))
Ns = np.array((published_data['N (t=0)']*100**3))
idx = np.where(Ns<0)
Ns[idx[0]]=0

ABS_trajectory_ensemble = simulate_IEPOX_chemistry(mu_star,
        t_end=7200.0, dt=30.0, updraft_velocity=0.0,
        Ddry=Ddrys, Ntot=Ns, Npart=len(Ddrys),
        S0=S0, P0=P0, T0=T0,pH0=pH0,
        accom=1., verbosity=50,
        species_names=['ABS'], mass_fractions=np.array([1.0]),
        gas_names=['IEPOX'], gas_conc=[500.0],
        radius_scale='lin',solver='ode15s',
        specdata_path='species_data/', mechanism_data_path='mechanisms/',
        condensation = True, 
        collisions = False, settling = False,
        cocondensation = True, chemistry = ['IEPOX'], freezing = False) # kg/m^3/s
'''




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

