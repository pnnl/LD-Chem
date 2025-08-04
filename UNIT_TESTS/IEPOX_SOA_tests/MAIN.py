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

import shutil, os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.integrate import trapz
from numba.typed import Dict
from numba import types

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
from SPLAT_initialization import Nmodal_lognormal
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
    OneParticle = make_particle(Dp, aero_spec_names, aero_spec_fracs, specdata_path='species_data/')
    population0 = ParticlePopulation(particles=[OneParticle], num_concs=[1.0], ids=[0])
    population0 = water_uptake.equilibrate_water(population0, S0, T0, P0, pH0)
    return population0.particles[0].get_Dwet()



# %% optimize friction velocity for the AS runs

# AS
# Npart = 30
# S0 = 0.51
# T0 = 297
# P0 = 101325
# pH0 = 1.5

# def optimize_friction_velocity(ts, mu):
    
#     print(mu)
#     # fit the measured initial AS size distribution
#     published_data = pd.read_excel('published_data.xls', sheet_name='SizeDists_AS')               
#     measured_Ddrys = dry_SizeDist(np.array((published_data['Dp (t=0)']))*1e-9, 'AS', S0, T0, P0, pH0)
#     Ddrys = np.array((measured_Ddrys))
#     Ns = np.array((published_data['N (t=0)']*100**3))
#     idx = np.where(Ns<0)
#     Ns[idx[0]]=0
    
#     trajectory_ensemble = simulate_IEPOX_chemistry(mu,
#             t_end=7200.0, dt=30.0, updraft_velocity=0.0,
#             Ddry=Ddrys, Ntot=Ns,
#             S0=S0, P0=P0, T0=T0,pH0=pH0,
#             accom=1., verbosity=50,
#             species_names=['AS'], mass_fractions=np.array([1.]),
#             gas_names=['IEPOX'], gas_conc=[1.0],
#             radius_scale='lin',solver='CVODE',
#             specdata_path='species_data/', mechanism_data_path='mechanisms/',
#             condensation = True, 
#             collisions = False, settling = False,
#             cocondensation = True, chemistry = ['IEPOX'], freezing = False) # kg/m^3/s

#     model_time = np.zeros(len(trajectory_ensemble[0].parcel_states))
#     model_Ntot = np.zeros(len(trajectory_ensemble[0].parcel_states))
#     for ii,(parcelstate) in enumerate(trajectory_ensemble[0].parcel_states):
#         particle_population=parcelstate.particle_population
#         model_time[ii]=trajectory_ensemble[0].ts[ii]/60 # minutes
#         Dps = np.zeros(len(particle_population.particles))
#         Ns = np.zeros(len(particle_population.particles))
#         for jj,(particle,num_conc) in enumerate(zip(particle_population.particles, particle_population.num_concs)):
#             Ns[jj]=num_conc # 1/m^3
#             Dps[jj]=particle.get_Dwet()
#         model_Ntot[ii]=trapz(Ns/100**3, x=np.log10(Dps)) # 1/cm^3
        
#     output_Ntot = np.zeros(len(ts))
#     for ii in range(len(ts)):
#         output_Ntot[ii]=np.interp(ts[ii], xp=model_time, fp=model_Ntot)
        
#     return output_Ntot*1e3
    
# published_data = pd.read_excel('published_data.xls', sheet_name='wall_losses_AS')      
# p0 = 0.7939091250408041
# pars, cov = curve_fit(optimize_friction_velocity, xdata=published_data['minutes'], ydata=published_data['Ntot'], p0=p0)
# mu_star = pars[0]

# plt.plot(published_data['minutes'], published_data['Ntot'], 'ro')
# ts = np.linspace(0, 120, 1000)
# Ntot = optimize_friction_velocity(ts, pars[0])
# plt.plot(ts, Ntot, '-k')
# plt.show()

# print('mu =', mu_star)


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

AS_trajectory_ensemble = simulate_IEPOX_chemistry(mu_star,
        t_end=7200.0, dt=30.0, updraft_velocity=0.0,
        Ddry=Ddrys, Ntot=Ns, Npart=len(Ddrys),
        S0=S0, P0=P0, T0=T0,pH0=pH0,
        accom=1., verbosity=50,
        species_names=['AS'], mass_fractions=np.array([1.0]),
        gas_names=['IEPOX'], gas_conc=[500.0],
        radius_scale='lin',solver='ode15s',
        specdata_path='species_data/', mechanism_data_path='mechanisms/',
        condensation = True, 
        collisions = False, settling = False,
        cocondensation = True, chemistry = ['IEPOX'], freezing = False) # kg/m^3/s




#%% make the plots

axis_label_fontsize=12
axis_tick_fontsize=11
legend_fontsize=11
markersize=7

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
model_Dps = np.zeros(len(AS_trajectory_ensemble[0].parcel_states[0].particle_population.particles))
model_Ns = np.zeros(len(AS_trajectory_ensemble[0].parcel_states[0].particle_population.particles))
for ii,(particle, num_conc) in enumerate(zip(AS_trajectory_ensemble[0].parcel_states[0].particle_population.particles, AS_trajectory_ensemble[0].parcel_states[0].particle_population.num_concs)):
    model_Dps[ii]=particle.get_Dwet()
    model_Ns[ii]=num_conc

published_data = pd.read_excel('published_data.xls', sheet_name='SizeDists_AS') 
sd.plot(published_data['Dp (t=0)'], published_data['N (t=0)'], '-', color='grey', label='t = 0 min (measured)')
sd.plot(published_data['Dp (t=120)'], published_data['N (t=120)'], '-', color='r', label='t = 120 min (measured)')

model_Dps = np.zeros(len(AS_trajectory_ensemble[0].parcel_states[-1].particle_population.particles))
model_Ns = np.zeros(len(AS_trajectory_ensemble[0].parcel_states[-1].particle_population.particles))
for ii,(particle, num_conc) in enumerate(zip(AS_trajectory_ensemble[0].parcel_states[-1].particle_population.particles, AS_trajectory_ensemble[0].parcel_states[-1].particle_population.num_concs)):
    model_Dps[ii]=particle.get_Dwet()
    model_Ns[ii]=num_conc

sd.plot(model_Dps*1e9, model_Ns/100**3, 'ro', label = 't = 120 min (modeled)')
sd.set_xscale('log')
sd.set_ylabel(r'dN/dlogdp (cm$^{-3}$)', fontsize=axis_label_fontsize, labelpad=15)
sd.set_ylabel(r'dN/dlogdp (cm$^{-3}$)', fontsize=axis_label_fontsize, labelpad=15)
sd.set_xlabel('particle diameter (nm)', fontsize=axis_label_fontsize, labelpad=15)
sd.set_xlim(10, 1000)
sd.set_ylim(0,0.035)
sd.legend(loc='center', ncol=2, bbox_to_anchor=(0.5, 1.1), frameon=False, fontsize=legend_fontsize)

    
tetrol_mass = 0
tetrol_olig_mass = 0
IEPOX_OS_mass = 0
total_mass = 0
Ns = []
pHs = []
for ii,(particle, num_conc) in enumerate(zip(AS_trajectory_ensemble[0].parcel_states[-1].particle_population.particles, AS_trajectory_ensemble[0].parcel_states[-1].particle_population.num_concs)):
    tetrol_mass+=particle.masses[particle.get_species_idx('tetrol')]*num_conc
    tetrol_olig_mass+=particle.masses[particle.get_species_idx('tetrol_olig')]*num_conc
    IEPOX_OS_mass+=particle.masses[particle.get_species_idx('IEPOX_OS')]*num_conc
    total_mass+=(particle.masses[particle.get_species_idx('tetrol')]+particle.masses[particle.get_species_idx('tetrol_olig')]+particle.masses[particle.get_species_idx('IEPOX_OS')])*num_conc    
    pHs.append(particle.get_pH())
    Ns.append(num_conc)
print()
print('AS runs: avg pH =', np.average(pHs, weights=Ns))
print()  

number_mean_diameters = np.zeros(len(AS_trajectory_ensemble[0].parcel_states))
model_time = np.zeros(len(AS_trajectory_ensemble[0].parcel_states))
for ii in range(len(AS_trajectory_ensemble[0].parcel_states)):
    model_time[ii]=AS_trajectory_ensemble[0].ts[ii]
    temp_Dps = np.zeros(len(AS_trajectory_ensemble[0].parcel_states[ii].particle_population.particles))
    temp_Ns = np.zeros(len(AS_trajectory_ensemble[0].parcel_states[ii].particle_population.particles))
    for jj,(particle, num_conc) in enumerate(zip(AS_trajectory_ensemble[0].parcel_states[ii].particle_population.particles, AS_trajectory_ensemble[0].parcel_states[ii].particle_population.num_concs)):
        temp_Dps[jj]=particle.get_Dwet()
        temp_Ns[jj]=num_conc
    number_mean_diameters[ii]=np.average(temp_Dps, weights=temp_Ns)
    
published_data = pd.read_excel('published_data.xls', sheet_name='number_mean_AS') 
nmean.plot(published_data['time'], published_data['Dp'], 'ko', label='measured')
nmean.plot(model_time/60, number_mean_diameters*1e9, '-r', label='modeled')
nmean.set_xlim(-10, 130)
nmean.set_xticks(np.arange(0, 140, 20))
nmean.set_ylim(80, 140)
nmean.set_ylabel('number mean diameter (nm)', fontsize=axis_label_fontsize, labelpad=15)
nmean.set_xlabel('time (minutes)', fontsize=axis_label_fontsize, labelpad=15)
nmean.legend(loc='center', ncol=2, bbox_to_anchor=(0.5, 1.1), frameon=False, fontsize=legend_fontsize)

comp.bar([0], [0.5], bottom=[0], color='k', edgecolor='k', label='non-volatile')
comp.bar([0], [0.5], bottom=[0.5], color='w', edgecolor='k', label='semi-volatile')
comp.text(0, 0.25, '50%', ha='center', va='center', fontsize=legend_fontsize, color='w')
comp.text(0, 0.75, '50%', ha='center', va='center', fontsize=legend_fontsize, color='k')
comp.bar([-10], [0.25], color='none', edgecolor='none', label=' ')

bottom = 0
comp.bar([1], [IEPOX_OS_mass/total_mass], bottom=[bottom], color='b', edgecolor='k', label='IEPOX OS')
comp.text(1, bottom + 0.5*(IEPOX_OS_mass/total_mass), str(int(100*IEPOX_OS_mass/total_mass))+'%', ha='center', va='center', fontsize=legend_fontsize, color='w')
bottom += IEPOX_OS_mass/total_mass
comp.bar([1], [tetrol_olig_mass/total_mass], bottom=[bottom], color='darkorange', edgecolor='k', label='tetrol oligomer')
comp.text(1.6, bottom + 0.5*(tetrol_olig_mass/total_mass), str(int(100*tetrol_olig_mass/total_mass))+'%', ha='center', va='center', fontsize=legend_fontsize, color='k')
bottom += tetrol_olig_mass/total_mass
comp.bar([1], [tetrol_mass/total_mass], bottom=[bottom], color='w', edgecolor='k', label='tetrol')
comp.text(1, bottom + 0.5*(tetrol_mass/total_mass), str(100-int(100*IEPOX_OS_mass/total_mass)-int(100*tetrol_olig_mass/total_mass))+'%', ha='center', va='center', fontsize=legend_fontsize, color='k')
comp.set_ylabel('SOA mass fraction', fontsize=axis_label_fontsize, labelpad=15)
comp.set_xticks([0, 1])
comp.set_xticklabels(['measured', 'modeled'])
comp.set_xlim(-0.5, 1.5)
comp.legend(loc='center', ncol=2, bbox_to_anchor=(0.5, 1.15), frameon=False, fontsize=legend_fontsize)

MeanDp_fig.savefig('SizeDists.png', dpi=200, bbox_inches='tight')


plt.show()

'''
# %% plot the uptake resistors

published_data = pd.read_excel('published_data.xls', sheet_name='Gammas_AS') 

IEPOX.plot(published_data['time'], published_data['IEPOX'], 'ro', label='Zhang et. al. 2023')
IEPOX.set_yscale('log')
IEPOX.set_ylabel(r'$\gamma_{IEPOX}$', fontsize=axis_label_fontsize, labelpad=10)
IEPOX.set_ylim(1e-5, 1e-4)

resistors.plot(published_data['time'], published_data['1/coat'], 'wo', mec='r', label='Zhang et. al. 2023')
resistors.plot(published_data['time'], published_data['1/aq'], 'ro', label='Zhang et. al. 2023')
resistors.set_yscale('log')
resistors.set_ylabel('uptake resisitors', fontsize=axis_label_fontsize, labelpad=10)
resistors.set_xlabel('time (minutes)', fontsize=axis_label_fontsize, labelpad=10)
resistors.set_ylim(1e-1, 1e6)

pH.plot(published_data['time'], published_data['Hplus conc'], 'wo', mec='r', label='Zhang et. al. 2023')
pH.set_yscale('log')
pH.set_ylim(1e-2, 1e1)

particle = AS_trajectory_ensemble[0].parcel_states[0].particle_population.particles[17]


model_time = np.zeros(len(AS_trajectory_ensemble[0].parcel_states))
model_Gamma_aq_inv = np.zeros(len(AS_trajectory_ensemble[0].parcel_states))
model_Gamma_org_inv = np.zeros(len(AS_trajectory_ensemble[0].parcel_states))
model_Gamma_IEPOX = np.zeros(len(AS_trajectory_ensemble[0].parcel_states))
model_pH = np.zeros(len(AS_trajectory_ensemble[0].parcel_states))

for ii in range(len(AS_trajectory_ensemble[0].parcel_states)):
    model_time[ii]=AS_trajectory_ensemble[0].ts[ii]
    particle = AS_trajectory_ensemble[0].parcel_states[ii].particle_population.particles[17]
    model_pH[ii] = particle.get_pH()
    
    # get the aqueous concentrations
    water_volume = particle.get_vol_tot()-particle.get_vol_dry()
    Caq = np.empty(0)
    aq_names = Dict.empty(key_type=types.unicode_type, value_type=types.int32)
    for jj, (species) in enumerate(particle.species):
        aq_names[species.name]=jj
        Caq=np.append(Caq, (particle.masses[particle.get_species_idx(species.name)]/species.molar_mass)/water_volume) # mol/m^3
    
    # these are needed if there is IEPOX condensation
    V_Org = 0
    for species, mass in zip(particle.species, particle.masses):
        if species.name in ['IEPOX_OS', 'tetrol', 'tetrol_olig', 'OC']:
            V_Org += mass/species.density
    V_NonOrg = particle.get_vol_dry() - V_Org
    h2o_NonOrg_radius = ((3*(water_volume+V_NonOrg))/(4.0*np.pi))**(1.0/3.0)
    inorg_radius = ((3*V_NonOrg)/(4.0*np.pi))**(1.0/3.0)
    radius = particle.get_Dwet()/2.0
    l_org = radius-h2o_NonOrg_radius # m
    
    for jj, (name) in enumerate(aq_names):
        if name == 'H2O':
            H2O_conc = 0.001*Caq[jj] # mol/L
        elif name == 'H+':
            Hplus_conc = 0.001*Caq[jj] # mol/L
        elif name == 'HSO4':
            HSO4_conc = 0.001*Caq[jj] # mol/L
        elif name == 'NH4':
            NH4_conc = 0.001*Caq[jj] # mol/L
        elif name == 'SO4':
            SO4_conc = 0.001*Caq[jj] # mol/L
    
    if 'HSO4' not in aq_names:
        HSO4_conc = 0
    if 'NH4' not in aq_names:
        NH4_conc = 0
    if 'SO4' not in aq_names:
        SO4_conc = 0
        
    kaqs = [1.8e-4, 2.62e-6, 6.2e-8, 1.91e-4]
    kaq = kaqs[0]*Hplus_conc*H2O_conc + kaqs[1]*HSO4_conc*H2O_conc + kaqs[2]*NH4_conc*H2O_conc + kaqs[3]*Hplus_conc*SO4_conc # 1/s
    
    T = AS_trajectory_ensemble[0].parcel_states[ii].T
    S = AS_trajectory_ensemble[0].parcel_states[ii].S
    V = (4.0/3.0)*np.pi*radius**3 # m^3
    
    Haq=5.5e14*(1000/101325) # mol/m^3 Pa, AS: 3.0e4*(1000/101325)
    Horg=5.5E6*(1000/101325) # mol/m^3 Pa, AS: 2.0e3*(1000/101325)
    
    Ap = 4.0*np.pi*radius**2 # m^2
    w = np.sqrt((8*c.R*T)/(np.pi*118e-3)) # thermal velocity, m/s
    Dg = (1/100**2)*1.9*np.power(118e-3, (-2/3)) # m^2/s
    Gamma_aq_inv = (4*V*c.R*T*Haq*kaq)/(Ap*w) # unitless
    Eta_org=6.92448e9*np.exp(-2.48362e1*S) # Pa*s (fit of table S3 in "Effect of the Aerosol-Phase State on Secondary Organic Aerosol Formation from the Reactive Uptake of Isoprene-Derived Epoxydiols (IEPOX)")
    Dorg=(1.380649E-23*T)/(6*np.pi*1e-10*Eta_org)
    Gamma_org_inv = ((w*l_org)/(4*c.R*T*Horg*Dorg))*(radius/inorg_radius)
    Gamma_IEPOX = np.power(((w*radius)/(4.0*Dg))+(1/0.001)+Gamma_aq_inv+Gamma_org_inv, -1.0)
    
    model_Gamma_aq_inv[ii]=Gamma_aq_inv
    model_Gamma_org_inv[ii]=Gamma_org_inv
    model_Gamma_IEPOX[ii]=Gamma_IEPOX    

for ii in range(len(model_time)):
    print(model_time[ii], model_Gamma_aq_inv[ii], model_Gamma_org_inv[ii])

IEPOX.plot(model_time/60, model_Gamma_IEPOX, '-r', label='simulated')
resistors.plot(model_time/60, model_Gamma_org_inv, '--r', label='simulated')
resistors.plot(model_time/60, model_Gamma_aq_inv, '-r', label='simulated')

pH.plot(model_time/60, 10**(-1.0*model_pH), '-r')


Gamma_fig.savefig('TEST.png', bbox_inches='tight')
plt.show()




# %% ABS runs

# ABS
S0 = 0.58
T0 = 297
P0 = 101325
pH0 = -0.9

def optimize_friction_velocity(ts, mu):
    
    print(mu)
    # fit the measured initial AS size distribution
    published_data = pd.read_excel('published_data.xls', sheet_name='SizeDists_ABS')               
    measured_Ddrys = dry_SizeDist(np.array((published_data['Dp (t=0)']))*1e-9, 'ABS', S0, T0, P0, pH0)
    idx = np.where(abs(measured_Ddrys) > 0)
    measured_Ddrys = measured_Ddrys[idx[0]]
    Ddrys = np.array((measured_Ddrys))
    Ns = np.array((published_data['N (t=0)']*100**3))
    idx = np.where(Ns<0)
    Ns[idx[0]]=0
    
    trajectory_ensemble = simulate_IEPOX_chemistry(mu,
            t_end=7200.0, dt=30.0, updraft_velocity=0.0,
            Ddry=Ddrys, Ntot=Ns,
            S0=S0, P0=P0, T0=T0,pH0=pH0,
            accom=1., verbosity=50,
            species_names=['AS'], mass_fractions=np.array([1.]),
            gas_names=['IEPOX'], gas_conc=[1.0],
            radius_scale='lin',solver='CVODE',
            specdata_path='species_data/', mechanism_data_path='mechanisms/',
            condensation = True, 
            collisions = False, settling = False,
            cocondensation = True, chemistry = ['IEPOX'], freezing = False) # kg/m^3/s

    model_time = np.zeros(len(trajectory_ensemble[0].parcel_states))
    model_Ntot = np.zeros(len(trajectory_ensemble[0].parcel_states))
    for ii,(parcelstate) in enumerate(trajectory_ensemble[0].parcel_states):
        particle_population=parcelstate.particle_population
        model_time[ii]=trajectory_ensemble[0].ts[ii]/60 # minutes
        Dps = np.zeros(len(particle_population.particles))
        Ns = np.zeros(len(particle_population.particles))
        for jj,(particle,num_conc) in enumerate(zip(particle_population.particles, particle_population.num_concs)):
            Ns[jj]=num_conc # 1/m^3
            Dps[jj]=particle.get_Dwet()
        model_Ntot[ii]=trapz(Ns/100**3, x=np.log10(Dps)) # 1/cm^3
        
    output_Ntot = np.zeros(len(ts))
    for ii in range(len(ts)):
        output_Ntot[ii]=np.interp(ts[ii], xp=model_time, fp=model_Ntot)
        
    return output_Ntot*1e3
    
published_data = pd.read_excel('published_data.xls', sheet_name='wall_losses_ABS')      
p0 = 0.7939091250408041
pars, cov = curve_fit(optimize_friction_velocity, xdata=published_data['minutes'], ydata=published_data['Ntot'], p0=p0)
mu_star = pars[0]

plt.plot(published_data['minutes'], published_data['Ntot'], 'ro')
ts = np.linspace(0, 120, 1000)
Ntot = optimize_friction_velocity(ts, pars[0])
plt.plot(ts, Ntot, '-k')
plt.show()

print('mu =', mu_star)

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

