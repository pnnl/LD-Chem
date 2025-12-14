#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 07:51:21 2025

@author: beel083
"""
import pandas as pd
import numpy as np
from scenario import get_aero_spec_fracs
from particles import make_particle, ParticlePopulation
from processes import water_uptake
from scipy.optimize import fsolve
from UnitTests_driver import simulate_IEPOX_chemistry
import pickle
from scipy.integrate import trapz
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# AS
Npart = 30
S0 = 0.51
T0 = 297
P0 = 101325
pH0 = 1.5

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

def dry_SizeDist(Dwets, species, S0, T0, P0, pH0):
    Ddrys = []
    for Dwet in Dwets:
        f = lambda d: get_wet_diameter(d, species, S0, T0, P0, pH0) - Dwet
        d_dry = fsolve(f, 0.5*Dwet)[0]
        Ddrys.append(d_dry)
    return np.array((Ddrys))

def optimize_friction_velocity(ts, mu):
    
    # fit the measured initial AS size distribution
    published_data = pd.read_excel('published_data.xls', sheet_name='SizeDists_AS')               
    measured_Ddrys = dry_SizeDist(np.array((published_data['Dp (t=0)']))*1e-9, 'AS', S0, T0, P0, pH0)
    Ddrys = np.array((measured_Ddrys))
    Ns = np.array((published_data['N (t=0)']*100**3))
    idx = np.where(Ns<0)
    Ns[idx[0]]=0
    
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

    simulate_IEPOX_chemistry(mu,
            t_end=7200.0, dt=30.0, updraft_velocity=0.0,
            Ddry=Ddrys, Ntot=Ns,
            S0=S0, P0=P0, T0=T0,pH0=pH0,
            accom=1., verbosity=50,
            species_names=['AS'], mass_fractions=np.array([1.]),
            gas_names=['IEPOX'], gas_conc=[1.0],
            radius_scale='lin',solver='CVODE',
            specdata_path='species_data/', mechanism_data_path='mechanisms/',
            condensation = True, 
            collisions = False, settling = False, gas_chemistry=False, entrainment=False,
            cocondensation = True, aq_chemistry = None, freezing = False,
            relaxation_time=None, output_path='mu_fitting',
            write_every=30.0)

    trajectory = pickle.load(open('mu_fitting/trajectory.pkl', 'rb'))
    Ns = trajectory['particles'][:,:,np.where(trajectory['particle species']=='num conc')[0][0]]
    Dps = trajectory['particles'][:,:,np.where(trajectory['particle species']=='Dwet')[0][0]]
    model_Ntot=trapz(Ns/100**3, x=np.log10(Dps), axis=1) # 1/cm^3
        
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
        
    output_Ntot = np.zeros(len(ts))
    for ii in range(len(ts)):
        output_Ntot[ii]=np.interp(ts[ii], xp=trajectory['times'], fp=model_Ntot)
        
    return output_Ntot*1e3
    



published_data = pd.read_excel('published_data.xls', sheet_name='wall_losses_AS')      
p0 = 0.7939091250408041
pars, cov = curve_fit(optimize_friction_velocity, xdata=published_data['minutes'], ydata=published_data['Ntot'], p0=p0)
mu_star = pars[0]

plt.plot(published_data['minutes'], published_data['Ntot'], 'ro')
ts = np.linspace(0, 120, 1000)
Ntot = optimize_friction_velocity(ts, pars[0])
plt.plot(ts, Ntot, '-k')
plt.show()

print('mu =', mu_star)

# %% ABS runs
'''
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
'''