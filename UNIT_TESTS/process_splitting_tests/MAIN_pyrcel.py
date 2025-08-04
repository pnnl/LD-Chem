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

import shutil, os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyrcel as pm
import warnings
from pyrcel import binned_activation
warnings.simplefilter('ignore')

P0 = 100000. # Pressure, Pa
T0 = 279.    # Temperature, K
S0 = -0.2   # Supersaturation, 1-RH
Nbins = 50
accom = 0.1
dt = 1.0
output_timestep=1.0 # s 

# %% changing updraft velocity

Vs = np.logspace(-1.0, 1.0, 10) # 0.1 - 5.0 m/s
smaxes, act_fracs = np.zeros(len(Vs)), np.zeros(len(Vs))
parcel_trajectory={}
print()

for ii, (V) in enumerate(Vs):
    
    print(ii, V)
    
    # Initialize the model
    aer =  pm.AerosolSpecies('ammonium sulfate',
                              pm.Lognorm(mu=0.05, sigma=2.0, N=1000.),
                              kappa=0.65, bins=Nbins)
    model = pm.ParcelModel([aer,], V, T0, S0, P0, accom=accom, console=False)
    par_out, aer_out = model.run(1000./V, dt, solver='cvode',
                                 output='dataframes', terminate=False)

    # Extract the supersaturation/activation details from the model
    # output
    S_max = par_out['S'].max()
    time_at_Smax = par_out['S'].argmax()
    wet_sizes_at_Smax = []
    for kk in aer_out['ammonium sulfate'].keys():
        wet_sizes_at_Smax.append(np.array(aer_out['ammonium sulfate'][kk])[time_at_Smax])
    wet_sizes_at_Smax = np.array(wet_sizes_at_Smax)
    act_fracs[ii], _, _, _ = binned_activation(S_max, T0, wet_sizes_at_Smax, aer)
    smaxes[ii]=S_max
    
    
    didx = int(output_timestep/dt)
    parcel_trajectory={}
    parcel_trajectory['t']=np.array(par_out['z'][::didx])/V
    parcel_trajectory['x']=np.zeros(len(par_out['z'][::didx]))
    parcel_trajectory['y']=np.zeros(len(par_out['z'][::didx]))
    parcel_trajectory['z']=np.array(par_out['z'][::didx])
    parcel_trajectory['P']=np.array(par_out['P'][::didx])
    parcel_trajectory['T']=np.array(par_out['T'][::didx])
    parcel_trajectory['w']=np.zeros(len(par_out['z'][::didx]))
    parcel_trajectory['qvapor']=np.zeros(len(par_out['z'][::didx]))
    parcel_trajectory['qcloud']=np.zeros(len(par_out['z'][::didx]))
    parcel_trajectory['s']=np.array(par_out['S'][::didx])+1.0
    
    file=open('pyrcel_changing_velocity/V_'+str(V)+'_trajectory.txt', 'w')
    for kk in parcel_trajectory.keys():
        file.write(kk)
        file.write(' ')
    file.write('\n')
    for jj in range(len(parcel_trajectory['t'])):
        for kk in parcel_trajectory.keys():
            file.write(str(parcel_trajectory[kk][jj]))
            file.write(' ')
        file.write('\n')
    file.close()
    
    file=open('pyrcel_changing_velocity/V_'+str(V)+'_diameters.txt', 'w')
    for jj in range(len(np.array(2.0*aer.r_drys))):
        file.write(str(np.array(2.0*aer.r_drys)[jj]))
        file.write('\n')
    file.close()
    
    file=open('pyrcel_changing_velocity/V_'+str(V)+'_num_concs.txt', 'w')
    Ns=np.array(aer.Nis)*100**3
    for jj in range(len(Ns)):
        file.write(str(Ns[jj]))
        file.write('\n')
    file.close()
    
    file=open('pyrcel_changing_velocity/V_'+str(V)+'_aero_spec_names.txt', 'w')
    spec_names=len(Ns)*['AS']
    for jj in range(len(spec_names)):
        file.write(str(spec_names[jj]))
        file.write('\n')
    file.close()
    
    file=open('pyrcel_changing_velocity/V_'+str(V)+'_aero_spec_fracs.txt', 'w')
    spec_fracs=len(Ns)*[1.0]
    for jj in range(len(spec_fracs)):
        file.write(str(spec_fracs[jj]))
        file.write('\n')
    file.close()
    
plt.plot(Vs, smaxes*100, 'wo', mec='k')
plt.plot(Vs, act_fracs, 'wo', mec='k')
plt.show()

file=open('pyrcel_changing_velocity/pyrcel_results.txt', 'w')
file.write('V activated_fraction Smax')
file.write('\n')
for jj in range(len(smaxes)):
    file.write(str(Vs[jj]))
    file.write(' ')
    file.write(str(act_fracs[jj]))
    file.write(' ')
    file.write(str(smaxes[jj]))
    file.write('\n')
file.close()


# %% changing Ntot

Ntots = np.logspace(2, 4, 10) # 1/cm^3
V = 0.5
smaxes, act_fracs = np.zeros(len(Ntots)), np.zeros(len(Ntots))
print()

for ii, (Ntot) in enumerate(Ntots):
    
    print(ii, Ntot)
    
    # Initialize the model
    aer =  pm.AerosolSpecies('ammonium sulfate',
                              pm.Lognorm(mu=0.05, sigma=2.0, N=Ntot),
                              kappa=0.65, bins=Nbins)
    model = pm.ParcelModel([aer,], V, T0, S0, P0, accom=accom, console=False)
    par_out, aer_out = model.run(1500./V, dt, solver='cvode',
                                  output='dataframes', terminate=False)

    # Extract the supersaturation/activation details from the model
    # output
    S_max = par_out['S'].max()
    time_at_Smax = par_out['S'].argmax()
    wet_sizes_at_Smax = []
    for kk in aer_out['ammonium sulfate'].keys():
        wet_sizes_at_Smax.append(np.array(aer_out['ammonium sulfate'][kk])[time_at_Smax])
    wet_sizes_at_Smax = np.array(wet_sizes_at_Smax)
    act_fracs[ii], _, _, _ = binned_activation(S_max, T0, wet_sizes_at_Smax, aer)
    smaxes[ii]=S_max
    
    didx = int(output_timestep/dt)
    parcel_trajectory={}
    parcel_trajectory['t']=np.array(par_out['z'][::didx])/V
    parcel_trajectory['x']=np.zeros(len(par_out['z'][::didx]))
    parcel_trajectory['y']=np.zeros(len(par_out['z'][::didx]))
    parcel_trajectory['z']=np.array(par_out['z'][::didx])
    parcel_trajectory['P']=np.array(par_out['P'][::didx])
    parcel_trajectory['T']=np.array(par_out['T'][::didx])
    parcel_trajectory['w']=np.zeros(len(par_out['z'][::didx]))
    parcel_trajectory['qvapor']=np.zeros(len(par_out['z'][::didx]))
    parcel_trajectory['qcloud']=np.zeros(len(par_out['z'][::didx]))
    parcel_trajectory['s']=np.array(par_out['S'][::didx])+1.0
    
    file=open('pyrcel_changing_Ntot/Ntot_'+str(Ntot)+'_trajectory.txt', 'w')
    for kk in parcel_trajectory.keys():
        file.write(kk)
        file.write(' ')
    file.write('\n')
    for jj in range(len(parcel_trajectory['t'])):
        for kk in parcel_trajectory.keys():
            file.write(str(parcel_trajectory[kk][jj]))
            file.write(' ')
        file.write('\n')
    file.close()
    
    file=open('pyrcel_changing_Ntot/Ntot_'+str(Ntot)+'_diameters.txt', 'w')
    for jj in range(len(np.array(2.0*aer.r_drys))):
        file.write(str(np.array(2.0*aer.r_drys)[jj]))
        file.write('\n')
    file.close()
    
    file=open('pyrcel_changing_Ntot/Ntot_'+str(Ntot)+'_num_concs.txt', 'w')
    Ns=np.array(aer.Nis)*100**3
    for jj in range(len(Ns)):
        file.write(str(Ns[jj]))
        file.write('\n')
    file.close()
    
    file=open('pyrcel_changing_Ntot/Ntot_'+str(Ntot)+'_aero_spec_names.txt', 'w')
    spec_names=len(Ns)*['AS']
    for jj in range(len(spec_names)):
        file.write(str(spec_names[jj]))
        file.write('\n')
    file.close()
    
    file=open('pyrcel_changing_Ntot/Ntot_'+str(Ntot)+'_aero_spec_fracs.txt', 'w')
    spec_fracs=len(Ns)*[1.0]
    for jj in range(len(spec_fracs)):
        file.write(str(spec_fracs[jj]))
        file.write('\n')
    file.close()
    
file=open('pyrcel_changing_Ntot/pyrcel_results.txt', 'w')
file.write('Ntot activated_fraction Smax')
file.write('\n')
for jj in range(len(smaxes)):
    file.write(str(Ntots[jj]))
    file.write(' ')
    file.write(str(act_fracs[jj]))
    file.write(' ')
    file.write(str(smaxes[jj]))
    file.write('\n')
file.close()


# %% changing mean radius


mean_radii = np.logspace(-8,-6.698970004336019,10) # m
Ntot = 1e9 # 1/cm^3
V = 0.5
smaxes, act_fracs = np.zeros(len(mean_radii)), np.zeros(len(mean_radii))
print()

for ii, (mean_radius) in enumerate(mean_radii):
    
    print(ii, mean_radius*1e6, Ntot/100**3)
    
    # Initialize the model
    aer =  pm.AerosolSpecies('ammonium sulfate',
                              pm.Lognorm(mu=mean_radius*1e6, sigma=2.0, N=Ntot/100**3),
                              kappa=0.65, bins=Nbins)
    model = pm.ParcelModel([aer,], V, T0, S0, P0, accom=accom, console=False)
    par_out, aer_out = model.run(1500./V, dt, solver='cvode',
                                  output='dataframes', terminate=False)

    # Extract the supersaturation/activation details from the model
    # output
    S_max = par_out['S'].max()
    time_at_Smax = par_out['S'].argmax()
    wet_sizes_at_Smax = []
    for kk in aer_out['ammonium sulfate'].keys():
        wet_sizes_at_Smax.append(np.array(aer_out['ammonium sulfate'][kk])[time_at_Smax])
    wet_sizes_at_Smax = np.array(wet_sizes_at_Smax)
    act_fracs[ii], _, _, _ = binned_activation(S_max, T0, wet_sizes_at_Smax, aer)
    smaxes[ii]=S_max
    
    didx = int(output_timestep/dt)
    parcel_trajectory={}
    parcel_trajectory['t']=np.array(par_out['z'][::didx])/V
    parcel_trajectory['x']=np.zeros(len(par_out['z'][::didx]))
    parcel_trajectory['y']=np.zeros(len(par_out['z'][::didx]))
    parcel_trajectory['z']=np.array(par_out['z'][::didx])
    parcel_trajectory['P']=np.array(par_out['P'][::didx])
    parcel_trajectory['T']=np.array(par_out['T'][::didx])
    parcel_trajectory['w']=np.zeros(len(par_out['z'][::didx]))
    parcel_trajectory['qvapor']=np.zeros(len(par_out['z'][::didx]))
    parcel_trajectory['qcloud']=np.zeros(len(par_out['z'][::didx]))
    parcel_trajectory['s']=np.array(par_out['S'][::didx])+1.0
    
    file=open('pyrcel_changing_radius/radius_'+str(mean_radius*1e6)+'_trajectory.txt', 'w')
    for kk in parcel_trajectory.keys():
        file.write(kk)
        file.write(' ')
    file.write('\n')
    for jj in range(len(parcel_trajectory['t'])):
        for kk in parcel_trajectory.keys():
            file.write(str(parcel_trajectory[kk][jj]))
            file.write(' ')
        file.write('\n')
    file.close()
    
    file=open('pyrcel_changing_radius/radius_'+str(mean_radius*1e6)+'_diameters.txt', 'w')
    for jj in range(len(np.array(2.0*aer.r_drys))):
        file.write(str(np.array(2.0*aer.r_drys)[jj]))
        file.write('\n')
    file.close()
    
    file=open('pyrcel_changing_radius/radius_'+str(mean_radius*1e6)+'_num_concs.txt', 'w')
    Ns=np.array(aer.Nis)*100**3
    for jj in range(len(Ns)):
        file.write(str(Ns[jj]))
        file.write('\n')
    file.close()
    
    file=open('pyrcel_changing_radius/radius_'+str(mean_radius*1e6)+'_aero_spec_names.txt', 'w')
    spec_names=len(Ns)*['AS']
    for jj in range(len(spec_names)):
        file.write(str(spec_names[jj]))
        file.write('\n')
    file.close()
    
    file=open('pyrcel_changing_radius/radius_'+str(mean_radius*1e6)+'_aero_spec_fracs.txt', 'w')
    spec_fracs=len(Ns)*[1.0]
    for jj in range(len(spec_fracs)):
        file.write(str(spec_fracs[jj]))
        file.write('\n')
    file.close()
    
file=open('pyrcel_changing_radius/pyrcel_results.txt', 'w')
file.write('mean_radius activated_fraction Smax')
file.write('\n')
for jj in range(len(smaxes)):
    file.write(str(mean_radii[jj]))
    file.write(' ')
    file.write(str(act_fracs[jj]))
    file.write(' ')
    file.write(str(smaxes[jj]))
    file.write('\n')
file.close()




