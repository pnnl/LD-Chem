#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 17:46:47 2025

@author: beel083
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt




# species='OHrad'
# traj=pickle.load(open('/Users/beel083/Library/CloudStorage/OneDrive-PNNL/Desktop/multipart_archived-main/TEST/trajectory_0/trajectory_000458.pkl', 'rb'))
# idx=np.where(traj['particle species']==species)[0][0]
# plt.plot(traj['times']/60, traj['particles'][:,0,idx], '-r')
# plt.show()

# IEPOX SOA products
species=['tetrol', 'tetrol_olig', 'IEPOX_OS']
traj=pickle.load(open('/Users/beel083/Library/CloudStorage/OneDrive-PNNL/Desktop/multipart_archived-main/TEST_100p/trajectory_0/trajectory_000458.pkl', 'rb'))
total_SO4_products = np.zeros((len(traj['times']), len(traj['particles'][0,:,0])))
for ii in species:
    idx = np.where(traj['particle species']==ii)[0][0]
    total_SO4_products += traj['particles'][:,:,idx]

particle=23
plt.plot(traj['times']/60, total_SO4_products[:,particle], '-r')
idx = np.where(traj['particle species']=='IEPOX_OH_SOA')[0][0]
plt.plot(traj['times']/60, traj['particles'][:,particle,idx], '-b')
plt.yscale('log')
plt.show()

fig, ax1 = plt.subplots(1,1)
ax2=plt.twinx()
ax1.plot(traj['times']/60, traj['particles'][:,particle,idx]/(traj['particles'][:,particle,idx]+total_SO4_products[:,particle]), '-k')
ax2.plot(traj['times']/60, traj['S'], '-b')
ax2.set_ylim(1,)


# pH and pOH plot
# traj=pickle.load(open('/Users/beel083/Library/CloudStorage/OneDrive-PNNL/Desktop/multipart_archived-main/TEST/trajectory_0/trajectory_000458.pkl', 'rb'))
# for ii in range(traj['particles'].shape[1]):
#     idx=np.where(traj['particle species']=='H2O')[0][0]
#     water_volume=1000*(traj['particles'][:,ii,idx]/1000.0) # L
#     idx=np.where(traj['particle species']=='H+')[0][0]
#     Hplus_moles=traj['particles'][:,ii,idx]/1e-3 # moles
#     idx=np.where(traj['particle species']=='OHrad')[0][0]
#     OHrad_moles=traj['particles'][:,ii,idx]/17e-3 # moles
#     plt.plot(traj['times']/60, -1.0*np.log10(Hplus_moles/water_volume), '-r')
#     plt.plot(traj['times']/60, -1.0*np.log10(OHrad_moles/water_volume), '-b')


    
#     plt.plot(traj['times']/60, traj['particles'][:,ii,idx], '-r')