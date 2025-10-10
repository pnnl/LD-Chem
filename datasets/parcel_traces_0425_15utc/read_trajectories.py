#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 09:30:25 2022

@author: fier887
"""

import numpy as np
import os, sys
import pickle
import matplotlib.pyplot as plt

working_dir = '/Users/beel083/Library/CloudStorage/OneDrive-PNNL/Desktop/multipart_archived-main/datasets/parcel_traces_0425_15utc/'
start_time = 15
read_dir = working_dir + 'trajectories_' + str(start_time) + 'utc/'
filenames = os.listdir(read_dir)
  
for jj, (file) in enumerate(filenames):
    print(jj)
    data = np.loadtxt(read_dir+file)    
    dict_of_things = {}
    dict_of_things['t'] = data[:,1]*3600
    dict_of_things['x'] = data[:,2]
    dict_of_things['y'] = data[:,3]
    dict_of_things['z'] = data[:,4]
    dict_of_things['P'] = data[:,6]*100 # Pa
    dict_of_things['T'] = data[:,7]
    dict_of_things['w'] = data[:,9]
    dict_of_things['qvapor'] = data[:,10] # g/kg
    dict_of_things['qcloud'] = data[:,11] # g/kg
    qvapor=data[:,10]
    temp=data[:,7]-273.15 # degrees C
    pres=data[:,6]*100 # Pa
    es = 611.2 * np.exp(17.67 * temp / (temp + 243.5)) # Pa
    qs = 622*es/(pres-(1.-0.622)*es)
    traj_rh=100.0*(qvapor/qs) # %
    dict_of_things['s'] = traj_rh/100
 
    # FROM JEROME'S OUTPUT
    # # = 0 
    # time (utc) = 1 
    # long = 2 
    # lat = 3 
    # altitude (agl) = 4 
    # altitude (msl) = 5 
    # pressure (mb) = 6
    # temperature (K) = 7
    # RH (%) = 8
    # w (m/s) = 9
    # water vapor mixing ratio (g/kg) = 10 
    # cloud mixing ratio (g/kg) = 11
    
    output_filename = working_dir + 'parcel_traces_' + str(jj).zfill(6) + '.pkl'
    pickle.dump(dict_of_things, open(output_filename, 'wb'))

