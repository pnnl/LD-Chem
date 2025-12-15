#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# example FLEXPART
'''
LES_traj = pickle.load(open('separate_tools/datasets/parcel_traces_0425_15utc/parcel_traces_000000.pkl', 'rb'))

print(LES_traj.keys())
output = {}

output['x'] = LES_traj['x']
output['y'] = LES_traj['y']
output['z'] = LES_traj['z']
output['t'] = LES_traj['t']-np.min(LES_traj['t'])
output['s'] = LES_traj['s']
output['T'] = LES_traj['T']
output['P'] = LES_traj['P']

gas_data = pickle.load(open('TEST_TRAJECTORY/trajectory_0/gas_data', 'rb'))
output['gas'] = {}
for kk in gas_data.keys():
    output['gas'][kk] = np.zeros(len(LES_traj['t']))
    for ii in range(len(LES_traj['t'])):
        if LES_traj['z'][ii] < np.min(gas_data[kk]['alt']):
            f = lambda x, a, b: a*x**b
            params, covariance = opt.curve_fit(f, gas_data[kk]['alt'][:2], gas_data[kk]['ppb'][:2], p0=[1, 0.1])
            output['gas'][kk][ii] = f(LES_traj['z'][ii], params[0], params[1])
        else:
            output['gas'][kk][ii] = np.interp(LES_traj['z'][ii], xp=gas_data[kk]['alt'], fp=gas_data[kk]['ppb'])


pickle.dump(output, open('datasets/example_trajectory_000000.pkl', 'wb'))
'''

# example PiChamber
ts,x,y,z,u,v,w,T,Qv,S = np.loadtxt('/Users/beel083/Library/CloudStorage/OneDrive-PNNL/Desktop/multipart_archived-main/datasets/PiChamber_NaCl_100cc_9K_50nm_2p5rate_trajectories/000000.txt', unpack=True)
output = {}
output['x'] = x
output['y'] = y
output['z'] = z
output['t'] = ts
output['s'] = S+1.0
output['T'] = T
output['P'] = np.repeat(101325, len(T))
output['gas'] = None
pickle.dump(output, open('../examples/example_datasets/example_pichamber_trajectory.pkl', 'wb'))


# x0=LES_data['x'][0],y0=LES_data['y'][0],z0=LES_data['z'][0],u0=None,
# v0=None,w0=None,
# S0=None, T0=None, P0=None,
# u_data=None,
# v_data=None,
# w_data=None,
# t_data=ts,
# x_data=LES_data['x'],
# y_data=LES_data['y'],
# z_data=LES_data['z'],
# S_data=LES_data['s'],
# P_data=LES_data['P'],
# T_data=LES_data['T'],
# population0=aerosol_population, gas0=TraceGas_population)
