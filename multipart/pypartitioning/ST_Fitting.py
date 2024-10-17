# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:03:06 2024

@author: Nahin Ferdousi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.constants import R
from mpl_toolkits.mplot3d import Axes3D



data_path = r'C:\Users\Nahin Ferdousi\OneDrive\Desktop\Partitioning Model Data\ST_Data.csv'

molarity_vals = []
ST_vals = []
with open(data_path) as data_file:
  next(data_file)
  for line in data_file:
      # mass_2MGA_mg, vol_H2O_mL, Molarity_M, ST_mN_per_m, std_dev_mN_per_m = print(line.split(','))
      output_oneline = line.split(',')
      molarity_vals.append(float(output_oneline[2]))
      ST_vals.append(float(output_oneline[3]))

molarity_vals_data = np.array(molarity_vals)
molarity_vals = molarity_vals_data[1:]

ST_vals_data = np.array(ST_vals)
ST_vals = ST_vals_data[1:]
ST_data = ST_vals/1000 #N/m


def ST_function(molarities,M_thresh,ST_dilute,slope):
  # ST is in mN/m
  # molarities = mols/L
  if type(molarities) == type(1e-7) or type(molarities) == type(np.array([0.])[0]):
    is_float = True
    molarities = np.array([molarities])
  else:
    is_float = False
    
  ST_CMC = np.min(ST_data)
  CMC_indx = np.argmin(ST_data)
  
  b = ST_dilute - slope*np.log(M_thresh)
  ST =  slope*np.log(molarities) + b 
 # ST = min(ST_func, 0.040)
 
 
  M_thresh_CMC = 20
 
  if is_float:
    if ST<=M_thresh:
      ST = ST_dilute
    #elif ST>=M_thresh_CMC:
     # ST = ST_CMC
  else:
    ST[molarities<=M_thresh] = ST_dilute
    #ST[molarities>=M_thresh_CMC] = ST_CMC
  return ST

fit_output = opt.curve_fit(ST_function, molarity_vals, ST_data)
M_thresh = fit_output[0][0]

ST_dilute = fit_output[0][1]

slope = fit_output[0][2]


#print ('final equation for  slope * np.log(molarities) + (ST_dilute - slope* np.log(M_thresh))')

#print('M thresh', M_thresh)
#print('ST_dilute', ST_dilute)
#print('slope', slope)

molarity_test_xvals = np.arange(0,40, 0.001)


plt.figure(1)
plt.scatter(molarity_vals,ST_data, label = 'Tensiometer Data') #; plt.xscale('log')

plt.xlim(0, 40)

plt.plot(molarity_test_xvals,ST_function(molarity_test_xvals,M_thresh,ST_dilute,slope), linestyle = '-.', color = 'red', label = 'ST Model Fit')#; plt.xscale('log')
plt.axvline(x = 20, color = 'black', linestyle = 'dashed')
plt.text(20, 0.055, 'CMC', color='blue', fontsize=12, rotation=90)
#ST_test = ST_function(molarity_test_xvals,M_thresh,ST_dilute,slope)
#plt.scatter(molarity_test_xvals, ST_test)#; plt.xscale('log')

plt.xlabel('Organic Molarity (mol/L)')
plt.ylabel('Surface Tension (N/m)')
plt.title('Figure 1a. Measured Surface Tension vs. 2-MGA Droplet Concentration')
plt.legend()
# plt.scatter(molarity_vals,ST_function_ln(molarity_vals,lnM_thresh2,ST_dilute2,slope2)); plt.xscale('log')







