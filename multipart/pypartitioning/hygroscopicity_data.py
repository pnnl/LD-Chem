# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:51:21 2024

@author: Nahin Ferdousi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from constants_pypartition import R
from mpl_toolkits.mplot3d import Axes3D



data_path_HTDMA = r'C:\Users\Nahin Ferdousi\OneDrive\Desktop\Partitioning Model Data\HTDMA_Data.csv'

data_HTDMA = np.loadtxt(data_path_HTDMA,skiprows=1,delimiter=',',usecols=None)
D_dry = data_HTDMA[:,0]*1e-9 # in meters
D_wet = data_HTDMA[:,1]*1e-9 # in meters
total_fraction_2MGA = data_HTDMA[:,2]
RH_data = data_HTDMA[:,3]
GF_data = data_HTDMA[:,4]


print("Ddry", D_dry)
print("Dwet", D_wet)
print("mass frac 2MGA", total_fraction_2MGA)
print('RH', RH_data)
print('GF', GF_data)




data_path_CCNC = r'C:\Users\Nahin Ferdousi\OneDrive\Desktop\Partitioning Model Data\2MGA_Ddry_vs_Sc.csv'
data_CCNC = np.loadtxt(data_path_CCNC,skiprows=1,delimiter=',',usecols=None)
total_fraction_2MGA_CCN = data_CCNC[:,0]
D_dry_CCN = data_CCNC[:,1]
Sc_CCN = data_CCNC[:,2]

print("mass frac 2MGA CCN", total_fraction_2MGA_CCN)
print('Ddry CCN', D_dry_CCN ) 
print('Crit SS', Sc_CCN)


