#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:06:32 2024

@author: beel083
"""
import numpy as np
import sys
import matplotlib.pyplot as plt

def splat_setup(Npart=1, size_distribution_file=None, splat_file=None,
                splat_species=None, ams_file=None, start_time=None, 
                end_time=None, cloud_flag=0, CVI_flag=0):
    
    diameters=np.zeros(Npart)
    if not start_time or not end_time:
        print('WARNING: Need start and end time for miniSPLAT averaging!')
        sys.exit()
        
        
    Dp_lowers, Dp_uppers, N, N_error = read_FIMS(size_distribution_file, 
                                                 start_time, end_time, cloud_flag, 
                                                 CVI_flag) # diameters in nm and N in #/cm^3
   
    avg_number_fraction, number_fraction_error = splat_number_fractions(splat_file, splat_species, start_time, end_time)    
    ams_mass_fractions, measured_total_mass, measured_total_mass_error = ams_mass_fraction(ams_file, start_time, end_time)    
    
    print(ams_mass_fractions)
    
    # plt.plot(Dp_lowers, N, '-o')
    # plt.xscale('log')
    # plt.show()
    
    return diameters
    

def ams_mass_fraction(ams_file, start_time, end_time):
    
    
    # read the AMS data from file
    filename = ams_file
    raw_data = np.loadtxt(filename, delimiter = ',', dtype='str', skiprows = 42)
    for i in range(0, len(raw_data)):
        for j in range(0, len(raw_data[0])):
            raw_data[i, j] = raw_data[i, j].strip()
    ams_data = {}
    for i in range(0, len(raw_data[0])): 
        ams_data[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
    
    # find indices in the averaging time
    indices = np.where(np.logical_and(ams_data['dat_ams_utc'] >= start_time, 
                                      np.logical_and(ams_data['dat_ams_utc'] < end_time,
                                      ams_data['flag'] < 0.5)))
    
    # pull out data within the averaging time
    ams_subdata = {}
    for key in ams_data:
        ams_subdata[key] = ams_data[key][indices[0][0]]
    for i in range(1, len(indices[0])):
        for key in ams_data:
            ams_subdata[key] = np.append(ams_subdata[key], ams_data[key][indices[0][i]])
    
    # find total mass concentrations
    ams_subdata['total_mass'] = np.zeros(len(ams_subdata['dat_ams_utc']))
    ams_subdata['total_mass_error'] = np.zeros(len(ams_subdata['dat_ams_utc']))
    for i in range(0, len(ams_subdata['dat_ams_utc'])):
        for key in ['Org', 'NO3', 'SO4', 'NH4', 'Chl']:
            ams_subdata['total_mass'][i] += ams_subdata[key][i]
            ams_subdata['total_mass_error'][i] += np.sqrt(np.power(ams_subdata[key + '_err'][i], 2))    

    # convert mass concentrations to mass fractions
    mass_fractions = {}
    for key in ['Org', 'NO3', 'SO4', 'NH4', 'Chl']:
            mass_fractions[key] = ams_subdata[key]/ams_subdata['total_mass']
            mass_fractions[key+'_err'] = mass_fractions[key]*np.sqrt(np.power(ams_subdata[key+'_err']/ams_subdata[key], 2) + np.power(ams_subdata['total_mass_error']/ams_subdata['total_mass'], 2))
            
    # find average mass fractions
    output = {}
    for key in ['Org', 'NO3', 'SO4', 'NH4', 'Chl']:
        output[key] = np.nanmean(mass_fractions[key])
        output[key+'_err'] = np.nanstd(mass_fractions[key])
        
    # format output
    mass_frac = {}
    mass_frac['sulfate'] = output['SO4']
    mass_frac['sulfate_error'] = output['SO4_err']
    mass_frac['nitrate'] = output['NO3']
    mass_frac['nitrate_error'] = output['NO3_err']
    mass_frac['organics'] = output['Org']
    mass_frac['organics_error'] = output['Org_err']
    mass_frac['ammonium'] = output['NH4']
    mass_frac['ammonium_error'] = output['NH4_err']
    mass_frac['chloride'] = output['Chl']
    mass_frac['chloride_error'] = output['Chl_err']
    
    return mass_frac, np.mean(ams_subdata['total_mass']), np.std(ams_subdata['total_mass'])

def splat_number_fractions(splat_file, splat_species, start_time, end_time):
    
    # read miniSPLAT data from file
    filename = splat_file
    raw_data = np.loadtxt(filename, dtype='str')
    full_SPLAT_data = {}
    SPLAT_subdata = {}
    for i in range(0, len(raw_data[0])): 
        full_SPLAT_data[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
        SPLAT_subdata[str(raw_data[0, i])] = np.zeros(0)
    
    # pull out the miniSPLAT data corresponding to initialization times
    idx = np.where(np.logical_and(full_SPLAT_data['Time']>start_time, full_SPLAT_data['Time']<=end_time))
    for key in full_SPLAT_data.keys():
        SPLAT_subdata[key] = np.array((full_SPLAT_data[key][idx[0]]))
    
    # create dict of all the species in the miniSPLAT data
    minisplat_species = []
    for key in SPLAT_subdata.keys():
        if key != 'Time':
            minisplat_species.append(key)

    # find the full time series of number fraction for each class
    reduced_fraction = {}
    for reduced_species in splat_species.keys():
        summation = np.zeros(len(full_SPLAT_data['Time']))
        for species in splat_species[reduced_species]:
            for i in range(0, len(summation)):
                summation[i] = summation[i] + full_SPLAT_data[species][i]
        reduced_fraction[reduced_species] = summation
        
    # find the average number fraction for each class during initialization times
    avg_comp = {}
    comp_error = {}
    for reduced_species in splat_species.keys():
        summation = np.zeros(len(SPLAT_subdata['Time']))
        for species in splat_species[reduced_species]:
            for i in range(0, len(summation)):
                summation[i] = summation[i] + SPLAT_subdata[species][i]
        avg_comp[reduced_species] = np.mean(summation)
        comp_error[reduced_species] = np.std(summation)
            
    return avg_comp, comp_error

    
def read_FIMS(FIMS_file, start_time, end_time, cloud_flag, CVI_flag):
    
    # read FIMS data from file   
    filename = FIMS_file
    raw_data = np.loadtxt(filename, delimiter = ',', dtype='str', skiprows = 100)    
    for i in range(0, len(raw_data)):
        for j in range(0, len(raw_data[0])):
            raw_data[i, j] = raw_data[i, j].strip()
    FIMS_data = {}
    FIMS_data[str(raw_data[0, 0])] = np.array(raw_data[1:, 0], dtype = 'float64')
    for i in range(56, len(raw_data[0])): 
        FIMS_data[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
    FIMS_data['N_Dp'] = np.array(raw_data[1:, 1:56], dtype = 'float64')
    
    # get the size distribution bins
    raw_data = np.loadtxt(filename, delimiter = ',', dtype='str', skiprows = 84, max_rows = 3)
    for i in range(0, len(raw_data)):
        name = raw_data[i, 0][:-5]
        name = name.replace(':', '')
        FIMS_data[name.strip()] = np.zeros(len(FIMS_data['N_Dp'][0]))
        value = raw_data[i, 0][-5:]
        FIMS_data[name.strip()][0] = np.float64(value.strip())
        FIMS_data[name.strip()][1:] = np.array(raw_data[i, 1:], dtype = 'float64')
    
    # find indices in array corresponding to initialization times
    indices = np.where(np.logical_and(FIMS_data['Start_UTC'] >= start_time, 
                                      np.logical_and(FIMS_data['Start_UTC'] < end_time,
                                      FIMS_data['Cloud_flag'] == cloud_flag)))
    
    # pull out data that is within the sampling time
    FIMS_subdata = FIMS_data['N_Dp'][indices[0][0], :]
    for i in range(1, len(indices[0])):
        FIMS_subdata = np.vstack((FIMS_subdata, FIMS_data['N_Dp'][indices[0][i], :]))   
    FIMS_subdata = np.exp(np.log(FIMS_subdata))
    
    # calculate average size distribution in averaging window
    avg_N = np.nanmean(FIMS_subdata, axis = 0)
    error_N = np.nanstd(FIMS_subdata, axis = 0)
        
    return FIMS_data['LOWER_BIN_SIZE_nanometer'], FIMS_data['UPPER_BIN_SIZE_nanometer'], avg_N, error_N
