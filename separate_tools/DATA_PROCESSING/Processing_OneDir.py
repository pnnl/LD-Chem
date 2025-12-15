#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 09:53:00 2025

@author: beel083
"""

# %% import files
import pickle
import numpy as np
from scipy.integrate import trapezoid
import os, shutil, time
multipart_directory='../multipart/'
files = ['particles.py', 'HISCALE_data_processing.py', 'SPLAT_initialization.py', 'scenario.py',
          'TraceGases.py', 'Reactions.py', 'constants.py', 'aerosol_species.py', 'utilities.py',
          'systems.py']
directories=['processes']
for file in files:
    shutil.copy(multipart_directory+file, os.getcwd())
for directory in directories:
    shutil.copytree(multipart_directory+directory, os.getcwd()+'/'+directory)
from HISCALE_data_processing import classify, get_CD_status, Particle_MassFracs, Particle_Concentrations

def get_files(*dirs, extension=None):
    """
    Return sorted list of common files across multiple directories.
    Optionally filters by file extension (e.g., '.txt', '.py').
    """
    file_sets = []
    for d in dirs:
        try:
            files = {
                f for f in os.listdir(d)
                if os.path.isfile(os.path.join(d, f))
                and (extension is None or f.endswith(extension))
            }
            file_sets.append(files)
        except FileNotFoundError:
            print(f"Directory not found: {d}")
            return []
        except PermissionError:
            print(f"Permission denied: {d}")
            return []
    
    if not file_sets:
        return []
    
    common_files = set.intersection(*file_sets)
    return sorted(common_files)

def IEPOX_OH_chemistry(Caq_0, aq_names, T):
    
    for ii, (name) in enumerate(aq_names):
        if name == 'OHrad':
            OHrad_conc=Caq_0[:,:,ii] # mol/m^3
            OHrad_idx=ii
        elif name == 'IEPOX':
            IEPOX_conc=Caq_0[:,:,ii] # mol/m^3
            IEPOX_idx=ii
        elif name == 'IEPOX_OH_SOA':
            SOA_idx=ii
        elif name == 'HO2':
            HO2_idx=ii
    
    # IEPOX_OH_SOA is made of:
    # .0006% DHBO
    # 35.4% DHMP
    # 13.8% glycolaldehyde and methylgloxal
    # 6.3% glyoxyl and hydroxyacetone
    # 25.9% oxygenated IEPOX species
    # 18.6% HBDO
    # by moles
    
    rate = np.repeat(2.4E8*np.exp(-1520/T), len(IEPOX_conc[0])).reshape(len(T), len(IEPOX_conc[0])) # m^3/mol/s
    dCaq_SOA=rate*IEPOX_conc*OHrad_conc # total for all SOA products
    
    return dCaq_SOA



# %% get the data

mass_thresholds={'IEPOX': [[0.3,0.5,0.1], ['IEPOX_OS','tetrol','tetrol_olig', 'IEPOX_OH_SOA']],
                'AS': [[0.5,0.7,0.1], ['SO4']],
                'AN': [[0.5,0.7,0.1], ['NO3']],
                'OC': [[0.5,0.7,0.1], ['OC']],
                'BC': [[0.5,0.7,0.1], ['BC']],
                'OIN': [[0.5,0.7,0.1], ['OIN']],
                'NH4': [[0.5,0.7,0.1], ['NH4']]}

WindDivergence = pickle.load(open('../datasets/parcel_traces_0425_15utc/WindDivergence.pkl', 'rb'))
traj_dirs = ['../LES_runs_tau24']
output_directory = 'LES'
dir = traj_dirs[0]

if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

trajfiles = get_files(*traj_dirs, extension='.pkl')
trajectory = pickle.load(open(dir+'/'+trajfiles[0], 'rb'))

mass_fractions = {}
masses = {}
for ptype in mass_thresholds.keys():
    mass_fractions[ptype]=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
    masses[ptype]=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))

activations=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
deactivations=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
CRT=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
cloud_state=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
NumConcs=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
dry_diameters=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
diameters=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
pHs=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
activated_fraction=np.zeros((len(trajectory['particles']), len(trajfiles)))
altitude=np.zeros((len(trajectory['particles']), len(trajfiles)))
temperature=np.zeros((len(trajectory['particles']), len(trajfiles)))
pressure=np.zeros((len(trajectory['particles']), len(trajfiles)))
S=np.zeros((len(trajectory['particles']), len(trajfiles)))
IEPOX_gas=np.zeros((len(trajectory['particles']), len(trajfiles)))
H2SO4_gas=np.zeros((len(trajectory['particles']), len(trajfiles)))
SO2_gas=np.zeros((len(trajectory['particles']), len(trajfiles)))
IEPOX_formation_rates=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(trajfiles)))
trajectory_x=np.zeros((len(trajectory['particles']), len(trajfiles)))
trajectory_y=np.zeros((len(trajectory['particles']), len(trajfiles)))

for FileNumber, (file) in enumerate(trajfiles):
    steptime0 = time.time()
    trajectory = pickle.load(open(dir+'/'+file, 'rb'))
    
    NumConcs[:,:,FileNumber]=trajectory['particles'][:,:,np.where(trajectory['particle species']=='num conc')[0][0]]
    dry_diameters[:,:,FileNumber]=trajectory['particles'][:,:,np.where(trajectory['particle species']=='Ddry')[0][0]]
    diameters[:,:,FileNumber]=trajectory['particles'][:,:,np.where(trajectory['particle species']=='Dwet')[0][0]]
    water_volume=trajectory['particles'][:,:,np.where(trajectory['particle species']=='H2O')[0][0]]/1000.0
    moles_Hplus=trajectory['particles'][:,:,np.where(trajectory['particle species']=='H+')[0][0]]/1e-3
    pHs[:,:,FileNumber]=-1.0*np.log10(moles_Hplus/(1000*water_volume))
    activated_fraction[:,FileNumber]=trajectory['activated fraction']
    trajectory_x[:,FileNumber]=trajectory['x']
    trajectory_y[:,FileNumber]=trajectory['y']
    altitude[:,FileNumber]=trajectory['z']
    temperature[:,FileNumber]=trajectory['T']
    pressure[:,FileNumber]=trajectory['P']
    S[:,FileNumber]=trajectory['S']
    IEPOX_gas[:,FileNumber]=trajectory['gases'][:,np.where(trajectory['gas species']=='IEPOX')[0][0]]
    H2SO4_gas[:,FileNumber]=trajectory['gases'][:,np.where(trajectory['gas species']=='H2SO4')[0][0]]
    SO2_gas[:,FileNumber]=trajectory['gases'][:,np.where(trajectory['gas species']=='SO2')[0][0]]
    particle_dry_MassFracs, dry_species = Particle_MassFracs(trajectory['particles'], trajectory['particle species'],
                                                                 specdata_path='../species_data/')
    '''
    for ptype in mass_thresholds.keys():
        for species in mass_thresholds[ptype][1]:
            species_idx = np.where(dry_species==species)[0][0]
            mass_fractions[ptype][:,:,FileNumber]+=particle_dry_MassFracs[:,:,species_idx]#.reshape(-1)
            species_idx = np.where(trajectory['particle species']==species)[0][0]
            masses[ptype][:,:,FileNumber]+=trajectory['particles'][:,:,species_idx]#.reshape(-1)
    
    # get the IEPOX formation rates
    particle_concs = Particle_Concentrations(trajectory['particles'],
                                              trajectory['particle species'],
                                              specdata_path='../species_data/') # mol/m^3
    IEPOX = particle_concs[:,:,np.where(trajectory['particle species']=='IEPOX')[0][0]]
    H2O = particle_concs[:,:,np.where(trajectory['particle species']=='H2O')[0][0]]
    Hplus = particle_concs[:,:,np.where(trajectory['particle species']=='H+')[0][0]]
    HSO4 = particle_concs[:,:,np.where(trajectory['particle species']=='HSO4')[0][0]]
    SO4 = particle_concs[:,:,np.where(trajectory['particle species']=='SO4')[0][0]]
    tetrol = particle_concs[:,:,np.where(trajectory['particle species']=='tetrol')[0][0]]
    dtetrol = 1e-8*H2O*IEPOX*Hplus - 3.54e-2*IEPOX*tetrol
    dIEPOX_OS = 1.46e-7*IEPOX*HSO4 + 1.06e-8*IEPOX*SO4*Hplus
    dtetrol_olig = 3.54e-2*IEPOX*tetrol
    dIEPOX_OH=IEPOX_OH_chemistry(particle_concs, trajectory['particle species'], trajectory['T'])
    formation_rate=np.zeros(dIEPOX_OS.shape)
    species_idx = np.where(trajectory['particle species']=='H2O')[0][0]
    water_volume = trajectory['particles'][:,:,species_idx]/1000.0 # m^3
    for species, dCaq, molar_mass in zip(['IEPOX_OS','tetrol','tetrol_olig','IEPOX_OH_SOA'], [dIEPOX_OS, dtetrol,
        dtetrol_olig, dIEPOX_OH], [215e-3, 136e-3, 254e-3, 84.4e-3]):
        formation_rate += dCaq*molar_mass*water_volume # kg/s
    IEPOX_formation_rates[:,:,FileNumber]=formation_rate


    for TimeStep in range(len(trajectory['times'])):
        
        x = trajectory['times'][:TimeStep+1]
        y = np.interp(x, 60.0*np.arange(0, len(WindDivergence), 1), WindDivergence[:,int(file[-10:-4])])
        integrated_u=trapezoid(y, x=x)
        NumConcs[TimeStep,:,FileNumber]=NumConcs[0,:,FileNumber]*np.exp(-1.0*integrated_u)
    
        if trajectory['activated fraction'][TimeStep]>0:
            cloud_droplets = get_CD_status(trajectory['particles'][TimeStep,:,np.where(trajectory['particle species']=='Dwet')[0][0]],
                                                trajectory['particles'][TimeStep,:,np.where(trajectory['particle species']=='Ddry')[0][0]],
                                                trajectory['particles'][TimeStep,:,np.where(trajectory['particle species']=='kappa')[0][0]],
                                                trajectory['particles'][TimeStep,:,np.where(trajectory['particle species']=='num conc')[0][0]],
                                                trajectory['T'][TimeStep])
            idx=np.where(cloud_droplets>0)[0]
            cloud_state[TimeStep,idx,FileNumber]=1
            CRT[TimeStep+1:,idx,FileNumber]+=1
    
    for pNumber in range(len(trajectory['particles'][0])):
        switches=cloud_state[1:,pNumber,FileNumber]-cloud_state[:-1,pNumber,FileNumber]
        activation_events=np.where(switches>0)[0]
        deactivation_events=np.where(switches<0)[0]
        for TimeStep in activation_events:
            activations[TimeStep+1:,pNumber,FileNumber]+=1
        for TimeStep in deactivation_events:
            deactivations[TimeStep+1:,pNumber,FileNumber]+=1
    '''
    print(str(dir+'/'+file), trajectory['particles'][:,:,np.where(trajectory['particle species']=='num conc')[0][0]].shape, str(FileNumber+1)+'/'+str(len(trajfiles))+' -- '+str(round(time.time() - steptime0, 2))+ 's/it')

aerosol_Ns = np.where(cloud_state == 0, NumConcs, 0)
air_density=(pressure*0.0289652)/(8.314*temperature)
total_MassConc = np.zeros(S.shape)
for ptype in mass_thresholds.keys():
    total_MassConc += 1e9*np.sum(aerosol_Ns*masses[ptype], axis=1)#/air_density
idx = np.where((altitude>500) & (altitude<=700))
NumConcs *= 4.441062694762885/(np.median(total_MassConc[idx]/air_density[idx])) # scale the number concentration so that the mixing ratio below cloud is equal to the measurements

print()
print(np.__version__)
import sys
print(sys.version)

#pickle.dump(cloud_state, open(output_directory+'/CloudState.pkl', 'wb'))
#pickle.dump(CRT, open(output_directory+'/CRT.pkl', 'wb'))
#pickle.dump(activations, open(output_directory+'/activations.pkl', 'wb'))
#pickle.dump(deactivations, open(output_directory+'/deactivations.pkl', 'wb'))
#pickle.dump(NumConcs, open(output_directory+'/NumConcs.pkl', 'wb'))
#pickle.dump(mass_fractions, open(output_directory+'/MassFracs.pkl', 'wb'))
#pickle.dump(masses, open(output_directory+'/Masses.pkl', 'wb'))
#pickle.dump(dry_diameters, open(output_directory+'/DryDiameters.pkl', 'wb'))
#pickle.dump(diameters, open(output_directory+'/Diameters.pkl', 'wb'))
#pickle.dump(activated_fraction, open(output_directory+'/ActFraction.pkl', 'wb'))
#pickle.dump(altitude, open(output_directory+'/Altitudes.pkl', 'wb'))
#pickle.dump(temperature, open(output_directory+'/Temperatures.pkl', 'wb'))
#pickle.dump(pressure, open(output_directory+'/Pressures.pkl', 'wb'))
#pickle.dump(S, open(output_directory+'/Saturation.pkl', 'wb'))

pickle.dump(IEPOX_gas, open(output_directory+'/IEPOX_gas.pkl', 'wb'))
pickle.dump(H2SO4_gas, open(output_directory+'/H2SO4_gas.pkl', 'wb'))
pickle.dump(SO2_gas, open(output_directory+'/SO2_gas.pkl', 'wb'))

#pickle.dump(pHs, open(output_directory+'/pHs.pkl', 'wb'))
#pickle.dump(IEPOX_formation_rates, open(output_directory+'/IEPOX_formation.pkl', 'wb'))
#pickle.dump(trajectory_y, open(output_directory+'/Latitudes.pkl', 'wb'))
#pickle.dump(trajectory_x, open(output_directory+'/Longitudes.pkl', 'wb'))

# %% remove files

for file in files:
    os.remove(file)
for directory in directories:
    shutil.rmtree(directory)

