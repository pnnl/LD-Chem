
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 12:39:14 2025

@author: beel083
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.font_manager as font_manager
import os, shutil, time
from numba.typed import Dict
from numba import types
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
from SPLAT_initialization import read_FIMS, splat_number_fractions
from scenario import make_AqReactions
from processes import aqueous_chemistry
from aerosol_species import retrieve_one_species

# %% functions for IEPOX formation

def IEPOX_SOA_chemistry(Caq_0, aq_names, T):
    
    HSO4_conc = 0
    NH4_conc = 0
    SO4_conc = 0
    for ii, (name) in enumerate(aq_names):
        if name == 'H2O':
            H2O_conc=0.001*Caq_0[:,ii]
            H2O_idx = ii
        elif name == 'H+':
            Hplus_conc=0.001*Caq_0[:,ii]
        elif name == 'HSO4':
            HSO4_conc=0.001*Caq_0[:,ii]
            HSO4_idx = ii
        elif name == 'SO4':
            SO4_conc=0.001*Caq_0[:,ii]
            SO4_idx = ii
        elif name == 'NH4':
            NH4_conc=0.001*Caq_0[:,ii]
            NH4_idx = ii
        elif name == 'IEPOX':
            IEPOX_conc=Caq_0[:,ii]
            IEPOX_idx = ii
        elif name == 'IEPOX_OS':
            IEPOX_OS_idx = ii
        elif name == 'tetrol':
            tetrol_conc = Caq_0[:,ii]
            tetrol_idx = ii
        elif name == 'tetrol_olig':
            tetrol_olig_idx = ii
                
    kaqs = [1.8e-4, 2.62e-6, 6.2e-8, 1.91e-4]
    kaq = kaqs[0]*Hplus_conc*H2O_conc + kaqs[1]*HSO4_conc*H2O_conc + kaqs[2]*NH4_conc*H2O_conc + kaqs[3]*Hplus_conc*SO4_conc # 1/s
            
    tau_olig=24 # AS: 12, ABS: 1.5
    BETA=0.35 # AS: 0.35, ABS: 0.6
    
    #dCaq_dt_all[IEPOX_idx] -= kaq*IEPOX_conc # mol/m^3*s
    dIEPOX_OS = BETA*kaq*IEPOX_conc # mol/m^3*s
    dtetrol = (1-BETA)*kaq*IEPOX_conc # mol/m^3*s
    dtetrol_olig = (1/(tau_olig*3600))*tetrol_conc # mol/m^3*s
    dtetrol -= (1/(tau_olig*3600))*tetrol_conc # mol/m^3*s
    
    return dIEPOX_OS, dtetrol, dtetrol_olig

def IEPOX_OH_chemistry(Caq_0, aq_names, T):
    
    for ii, (name) in enumerate(aq_names):
        if name == 'OHrad':
            OHrad_conc=Caq_0[:,ii] # mol/m^3
            OHrad_idx=ii
        elif name == 'IEPOX':
            IEPOX_conc=Caq_0[:,ii] # mol/m^3
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
    
    rate = 2.4E8*np.exp(-1520/T) # m^3/mol/s
    
    dCaq_SOA=rate*IEPOX_conc*OHrad_conc # total for all SOA products
    #dCaq_dt_all[IEPOX_idx] -= dCaq_SOA # mol/m^3*s
    #dCaq_dt_all[OHrad_idx] -= dCaq_SOA # mol/m^3*s
    #dCaq_dt_all[SOA_idx] += dCaq_SOA # mol/m^3*s
    #dCaq_dt_all[OHrad_idx] += (6E-5+0.354)*dCaq_SOA
    #dCaq_dt_all[HO2_idx] += (0.138+0.063+0.259+0.186)*dCaq_SOA
    
    return dCaq_SOA

# %% set up runs
mass_thresholds={'IEPOX': [[0.3,0.5,0.1], ['IEPOX_OS','tetrol','tetrol_olig', 'IEPOX_OH_SOA']],
                'AS': [[0.5,0.7,0.1], ['SO4']],
                'AN': [[0.5,0.7,0.1], ['NO3']],
                'OC': [[0.5,0.7,0.1], ['OC']],
                'BC': [[0.5,0.7,0.1], ['BC']],
                'OIN': [[0.5,0.7,0.1], ['OIN']],
                'NH4': [[0.5,0.7,0.1], ['NH4']]}

# get the files that are common across directories
single_act_dir = '/rcfs/projects/partikkel/multipart/old_data/paper_runs_accom=1.0/0425_15utc_single_activation'
les_dir = '/rcfs/projects/partikkel/multipart/old_data/paper_runs_accom=1.0/0425_15utc_ALL'
constant_updraft_dir = '/rcfs/projects/partikkel/multipart/old_data/paper_runs_accom=1.0/0425_15utc_constant_updraft'

files1 = set(f for f in os.listdir(single_act_dir) if f.endswith('.pkl'))
files2 = set(f for f in os.listdir(les_dir) if f.endswith('.pkl'))
files3 = set(f for f in os.listdir(constant_updraft_dir) if f.endswith('.pkl'))
common_files = list(files1.intersection(files2, files3))

#les_dir='../new_accom_test/updated_pH'
#files1 = set(f for f in os.listdir(les_dir) if f.endswith('.pkl'))
#common_files = list(files1)

#with open('progress.out', 'w') as f:
print('Processing '+str(len(common_files))+' files...')#, file=f)


# %% one activation

#with open('progress.out', 'a') as f:
print('\n')#, file=f)
print('SINGLE ACTIVATION:')#, file=f)

trajectory = pickle.load(open(single_act_dir+'/'+common_files[0], 'rb'))

mass_fractions = {}
masses = {}
for ptype in mass_thresholds.keys():
    mass_fractions[ptype]=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
    masses[ptype]=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
activations=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
deactivations=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
CRT=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
cloud_state=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
NumConcs=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
dry_diameters=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
diameters=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
pHs=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
activated_fraction=np.zeros((len(trajectory['particles']), len(common_files)))
altitude=np.zeros((len(trajectory['particles']), len(common_files)))
temperature=np.zeros((len(trajectory['particles']), len(common_files)))
pressure=np.zeros((len(trajectory['particles']), len(common_files)))
S=np.zeros((len(trajectory['particles']), len(common_files)))

for FileNumber, (file) in enumerate(common_files):
    steptime0 = time.time()
    trajectory = pickle.load(open(single_act_dir+'/'+file, 'rb'))
    
    NumConcs[:,:,FileNumber]=trajectory['particles'][:,:,np.where(trajectory['particle species']=='num conc')[0][0]]
    dry_diameters[:,:,FileNumber]=trajectory['particles'][:,:,np.where(trajectory['particle species']=='Ddry')[0][0]]
    diameters[:,:,FileNumber]=trajectory['particles'][:,:,np.where(trajectory['particle species']=='Dwet')[0][0]]
    water_volume=trajectory['particles'][:,:,np.where(trajectory['particle species']=='H2O')[0][0]]/1000.0
    moles_Hplus=trajectory['particles'][:,:,np.where(trajectory['particle species']=='H+')[0][0]]/1e-3
    pHs[:,:,FileNumber]=-1.0*np.log10(moles_Hplus/(1000*water_volume))
    activated_fraction[:,FileNumber]=trajectory['activated fraction']
    altitude[:,FileNumber]=trajectory['z']
    temperature[:,FileNumber]=trajectory['T']
    pressure[:,FileNumber]=trajectory['P']
    S[:,FileNumber]=trajectory['S']
    particle_dry_MassFracs, dry_species = Particle_MassFracs(trajectory['particles'], trajectory['particle species'],
                                                                 specdata_path='../species_data/')
    '''
    for ptype in mass_thresholds.keys():
        for species in mass_thresholds[ptype][1]:
            species_idx = np.where(dry_species==species)[0][0]
            mass_fractions[ptype][:,:,FileNumber]+=particle_dry_MassFracs[:,:,species_idx]#.reshape(-1)
            species_idx = np.where(trajectory['particle species']==species)[0][0]
            masses[ptype][:,:,FileNumber]+=trajectory['particles'][:,:,species_idx]#.reshape(-1)
    

    for TimeStep in range(len(trajectory['times'])):
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
    #with open('progress.out', 'a') as f:
    print(str(file), trajectory['particles'][:,:,np.where(trajectory['particle species']=='num conc')[0][0]].shape, str(FileNumber)+'/'+str(len(common_files))+' -- '+str(round(time.time() - steptime0, 2))+ 's/it')#, file=f)
    
#pickle.dump(cloud_state, open('CloudState_single_act.pkl', 'wb'))
#pickle.dump(CRT, open('CRT_single_act.pkl', 'wb'))
#pickle.dump(activations, open('activations_single_act.pkl', 'wb'))
#pickle.dump(deactivations, open('deactivations_single_act.pkl', 'wb'))
#pickle.dump(NumConcs, open('NumConcs_single_act.pkl', 'wb'))
#pickle.dump(mass_fractions, open('MassFracs_single_act.pkl', 'wb'))
#pickle.dump(masses, open('Masses_single_act.pkl', 'wb'))
#pickle.dump(dry_diameters, open('DryDiameters_single_act.pkl', 'wb'))
#pickle.dump(diameters, open('Diameters_single_act.pkl', 'wb'))
#pickle.dump(activated_fraction, open('ActFraction_single_act.pkl', 'wb'))
#pickle.dump(altitude, open('Altitudes_single_act.pkl', 'wb'))
#pickle.dump(temperature, open('Temperatures_single_act.pkl', 'wb'))
#pickle.dump(pressure, open('Pressures_single_act.pkl', 'wb'))
#pickle.dump(S, open('Saturation_single_act.pkl', 'wb'))
pickle.dump(pHs, open('pHs_single_act.pkl', 'wb'))


# %% constant updraft

#with open('progress.out', 'a') as f:
print('\n')#, file=f)
print('CONSTANT UPDRAFT:')#, file=f)

trajectory = pickle.load(open(single_act_dir+'/'+common_files[0], 'rb'))

mass_fractions = {}
masses = {}
for ptype in mass_thresholds.keys():
    mass_fractions[ptype]=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
    masses[ptype]=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
activations=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
deactivations=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
CRT=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
cloud_state=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
NumConcs=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
dry_diameters=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
diameters=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
pHs=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
activated_fraction=np.zeros((len(trajectory['particles']), len(common_files)))
altitude=np.zeros((len(trajectory['particles']), len(common_files)))
trajectory_x=np.zeros((len(trajectory['particles']), len(common_files)))
trajectory_y=np.zeros((len(trajectory['particles']), len(common_files)))
temperature=np.zeros((len(trajectory['particles']), len(common_files)))
pressure=np.zeros((len(trajectory['particles']), len(common_files)))
S=np.zeros((len(trajectory['particles']), len(common_files)))

for FileNumber, (file) in enumerate(common_files):
    steptime0 = time.time()
    trajectory = pickle.load(open(constant_updraft_dir+'/'+file, 'rb'))
    
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
    particle_dry_MassFracs, dry_species = Particle_MassFracs(trajectory['particles'], trajectory['particle species'],
                                                                 specdata_path='../species_data/')
    '''
    for ptype in mass_thresholds.keys():
        for species in mass_thresholds[ptype][1]:
            species_idx = np.where(dry_species==species)[0][0]
            mass_fractions[ptype][:,:,FileNumber]+=particle_dry_MassFracs[:,:,species_idx]#.reshape(-1)
            species_idx = np.where(trajectory['particle species']==species)[0][0]
            masses[ptype][:,:,FileNumber]+=trajectory['particles'][:,:,species_idx]#.reshape(-1)
    

    for TimeStep in range(len(trajectory['times'])):
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
    #with open('progress.out', 'a') as f:
    print(str(file), trajectory['particles'][:,:,np.where(trajectory['particle species']=='num conc')[0][0]].shape, str(FileNumber)+'/'+str(len(common_files))+' -- '+str(round(time.time() - steptime0, 2))+ 's/it')#, file=f)

#pickle.dump(cloud_state, open('CloudState_constant_updraft.pkl', 'wb'))
#pickle.dump(CRT, open('CRT_constant_updraft.pkl', 'wb'))
#pickle.dump(activations, open('activations_constant_updraft.pkl', 'wb'))
#pickle.dump(deactivations, open('deactivations_constant_updraft.pkl', 'wb'))
#pickle.dump(NumConcs, open('NumConcs_constant_updraft.pkl', 'wb'))
#pickle.dump(mass_fractions, open('MassFracs_constant_updraft.pkl', 'wb'))
#pickle.dump(masses, open('Masses_constant_updraft.pkl', 'wb'))
#pickle.dump(dry_diameters, open('DryDiameters_constant_updraft.pkl', 'wb'))
#pickle.dump(diameters, open('Diameters_constant_updraft.pkl', 'wb'))
#pickle.dump(activated_fraction, open('ActFraction_constant_updraft.pkl', 'wb'))
#pickle.dump(altitude, open('Altitudes_constant_updraft.pkl', 'wb'))
#pickle.dump(trajectory_y, open('Latitudes_constant_updraft.pkl', 'wb'))
#pickle.dump(trajectory_x, open('Longitudes_constant_updraft.pkl', 'wb'))
#pickle.dump(temperature, open('Temperatures_constant_updraft.pkl', 'wb'))
#pickle.dump(pressure, open('Pressures_constant_updraft.pkl', 'wb'))
#pickle.dump(S, open('Saturation_constant_updraft.pkl', 'wb'))
pickle.dump(pHs, open('pHs_constant_updraft.pkl', 'wb'))


# %% multiple activations

#with open('progress.out', 'a') as f:
print('\n')#, file=f)
print('LES:')#, file=f)

trajectory = pickle.load(open(les_dir+'/'+common_files[0], 'rb'))
mass_fractions = {}
masses = {}
for ptype in mass_thresholds.keys():
    mass_fractions[ptype]=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
    masses[ptype]=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
activations=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
deactivations=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
CRT=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
cloud_state=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
NumConcs=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
dry_diameters=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
diameters=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
pHs=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
activated_fraction=np.zeros((len(trajectory['particles']), len(common_files)))
altitude=np.zeros((len(trajectory['particles']), len(common_files)))
temperature=np.zeros((len(trajectory['particles']), len(common_files)))
pressure=np.zeros((len(trajectory['particles']), len(common_files)))
S=np.zeros((len(trajectory['particles']), len(common_files)))
IEPOX_formation_rates=np.zeros((len(trajectory['particles']), len(trajectory['particles'][0]), len(common_files)))
trajectory_x=np.zeros((len(trajectory['particles']), len(common_files)))
trajectory_y=np.zeros((len(trajectory['particles']), len(common_files)))

for FileNumber, (file) in enumerate(common_files):

    steptime0 = time.time()
    trajectory = pickle.load(open(les_dir+'/'+file, 'rb'))
    
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
    
    '''
    particle_dry_MassFracs, dry_species = Particle_MassFracs(trajectory['particles'], trajectory['particle species'],
                                                                 specdata_path='../species_data/')
    for ptype in mass_thresholds.keys():
        for species in mass_thresholds[ptype][1]:
            species_idx = np.where(dry_species==species)[0][0]
            mass_fractions[ptype][:,:,FileNumber]+=particle_dry_MassFracs[:,:,species_idx]#.reshape(-1)
            species_idx = np.where(trajectory['particle species']==species)[0][0]
            masses[ptype][:,:,FileNumber]+=trajectory['particles'][:,:,species_idx]#.reshape(-1)
    
    for TimeStep in range(len(trajectory['times'])):
        if trajectory['activated fraction'][TimeStep]>0:
            cloud_droplets = get_CD_status(trajectory['particles'][TimeStep,:,np.where(trajectory['particle species']=='Dwet')[0][0]], 
                                                trajectory['particles'][TimeStep,:,np.where(trajectory['particle species']=='Ddry')[0][0]], 
                                                trajectory['particles'][TimeStep,:,np.where(trajectory['particle species']=='kappa')[0][0]],
                                                trajectory['particles'][TimeStep,:,np.where(trajectory['particle species']=='num conc')[0][0]],
                                                trajectory['T'][TimeStep])
            idx=np.where(cloud_droplets>0)[0]
            cloud_state[TimeStep,idx,FileNumber]=1
            CRT[TimeStep+1:,idx,FileNumber]+=1
        
        # get the formation rates
        particle_concs = Particle_Concentrations(trajectory['particles'][TimeStep],
                                                  trajectory['particle species'],
                                                  specdata_path='../species_data/') # mol/m^3
        dIEPOX_OS, dtetrol, dtetrol_olig=IEPOX_SOA_chemistry(particle_concs, trajectory['particle species'], trajectory['T'][TimeStep])
        dIEPOX_OH=IEPOX_OH_chemistry(particle_concs, trajectory['particle species'], trajectory['T'][TimeStep])
        formation_rate=np.zeros(dIEPOX_OS.shape)
        water_idx = np.where(trajectory['particle species']=='H2O')[0][0]
        water_volume = trajectory['particles'][TimeStep,:,water_idx]/1000.0 # m^3
        for species, dCaq in zip(['IEPOX_OS','tetrol','tetrol_olig','IEPOX_OH_SOA'], [dIEPOX_OS, dtetrol,
            dtetrol_olig, dIEPOX_OH]):
            species_info=retrieve_one_species(species, specdata_path='../species_data/')
            formation_rate += dCaq*species_info.molar_mass*water_volume # kg/s
        IEPOX_formation_rates[TimeStep,:,FileNumber]=formation_rate
        
    for pNumber in range(len(trajectory['particles'][0])):
        switches=cloud_state[1:,pNumber,FileNumber]-cloud_state[:-1,pNumber,FileNumber]
        activation_events=np.where(switches>0)[0]
        deactivation_events=np.where(switches<0)[0]
        for TimeStep in activation_events:
            activations[TimeStep+1:,pNumber,FileNumber]+=1  
        for TimeStep in deactivation_events:
            deactivations[TimeStep+1:,pNumber,FileNumber]+=1    
    '''
    #with open('progress.out', 'a') as f:
    print(str(file), trajectory['particles'][:,:,np.where(trajectory['particle species']=='num conc')[0][0]].shape, str(FileNumber)+'/'+str(len(common_files))+' -- '+str(round(time.time() - steptime0, 2))+ 's/it')#, file=f)
    
#pickle.dump(cloud_state, open('CloudState_multiple_acts.pkl', 'wb'))
#pickle.dump(CRT, open('CRT_multiple_acts.pkl', 'wb'))
#pickle.dump(activations, open('activations_multiple_acts.pkl', 'wb'))
#pickle.dump(deactivations, open('deactivations_multiple_acts.pkl', 'wb'))
#pickle.dump(NumConcs, open('NumConcs_multiple_acts.pkl', 'wb'))
#pickle.dump(mass_fractions, open('MassFracs_multiple_acts.pkl', 'wb'))
#pickle.dump(masses, open('Masses_multiple_acts.pkl', 'wb'))
#pickle.dump(dry_diameters, open('DryDiameters_multiple_acts.pkl', 'wb'))
#pickle.dump(diameters, open('Diameters_multiple_acts.pkl', 'wb'))
#pickle.dump(activated_fraction, open('ActFraction_multiple_acts.pkl', 'wb'))
#pickle.dump(IEPOX_formation_rates, open('IEPOX_formation_multiple_acts.pkl', 'wb'))
#pickle.dump(trajectory_y, open('Latitudes_multiple_acts.pkl', 'wb'))
#pickle.dump(trajectory_x, open('Longitudes_multiple_acts.pkl', 'wb'))
#pickle.dump(altitude, open('Altitudes_multiple_acts.pkl', 'wb'))
#pickle.dump(temperature, open('Temperatures_multiple_acts.pkl', 'wb'))
#pickle.dump(pressure, open('Pressures_multiple_acts.pkl', 'wb'))
#pickle.dump(S, open('Saturation_multiple_acts.pkl', 'wb'))
pickle.dump(pHs, open('pHs_multiple_acts.pkl', 'wb'))

# %% remove files

for file in files:
    os.remove(file)
for directory in directories:
    shutil.rmtree(directory)
