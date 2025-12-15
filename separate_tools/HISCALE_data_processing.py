#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:55:53 2024

@author: beel083
"""
import numpy as np
import matplotlib.pyplot as plt
from SPLAT_initialization import read_FIMS, splat_number_fractions, ams_mass_fraction
import sys, tqdm, particles
from processes import water_uptake
from scipy.optimize import fminbound


def plot_diameters(trajectory, axis='height', resolution=60):
    
    # fix this -- make these into time series
    d_drys = []
    d_wets = []
    z = []
    S = []
    t = []
    dt=trajectory['times'][1]-trajectory['times'][0]
    didx=int(resolution/dt)
    for pState in range(0, len(trajectory['particles']), didx):
        S.append(trajectory['S'][pState])
        z.append(trajectory['z'][pState])
        t.append(trajectory['times'][pState])
        
        temp_dry = []
        temp_wet = []
        Dwet_idx = np.where(trajectory['particle species']=='Dwet')
        Ddry_idx = np.where(trajectory['particle species']=='Ddry')
        for pNumber in range(len(trajectory['particles'][pState])):
            d_dry = float(trajectory['particles'][pState, pNumber, Ddry_idx])
            d_wet = float(trajectory['particles'][pState, pNumber, Dwet_idx])
            temp_wet.append(d_wet)
            temp_dry.append(d_dry)
        d_drys.append(temp_dry)
        d_wets.append(temp_wet)
    
    if axis == 'height':
        fig, ax = plt.subplots(1, 1)
        ax2=ax.twiny()
        ax2.spines['bottom'].set_color('blue')
        ax2.spines['top'].set_color('red')
        ax.tick_params(axis='x', which="both",color='blue', labelcolor='blue')
        ax2.tick_params(axis='x', which="both",color='red', labelcolor='red')
        ax.plot(np.array((d_wets))*1e6, z, '-b')
        ax2.plot(S, z, '-r')
        ax.set_xlabel(r'wet diameter ($\mu$m)', color='blue')
        ax.set_xscale('log')
        ax2.set_xlabel('saturation ratio', color='red')
        ax2.set_xlim(1.0,)
        ax.set_ylabel('altitude (m)')

    
    elif axis == 'time':
        fig, ax = plt.subplots(1, 1)
        ax2=ax.twinx()
        ax2.spines['left'].set_color('blue')
        ax2.spines['right'].set_color('red')
        ax.tick_params(axis='y', which="both",color='blue', labelcolor='blue')
        ax2.tick_params(axis='y', which="both",color='red', labelcolor='red')
        ax.plot(np.array((t))/60, np.array((d_wets))*1e6, '-b')
        ax.set_yscale('log')
        ax2.plot(np.array((t))/60, S, '-r')
        ax.set_ylabel(r'wet diameter ($\mu$m)', color='blue')
        ax2.set_ylabel('saturation ratio', color='red')
        ax.set_xlabel('time (min)')    

    return fig

def plot_aq_species(trajectory, species, axis='time'):
    
    t = trajectory['times']
    z = trajectory['z']
    
    fig1, ax1 = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)      
    
    traj_idx = np.where(trajectory['particle species']=='num conc')[0][0]
    num_concs = trajectory['particles'][:, :, traj_idx]
        
    for ii, (s) in enumerate(species):
        traj_idx = np.where(trajectory['particle species']==s)[0][0]
        masses = trajectory['particles'][:, :, traj_idx]
        color='C'+str(ii)
        
        if axis == 'height':
            ax1.plot(masses*1e9, z, '-', color=color, label='_nolabel_')            
            ax2.plot(np.nansum(num_concs*masses*1e9, axis=1), z, '-', color=color, label=s, linewidth=2)   
        elif axis == 'time':
            ax1.plot(t/60, masses*1e9, '-', color=color, label='_nolabel_')
            ax2.plot(t/60, np.nansum(num_concs*masses*1e9, axis=1), '-', color=color, label=s, linewidth=2)   
            
    for ii, (s) in enumerate(species):
        color='C'+str(ii)
        ax1.plot(-10, -10, '-', color=color, label=s)
        
    if axis == 'height':
        ax1.set_xscale('log')
        ax1.set_xlabel('mass ($\mu$g)')
        ax1.legend()
        ax1.set_ylabel('altitude (m)')
        
        ax2.set_xlabel(r'mass concentration ($\mu$g/m$^3$)')
        ax2.set_ylabel('altitude (m)')
        ax2.legend()
    
    elif axis == 'time':
        ax1.set_ylabel(r' mass ($\mu$g)')
        ax1.set_xlabel('time (min)')
        ax1.set_xlim(0,)
        ax1.set_yscale('log')
        
        ax2.set_ylabel(r'mass concentration ($\mu$g/m$^3$)')
        ax2.set_xlabel('time (min)')
        ax2.legend()
        ax2.set_xlim(0,)

    return fig1, fig2

def plot_activated_fraction(trajectory):
      
    fig, ax = plt.subplots(1, 1)
    ax2=ax.twinx()
    ax2.spines['left'].set_color('blue')
    ax2.spines['right'].set_color('red')
    ax.tick_params(axis='y', which="both",color='blue', labelcolor='blue')
    ax2.tick_params(axis='y', which="both",color='red', labelcolor='red')
    ax.plot(trajectory['times']/60, trajectory['S'], '-b')
    ax2.plot(trajectory['times']/60, trajectory['activated fraction'], '-r')
    ax2.set_ylim(0,1)
    ax2.set_ylabel('activated fraction', color='red')
    ax.set_ylabel('saturation ratio', color='blue')
    ax.set_xlabel('time (min)')
    
    return fig


def initial_SizeDist(trajectory, size_distribution_file, splat_species,
                     mass_thresholds, start_time=None, end_time=None):
        
    Dp_lowers, Dp_uppers, N_measured, N_error = read_FIMS(size_distribution_file, 
                                                  start_time, end_time, 0.0, 
                                                  0.0) # diameters in nm and N in #/cm^3
    
    particle_classes = classify(trajectory['particles'][0], trajectory['particle species'], splat_species, mass_thresholds)
    
    bins=Dp_uppers[::2]
    histogram_Dps = bins[:-1]
    histogram_Ns={}
    for t in splat_species.keys():
        histogram_Ns[t]=np.zeros((len(trajectory), len(histogram_Dps))) 
    
    Ddrys=trajectory['particles'][0][:, np.where(trajectory['particle species']=='Ddry')[0][0]]
    Ns=trajectory['particles'][0][:, np.where(trajectory['particle species']=='num conc')[0][0]]
        
    for t in splat_species.keys():
        idx=np.where(particle_classes==t)
        hist=np.histogram(1e9*Ddrys[idx[0]], bins=bins, weights=18.10*Ns[idx[0]]/100**3)   
        histogram_Ns[t]=hist[0]
    
    fig, ax = plt.subplots(1, 1)
    bottom=np.zeros(len(bins)-1)
    hist=np.histogram(Dp_uppers, bins=bins, weights=N_measured*18.10)
    widths=hist[1][1:]-hist[1][:-1]
    ax.plot(hist[1][:-1], hist[0], '-ko', label='measured')
    
    for ii, (t, c, l) in enumerate(zip(splat_species.keys(), ['grey','gold','r','b','g','C6'], ['black carbon', 'dust', 'sulfate rich', 'nitrate rich', 'organics', 'IEPOX SOA'])):
        ax.bar(histogram_Dps, histogram_Ns[t], 
                width=widths, bottom=bottom, facecolor=c, edgecolor='k', 
                label=l, align='edge')
        bottom+=histogram_Ns[t]
        
    ax.set_ylabel(r'dN/dlog($d_p$) (cm$^{-3}$)', labelpad=15)
    ax.set_xlabel('dry diameter (nm)', labelpad=15)
    ax.set_xscale('log')
    ax.set_ylim(0,)
    ax.legend()
        
    return fig


def cloud_composition(trajectory, splat_file, splat_species, size_distribution_file,
                      mass_thresholds, resolution=60):
    
    
    # read miniSPLAT data from file
    filename = splat_file
    raw_data = np.loadtxt(filename, dtype='str')
    full_SPLAT_data = {}
    SPLAT_subdata = {}
    for i in range(0, len(raw_data[0])): 
        full_SPLAT_data[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
        SPLAT_subdata[str(raw_data[0, i])] = np.zeros(0)
    
    # pull out the miniSPLAT data corresponding to initialization times
    idx = np.where(np.logical_and(full_SPLAT_data['Time']>0, full_SPLAT_data['Time']<=86400))
    for key in full_SPLAT_data.keys():
        SPLAT_subdata[key] = np.array((full_SPLAT_data[key][idx[0]]))
    
    # create dict of all the species in the miniSPLAT data
    minisplat_species = []
    for key in SPLAT_subdata.keys():
        if key != 'Time':
            minisplat_species.append(key)

    # find the full time series of number fraction for each class
    minisplat_fraction = {}
    for reduced_species in splat_species.keys():
        summation = np.zeros(len(full_SPLAT_data['Time']))
        for species in splat_species[reduced_species]:
            for i in range(0, len(summation)):
                summation[i] = summation[i] + full_SPLAT_data[species][i]
        minisplat_fraction[reduced_species] = summation
    
    # read FIMS data from file   
    filename = size_distribution_file
    raw_data = np.loadtxt(filename, delimiter = ',', dtype='str', skiprows = 100)    
    for i in range(0, len(raw_data)):
        for j in range(0, len(raw_data[0])):
            raw_data[i, j] = raw_data[i, j].strip()
    FIMS_data = {}
    FIMS_data[str(raw_data[0, 0])] = np.array(raw_data[1:, 0], dtype = 'float64')
    for i in range(56, len(raw_data[0])): 
        FIMS_data[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
    FIMS_data['N_Dp'] = np.array(raw_data[1:, 1:56], dtype = 'float64')

    # get splat data only in cloud and CVI inlet
    idx = np.where(np.logical_and(FIMS_data['Cloud_flag']==1.0, FIMS_data['CVI_flag']==1.0))
    times = FIMS_data['Start_UTC'][idx[0]] 
    
    # set up dict for measured values
    measured_CDRs={}
    for ptype in splat_species.keys():
        measured_CDRs[ptype]=np.zeros(0)
        for ii, (t) in enumerate(times):
            idx = np.where(SPLAT_subdata['Time']==t)
            if len(idx[0])>0:
                measured_CDRs[ptype]=np.append(measured_CDRs[ptype], minisplat_fraction[ptype][idx[0][0]])

    # get splat data only in cloud and aerosol inlet
    idx = np.where(np.logical_and(FIMS_data['Cloud_flag']==1.0, FIMS_data['CVI_flag']==0.0))
    times = FIMS_data['Start_UTC'][idx[0]] 
    
    # set up dict for measured values
    measured_interstitials={}
    for ptype in splat_species.keys():
        measured_interstitials[ptype]=np.zeros(0)
        for ii, (t) in enumerate(times):
            idx = np.where(SPLAT_subdata['Time']==t)
            if len(idx[0])>0:
                measured_interstitials[ptype]=np.append(measured_interstitials[ptype], minisplat_fraction[ptype][idx[0][0]])    
    
    
    # get the modeled values
    dt=trajectory['times'][1]-trajectory['times'][0]
    didx=int(resolution/dt)
    
    modeled_CDRs = {}
    for ptype in splat_species.keys():
        modeled_CDRs[ptype]=np.zeros(len(trajectory['times']))
    for ptype in splat_species.keys():
        modeled_CDRs[ptype][:]=np.nan
        
    modeled_interstitials = {}
    for ptype in splat_species.keys():
        modeled_interstitials[ptype]=np.zeros(len(trajectory['times']))
    for ptype in splat_species.keys():
        modeled_interstitials[ptype][:]=np.nan

    print('plotting cloud composition...')
    pbar = tqdm.tqdm(total = len(trajectory['times'][::didx]))
    for pState in range(0, len(trajectory['times']), didx):        
        F_activated = trajectory['activated fraction'][pState]
        if F_activated>0.0:
            cloud_droplets = get_CD_status(trajectory['particles'][pState, :, np.where(trajectory['particle species']=='Dwet')[0][0]], 
                                           trajectory['particles'][pState, :, np.where(trajectory['particle species']=='Ddry')[0][0]], 
                                           trajectory['particles'][pState, :, np.where(trajectory['particle species']=='kappa')[0][0]],
                                           trajectory['particles'][pState, :, np.where(trajectory['particle species']=='num conc')[0][0]],
                                           trajectory['T'][pState])

            particle_classes = classify(trajectory['particles'][pState], trajectory['particle species'], splat_species, mass_thresholds)            
            idx = np.where(cloud_droplets>0.0)
            Ntot_activated = np.sum(cloud_droplets[idx[0]])
            idx = np.where(cloud_droplets<0.0)
            Ntot_unactivated = -1.0*np.sum(cloud_droplets[idx[0]])
            for ptype in splat_species.keys():
                idx = np.where(np.logical_and(particle_classes==ptype, cloud_droplets>0.0))                 
                modeled_CDRs[ptype][pState]=np.sum(cloud_droplets[idx[0]])/Ntot_activated
                idx = np.where(np.logical_and(particle_classes==ptype, cloud_droplets<0.0))
                modeled_interstitials[ptype][pState]=(-1.0*np.sum(cloud_droplets[idx[0]]))/Ntot_unactivated  
                 
        pbar.update(1)
    
    pbar.close()
    
    # print()
    # for k in modeled_CDRs.keys():
    #     print(k, modeled_CDRs[k].shape)
    # print()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.0*6.4, 1.0*4.8), sharey=True)
    ax1.grid(axis='y')
    ax2.grid(axis='y')
    
    # plot the CDRs
    locations = np.zeros(len(measured_CDRs.keys()))
    measured_values = np.zeros(len(measured_CDRs.keys()))
    measured_errors = np.zeros(len(measured_CDRs.keys()))
    modeled_values = np.zeros(len(modeled_CDRs.keys()))
    modeled_errors = np.zeros(len(modeled_CDRs.keys()))
    for ii, (ptype) in enumerate(measured_CDRs.keys()):
        locations[ii]=ii
        measured_values[ii]=np.mean(measured_CDRs[ptype])
        measured_errors[ii]=np.std(measured_CDRs[ptype])
        
        modeled_values[ii]=np.nanmean(modeled_CDRs[ptype])
        modeled_errors[ii]=0 #np.nanstd(modeled_CDRs[ptype])
    
    ax1.bar(locations-0.3, measured_values, width=0.3, align='edge', color='grey', yerr=measured_errors, edgecolor='k', label='measured')
    ax1.bar(locations, modeled_values, width=0.3, align='edge', color='w', yerr=modeled_errors, edgecolor='k', label='modeled')
    ax1.set_xticks(locations)
    ax1.set_xticklabels(measured_CDRs.keys())
    ax1.set_title('cloud droplet residuals', pad=15)
    
    # plot the interstitials
    locations = np.zeros(len(measured_interstitials.keys()))
    measured_values = np.zeros(len(measured_interstitials.keys()))
    measured_errors = np.zeros(len(measured_interstitials.keys()))
    modeled_values = np.zeros(len(modeled_interstitials.keys()))
    modeled_errors = np.zeros(len(modeled_interstitials.keys()))
    for ii, (ptype) in enumerate(measured_interstitials.keys()):
        locations[ii]=ii
        measured_values[ii]=np.mean(measured_interstitials[ptype])
        measured_errors[ii]=np.std(measured_interstitials[ptype])
        
        modeled_values[ii]=np.nanmean(modeled_interstitials[ptype])
        modeled_errors[ii]=0 #np.nanstd(modeled_interstitials[ptype])

    ax2.bar(locations-0.3, measured_values, width=0.3, align='edge', color='grey', yerr=measured_errors, edgecolor='k', label='measured')
    ax2.bar(locations, modeled_values, width=0.3, align='edge', color='w', yerr=modeled_errors, edgecolor='k', label='modeled')
    ax2.set_xticks(locations)
    ax2.set_xticklabels(measured_interstitials.keys())
    ax2.set_title('interstitials', pad=15)
    
    ax1.set_ylabel('number fraction', labelpad=15)
    ax1.set_ylim(0,1)
    ax2.legend()
        
    return fig
   
    
def Particle_MassFracs(particle_masses, particle_species, specdata_path='../species_data'):
    
    dry_species = []
    dry_idx = []
    for species in particle_species:
        try:
            SpeciesData = particles.retrieve_one_species(species, specdata_path=specdata_path)
            if SpeciesData.density > 0.0 and species != 'H2O':
                dry_species.append(species)
                dry_idx.append(np.where(particle_species==species)[0][0])
        except Exception:
            pass
    dry_species = np.array(dry_species)
    particle_dry_masses = particle_masses[:,:,dry_idx] # this pulls out only the aerosol species
    particle_dry_MassFracs = np.zeros(particle_dry_masses.shape)
    total_dry_masses=np.sum(particle_dry_masses, axis=2)
    total_dry_masses = np.repeat(total_dry_masses[:,:,np.newaxis], len(dry_species), axis=2)
    for ii in range((len(particle_dry_masses[0]))):
        particle_dry_MassFracs[:,ii,:]=particle_dry_masses[:,ii,:]/total_dry_masses[:,ii,:]
    
    return particle_dry_MassFracs, dry_species

def Particle_Concentrations(particle_masses, particle_species, specdata_path='../species_data'):
    
    SpeciesData = particles.retrieve_one_species('H2O', specdata_path=specdata_path)
    H2O_idx = np.where(particle_species=='H2O')[0][0]
    water_volumes = particle_masses[:,:,H2O_idx]/SpeciesData.density
    concentrations=np.zeros(particle_masses.shape)
    concentrations[:]=np.nan
    for ii, (species) in enumerate(particle_species):
        try:
            SpeciesData = particles.retrieve_one_species(species, specdata_path=specdata_path)
            moles_x = particle_masses[:,:,ii]/SpeciesData.molar_mass
            concentrations[:,:,ii]=moles_x/water_volumes
        except Exception:
            pass
        
    return concentrations # mol/m^3


def NumFracs_MassFracs(particles, particle_species, splat_file, splat_species,
                        mass_thresholds, splat_cutoff=85, specdata_path='../species_data'):

    particle_classes = classify(particles, particle_species, splat_species, mass_thresholds, specdata_path=specdata_path) 
    
    # model number fraction
    N_idx = np.where(particle_species=='num conc')[0][0]
    particle_num_concs = particles[:, N_idx]
    Ddry_idx = np.where(particle_species=='Ddry')[0][0]
    particle_diameters = particles[:, Ddry_idx]
    number_fraction={}
    idx=np.where(particle_diameters*1e9>=splat_cutoff)    
    Ntot=np.sum(particle_num_concs[idx[0]])
    for spec in mass_thresholds.keys():
        idx=np.where(np.logical_and(particle_classes==spec, particle_diameters*1e9>=splat_cutoff))
        number_fraction[spec]=np.sum(particle_num_concs[idx[0]])/Ntot

    # model mass fraction
    masses={}
    for group in mass_thresholds.keys():
        masses[group]=0
    for group in mass_thresholds.keys():
        for species in mass_thresholds[group][1]:
            idx = np.where(particle_species==species)[0][0]
            masses[group]+=np.sum(particle_num_concs*particles[:, idx]) 
    total_mass=np.sum(list(masses.values()))
    mass_fraction={}
    for group in masses.keys():
        mass_fraction[group]=masses[group]/total_mass

    return number_fraction, mass_fraction


def ModelComposition(trajectory, mass_thresholds, splat_species, resolution=60):
    
    dt=trajectory['times'][1]-trajectory['times'][0]
    didx=int(resolution/dt)
    
    CDRs={}
    interstitials={}
    total_aerosol={}
    for spec in mass_thresholds.keys():
        interstitials[spec]=np.zeros(len(trajectory['times'][::didx]))
        CDRs[spec]=np.zeros(len(trajectory['times'][::didx]))
        total_aerosol[spec]=np.zeros(len(trajectory['times'][::didx]))
    
    # z=[]
    # t=[]
    # F_activated = []
    N_idx = np.where(trajectory['particle species']=='num conc')[0][0]
    print('plotting composition...')
    pbar = tqdm.tqdm(total = len(trajectory['times'][::didx]))
    for pState in range(0, len(trajectory['times']), didx):
        
        particle_classes = classify(trajectory['particles'][pState], trajectory['particle species'], splat_species, mass_thresholds)
        cloud_droplets = get_CD_status(trajectory['particles'][pState, :, np.where(trajectory['particle species']=='Dwet')[0][0]], 
                                       trajectory['particles'][pState, :, np.where(trajectory['particle species']=='Ddry')[0][0]], 
                                       trajectory['particles'][pState, :, np.where(trajectory['particle species']=='kappa')[0][0]],
                                       trajectory['particles'][pState, :, np.where(trajectory['particle species']=='num conc')[0][0]],
                                       trajectory['T'][pState])
        

        for ptype in mass_thresholds.keys():
            idx = np.where(np.logical_and(particle_classes==ptype, cloud_droplets>0)) # location of cloud droplets of each type
            CDRs[ptype][int(pState/didx)]+=np.sum(trajectory['particles'][pState, idx[0], N_idx])
            idx = np.where(np.logical_and(particle_classes==ptype, cloud_droplets<0)) # location of interstitials of each type
            interstitials[ptype][int(pState/didx)]+=np.sum(trajectory['particles'][pState, idx[0], N_idx])
            idx = np.where(particle_classes==ptype)
            total_aerosol[ptype][int(pState/didx)]=np.sum(trajectory['particles'][pState, idx[0], N_idx])
        
        
        Ntot_all = np.sum(np.array((list(total_aerosol.values())))[:, int(pState/didx)])
        Ntot_CDRs = np.sum(np.array((list(CDRs.values())))[:, int(pState/didx)])
        Ntot_interstitials = np.sum(np.array((list(interstitials.values())))[:, int(pState/didx)])
    
        for ptype in mass_thresholds.keys():
            CDRs[ptype][int(pState/didx)]/=Ntot_CDRs
            interstitials[ptype][int(pState/didx)]/=Ntot_interstitials
            total_aerosol[ptype][int(pState/didx)]/=Ntot_all

        pbar.update(1)
    pbar.close()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3.0*6.4, 1.0*4.8), sharey=True)
    
    # total aerosol
    t = trajectory['times'][::didx]
    bottom=np.zeros(len(t))
    for ptype, color in zip(splat_species.keys(), ['grey','gold','r','b','g','C6']):
        ax1.fill_between(t/60, bottom, bottom+total_aerosol[ptype], color=color)
        bottom+=total_aerosol[ptype]
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0,np.max(t)/60)
    ax1.set_xlabel('time (minutes)', labelpad=15)
    ax1.set_ylabel('fraction', labelpad=15)
    ax1.set_title('total aerosol', pad=15)
    ax1.plot(t/60, trajectory['activated fraction'][::didx], '-k', linewidth=2)
    
    # interstitials
    bottom=np.zeros(len(t))
    for ptype, color in zip(splat_species.keys(), ['grey','gold','r','b','g','C6']):
        ax2.fill_between(t/60, bottom, bottom+interstitials[ptype], color=color)
        bottom+=interstitials[ptype]
    ax2.set_xlim(0,np.max(t)/60)
    ax2.set_title('interstitials', pad=15)
    ax2.set_xlabel('time (minutes)', labelpad=15)
    ax2.plot(t/60, trajectory['activated fraction'][::didx], '-k', linewidth=2)
    
    # CDRs
    bottom=np.zeros(len(t))
    for ptype, color in zip(splat_species.keys(), ['grey','gold','r','b','g','C6']):
        ax3.fill_between(t/60, bottom, bottom+CDRs[ptype], color=color, label=ptype)
        bottom+=CDRs[ptype]
    ax3.set_xlim(0,np.max(t)/60)
    ax3.set_title('cloud droplet residuals', pad=15)
    ax3.set_xlabel('time (minutes)', labelpad=15)
    ax3.plot(t/60, trajectory['activated fraction'][::didx], '-k', linewidth=2, label='activated'+'\n'+'fraction')
    ax3.legend(loc='center', bbox_to_anchor=(1.2, 0.5)) 
            
    return fig

def MassFraction_TimeSeries(trajectory, mass_thresholds, ptype, splat_species, resolution=60):
    
    dt=trajectory['times'][1]-trajectory['times'][0]
    didx=int(resolution/dt)
    
    MassFrac_unactivated=np.zeros((len(trajectory['times'][::didx]), len(trajectory['particles'][0])))
    MassFrac_activated=np.zeros((len(trajectory['times'][::didx]), len(trajectory['particles'][0])))
    
    MassFrac_unactivated[:]=np.nan
    MassFrac_activated[:]=np.nan
    
    dry_species = []
    dry_idx = []
    for species in trajectory['particle species']:
        try: 
            SpeciesData = particles.retrieve_one_species(species)
            if SpeciesData.density > 0.0 and species != 'H2O':
                dry_species.append(species)
                dry_idx.append(np.where(trajectory['particle species']==species)[0][0])
        except:
            temp = 1
    
    dry_species = np.array(dry_species)
    particle_dry_masses = trajectory['particles'][:, :, dry_idx] # this pulls out only the aerosol species

    ptype_idx = []
    for spec in mass_thresholds[ptype][1]:
        idx = np.where(dry_species==spec)
        ptype_idx.append(idx[0][0])

    
    print('plotting '+ptype+' mass fractions...')
    pbar = tqdm.tqdm(total = len(trajectory['times'][::didx]))
    for pState in range(0, len(trajectory['times']), didx):
                        
        cloud_droplets = get_CD_status(trajectory['particles'][pState, :, np.where(trajectory['particle species']=='Dwet')[0][0]], 
                                        trajectory['particles'][pState, :, np.where(trajectory['particle species']=='Ddry')[0][0]], 
                                        trajectory['particles'][pState, :, np.where(trajectory['particle species']=='kappa')[0][0]],
                                        trajectory['particles'][pState, :, np.where(trajectory['particle species']=='num conc')[0][0]],
                                        trajectory['T'][pState])
        
        TotalMass = np.sum(particle_dry_masses[pState], axis=1, keepdims=True)
        MassFrac = particle_dry_masses[pState]/TotalMass
        ptype_MassFrac = np.sum(MassFrac[:, ptype_idx], axis=1)
        
        idx = np.where(cloud_droplets<0)[0]
        MassFrac_unactivated[int(pState/didx), idx]=ptype_MassFrac[idx]
        
        idx = np.where(cloud_droplets>0)[0]
        MassFrac_activated[int(pState/didx), idx]=ptype_MassFrac[idx]
        
        pbar.update(1)
    
    pbar.close()
                
    fig, (ax) = plt.subplots(1, 1, figsize=(1.0*6.4, 1.0*4.8))
    ax.plot(trajectory['times'][::didx]/60, MassFrac_unactivated, '-r', linewidth=0.5)
    ax.plot(trajectory['times'][::didx]/60, MassFrac_activated, '-b', linewidth=0.5)
    ax.set_xlim(0,np.max(trajectory['times'])/60)
    ax.set_ylim(0,1)
    ax.set_xlabel('time (minutes)', labelpad=15)
    ax.set_ylabel(ptype+' mass fraction', labelpad=15)
    
    return fig
    
def VerticalComposition(trajectory, mass_thresholds, splat_species, splat_file,
                        aimms_file, bins=10):
    
    # read miniSPLAT data from file
    filename = splat_file
    raw_data = np.loadtxt(filename, dtype='str')
    full_SPLAT_data = {}
    SPLAT_subdata = {}
    for i in range(0, len(raw_data[0])): 
        full_SPLAT_data[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
        SPLAT_subdata[str(raw_data[0, i])] = np.zeros(0)

    # find the full time series of number fraction for each class
    minisplat_fraction = {}
    for reduced_species in splat_species.keys():
        summation = np.zeros(len(full_SPLAT_data['Time']))
        for species in splat_species[reduced_species]:
            for i in range(0, len(summation)):
                summation[i] = summation[i] + full_SPLAT_data[species][i]
        minisplat_fraction[reduced_species] = summation
        
    # read AIMMS_file data from file
    filename = aimms_file
    raw_data = np.loadtxt(filename, delimiter = ',', skiprows = 53, dtype='str')
    aimms_data = {}
    for i in range(0, len(raw_data[0])): 
        aimms_data[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
      
    # get the altitude of minisplat measurements
    minisplat_fraction['altitude']=np.zeros(len(full_SPLAT_data['Time']))
    minisplat_fraction['S']=np.zeros(len(full_SPLAT_data['Time']))
    for ii, (time) in enumerate(full_SPLAT_data['Time']):
        idx = np.argmin(abs(aimms_data['Time(UTC)']-time))
        minisplat_fraction['altitude'][ii]=aimms_data['Alt'][idx]
        minisplat_fraction['S'][ii]=aimms_data['Humidity'][idx]

    # set up modeled value arrays
    N_idx=np.where(trajectory['particle species']=='num conc')[0][0]
    NumFractions_all={}
    modeled_avg_composition={}
    measured_avg_composition={}
    NumFractions_all['S']=np.zeros(len(trajectory['times']))
    for ptype in mass_thresholds.keys():
        NumFractions_all[ptype]=np.zeros(len(trajectory['times']))
        modeled_avg_composition[ptype]=np.zeros(bins)
        measured_avg_composition[ptype]=np.zeros(bins)
    measured_avg_composition['S']=np.zeros(bins)
    modeled_avg_composition['S']=np.zeros(bins)
    
    print('getting vertical number fractions...')
    pbar = tqdm.tqdm(total = len(trajectory['times']))
    for pState in range(0, len(trajectory['times'])):
        NumFractions_all['S'][pState]=trajectory['S'][pState]
        particle_classes = classify(trajectory['particles'][pState], trajectory['particle species'], splat_species, mass_thresholds)
        for ptype in mass_thresholds.keys():
            idx = np.where(particle_classes==ptype)[0]
            NumFractions_all[ptype][pState]=np.sum(trajectory['particles'][pState, idx, N_idx])/np.sum(trajectory['particles'][pState, :, N_idx])
        pbar.update(1)
    pbar.close()

    # set up dict for coarse z-grid values
    z_grid_coarse = np.linspace(np.min(trajectory['z']), np.max(trajectory['z']), bins+1)
    heights = z_grid_coarse[1:]-z_grid_coarse[:-1]
    z_mids = z_grid_coarse[:-1]+0.5*heights
    modeled_composition_coarse = {}
    measured_composition_coarse = {}
    for ptype in mass_thresholds.keys():
        modeled_composition_coarse[ptype]=list(np.zeros(len(z_grid_coarse)-1))
        measured_composition_coarse[ptype]=list(np.zeros(len(z_grid_coarse)-1))
    measured_composition_coarse['S']=list(np.zeros(len(z_grid_coarse)-1))
    modeled_composition_coarse['S']=list(np.zeros(len(z_grid_coarse)-1))
        
    # get values within the grid points
    for rr in range(1, len(z_grid_coarse)):
        model_idx, = np.nonzero((trajectory['z']>z_grid_coarse[rr-1]) & (trajectory['z']<=z_grid_coarse[rr]))
        measured_idx, = np.nonzero((minisplat_fraction['altitude']>z_grid_coarse[rr-1]) & (minisplat_fraction['altitude']<=z_grid_coarse[rr]))
        measured_composition_coarse['S'][rr-1]=minisplat_fraction['S'][measured_idx]
        modeled_composition_coarse['S'][rr-1]=NumFractions_all['S'][model_idx]
        for ptype in mass_thresholds.keys():
            modeled_composition_coarse[ptype][rr-1]=NumFractions_all[ptype][model_idx]
            measured_composition_coarse[ptype][rr-1]=minisplat_fraction[ptype][measured_idx]
    
    # average in each grid point
    for ii in range(0, bins):
        if len(measured_composition_coarse['S'][ii])>0:
            measured_avg_composition['S'][ii]=np.mean(measured_composition_coarse['S'][ii])
        if len(modeled_composition_coarse['S'][ii])>0:
            modeled_avg_composition['S'][ii]=np.mean(modeled_composition_coarse['S'][ii])
        for ptype in mass_thresholds.keys():
            if len(modeled_composition_coarse[ptype][ii])>0:
                modeled_avg_composition[ptype][ii]=np.mean(modeled_composition_coarse[ptype][ii])
            if len(measured_composition_coarse[ptype][ii])>0:
                measured_avg_composition[ptype][ii]=np.mean(measured_composition_coarse[ptype][ii])
      
    # plot
    fig, (meas_ax, mod_ax) = plt.subplots(1, 2, figsize=(2.0*6.4, 1.0*4.8), sharey=True)    
    
    meas_S_ax = meas_ax.twiny()
    meas_S_ax.plot(measured_avg_composition['S'], z_mids, '-k')
    
    mod_S_ax = mod_ax.twiny()
    mod_S_ax.plot(modeled_avg_composition['S'], z_mids, '-k')
    
    mod_left=np.zeros(bins)
    meas_left=np.zeros(bins)
    for ii, (t, c, l) in enumerate(zip(splat_species.keys(), ['grey','gold','r','b','g','C6'], ['black carbon', 'dust', 'sulfate rich', 'nitrate rich', 'organics', 'IEPOX SOA'])):
        mod_ax.barh(z_mids, modeled_avg_composition[t], 
                    height=heights, left=mod_left, facecolor=c, 
                    edgecolor='k', label=l, align='center')
        mod_left+=modeled_avg_composition[t]
        
        meas_ax.barh(z_mids, measured_avg_composition[t], 
                    height=heights, left=meas_left, facecolor=c, 
                    edgecolor='k', label=l, align='center')
        meas_left+=measured_avg_composition[t]
    
    
    meas_S_ax.set_xlim(0, 1.2)
    mod_S_ax.set_xlim(0, 1.2)
    meas_ax.set_xlim(0,1)
    meas_S_ax.set_xlabel('saturation ratio', labelpad=15)
    mod_S_ax.set_xlabel('saturation ratio', labelpad=15)
    meas_ax.set_xlabel('number fraction', labelpad=15)
    mod_ax.set_xlabel('number fraction', labelpad=15)
    meas_ax.set_ylabel('altitude (m)', labelpad=15)
    mod_ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)
    
    return fig

    

def classify(particle_masses, particle_species, splat_species, mass_thresholds,
             specdata_path='../species_data',
             reassignment=False):
    
    """
    Parameters
    ----------
    particle_masses : array
        N x M numpy array containing the mass of species 
        in each particle. N is the number of particles 
        and M is the number of chemical species.
    particle_species : array or list
        Name of species corresponding to each column 
        in particle_masses.
    splat_species : dict
        Dict of miniSPLAT species which belong 
        to each particle type. For example, grouping 
        of all organics in one class.
    mass_thresholds : dict
        Minimum, average, and standard deviation for mass 
        fraction of species necessary for classification. 
        Each dict element must have format 
        [[minimum mass fraction, average mass fraction, 
          standard deviation of mass fraction], 
         [species 1 included, species 2 included, ...]].
    specdata_path : string, optional
        Where aerosol species informatio is stored. 
        Default is ../species_data.
    reassignment : boolean
        Determines classification of particles which do not 
        meet the minimum mass threshold of any class. If 
        true, particles are assigned to the class for which 
        they have the highest mass fraction. If false, 
        these particles are assigned class of 'none'.

    Returns
    -------
    classes : array
        N-shaped array of particle classes.
    """
    
    dry_species = []
    dry_idx = []
    for species in particle_species:
        try: 
            SpeciesData = particles.retrieve_one_species(species, specdata_path=specdata_path)
            if SpeciesData.density > 0.0 and species != 'H2O':
                dry_species.append(species)
                dry_idx.append(np.where(particle_species==species)[0][0])
        except:
            temp = 1
    
    dry_species = np.array(dry_species)
    particle_dry_masses = particle_masses[:, dry_idx] # this pulls out only the aerosol species
    
    classes=[]
    for masses in particle_dry_masses:
        ptype=None
        mass_fraction = masses/np.sum(masses)
        
        if all_real(mass_fraction):        
            for pclass in mass_thresholds.keys():
                mass=0
                for spec in mass_thresholds[pclass][1]:
                    idx = np.where(dry_species==spec)
                    mass+=mass_fraction[idx[0][0]]
                if mass>=mass_thresholds[pclass][0][0]:
                    ptype=pclass
                    break

            if not ptype:
                if reassignment:
                    difference={}
                    for pclass in mass_thresholds.keys():
                        mass=0
                        for spec in mass_thresholds[pclass][1]:
                            idx = np.where(dry_species==spec)
                            mass+=abs(mass_fraction[idx[0][0]])
                        difference[pclass]=mass_thresholds[pclass][0][0]-mass
                    ptype=min(difference, key=difference.get)
                else:
                    ptype='none'
            
        classes.append(ptype)

    return np.array((classes))
    
def all_real(mass_fractions):
  return all(x >= 0 for x in abs(mass_fractions))

def get_CD_status(wet_diameters, dry_diameters, kappas, Ns, T):
    
    CD_status=np.zeros(len(wet_diameters))
    for ii, (Ddry, Dwet, kappa, N) in enumerate(zip(dry_diameters, wet_diameters, kappas, Ns)):
        r=Dwet/2.
        r_dry=Ddry/2.
        neg_Seq = lambda r: -1.0 * water_uptake.Seq(r, r_dry, T, kappa)
        out = fminbound(neg_Seq, r_dry, r_dry * 1e4, xtol=1e-10, full_output=True, disp=0)
        r_crit, s_crit = out[:2]
        s_crit *= -1.0  # multiply by -1 to undo negative flag for Seq
        if r >= r_crit:
            CD_status[ii]=N 
        else:
            CD_status[ii]=-1.0*N 
    return CD_status # negative values are -1*number concentration for interstitials, positive values are number concentration of cloud droplet residuals
    
    
def miniSPLAT_CloudComposition(splat_file, size_distribution_file, splat_species):
    
    # read miniSPLAT data from file
    filename = splat_file
    raw_data = np.loadtxt(filename, dtype='str')
    full_SPLAT_data = {}
    # SPLAT_subdata = {}
    for i in range(0, len(raw_data[0])):
        full_SPLAT_data[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
        # SPLAT_subdata[str(raw_data[0, i])] = np.zeros(0)
    
    # pull out the miniSPLAT data corresponding to initialization times
    # idx = np.where(np.logical_and(full_SPLAT_data['Time']>0, full_SPLAT_data['Time']<=86400))
    # for key in full_SPLAT_data.keys():
    #     SPLAT_subdata[key] = np.array((full_SPLAT_data[key][idx[0]]))
    
    # create dict of all the species in the miniSPLAT data
    minisplat_species = []
    for key in full_SPLAT_data.keys():
        # if key != 'Time':
        minisplat_species.append(key)

    # find the full time series of number fraction for each class
    minisplat_fraction = {}
    minisplat_fraction['Time']=full_SPLAT_data['Time']
    for reduced_species in splat_species.keys():
        summation = np.zeros(len(full_SPLAT_data['Time']))
        for species in splat_species[reduced_species]:
            for i in range(0, len(summation)):
                summation[i] = summation[i] + full_SPLAT_data[species][i]
        minisplat_fraction[reduced_species] = summation
    
    # read FIMS data from file
    filename = size_distribution_file
    raw_data = np.loadtxt(filename, delimiter = ',', dtype='str', skiprows = 100)
    for i in range(0, len(raw_data)):
        for j in range(0, len(raw_data[0])):
            raw_data[i, j] = raw_data[i, j].strip()
    FIMS_data = {}
    FIMS_data[str(raw_data[0, 0])] = np.array(raw_data[1:, 0], dtype = 'float64')
    for i in range(56, len(raw_data[0])):
        FIMS_data[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
    FIMS_data['N_Dp'] = np.array(raw_data[1:, 1:56], dtype = 'float64')

    # get splat data only in cloud and CVI inlet
    CDR_idx = np.where(np.logical_and(FIMS_data['Cloud_flag']==1.0, FIMS_data['CVI_flag']==1.0))
    CDR_times = FIMS_data['Start_UTC'][CDR_idx[0]] 
    interstitial_idx = np.where(np.logical_and(FIMS_data['Cloud_flag']==1.0, FIMS_data['CVI_flag']==0.0))
    interstitial_times = FIMS_data['Start_UTC'][interstitial_idx[0]] 
    
    CDRs = {}
    interstitials = {}
    for species in splat_species.keys():
        CDRs[species]=[]
        interstitials[species]=[]
    
    for t in CDR_times:
        idx = np.where(np.round(minisplat_fraction['Time'],0)==np.round(t,0))[0]
        if len(idx)>0:
            for species in splat_species.keys():
                CDRs[species].append(minisplat_fraction[species][idx[0]])
    for species in splat_species.keys():
        CDRs[species]=np.array(CDRs[species])
        
    for t in interstitial_times:
        idx = np.where(np.round(minisplat_fraction['Time'],0)==np.round(t,0))[0]
        if len(idx)>0:
            for species in splat_species.keys():
                interstitials[species].append(minisplat_fraction[species][idx[0]])
    for species in splat_species.keys():
        interstitials[species]=np.array(interstitials[species])
    
    return CDRs, interstitials
 

    
        
        
        
