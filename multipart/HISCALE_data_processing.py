#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:55:53 2024

@author: beel083
"""
import numpy as np
import matplotlib.pyplot as plt
from initialization import read_FIMS, splat_number_fractions, ams_mass_fraction
import sys, tqdm
from processes import water_uptake
from scipy.optimize import fminbound

def initial_SizeDist(trajectory_ensemble, size_distribution_file, splat_species,
                     mass_thresholds, start_time=None, end_time=None):
        
    Dp_lowers, Dp_uppers, N_measured, N_error = read_FIMS(size_distribution_file, 
                                                  start_time, end_time, 0.0, 
                                                  0.0) # diameters in nm and N in #/cm^3 
    
    bins=Dp_uppers[::2]
    histogram_Dps = bins[:-1]
    histogram_Ns={}
    for t in splat_species.keys():
        histogram_Ns[t]=np.zeros((len(trajectory_ensemble), len(histogram_Dps))) 
    
    print('reading initial size distributions...')
    pbar = tqdm.tqdm(total = len(trajectory_ensemble))
    for ii, (trajectory) in enumerate(trajectory_ensemble):
        particle_population = trajectory.parcel_states[0].particle_population
        particle_classes = classify(particle_population, splat_species, mass_thresholds)
        particle_diameters = np.zeros(len(particle_population.particles))
        particle_num_concs = np.zeros(len(particle_population.particles))
        for jj,(particle,num_conc) in enumerate(zip(particle_population.particles,particle_population.num_concs)):
            d_dry = particle.get_Ddry()
            particle_diameters[jj]=d_dry
            particle_num_concs[jj]=num_conc
        for t in splat_species.keys():
            idx=np.where(np.array([particle_classes])==t)
            hist=np.histogram(1e9*particle_diameters[idx[1]], bins=bins, weights=18.10*particle_num_concs[idx[1]]/100**3)   
            histogram_Ns[t][ii]=hist[0]
        pbar.update(1)
    pbar.close()
    
    fig, ax = plt.subplots(1, 1)
    bottom=np.zeros(len(bins)-1)
    yerr=np.zeros(len(bins)-1)
    hist=np.histogram(Dp_uppers, bins=bins, weights=N_measured*18.10)
    widths=hist[1][1:]-hist[1][:-1]
    ax.plot(hist[1][:-1], hist[0], '-ko', label='measured')
    for ii, (t, c) in enumerate(zip(splat_species.keys(), ['grey','gold','r','b','g','C6'])):
        if ii == len(splat_species.keys())-1:
            yerr+=np.std(histogram_Ns[t], axis=0)
            ax.bar(histogram_Dps, np.average(histogram_Ns[t], axis=0), 
                    width=widths, bottom=bottom, facecolor=c, edgecolor='k', 
                    label=t, align='edge', yerr=yerr)
            bottom+=np.average(histogram_Ns[t], axis=0)
        else:
            yerr+=np.std(histogram_Ns[t], axis=0)
            ax.bar(histogram_Dps, np.average(histogram_Ns[t], axis=0), 
                    width=widths, bottom=bottom, facecolor=c, edgecolor='k', 
                    label=t, align='edge')
            bottom+=np.average(histogram_Ns[t], axis=0)
            
        
    ax.set_ylabel('dN/dlog($d_p$)', labelpad=15)
    ax.set_xlabel('dry diameter (nm)', labelpad=15)
    ax.set_xscale('log')
    ax.set_ylim(0,)
    ax.legend()
    
    return fig


def cloud_composition(trajectory_ensemble, splat_file, splat_species, size_distribution_file,
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
    
    
    
    # get the measured values
    
    dt=trajectory_ensemble[0].ts[1]-trajectory_ensemble[0].ts[0]
    didx=int(resolution/dt)
    
    modeled_CDRs = {}
    for ptype in splat_species.keys():
        modeled_CDRs[ptype]=np.zeros((len(trajectory_ensemble), len(trajectory_ensemble[0].parcel_states)))
    for ptype in splat_species.keys():
        modeled_CDRs[ptype][:]=np.nan
        
    modeled_interstitials = {}
    for ptype in splat_species.keys():
        modeled_interstitials[ptype]=np.zeros((len(trajectory_ensemble), len(trajectory_ensemble[0].parcel_states)))
    for ptype in splat_species.keys():
        modeled_interstitials[ptype][:]=np.nan
     
        
    print('plotting cloud composition...')
    pbar = tqdm.tqdm(total = len(trajectory_ensemble)*len(trajectory_ensemble[0].parcel_states[::didx]))
    for ii in range(len(trajectory_ensemble)):
        
        trajectory = trajectory_ensemble[ii]
        # temp_CDRs={}
        # for ptype in splat_species.keys():
        #     temp_CDRs[ptype]=np.zeros(0)
        # temp_interstitials={}
        # for ptype in splat_species.keys():
        #     temp_interstitials[ptype]=np.zeros(0)
        
        for jj in range(0, len(trajectory.parcel_states), didx):
            F_activated = trajectory.parcel_states[jj].get_activated_fraction()
            if F_activated>0.0:
                particle_population=trajectory.parcel_states[jj].particle_population
                particle_classes = classify(particle_population, splat_species, mass_thresholds)
                cloud_droplets = get_CD_status(particle_population, trajectory.parcel_states[jj].T)
                idx = np.where(cloud_droplets>0.0)
                Ntot_activated = np.sum(cloud_droplets[idx[0]])
                idx = np.where(cloud_droplets<0.0)
                Ntot_unactivated = -1.0*np.sum(cloud_droplets[idx[0]])
                for ptype in splat_species.keys():
                    idx = np.where(np.logical_and(np.array((particle_classes))==ptype, cloud_droplets>0.0))
                    modeled_CDRs[ptype][ii,jj]=np.sum(cloud_droplets[idx[0]])/Ntot_activated
                    idx = np.where(np.logical_and(np.array((particle_classes))==ptype, cloud_droplets<0.0))
                    modeled_interstitials[ptype][ii,jj]=(-1.0*np.sum(cloud_droplets[idx[0]]))/Ntot_unactivated  
                 
            pbar.update(1)
    
    pbar.close()

    print()
    for k in modeled_CDRs.keys():
        print(k, modeled_CDRs[k].shape)
    print()



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
        
        temp_avg=np.zeros(len(modeled_CDRs[ptype]))
        temp_error=np.zeros(len(modeled_CDRs[ptype]))
        for jj in range(len(trajectory_ensemble)):
            temp_avg[jj]=np.nanmean(modeled_CDRs[ptype][jj])
            temp_error[jj]=np.nanstd(modeled_CDRs[ptype][jj])
        modeled_values[ii]=np.nanmean(temp_avg)
        modeled_errors[ii]=np.sqrt(np.nansum((temp_error/temp_avg)**2))/float(len(np.where(temp_error>=0.0)[0]))

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
        
        temp_avg=np.zeros(len(modeled_interstitials[ptype]))
        temp_error=np.zeros(len(modeled_interstitials[ptype]))
        for jj in range(len(trajectory_ensemble)):
            temp_avg[jj]=np.nanmean(modeled_interstitials[ptype][jj])
            temp_error[jj]=np.nanstd(modeled_interstitials[ptype][jj])
        modeled_values[ii]=np.nanmean(temp_avg)
        modeled_errors[ii]=np.sqrt(np.nansum((temp_error/temp_avg)**2))/float(len(np.where(temp_error>=0.0)[0]))
    
    ax2.bar(locations-0.3, measured_values, width=0.3, align='edge', color='grey', yerr=measured_errors, edgecolor='k', label='measured')
    ax2.bar(locations, modeled_values, width=0.3, align='edge', color='w', yerr=modeled_errors, edgecolor='k', label='modeled')
    ax2.set_xticks(locations)
    ax2.set_xticklabels(measured_interstitials.keys())
    ax2.set_title('interstitials', pad=15)
    
    ax1.set_ylabel('number fraction', labelpad=15)
    ax1.set_ylim(0,1)
    ax2.legend()
    
    return fig
    
    


def initial_N_MassFracs(trajectory, ams_file, splat_file, splat_species,
                        mass_thresholds, start_time=None, end_time=None,
                        splat_cutoff=85):
    
    ams_mass_fractions, ams_mass_fraction_error, measured_total_mass, measured_total_mass_error = ams_mass_fraction(ams_file, start_time, end_time)       
    splat_number_fraction, splat_number_fraction_error = splat_number_fractions(splat_file, splat_species, start_time, end_time)   
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.0*6.4, 1.0*4.8), sharey=True)
    widths=0.4
    
    # measured values
    ax2_positions=np.arange(0, len(ams_mass_fractions.keys()), 1)
    ax2.bar(ax2_positions-widths, ams_mass_fractions.values(), yerr=ams_mass_fraction_error.values(), width=widths, facecolor='grey', edgecolor='k', label='measured', align='edge')
    ax1_positions=np.arange(0, len(splat_number_fraction.keys()), 1)
    ax1.bar(ax1_positions-widths, splat_number_fraction.values(), yerr=splat_number_fraction_error.values(), width=widths, facecolor='grey', edgecolor='k', align='edge')
    
    # modeled number fraction
    particle_population = trajectory.parcel_states[0].particle_population
    particle_classes = classify(particle_population, splat_species, mass_thresholds) 
    particle_num_concs = np.zeros(len(particle_population.particles))
    particle_diameters = np.zeros(len(particle_population.particles))
    for ii,(particle,num_conc) in enumerate(zip(particle_population.particles,particle_population.num_concs)):
        d_dry = particle.get_Ddry()
        particle_num_concs[ii]=num_conc
        particle_diameters[ii]=d_dry
    number_fraction={}
    idx=np.where(np.array((particle_diameters))*1e9>=splat_cutoff)
    Ntot=np.sum(particle_num_concs[idx[0]])
    for spec in mass_thresholds.keys():
        idx=np.where(np.logical_and(np.array((particle_classes))==spec, np.array((particle_diameters))*1e9>=splat_cutoff))
        number_fraction[spec]=np.sum(particle_num_concs[idx[0]])/Ntot

    # modeled mass fraction
    masses={}
    for species in ams_mass_fractions.keys():
        masses[species]=0
    for ii,(particle,num_conc) in enumerate(zip(particle_population.particles,particle_population.num_concs)):
        for species in ams_mass_fractions.keys():
            if type(particle.get_species_idx(species))==int:
                masses[species]+=particle_num_concs[ii]*particle.masses[particle.get_species_idx(species)]        
    total_mass=0
    for group in masses.keys():
        total_mass+=np.sum(np.array((masses[group])))
    mass_fraction={}
    for group in masses.keys():
        mass_fraction[group]=masses[group]/total_mass
        
    for spec, pos in zip(splat_number_fraction.keys(), ax1_positions):
        ax1.bar(pos, number_fraction[spec], width=widths, facecolor='w', edgecolor='k', align='edge')
    for spec, pos in zip(ams_mass_fractions.keys(), ax2_positions):
        if pos==0:
            ax2.bar(pos, mass_fraction[spec], width=widths, facecolor='w', edgecolor='k', align='edge', label='modeled')
        else:
            ax2.bar(pos, mass_fraction[spec], width=widths, facecolor='w', edgecolor='k', align='edge')

    ax1.set_title('number', pad=15)
    ax2.set_title('mass', pad=15)
    ax2.set_ylim(0, 1)
    ax1.set_ylabel('fraction', labelpad=15)
    ax1.set_xticks(ax1_positions)
    ax1.set_xticklabels(splat_number_fraction.keys())
    ax2.set_xticks(ax2_positions)
    ax2.set_xticklabels(ams_mass_fractions.keys())
    ax2.legend()
    
    return fig

# def composition_barcharts(trajectory_ensemble, resolution=60):



def ModelComposition(trajectory, mass_thresholds, splat_species, resolution=60):
    
    dt=trajectory.ts[1]-trajectory.ts[0]
    didx=int(resolution/dt)
    
    CDRs={}
    interstitials={}
    total_aerosol={}
    for spec in mass_thresholds.keys():
        interstitials[spec]=np.zeros(len(trajectory.parcel_states[::didx]))
        CDRs[spec]=np.zeros(len(trajectory.parcel_states[::didx]))
        total_aerosol[spec]=np.zeros(len(trajectory.parcel_states[::didx]))
    
    z=[]
    t=[]
    F_activated = []
    print('plotting composition...')
    pbar = tqdm.tqdm(total = len(trajectory.parcel_states[::didx]))
    for ii in range(0, len(trajectory.parcel_states), didx):
        particle_population = trajectory.parcel_states[ii].particle_population
        z.append(trajectory.parcel_states[ii].z)
        t.append(trajectory.ts[ii])
        F_activated.append(trajectory.parcel_states[ii].get_activated_fraction())
        
        particle_classes = classify(particle_population, splat_species, mass_thresholds)
        for jj,(particle,num_conc, ptype) in enumerate(zip(particle_population.particles,particle_population.num_concs, particle_classes)):
            r=particle.get_Dwet()/2.
            r_dry=particle.get_Ddry()/2.
            kappa=particle.get_tkappa()
            neg_Seq = lambda r: -1.0 * water_uptake.Seq(r, r_dry, trajectory.parcel_states[ii].T, kappa)
            out = fminbound(neg_Seq, r_dry, r_dry * 1e4, xtol=1e-10, full_output=True, disp=0)
            r_crit, s_crit = out[:2]
            s_crit *= -1.0  # multiply by -1 to undo negative flag for Seq
            if r>=r_crit:
                CDRs[ptype][int(ii/didx)]+=num_conc
            else:
                interstitials[ptype][int(ii/didx)]+=num_conc
        pbar.update(1)
    pbar.close()
    
    t=np.array((t))
    z=np.array((z))
    F_activated=np.array((F_activated))
    
    for ii in range(len(trajectory.parcel_states[::didx])):
        CDR_total=0
        int_total=0
        for spec in mass_thresholds.keys():
            CDR_total+=CDRs[spec][ii]
            int_total+=interstitials[spec][ii]
        for spec in mass_thresholds.keys():
            total_aerosol[spec][ii]=(CDRs[spec][ii]+interstitials[spec][ii])/(int_total+CDR_total)
            if CDR_total>0.0:
                CDRs[spec][ii]/=CDR_total
            else:
                CDRs[spec][ii]=0.0
            interstitials[spec][ii]/=int_total
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3.0*6.4, 1.0*4.8), sharey=True)
    
    # total aerosol
    bottom=np.zeros(len(trajectory.parcel_states[::didx]))
    for ptype, color in zip(splat_species.keys(), ['grey','gold','r','b','g','C6']):
        ax1.fill_between(t/60, bottom, bottom+total_aerosol[ptype], color=color)
        bottom+=total_aerosol[ptype]
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0,np.max(t)/60)
    ax1.set_xlabel('time (minutes)', labelpad=15)
    ax1.set_ylabel('fraction', labelpad=15)
    ax1.set_title('total aerosol', pad=15)
    ax1.plot(t/60, F_activated, '-k', linewidth=2)
    
    # interstitials
    bottom=np.zeros(len(trajectory.parcel_states[::didx]))
    for ptype, color in zip(splat_species.keys(), ['grey','gold','r','b','g','C6']):
        ax2.fill_between(t/60, bottom, bottom+interstitials[ptype], color=color)
        bottom+=interstitials[ptype]
    ax2.set_xlim(0,np.max(t)/60)
    ax2.set_title('interstitials', pad=15)
    ax2.set_xlabel('time (minutes)', labelpad=15)
    ax2.plot(t/60, F_activated, '-k', linewidth=2)
    
    # CDRs
    bottom=np.zeros(len(trajectory.parcel_states[::didx]))
    for ptype, color in zip(splat_species.keys(), ['grey','gold','r','b','g','C6']):
        ax3.fill_between(t/60, bottom, bottom+CDRs[ptype], color=color, label=ptype)
        bottom+=CDRs[ptype]
    ax3.set_xlim(0,np.max(t)/60)
    ax3.set_title('cloud droplet residuals', pad=15)
    ax3.set_xlabel('time (minutes)', labelpad=15)
    ax3.plot(t/60, F_activated, '-k', linewidth=2, label='activated'+'\n'+'fraction')
    ax3.legend(loc='center', bbox_to_anchor=(1.2, 0.5)) 
            
    return fig



def classify(particle_population, splat_species, mass_thresholds):
    
    classes=[]
    for particle in particle_population.particles:
        ptype=None
        mass_fraction = get_mass_fraction(particle)  
        for ii in mass_thresholds.keys():
            mass=0
            for jj in mass_thresholds[ii][1]:
                if jj in mass_fraction.keys():
                    mass+=mass_fraction[jj]
            if mass>=mass_thresholds[ii][0][0]:
                ptype=ii
        if not ptype:
            difference={}
            for ii in mass_thresholds.keys():
                mass=0
                for jj in mass_thresholds[ii][1]:
                    if jj in mass_fraction.keys():
                        mass+=mass_fraction[jj]
                difference[ii]=mass_thresholds[ii][0][0]-mass
            ptype=min(difference, key=difference.get)
        classes.append(ptype)
    return classes
    
    

def get_mass_fraction(particle):
    
    total_mass=0
    mass_fraction={}
    for spec, mass in zip(particle.species, particle.masses):
        if spec.density > 0 and spec.name != 'H2O':
            mass_fraction[spec.name]=mass
            total_mass+=mass
    
    for spec in mass_fraction.keys():
        mass_fraction[spec]/=total_mass
    
    return mass_fraction
    

def get_CD_status(particle_population, T):
    
    CD_status=np.zeros(len(particle_population.particles))
    for ii, (particle, num_conc) in enumerate(zip(particle_population.particles, particle_population.num_concs)):
        r=particle.get_Dwet()/2.
        r_dry=particle.get_Ddry()/2.
        kappa=particle.get_tkappa()
        neg_Seq = lambda r: -1.0 * water_uptake.Seq(r, r_dry, T, kappa)
        out = fminbound(neg_Seq, r_dry, r_dry * 1e4, xtol=1e-10, full_output=True, disp=0)
        r_crit, s_crit = out[:2]
        s_crit *= -1.0  # multiply by -1 to undo negative flag for Seq
        if r >= r_crit:
            CD_status[ii]=num_conc 
        else:
            CD_status[ii]=-1.0*num_conc 
    return CD_status # negative values are -1*number concentration for interstitials, positive values are number concentration of cloud droplet residuals
    

    
        
        
        
