#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:06:32 2024

@author: beel083
"""
import os

import numpy as np
import sys
import matplotlib.pyplot as plt
from scenario import get_aero_spec_fracs
from scipy.optimize import curve_fit
#import tqdm
from particles import make_particle
from scipy.special import erf
import pickle, shutil

def splat_setup(Npart=1, optimization_points=10000, mass_thresholds=None,
                size_distribution_file=None, splat_file=None, trace_gas_folder=None,
                splat_species=None, mass_fractions=None,ams_file=None,aimms_file=None, 
                les_path='../datasets/parcel_traces_0425_15utc', les_number='000000', dz=2.0,
                gas_names=None, specdata_path='../species_data/',
                override_matching=False, splat_cutoff=85):
    
    # diameters=np.zeros(Npart)
    if not les_path or not les_number:
        print('WARNING: Need path for the LES file!')
        sys.exit()
    if not size_distribution_file:
        print('WARNING: Need size distribution file!')
        sys.exit()
    if not ams_file:
        print('WARNING: Need AMS file!')
        sys.exit()
    if not size_distribution_file or not splat_species:
        print('WARNING: Need size SPLAT file!')
        sys.exit()
    if not mass_thresholds:
        print('WARNING: Need mass threshold information!')
        sys.exit()
    if gas_names and not trace_gas_folder:
        print('WARNING: Need to specify where the gas information is stored!')
        sys.exit()
    
    # read LES data from file   
    # les_data = pickle.load(open(les_path+'/parcel_traces_'+les_number+'.pkl', 'rb'))    
    z = 100.0 #les_data['z'][0] # m
    
    mode_fractions, fitting_params, N_multiplier=optimize_splat_size_distribution(datapoints=optimization_points,
                        size_distribution_file=size_distribution_file, 
                        ams_file=ams_file, splat_file=splat_file,
                        aimms_file=aimms_file,
                        mass_thresholds=mass_fractions,
                        splat_species=splat_species, splat_cutoff=splat_cutoff,
                        specdata_path=specdata_path, z=z, dz=dz)  
    print()
    print('Fitted size distribution:')
    print(fitting_params)
    print()
    print('Optimized number mode fractions:')
    for spec in mode_fractions:
        print(spec, mode_fractions[spec])
        modes=int(len(mode_fractions[spec]))
    print()
    
    # read in the measured data
    Dp_lowers, Dp_uppers, measured_N, N_error = read_FIMS(size_distribution_file, aimms_file, z, dz) # diameters in nm and N in #/cm^3
    
    idx=np.where(Dp_uppers<=1000)[0]
    Dp_lowers=Dp_lowers[idx]
    Dp_uppers=Dp_uppers[idx]
    measured_N=measured_N[idx]
    N_error=N_error[idx]
    Dp_mids = Dp_lowers + 0.5*(Dp_uppers - Dp_lowers)
    
    # measured_Ntot=np.sum(measured_N[idx[0]])
    avg_number_fraction, number_fraction_error = splat_number_fractions(splat_file, aimms_file, size_distribution_file, splat_species, z, dz)    
    ams_mass_fractions, ams_mass_fraction_error, measured_total_mass, measured_total_mass_error = ams_mass_fraction(ams_file, aimms_file, size_distribution_file, z, dz)       
    checks=[False]
        
    counter = 0
    maxcounter=100
    print('sampling', Npart, 'particles...')
    #pbar = tqdm.tqdm(total = maxcounter)
    while (sum(checks)!=len(checks) and counter<maxcounter):
        # ====================================================================
        
        # sample which particles are which
        particle_species=[]
        for species in avg_number_fraction.keys():
            particle_species.append(species)

        rand=np.floor(len(avg_number_fraction.keys())*np.random.rand(Npart))
        ptypes=['None']*Npart    
        for ii in range(len(particle_species)):
            idx=np.where(rand==ii)
            if len(idx[0])>0:
                for jj in idx[0]:
                    ptypes[jj]=particle_species[ii]
                
        # get the size distribution for each type
        particle_diameters=np.zeros(Npart)
        particle_num_concs=np.zeros(Npart)
        total_SizeDist=np.zeros(len(Dp_mids))
        for ii in range(len(particle_species)):
            
            particle_type=particle_species[ii]
            if particle_type=='BC':
                Dpg = 110
                sigma = 1.6
                params=[avg_number_fraction['BC'], Dpg, sigma]
                SizeDist=N_multiplier*size_dependent_composition(Dp_mids, measured_N, 1,
                                                        splat_cutoff, 
                                                        avg_number_fraction[particle_type], 
                                                        [1.0], params)
                total_SizeDist+=SizeDist
            elif particle_type=='OIN':
                Dpg = 110 # taken from accumulation mode of MAM4
                sigma = 1.6
                params=[avg_number_fraction['OIN'], Dpg, sigma]
                SizeDist=N_multiplier*size_dependent_composition(Dp_mids, measured_N, 1,
                                                        splat_cutoff, 
                                                        avg_number_fraction[particle_type], 
                                                        [1.0], params)
                total_SizeDist+=SizeDist
            else:
                params=[]
                for mode in range(modes):
                    params.append(avg_number_fraction[particle_type]*mode_fractions[particle_type][mode])
                    params.append(fitting_params[mode][1])
                    params.append(fitting_params[mode][2])
                SizeDist = N_multiplier*size_dependent_composition(Dp_mids, measured_N, modes,
                                                        splat_cutoff, 
                                                        avg_number_fraction[particle_type], 
                                                        mode_fractions[particle_type], params) # 1/m^3
                total_SizeDist+=SizeDist

            # sample parameters (force at least one particle in every bin)
            idx=np.where(np.array([ptypes])==particle_type)[1]
            bins_full=False
            while bins_full==False:
                rands=np.log10(np.min(Dp_uppers))+(np.log10(np.max(Dp_uppers))-np.log10(np.min(Dp_uppers)))*np.random.rand(len(idx))
                hist=np.histogram(10**rands, bins=Dp_uppers)
                if np.min(hist[0])>0:
                    bins_full=True
            sampled_Dps=10**rands # nm
            sampled_Ns=np.interp(sampled_Dps, xp=Dp_mids, fp=SizeDist) # cm^-3
            
            # sample parameters (do not force one in each bin)
            # idx=np.where(np.array([ptypes])==particle_type)[1]
            # rands=np.log10(np.min(Dp_uppers))+(np.log10(np.max(Dp_uppers))-np.log10(np.min(Dp_uppers)))*np.random.rand(len(idx))
            # sampled_Dps=10**rands # nm
            # sampled_Ns=np.interp(sampled_Dps, xp=Dp_mids, fp=SizeDist) # cm^-3
            
            # change number concentrations based on histogram
            for jj in range(0,len(Dp_uppers)):
                idx2=np.where(np.logical_and(sampled_Dps >= Dp_lowers[jj], sampled_Dps < Dp_uppers[jj]))
                N_in_bin = len(idx2[0])
                sampled_Ns[idx2[0]]/=N_in_bin

            # save the sampled values
            particle_diameters[idx]=sampled_Dps*1e-9 # m
            particle_num_concs[idx]=sampled_Ns*100**3 # m^-3
        
        # change number concentrations to match measurements
        mult={}
        for ptype in avg_number_fraction.keys():
            spec_idx=np.where(np.logical_and(np.array((ptypes))==ptype, particle_diameters>=splat_cutoff*1e-9))
            all_idx=np.where(particle_diameters>=splat_cutoff*1e-9)
            modeled_Nfraction=np.sum(particle_num_concs[spec_idx[0]])/np.sum(particle_num_concs[all_idx[0]])
            mult[ptype]=avg_number_fraction[ptype]/modeled_Nfraction
        for ptype in avg_number_fraction.keys():
            spec_idx=np.where(np.logical_and(np.array((ptypes))==ptype, particle_diameters>=splat_cutoff*1e-9))
            all_idx=np.where(particle_diameters>=splat_cutoff*1e-9)
            particle_num_concs[spec_idx[0]]*=mult[ptype]
            modeled_Nfraction=np.sum(particle_num_concs[spec_idx[0]])/np.sum(particle_num_concs[all_idx[0]])
        
        # match the measured number concentration
        #mult=np.sum(measured_N)/(np.sum(particle_num_concs)/100**3)
        #particle_num_concs*=mult
        
        # plot ==================================================
        #bottom=np.zeros(len(Dp_uppers)-1)
        #plt.errorbar(Dp_mids, measured_N, fmt='o', yerr=N_error, mfc='w', mec='k', ecolor='k')
        #plt.plot(Dp_mids, total_SizeDist, '-r', zorder=100)
        #for t, c in zip(avg_number_fraction.keys(), ['grey','gold','r','b','g','C6']):
        #  idx=np.where(np.array([ptypes])==t)
        #  hist=np.histogram(1e9*particle_diameters[idx[1]], bins=Dp_uppers, weights=particle_num_concs[idx[1]]/100**3)
        #  widths=hist[1][1:]-hist[1][:-1]
        #  plt.bar(Dp_uppers[:-1], hist[0], width=widths, align='edge', bottom=bottom, facecolor=c, edgecolor='k', label=t)
        #  bottom+=hist[0]
        #plt.xscale('log')
        #plt.legend()
        #plt.ylim(0,)
        #plt.savefig('SAMPLED_PARTICLES.png', bbox_inches='tight')
        #plt.close()
        # plot ==================================================
        
        
        # sample the mass fraction of species in each particle
        aero_spec_names=[]
        aero_spec_fracs=[]   
        aero_pHs=[]   
            
        for ii in range(len(ptypes)):
            
            included_species=[]
            included_mass=np.zeros(0)
            remaining_species=[]
            remaining_mass=np.zeros(0)
            breaker=False
        
            while breaker==False:
                
                total_incl_mass=np.random.normal(loc=mass_fractions[ptypes[ii]][0][1], scale=mass_fractions[ptypes[ii]][0][2])
            
                try:
                    spec, fracs = get_aero_spec_fracs(molecule_names=[ptypes[ii]], molecule_mass_fracs=np.array([1.]),specdata_path=specdata_path)
                    tmass=0
                    for s in mass_fractions[ptypes[ii]][1]:
                        idx=np.where(np.array((spec))==s)
                        tmass+=total_incl_mass*fracs[idx[0][0]]
                    
                    if total_incl_mass < 1.0 and tmass > mass_fractions[ptypes[ii]][0][0]:
                        breaker=True
                        included_mass=np.append(included_mass, total_incl_mass)
                        included_species.append(ptypes[ii])
                        
                except:
                    incl_fracs=np.random.rand(len(mass_fractions[ptypes[ii]][1]))
                    incl_fracs/=np.sum(incl_fracs)
                    spec, fracs = get_aero_spec_fracs(molecule_names=mass_fractions[ptypes[ii]][1], molecule_mass_fracs=np.array(incl_fracs),specdata_path=specdata_path)
                    
                    tmass=0
                    for s, f in zip(spec, fracs):
                        idx=np.where(np.array((mass_fractions[ptypes[ii]][1]))==s)
                        tmass+=f*total_incl_mass
                    
                    if tmass < 1.0 and tmass > mass_fractions[ptypes[ii]][0][0]:
                        breaker=True
                        for s, f in zip(mass_fractions[ptypes[ii]][1], incl_fracs):
                            included_mass=np.append(included_mass, total_incl_mass*f)
                            included_species.append(s)            
            
            other_species=[]
            for t in mass_thresholds.keys():
                if t != ptypes[ii]:
                    try:
                        spec, fracs = get_aero_spec_fracs(molecule_names=[t], molecule_mass_fracs=np.array([1.]),specdata_path=specdata_path)
                        other_species.append(t)
                    except:
                        for s in mass_thresholds[t][1]:
                            if s not in mass_fractions['IEPOX'][1]:
                                other_species.append(s)
                
            while np.sum(remaining_mass)<1-total_incl_mass:
                spec_name=other_species[int(len(other_species)*np.random.rand())]
                remaining_mass=np.append(remaining_mass, (1-total_incl_mass)*np.random.rand())
                remaining_species.append(spec_name)
            remaining_mass*=(1-total_incl_mass)/np.sum(remaining_mass)
            
            temp_names=[]
            temp_fracs=[]
            for jj in range(len(included_species)):
                temp_names.append(included_species[jj])
                temp_fracs.append(included_mass[jj])
            for jj in range(len(remaining_species)):
                temp_names.append(remaining_species[jj])
                temp_fracs.append(remaining_mass[jj])
            
            aero_spec_names.append(temp_names)
            aero_spec_fracs.append(temp_fracs)
            aero_pHs.append(np.random.normal(loc=3.0, scale=0.5))
        
        # check that the sampled mass fractions match measurements
        masses={}
        for species in ams_mass_fractions.keys():
            masses[species]=0

        for ii in range(len(particle_diameters)):
            aero_names_temp, aero_fracs_temp = get_aero_spec_fracs(
                molecule_names=aero_spec_names[ii], molecule_mass_fracs=aero_spec_fracs[ii],
                specdata_path=specdata_path)            
            OneParticle=make_particle(particle_diameters[ii], aero_names_temp, 
                                      aero_fracs_temp, specdata_path=specdata_path, 
                                      surface_tension=0.072, reactions=None, gases=None)
            for species in ams_mass_fractions.keys():
                if type(OneParticle.get_species_idx(species))==int:
                    masses[species]+=particle_num_concs[ii]*OneParticle.masses[OneParticle.get_species_idx(species)]        
        
        total_mass=0
        for group in masses.keys():
            total_mass+=np.sum(np.array((masses[group])))
        
        sampled_mass_fractions={}
        for group in masses.keys():
            sampled_mass_fractions[group]=masses[group]/total_mass
        
        # check that the mass fractions match the AMS measurement
        checks=[]
        for group in masses.keys():
            sampled=sampled_mass_fractions[group]
            measured=ams_mass_fractions[group]
            measured_error=ams_mass_fraction_error[group]
            if sampled >= measured-measured_error and sampled <= measured+measured_error:
                checks.append(True)
            else:
                checks.append(False)

        if override_matching==True:
            for ii in range(len(checks)):
                checks[ii]=True
        
        # check that the particle types match the minisplat measurement
        for species in splat_species.keys():
            spec_idx=np.where(np.logical_and(np.array((ptypes))==species, particle_diameters>=splat_cutoff*1e-9))
            all_idx=np.where(particle_diameters>=splat_cutoff*1e-9)
            sampled=np.sum(particle_num_concs[spec_idx[0]])/np.sum(particle_num_concs[all_idx[0]])
            measured=avg_number_fraction[species]
            measured_error=number_fraction_error[species]
            if sampled >= measured-measured_error and sampled <= measured+measured_error and len(spec_idx[0])>Npart/20:
                checks.append(True)
            else:
                checks.append(False)
        
        counter += 1
        print(str(counter)+'/'+str(maxcounter), flush=True)
        #pbar.update(1)
        
        # ====================================================================
    #pbar.close()
     
    # change the number concentrations so that the total
    # mass concentration matches the AMS measurements
    particle_num_concs*=measured_total_mass/(1e9*total_mass)
    
    # plot ==================================================
    bottom=np.zeros(len(Dp_uppers)-1)
    plt.errorbar(Dp_mids, measured_N/np.max(measured_N), fmt='o', yerr=N_error/np.max(measured_N), mfc='w', mec='k', ecolor='k')
    plt.plot(Dp_mids, total_SizeDist/np.max(total_SizeDist), '-r', zorder=100)
    hist=np.histogram(1e9*particle_diameters, bins=Dp_uppers, weights=particle_num_concs/100**3)
    hist_max=np.max(hist[0])
    for t, c in zip(avg_number_fraction.keys(), ['grey','gold','r','b','g','C6']):
        idx=np.where(np.array([ptypes])==t)
        hist=np.histogram(1e9*particle_diameters[idx[1]], bins=Dp_uppers, weights=particle_num_concs[idx[1]]/100**3)
        widths=hist[1][1:]-hist[1][:-1]
        plt.bar(Dp_uppers[:-1], hist[0]/hist_max, width=widths, align='edge', bottom=bottom, facecolor=c, edgecolor='k', label=t)
        bottom+=hist[0]/hist_max
    plt.xscale('log')
    plt.ylabel(r'Normalized Number Concentration (cm$^{-3}$)', labelpad=10)
    plt.xlabel('Dry Diameter (nm)', labelpad=10)
    plt.legend()
    plt.ylim(0,)
    plt.savefig('SAMPLED_PARTICLES.png', bbox_inches='tight')
    plt.close()
    # =======================================================

    if counter == maxcounter:
        print('Measurements not matched after', maxcounter, 'iterations...returning sampled values.')
    elif override_matching==True:
        print('Measurement override = True...returning sampled values.')
    print()
    print('Measured, modeled total number concentration:')
    print(np.sum(measured_N), np.sum(particle_num_concs)/100**3)
    print()
    print('Measured, modeled number fractions:')
    for species in splat_species.keys():
        spec_idx=np.where(np.logical_and(np.array((ptypes))==species, particle_diameters>=splat_cutoff*1e-9))
        all_idx=np.where(particle_diameters>=splat_cutoff*1e-9)
        sampled=np.sum(particle_num_concs[spec_idx[0]])/np.sum(particle_num_concs[all_idx[0]])
        measured=avg_number_fraction[species]
        measured_error=number_fraction_error[species]
        if sampled >= measured-measured_error and sampled <= measured+measured_error:
            print(species, measured, sampled, True, len(spec_idx[0]))
        else:
            print(species, measured, sampled, False, len(spec_idx[0]))
    
    # get the updated mass concentrations
    masses={}
    for species in ams_mass_fractions.keys():
        masses[species]=0
    for ii in range(len(particle_diameters)):
        aero_names_temp, aero_fracs_temp = get_aero_spec_fracs(
            molecule_names=aero_spec_names[ii], molecule_mass_fracs=aero_spec_fracs[ii],
            specdata_path=specdata_path)
        OneParticle=make_particle(particle_diameters[ii], aero_names_temp,
                                  aero_fracs_temp, specdata_path=specdata_path,
                                  surface_tension=0.072, reactions=None, gases=None)
        for species in ams_mass_fractions.keys():
            if type(OneParticle.get_species_idx(species))==int:
                masses[species]+=particle_num_concs[ii]*OneParticle.masses[OneParticle.get_species_idx(species)]
    
    print()
    print('Measured, modeled mass concentrations:')
    for group in masses.keys():
        sampled=1e9*masses[group]
        measured=ams_mass_fractions[group]*measured_total_mass
        measured_error=ams_mass_fraction_error[group]*measured_total_mass
        if sampled >= measured-measured_error and sampled <= measured+measured_error:
            print(group, measured, sampled, True)
        else:
            print(group, measured, sampled, False)
    
    if gas_names:
        gas_vertical_profiles = measured_gas_phase(trace_gas_folder, gas_names)
    else:
        gas_vertical_profiles=None
    
    return particle_diameters, particle_num_concs, aero_spec_names, aero_spec_fracs, aero_pHs, gas_vertical_profiles
    
def scale_height(z, ppb0, H):
    return ppb0*np.exp(-z/H)

def optimize_splat_size_distribution(datapoints=10000,size_distribution_file=None, 
                                     start_time=None, end_time=None, cloud_flag=None, 
                                     CVI_flag=None, modes=1, mass_thresholds=None,
                                     ams_species=None, splat_species=None, splat_file=None,
                                     ams_file=None, specdata_path='../species_data/',
                                     aimms_file=None, splat_cutoff=85, z=0.0, dz=2.0):
    
    pars = fit_Nmodal_distibution(size_distribution_file, aimms_file, z, dz)   
    modes = int(len(pars)/3)
    output_pars=[]
    for mode in range(modes):
        output_pars.append(pars[mode*3:mode*3+3])
    
    Dpg_BC = 110
    sigma_BC = 1.6
    Dpg_dust = 110 # taken from accumulation mode of MAM4
    sigma_dust = 1.6
    
    model_species=[]
    for spec in splat_species.keys():
        if spec!='BC' and spec!='OIN':
            model_species.append(spec)
    
    Nspec=len(model_species)
    mode_fractions = np.zeros((datapoints, Nspec*modes))
    for i in range(0, datapoints):
        for j in range(Nspec):
            mode_fractions[i, j*modes:(j+1)*modes] = np.random.rand(modes)
            mode_fractions[i, j*modes:(j+1)*modes] /= np.sum(mode_fractions[i, j*modes:(j+1)*modes])
    mode_fractions = np.unique(mode_fractions, axis = 0)      
    
    # get initial size distribution properties from measurements
    Dp_lower, Dp_upper, measured_N, measured_N_error = read_FIMS(size_distribution_file, aimms_file, z, dz)
    Dp_mid = Dp_lower + 0.5*(Dp_upper - Dp_lower)
    measured_Ntot = np.sum(measured_N) # N per volume
    measured_mean_size = np.average(Dp_mid, weights=measured_N)
    SA_dist = 4.0*np.pi*np.power((Dp_mid)/2, 2)*(measured_N) # aerosol SA/volume air
    measured_SA = np.sum(SA_dist) # total surface area per volume
    V_dist = (4.0/3.0)*np.pi*np.power((Dp_mid)/2, 3)*measured_N # volume aerosol/volume air
    measured_Vtot = np.sum(V_dist) # total surface area per volume
    avg_number_fraction, number_fraction_error = splat_number_fractions(splat_file, aimms_file, size_distribution_file, splat_species, z, dz)
    ams_mass_fractions, ams_mass_fraction_errors, measured_total_mass, measured_total_mass_error = ams_mass_fraction(ams_file, aimms_file, size_distribution_file, z, dz)       
    
    # set initial RSS
    min_RSS = 1e10
    line_to_save=-1
    
    # loop through mode fractions and find the error associated with each combination
    print()
    print('optimizing size distribution,', modes, 'modes...')
    counter=0
    #pbar = tqdm.tqdm(total = len(mode_fractions))
    for line in range(0, len(mode_fractions)):
        
        spec_masses={}
        for x in ams_mass_fractions.keys():
            spec_masses[x]=0
            
        total_mass=0
        total_Ns=np.zeros(len(Dp_mid))
        
        for spec in range(Nspec):
            
            params=[]
            for mode in range(modes):
                params.append(avg_number_fraction[model_species[spec]]*mode_fractions[line, spec*modes+mode])
                params.append(pars[mode*3+1])
                params.append(pars[mode*3+2])
            spec_Ns = size_dependent_composition(Dp_mid, measured_N, modes,
                                                    splat_cutoff, 
                                                    avg_number_fraction[model_species[spec]], 
                                                    mode_fractions[line, spec*modes:spec*modes+modes], params) # 1/m^3
            
            total_Ns+=spec_Ns
            
            try:
                x=model_species[spec]
                temp_names, temp_fracs = get_aero_spec_fracs(
                    molecule_names=[x], molecule_mass_fracs=[1.0],
                    specdata_path=specdata_path) 
                OneParticle=make_particle(100e-9, temp_names, temp_fracs, specdata_path=specdata_path, surface_tension=0.072, reactions=None, gases=None)
            except:
                x=mass_thresholds[model_species[spec]][1][0]
                temp_names, temp_fracs = get_aero_spec_fracs(
                    molecule_names=[x], molecule_mass_fracs=[1.0],
                    specdata_path=specdata_path) 
                OneParticle=make_particle(100e-9, temp_names, temp_fracs, specdata_path=specdata_path, surface_tension=0.072, reactions=None, gases=None)
            for s, f, m in zip(OneParticle.species, temp_fracs, OneParticle.masses):
                if s.name != 'H2O':
                    volume_frac=(m/s.density)/((4.0/3.0)*np.pi*(100e-9/2)**3)
                    Vtot=np.sum(spec_Ns*(4.0/3.0)*np.pi*((Dp_mid*1e-9)/2)**3)
                    try:
                        spec_masses[s.name]+=mass_thresholds[model_species[spec]][0][1]*s.density*volume_frac*Vtot
                        total_mass+=mass_thresholds[model_species[spec]][0][1]*s.density*volume_frac*Vtot
                    except:
                        total_mass+=mass_thresholds[model_species[spec]][0][1]*s.density*volume_frac*Vtot
        
        # sample the BC and dust
        params=[avg_number_fraction['BC'], Dpg_BC, sigma_BC]
        spec_Ns=size_dependent_composition(Dp_mid, measured_N, 1,
                                                splat_cutoff, 
                                                avg_number_fraction[model_species[spec]], 
                                                [1.0], params)
        total_Ns+=spec_Ns        
        params=[avg_number_fraction['OIN'], Dpg_dust, sigma_dust]
        spec_Ns=size_dependent_composition(Dp_mid, measured_N, 1,
                                                splat_cutoff, 
                                                avg_number_fraction[model_species[spec]], 
                                                [1.0], params)
        total_Ns+=spec_Ns
        
        # match the measured number concentration
        mult=np.sum(measured_N)/np.sum(total_Ns)
        total_Ns*=mult
        
        # get the calculated mass fraction
        calculated_mass_fraction={}
        for ii in ams_mass_fractions.keys():
            calculated_mass_fraction[ii]=spec_masses[ii]/total_mass
        
        # get the calculated Ntot, Vtot, and SAtot
        calculated_mean_size = np.average(Dp_mid, weights=total_Ns)
        calculated_Vtot = np.sum(total_Ns*(4/3)*np.pi*(0.5*Dp_mid)**3)
        calculated_SA = np.sum(total_Ns*4*np.pi*(0.5*Dp_mid)**2)
    
        # calculate RSS
        RSS=((calculated_mean_size - measured_mean_size)/measured_mean_size)**2\
            + ((calculated_SA - measured_SA)/measured_SA)**2\
            + ((calculated_Vtot - measured_Vtot)/measured_Vtot)**2
        for ii in ams_mass_fractions.keys():
            RSS+=((calculated_mass_fraction[ii]-ams_mass_fractions[ii])/ams_mass_fractions[ii])**2
        
        if RSS<min_RSS:
            min_RSS=RSS
            line_to_save=line
            mult_to_save=mult
        
        counter+=1
        #print(str(counter)+'/'+str(len(mode_fractions)), flush=True)
        sys.stdout.write(str(counter)+'/'+str(len(mode_fractions))+'\n')
        sys.stdout.flush()

    output={}
    for ii in range(len(model_species)):
        output[model_species[ii]]=mode_fractions[line_to_save, ii*modes:ii*modes+modes]
    
    
    
    # plotting ===========================================================================
    # total_Ns=np.zeros(len(Dp_mid))
    # for spec in range(Nspec):
    #     params=[]
    #     for mode in range(modes):
    #         params.append(avg_number_fraction[model_species[spec]]*mode_fractions[line_to_save, spec*modes+mode])
    #         params.append(pars[mode*3+1])
    #         params.append(pars[mode*3+2])
    #     spec_Ns = size_dependent_composition(Dp_mid, measured_N, modes,
    #                                             splat_cutoff, 
    #                                             avg_number_fraction[model_species[spec]], 
    #                                             mode_fractions[line_to_save, spec*modes:spec*modes+modes], params) # 1/m^3
    #     plt.plot(Dp_mid, mult_to_save*spec_Ns, label=model_species[spec])
    #     total_Ns+=mult_to_save*spec_Ns
    # params=[avg_number_fraction['BC'], Dpg_BC, sigma_BC]
    # spec_Ns=size_dependent_composition(Dp_mid, measured_N, 1,
    #                                         splat_cutoff, 
    #                                         avg_number_fraction[model_species[spec]], 
    #                                         [1.0], params)
    # plt.plot(Dp_mid, mult_to_save*spec_Ns, label='BC')
    # total_Ns+=mult_to_save*spec_Ns
    # params=[avg_number_fraction['OIN'], Dpg_dust, sigma_dust]
    # spec_Ns=size_dependent_composition(Dp_mid, measured_N, 1,
    #                                         splat_cutoff, 
    #                                         avg_number_fraction[model_species[spec]], 
    #                                         [1.0], params)
    # plt.plot(Dp_mid, mult_to_save*spec_Ns, label='dust')
    # total_Ns+=mult_to_save*spec_Ns
    # plt.plot(Dp_mid, measured_N, 'ko')
    # plt.plot(Dp_mid, total_Ns, '-ks')
    # plt.xscale('log')
    # plt.legend()
    # plt.ylim(0,)
    # plt.savefig('OPTIMIZATION.png', bbox_inches='tight')
    # plt.close()
    # print(np.sum(total_Ns), np.sum(measured_Ntot))
    # sys.exit()
    # =======================================================================================
    
    return output, output_pars, mult_to_save

def size_dependent_composition(Dps, measured_Ns, N_modes, cutoff, splat_fraction, mode_fractions, size_dist_params):
    
    CDF_calc=np.zeros(len(Dps))
    for mode in range(N_modes):
        Dpg=size_dist_params[mode*3+1]
        sigma=size_dist_params[mode*3+2]
        CDF_calc+= mode_fractions[mode]*0.5*(1+erf((np.log(Dps)-np.log(Dpg))/(np.sqrt(2)*np.log(sigma))))

    indices = np.where(Dps >= cutoff)
    Ntot_meas_gt_cutoff = np.sum(measured_Ns[indices[0][0]:])
    Fx_gt_cutoff = 1 - CDF_calc[indices[0][0]]
    Nx_gt_cutoff = Ntot_meas_gt_cutoff*splat_fraction
    Nx = Nx_gt_cutoff/Fx_gt_cutoff
    
    for mode in range(N_modes):
        size_dist_params[mode*3+0]=Nx*mode_fractions[mode]
    
    Ns=Nmodal_lognormal(Dps, *size_dist_params)
    mult = (Nx_gt_cutoff)/np.sum(Ns[indices[0][0]:])
    Ns*=mult
    
    return Ns


def fit_Nmodal_distibution(FIMS_file, AIMMS_file, z, dz):
    
    Dp_lower, Dp_upper, measured_N, measured_N_error = read_FIMS(FIMS_file, AIMMS_file, z, dz)
    modes=0
    r2=0
    while r2<0.9:
        modes+=1
        try:
            p0 = []
            lower_bounds = []
            upper_bounds = []
            for i in range(modes):
                p0.append(1.2e4)
                p0.append(100*(i+1))
                p0.append(1.3)
                lower_bounds.append(0)
                lower_bounds.append(0)
                lower_bounds.append(1.0)
                upper_bounds.append(np.inf)
                upper_bounds.append(np.inf)
                upper_bounds.append(np.inf)
            pars, cov = curve_fit(Nmodal_lognormal, xdata=Dp_upper, ydata=measured_N, p0=p0, bounds=[lower_bounds, upper_bounds])
            predicted_N = Nmodal_lognormal(Dp_upper, *pars)
                    
            rss = np.sum((measured_N - predicted_N) ** 2)
            tss = np.sum((measured_N - np.mean(measured_N)) ** 2)
            r2 = 1 - (rss / tss)
        except:
            r2 = 0
            
    # plt.plot(Dp_upper, measured_N, 'ko')
    # plt.plot(Dp_upper, predicted_N, '-r')
    # plt.xscale('log')
    #plt.savefig('TEST.png')
    # sys.exit()
            
    return pars
   

def Nmodal_lognormal(x, *params):
    n_modes=int(len(params)/3)
    N=np.zeros(len(x))
    for mode in range(n_modes):
        Ntot=params[mode*3+0]
        Dpg=params[mode*3+1]
        sigma=params[mode*3+2]
        N=N+lognormal_distribution(x, Ntot, Dpg, sigma)
    
    return N
        
    
def lognormal_distribution(x, Ntot, Dpg, sigma):
    prefactor = Ntot/(np.sqrt(2.0*np.pi)*np.log(sigma)*x)
    numerator = -1.0*np.power(np.log(x)-np.log(Dpg), 2)
    denominator = 2.0*np.log(sigma)*np.log(sigma)
    N = prefactor*np.exp(numerator/denominator)
    return N




def ams_mass_fraction(ams_file, aimms_file, fims_file, z, dz):
    
    # read the AMS data from file
    filename = ams_file
    raw_data = np.loadtxt(filename, delimiter = ',', dtype='str', skiprows = 42)
    for i in range(0, len(raw_data)):
        for j in range(0, len(raw_data[0])):
            raw_data[i, j] = raw_data[i, j].strip()
    ams_data = {}
    for i in range(0, len(raw_data[0])): 
        ams_data[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
    
    # read AIMMS data from file   
    raw_data = np.loadtxt(aimms_file, delimiter = ',', dtype='str', skiprows = 53)
    AIMMS_data = {}
    for i in range(len(raw_data[0])): 
        AIMMS_data[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
    
    # read FIMS data from file   
    raw_data = np.loadtxt(fims_file, delimiter = ',', dtype='str', skiprows = 100)    
    for i in range(0, len(raw_data)):
        for j in range(0, len(raw_data[0])):
            raw_data[i, j] = raw_data[i, j].strip()
    FIMS_data = {}
    FIMS_data[str(raw_data[0, 0])] = np.array(raw_data[1:, 0], dtype = 'float64')
    for i in range(56, len(raw_data[0])): 
        FIMS_data[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
    FIMS_data['N_Dp'] = np.array(raw_data[1:, 1:56], dtype = 'float64')
    
    # find times where altitude is between z-dz and z+dz
    AIMMS_indices = np.where(np.logical_and(AIMMS_data['Alt'] >= z-dz, 
                                      AIMMS_data['Alt'] < z+dz))
    if len(AIMMS_indices[0])==0 and z-dz>np.max(AIMMS_data['Alt']):
        AIMMS_indices = np.where(AIMMS_data['Alt'] >= np.max(AIMMS_data['Alt'])-dz)
    elif len(AIMMS_indices[0])==0 and z+dz<np.min(AIMMS_data['Alt']):
        AIMMS_indices = np.where(AIMMS_data['Alt'] <= np.min(AIMMS_data['Alt'])+dz)
    
    # get times between z-dz and z+dz out of cloud
    FIMS_times = []
    for ii in AIMMS_indices[0]:
        FIMS_index = np.where(np.logical_and(np.round(AIMMS_data['Time(UTC)'][ii],0)==FIMS_data['Start_UTC'],
                                             FIMS_data['Cloud_flag'] == 0))
        if len(FIMS_index[0])>0:
            for jj in FIMS_index[0]:
                FIMS_times.append(FIMS_data['Start_UTC'])
    FIMS_times=np.unique(FIMS_times)
    
    # get the SPLAT indices corresponding to the out of cloud heights
    AMS_indices = []
    for t in FIMS_times:
        AMS_index = np.where(np.logical_and(ams_data['dat_ams_utc']==t, ams_data['flag'] < 0.5))
        if len(AMS_index[0])>0:
            for jj in AMS_index[0]:
                AMS_indices.append(jj)
    AMS_indices=np.unique(AMS_indices)
    
    # pull out data within the averaging time
    ams_subdata = {}
    for key in ams_data:
        ams_subdata[key] = ams_data[key][AMS_indices[0]]
    for i in range(1, len(AMS_indices)):
        for key in ams_data:
            ams_subdata[key] = np.append(ams_subdata[key], ams_data[key][AMS_indices[i]])
    
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
    mass_frac_error = {}
    mass_frac['SO4'] = output['SO4']
    mass_frac_error['SO4'] = output['SO4_err']
    mass_frac['NO3'] = output['NO3']
    mass_frac_error['NO3'] = output['NO3_err']
    mass_frac['OC'] = output['Org']
    mass_frac_error['OC'] = output['Org_err']
    mass_frac['NH4'] = output['NH4']
    mass_frac_error['NH4'] = output['NH4_err']
    
    return mass_frac, mass_frac_error, np.mean(ams_subdata['total_mass']), np.std(ams_subdata['total_mass'])

def splat_number_fractions(splat_file, aimms_file, fims_file, splat_species, z, dz):
    
    # read miniSPLAT data from file
    raw_data = np.loadtxt(splat_file, dtype='str')
    full_SPLAT_data = {}
    SPLAT_subdata = {}
    for i in range(0, len(raw_data[0])): 
        full_SPLAT_data[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
        SPLAT_subdata[str(raw_data[0, i])] = np.zeros(0)
    
    # read AIMMS data from file   
    raw_data = np.loadtxt(aimms_file, delimiter = ',', dtype='str', skiprows = 53)
    AIMMS_data = {}
    for i in range(len(raw_data[0])): 
        AIMMS_data[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
    
    # read FIMS data from file   
    raw_data = np.loadtxt(fims_file, delimiter = ',', dtype='str', skiprows = 100)    
    for i in range(0, len(raw_data)):
        for j in range(0, len(raw_data[0])):
            raw_data[i, j] = raw_data[i, j].strip()
    FIMS_data = {}
    FIMS_data[str(raw_data[0, 0])] = np.array(raw_data[1:, 0], dtype = 'float64')
    for i in range(56, len(raw_data[0])): 
        FIMS_data[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
    FIMS_data['N_Dp'] = np.array(raw_data[1:, 1:56], dtype = 'float64')
    
    # find times where altitude is between z-dz and z+dz
    AIMMS_indices = np.where(np.logical_and(AIMMS_data['Alt'] >= z-dz, 
                                      AIMMS_data['Alt'] < z+dz))
    if len(AIMMS_indices[0])==0 and z-dz>np.max(AIMMS_data['Alt']):
        AIMMS_indices = np.where(AIMMS_data['Alt'] >= np.max(AIMMS_data['Alt'])-dz)
    elif len(AIMMS_indices[0])==0 and z+dz<np.min(AIMMS_data['Alt']):
        AIMMS_indices = np.where(AIMMS_data['Alt'] <= np.min(AIMMS_data['Alt'])+dz)
    
    # get times between z-dz and z+dz out of cloud
    FIMS_times = []
    for ii in AIMMS_indices[0]:
        FIMS_index = np.where(np.logical_and(np.round(AIMMS_data['Time(UTC)'][ii],0)==FIMS_data['Start_UTC'],
                                             FIMS_data['Cloud_flag'] == 0))
        if len(FIMS_index[0])>0:
            for jj in FIMS_index[0]:
                FIMS_times.append(FIMS_data['Start_UTC'])
    FIMS_times=np.unique(FIMS_times)
    
    # get the SPLAT indices corresponding to the out of cloud heights
    SPLAT_indices = []
    for t in FIMS_times:
        SPLAT_index = np.where(full_SPLAT_data['Time']==t)        
        if len(SPLAT_index[0])>0:
            for jj in SPLAT_index[0]:
                SPLAT_indices.append(jj)
    SPLAT_indices=np.unique(SPLAT_indices)
    
    # pull out the miniSPLAT data corresponding to initialization times
    for key in full_SPLAT_data.keys():
        SPLAT_subdata[key] = np.array((full_SPLAT_data[key][SPLAT_indices]))
    
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

    
def read_FIMS(FIMS_file, AIMMS_file, z, dz):
    
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
    
    # read AIMMS data from file   
    raw_data = np.loadtxt(AIMMS_file, delimiter = ',', dtype='str', skiprows = 53)
    AIMMS_data = {}
    for i in range(len(raw_data[0])): 
        AIMMS_data[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
    
    # find times where altitude is between z-dz and z+dz
    AIMMS_indices = np.where((AIMMS_data['Alt'] >= z-dz) & (AIMMS_data['Alt'] < z+dz) & (AIMMS_data['Lon']>-97.5) & (AIMMS_data['Lon']<-97.4) & (AIMMS_data['Lat']>36.05) & (AIMMS_data['Lat']<36.81))[0]
    leg_indices = np.where((AIMMS_data['Lon']>-97.5) & (AIMMS_data['Lon']<-97.4) & (AIMMS_data['Lat']>36.05) & (AIMMS_data['Lat']<36.81))[0]
    if len(AIMMS_indices)==0 and z-dz>np.max(AIMMS_data['Alt'][leg_indices]):
        AIMMS_indices = np.where((AIMMS_data['Alt'] > np.max(AIMMS_data['Alt'][leg_indices])-dz) & (AIMMS_data['Lon']>-97.5) & (AIMMS_data['Lon']<-97.4) & (AIMMS_data['Lat']>36.05) & (AIMMS_data['Lat']<36.81))[0]
    elif len(AIMMS_indices)==0 and z+dz<np.min(AIMMS_data['Alt'][leg_indices]):
        AIMMS_indices = np.where((AIMMS_data['Alt'] < np.min(AIMMS_data['Alt'][leg_indices])+dz) & (AIMMS_data['Lon']>-97.5) & (AIMMS_data['Lon']<-97.4) & (AIMMS_data['Lat']>36.05) & (AIMMS_data['Lat']<36.81))[0]
    
    # get the FIMS indices corresponding to the AIMMS heights
    FIMS_indices = []
    for ii in AIMMS_indices:
        FIMS_index = np.where(np.logical_and(np.round(AIMMS_data['Time(UTC)'][ii],0)==FIMS_data['Start_UTC'],
                                             FIMS_data['Cloud_flag'] == 0))
        if len(FIMS_index[0])>0:
            for jj in FIMS_index[0]:
                FIMS_indices.append(jj)
    FIMS_indices=np.unique(FIMS_indices)
    
    # pull out data that is within the sampling time
    FIMS_subdata = FIMS_data['N_Dp'][FIMS_indices[0], :]
    for i in range(1, len(FIMS_indices)):
        FIMS_subdata = np.vstack((FIMS_subdata, FIMS_data['N_Dp'][FIMS_indices[i], :]))   
    FIMS_subdata = np.exp(np.log(FIMS_subdata))
    
    # calculate average size distribution in averaging window
    avg_N = np.nanmean(FIMS_subdata, axis = 0)
    error_N = np.nanstd(FIMS_subdata, axis = 0)

    # make less columns
    grid=2
    if grid>1:
        Dp_uppers=[]
        Dp_lowers=[]
        avg=[]
        error=[]
        for i in range(1, len(avg_N), grid):
            Dp_uppers.append(FIMS_data['UPPER_BIN_SIZE_nanometer'][i])
            Dp_lowers.append(FIMS_data['LOWER_BIN_SIZE_nanometer'][i-grid+1])
            avg.append(avg_N[i-grid+1] + avg_N[i])
            error.append(error_N[i-grid+1] + error_N[i])
        avg=np.array((avg))
        Dp_uppers=np.array((Dp_uppers))
        Dp_lowers=np.array((Dp_lowers))
        error=np.array((error))
    else:
        avg=avg_N
        Dp_uppers=FIMS_data['UPPER_BIN_SIZE_nanometer']
        Dp_lowers=FIMS_data['LOWER_BIN_SIZE_nanometer']
        error=error_N
    
    return Dp_lowers, Dp_uppers, avg, error

def measured_gas_phase(trace_gas_folder, gas_names):
    
    # get the altitude grid used for interpolating
    #filename = os.path.join(trace_gas_folder, os.listdir(trace_gas_folder)[0])
    #raw_data = np.loadtxt(filename, delimiter = ',', dtype='str')
    #gas_data = {}
    #for ii in range(0, len(raw_data[0])):
    #    gas_data[str(raw_data[0, ii])] = np.array(raw_data[1:, ii], dtype = 'float64')
    #alt_grid = np.linspace(np.min(gas_data['Alt']), np.max(gas_data['Alt']), 16)
    #alt_mids = 0.5*(alt_grid[range(len(alt_grid)-1)] + alt_grid[range(1,len(alt_grid))])
    
    gas_data_all={}    
    for gas in gas_names:
                
        if gas == 'NO2': # NO2 = NOx-NO
            for f in os.listdir(trace_gas_folder):
                filename = os.path.join(trace_gas_folder, f)
                if os.path.isfile(filename):
                    temp_name=filename.split('_')
                    if temp_name[-2]=='NOx':
                        filename1=os.path.join(trace_gas_folder, f)
                    elif temp_name[-2]=='NO':
                        filename2=os.path.join(trace_gas_folder, f)
                        
            raw_data = np.loadtxt(filename1, delimiter = ',', dtype='str')
            NOx_data = {}
            for ii in range(0, len(raw_data[0])):
                NOx_data[str(raw_data[0, ii])] = np.array(raw_data[1:, ii], dtype = 'float64')
            
            raw_data = np.loadtxt(filename2, delimiter = ',', dtype='str')
            NO_data = {}
            for ii in range(0, len(raw_data[0])):
                NO_data[str(raw_data[0, ii])] = np.array(raw_data[1:, ii], dtype = 'float64')
            
            NO2_data={'Alt': NOx_data['Alt'], 'Lat': NOx_data['Lat'], 'Long': NOx_data['Long'],
                      'Value_ppb': NOx_data['Value_ppb']-NO_data['Value_ppb']}
            
            idx = np.where((NO2_data['Value_ppb']>=0.0))[0]
            NO2_alts=np.zeros(0)
            NO2_medians = np.zeros(0)
            if len(idx)>0:
                alt_grid = np.linspace(np.min(NO2_data['Alt'][idx]), np.max(NO2_data['Alt'][idx]), 11)
                alt_mids = 0.5*(alt_grid[range(len(alt_grid)-1)] + alt_grid[range(1,len(alt_grid))])
                for rr in range(1, len(alt_grid)):
                    idx = np.where((NO2_data['Alt']>alt_grid[rr-1])
                                      & (NO2_data['Alt']<=alt_grid[rr])
                                      & (NO2_data['Value_ppb']>=0.0)
                                      & (NO2_data['Long']>-97.5)
                                      & (NO2_data['Long']<-97.4)
                                      & (NO2_data['Lat']>36.05)
                                      & (NO2_data['Lat']<36.81))[0]
                    if len(idx)>0:
                        NO2_alts=np.append(NO2_alts,alt_mids[rr-1])
                        NO2_medians = np.append(NO2_medians, np.percentile(NO2_data['Value_ppb'][idx], 50))
                if len(NO2_medians)>0:
                    gas_data_all[gas]={}
                    gas_data_all[gas]['ppb']=NO2_medians
                    gas_data_all[gas]['alt']=NO2_alts
                else:
                    try:
                        alts, medians = get_AGFL_profile(gas, trace_gas_folder)
                        gas_data_all[gas]={}
                        gas_data_all[gas]['ppb']=medians
                        gas_data_all[gas]['alt']=alts
                    except:
                        print('WARNING: No data for', gas, '!')
                        sys.exit()
            else:
                try:
                    alts, medians = get_AGFL_profile(gas, trace_gas_folder)
                    gas_data_all[gas]={}
                    gas_data_all[gas]['ppb']=medians
                    gas_data_all[gas]['alt']=alts
                except:
                    print('WARNING: No data for', gas, '!')
                    sys.exit()
        
        else: # this is for all the other gases
            file_to_read=None
            for f in os.listdir(trace_gas_folder):
                filename = os.path.join(trace_gas_folder, f)
                if os.path.isfile(filename):
                    temp_name=filename.split('_')
                    if temp_name[-2]==gas:
                        file_to_read=filename
                        break

            if file_to_read:
                raw_data = np.loadtxt(file_to_read, delimiter = ',', dtype='str')
                gas_data = {}
                for ii in range(0, len(raw_data[0])): 
                    gas_data[str(raw_data[0, ii])] = np.array(raw_data[1:, ii], dtype = 'float64')
                
                idx = np.where((gas_data['Value_ppb']>=0.0))[0]
                alts=np.zeros(0)
                medians = np.zeros(0)
                if len(idx)>0:
                    alt_grid = np.linspace(np.min(gas_data['Alt'][idx]), np.max(gas_data['Alt'][idx]), 11)
                    alt_mids = 0.5*(alt_grid[range(len(alt_grid)-1)] + alt_grid[range(1,len(alt_grid))])
                    for rr in range(1, len(alt_grid)):
                        idx = np.where((gas_data['Alt']>alt_grid[rr-1])
                                          & (gas_data['Alt']<=alt_grid[rr])
                                          & (gas_data['Value_ppb']>=0.0)
                                          & (gas_data['Long']>-97.5)
                                          & (gas_data['Long']<-97.4)
                                          & (gas_data['Lat']>36.05)
                                          & (gas_data['Lat']<36.81))[0]
                        if len(idx)>0:
                            alts=np.append(alts,alt_mids[rr-1])
                            medians = np.append(medians, np.percentile(gas_data['Value_ppb'][idx], 50))
                
                if len(medians)>0:
                    gas_data_all[gas]={}
                    gas_data_all[gas]['ppb']=medians
                    gas_data_all[gas]['alt']=alts
                else:
                    try:
                        alts, medians = get_AGFL_profile(gas, trace_gas_folder)
                        gas_data_all[gas]={}
                        gas_data_all[gas]['ppb']=medians
                        gas_data_all[gas]['alt']=alts
                    except:
                        print('WARNING: No data for', gas, '!')
                        sys.exit()
            else:
                try:
                    alts, medians = get_AGFL_profile(gas, trace_gas_folder)
                    gas_data_all[gas]={}
                    gas_data_all[gas]['ppb']=medians
                    gas_data_all[gas]['alt']=alts
                except:
                    print('WARNING: No data for', gas, '!')
                    sys.exit()
                    
    #print(gas_data_all)
    #for gas in gas_names:
        #plt.plot(gas_data_all[gas]['ppb'], gas_data_all[gas]['alt']/1000, label=gas)
    #plt.xscale('log')
    #plt.legend()
    #plt.ylim(0, 2.5)
    #plt.xlim(1e-3, 1e2)
    #plt.savefig('GAS_PHASE.png')
    #print('gere')
    #sys.exit()
    
    return gas_data_all

def get_AGFL_profile(gas, trace_gas_folder):
    
    filename = os.path.join(trace_gas_folder, 'AGFL_atmosphere.txt')
    raw_data = np.loadtxt(filename, dtype='str')
    AGFL = {}
    AGFL[str(raw_data[0, 0])] = np.array(raw_data[1:, 0], dtype='float64')
    for i in range(0, len(raw_data[0])): 
        AGFL[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
        
    return 1000*AGFL['z'], 1000*AGFL[gas]

    
    

# ================================ main function ===============================

# do a try/except so I can use the functions without doing the initialization
try:
    total_Np = int(sys.argv[1])
    les_path = str(sys.argv[2])
    output_path = str(sys.argv[3])
    les_number = str(int(sys.argv[4])).zfill(6)

    #sys.stdout = open('output.log', 'w')

    splat_species = {'BC': ['soot'],
                      'OIN': ['Dust'],
                      'AS': ['sulfate_nitrate_org'],
                      'AN': ['nitrate_amine_org'],
                      'OC': ['org28', 'org30_43', 'BB_SOA', 'org_amines', 'BB', 'pyridine'],
                      'IEPOX': ['IEPOX_SOA']}

    mass_fractions={'IEPOX': [[0.3,0.5,0.1], ['IEPOX_OS','tetrol','tetrol_olig', 'IEPOX_OH_SOA']],
                    'AS': [[0.5,0.7,0.1], ['SO4']],
                    'AN': [[0.5,0.7,0.1], ['NO3']],
                    'OC': [[0.5,0.7,0.1], ['OC']],
                    'BC': [[0.5,0.7,0.1], ['BC']],
                    'OIN': [[0.5,0.7,0.1], ['OIN']]}

    gas_names = ['SO2', 'O3', 'H2O2', 'IEPOX', 'OH', 'HNO3', 'NO2', 'NO']

    diameters, num_concs, aero_spec_names, aero_spec_fracs, pHs, gas_data=splat_setup(Npart=total_Np,
                           optimization_points=1000, mass_thresholds=mass_fractions,
                           size_distribution_file='/rcfs/projects/partikkel/multipart/datasets/HISCALE_data_0425/BEASD_G1_20160425155810_R2_HISCALE_001s.txt',
                           splat_file='/rcfs/projects/partikkel/multipart/datasets/HISCALE_data_0425/Splat_Composition_25-Apr-2016.txt',
                           aimms_file='/rcfs/projects/partikkel/multipart/datasets/HISCALE_data_0425/AIMMS20_G1_20160425155810_R2_HISCALE020h.txt',
                           trace_gas_folder='/rcfs/projects/partikkel/multipart/datasets/HISCALE_data_0425/CIMS_data',
                           dz=100.0, splat_species=splat_species,
                           mass_fractions=mass_fractions,
                           gas_names=gas_names, les_path=les_path, les_number=les_number,
                           ams_file='/rcfs/projects/partikkel/multipart/datasets/HISCALE_data_0425/HiScaleAMS_G1_20160425_R0.txt',
                           override_matching=True,
                           specdata_path='/rcfs/projects/partikkel/multipart/species_data/')

    # write pickle files that save the initial aerosol properties
    f = open(output_path+'/diameters', 'wb')
    pickle.dump(diameters, f)
        
    f = open(output_path+'/num_concs', 'wb')
    pickle.dump(num_concs, f)

    f = open(output_path+'/aero_spec_names', 'wb')
    pickle.dump(aero_spec_names, f)

    f = open(output_path+'/aero_spec_fracs', 'wb')
    pickle.dump(aero_spec_fracs, f)

    f = open(output_path+'/pHs', 'wb')
    pickle.dump(pHs, f)

    f = open(output_path+'/gas_data', 'wb')
    pickle.dump(gas_data, f)

    f = open(output_path+'/trajectory_number', 'wb')
    pickle.dump(les_number, f)

    print()
    
except:
    pass

#import multiprocessing
#print(f"Number of processes: {multiprocessing.cpu_count()}")
#print(f"OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS')}")
#import threading
#print(f"Active threads: {threading.active_count()}")
#print(f"Number of active processes: {multiprocessing.current_process().name}")

