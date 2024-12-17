#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:06:32 2024

@author: beel083
"""
import numpy as np
import sys, os
import matplotlib.pyplot as plt
# from scipy import trapz
from scenario import get_aero_spec_fracs
from scipy.optimize import curve_fit
import tqdm
from particles import make_particle
# from scenario import get_aero_spec_fracs
from scipy.special import erf
import pickle, shutil

def splat_setup(Npart=1, optimization_points=10000, modes=2, mass_thresholds=None,
                size_distribution_file=None, splat_file=None, trace_gas_folder=None,
                splat_species=None, mass_fractions=None,ams_file=None,aimms_file=None, 
                start_time=None, end_time=None, cloud_flag=0, gas_names=None,
                CVI_flag=0, specdata_path='../species_data/',
                override_matching=False, splat_cutoff=85):
    
    # diameters=np.zeros(Npart)
    if not start_time or not end_time:
        print('WARNING: Need start and end time for miniSPLAT averaging!')
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
       
    mode_fractions, fitting_params=optimize_splat_size_distribution(datapoints=optimization_points,
                        size_distribution_file=size_distribution_file, 
                        splat_file=splat_file,
                        ams_file=ams_file,
                        start_time=start_time, end_time=end_time, 
                        cloud_flag=cloud_flag, CVI_flag=CVI_flag,
                        modes=modes, mass_thresholds=mass_fractions,
                        splat_species=splat_species, splat_cutoff=splat_cutoff)  
    
    print()
    print('Fitted size distribution modes:')
    print(fitting_params)
    print()
    print('Optimized number mode fractions:')
    for spec in mode_fractions:
        print(spec, mode_fractions[spec])
    print()
        
    # read in the measured data
    Dp_lowers, Dp_uppers, measured_N, N_error = read_FIMS(size_distribution_file, 
                                                 start_time, end_time, cloud_flag, 
                                                 CVI_flag) # diameters in nm and N in #/cm^3
    idx=np.where(Dp_lowers>splat_cutoff)
    measured_Ntot=np.sum(measured_N[idx[0]])
    avg_number_fraction, number_fraction_error = splat_number_fractions(splat_file, splat_species, start_time, end_time)    
    ams_mass_fractions, ams_mass_fraction_error, measured_total_mass, measured_total_mass_error = ams_mass_fraction(ams_file, start_time, end_time)       
    checks=[False]
    
    counter = 0
    maxcounter=100
    print('sampling', Npart, 'particles...')
    pbar = tqdm.tqdm(total = maxcounter)
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
                
        # sample the particle diameters
        particle_diameters=np.zeros(Npart)
        particle_num_concs=np.zeros(Npart)
        model_Dps=np.logspace(1, 3, 1000)
        for ii in range(len(particle_species)):
            particle_type=particle_species[ii] 
            CDF_calc=np.zeros(len(model_Dps))               
            if particle_type=='BC':
                Ntot=1.0
                Dpg = 110
                sigma = 1.6
                SD_params=[Ntot, Dpg, sigma]
                CDF_calc=0.5*(1+erf((np.log(model_Dps)-np.log(Dpg))/(np.sqrt(2)*np.log(sigma))))
            elif particle_type=='OIN':
                Ntot=1.0
                Dpg = 110 # taken from accumulation mode of MAM4
                sigma = 1.6
                SD_params=[Ntot, Dpg, sigma]
                CDF_calc=0.5*(1+erf((np.log(model_Dps)-np.log(Dpg))/(np.sqrt(2)*np.log(sigma))))
            else:
                SD_params=[]
                for mode in range(modes):
                    Ntot=mode_fractions[particle_type][mode]*fitting_params[mode][0]
                    Dpg=fitting_params[mode][1]
                    sigma=fitting_params[mode][2]
                    SD_params.append(Ntot)
                    SD_params.append(Dpg)
                    SD_params.append(sigma)
                    CDF_calc+=mode_fractions[particle_type][mode]*0.5*(1+erf((np.log(model_Dps)-np.log(Dpg))/(np.sqrt(2)*np.log(sigma))))
            
            # sample parameters
            idx=np.where(np.array([ptypes])==particle_type)
            rands=np.random.rand(len(idx[1]))
            sampled_Dps=np.interp(rands, xp=CDF_calc, fp=model_Dps) # nm
            sampled_Ns=Nmodal_lognormal(sampled_Dps, *SD_params)
            
            # change number concentrations based on histogram
            for jj in range(0,len(Dp_uppers)):
                idx2=np.where(np.logical_and(sampled_Dps >= Dp_lowers[jj], sampled_Dps < Dp_uppers[jj]))
                N_in_bin = len(idx2[0])
                sampled_Ns[idx2[0]]/=N_in_bin
            
            # change number concentrations to match measurements
            idx3=np.where(sampled_Dps>=splat_cutoff)
            modeled_Ntot=np.sum(sampled_Ns[idx3[0]])
            mult=(avg_number_fraction[particle_type]*measured_Ntot)/modeled_Ntot
            sampled_Ns*=mult
            
            # save the sampled values
            particle_diameters[idx[1]]=sampled_Dps*1e-9 # m
            particle_num_concs[idx[1]]=sampled_Ns*100**3
            
        # match the measured number concentration
        mult=np.sum(measured_N)/(np.sum(particle_num_concs)/100**3)
        particle_num_concs*=mult
        
        # plot
        # bottom=np.zeros(len(Dp_uppers)-1)
        # plt.plot(Dp_uppers, measured_N, '-ko')
        # params=[]
        # for mode in range(modes):
        #     params.append(fitting_params[mode][0])
        #     params.append(fitting_params[mode][1])
        #     params.append(fitting_params[mode][2])
        # plt.plot(model_Dps, Nmodal_lognormal(model_Dps, *params), '-r')
        # for t, c in zip(avg_number_fraction.keys(), ['grey','r','b','g','C6','gold']):
        #     idx=np.where(np.array([ptypes])==t)
        #     hist=np.histogram(1e9*particle_diameters[idx[1]], bins=Dp_uppers, weights=particle_num_concs[idx[1]]/100**3)
        #     widths=hist[1][1:]-hist[1][:-1]
        #     plt.bar(Dp_uppers[:-1], hist[0], width=widths, bottom=bottom, facecolor=c, edgecolor='k', label=t)
        #     bottom+=hist[0]
        # plt.xscale('log')
        # plt.show()  
        # print(np.sum(particle_num_concs)/100**3, np.sum(measured_N))
        # print()
            
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
                            # if s not in ['IEPOX_OS', 'tetrol', 'tetrol_olig']:
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
        
        # check that the particle types match the minisplat measurement
        checks=[]
        for species in splat_species.keys():
            spec_idx=np.where(np.logical_and(np.array((ptypes))==species, particle_diameters>=splat_cutoff*1e-9))
            all_idx=np.where(particle_diameters>=splat_cutoff*1e-9)
            sampled=np.sum(particle_num_concs[spec_idx[0]])/np.sum(particle_num_concs[all_idx[0]])
            measured=avg_number_fraction[species]
            measured_error=number_fraction_error[species]
            if sampled >= measured-measured_error and sampled <= measured+measured_error:
                checks.append(True)
            else:
                checks.append(False)
        
        # check that the mass fractions match the AMS measurement
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
        
        counter += 1
        pbar.update(1)
        
        # ====================================================================
    pbar.close()
    
    
    if counter == maxcounter:
        print('Measurements not matched after', maxcounter, 'iterations...returning sampled values.')
    elif override_matching==True:
        print('Measurement override = True...returning sampled values.')
    print()
    print('Measured, modeled number fractions:')
    for species in splat_species.keys():
        spec_idx=np.where(np.logical_and(np.array((ptypes))==species, particle_diameters>=splat_cutoff*1e-9))
        all_idx=np.where(particle_diameters>=splat_cutoff*1e-9)
        sampled=np.sum(particle_num_concs[spec_idx[0]])/np.sum(particle_num_concs[all_idx[0]])
        measured=avg_number_fraction[species]
        measured_error=number_fraction_error[species]
        if sampled >= measured-measured_error and sampled <= measured+measured_error:
            print(species, measured, sampled, True)
        else:
            print(species, measured, sampled, False)
    print()
    print('Modeled, measured mass fractions:')
    for group in masses.keys():
        sampled=sampled_mass_fractions[group]
        measured=ams_mass_fractions[group]
        measured_error=ams_mass_fraction_error[group]
        if sampled >= measured-measured_error and sampled <= measured+measured_error:
            print(group, sampled, measured, True)
        else:
            print(group, sampled, measured, False)    
    
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
                                     splat_cutoff=85):
    
    pars = fit_Nmodal_distibution(size_distribution_file, modes, start_time, end_time, cloud_flag, CVI_flag)
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
    Dp_lower, Dp_upper, measured_N, measured_N_error = read_FIMS(size_distribution_file, start_time, end_time, cloud_flag, CVI_flag)
    Dp_mid = Dp_lower + 0.5*(Dp_upper - Dp_lower)
    measured_Ntot = np.sum(measured_N) # N per volume
    SA_dist = 4.0*np.pi*np.power((Dp_mid)/2, 2)*(measured_N) # aerosol SA/volume air
    measured_SA = np.sum(SA_dist) # total surface area per volume
    V_dist = (4.0/3.0)*np.pi*np.power((Dp_mid)/2, 3)*measured_N # volume aerosol/volume air
    measured_Vtot = np.sum(V_dist) # total surface area per volume
    avg_number_fraction, number_fraction_error = splat_number_fractions(splat_file, splat_species, start_time, end_time)    
    ams_mass_fractions, ams_mass_fraction_errors, measured_total_mass, measured_total_mass_error = ams_mass_fraction(ams_file, start_time, end_time)       
    
    # set initial RSS
    min_RSS = 1e10
    line_to_save=-1
    
    # loop through mode fractions and find the error associated with each combination
    print()
    print('optimizing size distribution,', modes, 'modes...')
    pbar = tqdm.tqdm(total = len(mode_fractions))
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
        
        # get the calculated mass fraction
        calculated_mass_fraction={}
        for ii in ams_mass_fractions.keys():
            calculated_mass_fraction[ii]=spec_masses[ii]/total_mass
        
        # get the calculated Ntot, Vtot, and SAtot
        calculated_Ntot = np.sum(total_Ns)
        calculated_Vtot = np.sum(total_Ns*(4/3)*np.pi*(0.5*Dp_mid)**3)
        calculated_SA = np.sum(total_Ns*4*np.pi*(0.5*Dp_mid)**2)
    
        # calculate RSS
        RSS=((calculated_Ntot - measured_Ntot)/measured_Ntot)**2\
            + ((calculated_SA - measured_SA)/measured_SA)**2\
            + ((calculated_Vtot - measured_Vtot)/measured_Vtot)**2
        for ii in ams_mass_fractions.keys():
            RSS+=((calculated_mass_fraction[ii]-ams_mass_fractions[ii])/ams_mass_fractions[ii])**2
        
        if RSS<min_RSS:
            min_RSS=RSS
            line_to_save=line
        
        pbar.update(1)
    pbar.close()

    output={}
    for ii in range(len(model_species)):
        output[model_species[ii]]=mode_fractions[line_to_save, ii*modes:ii*modes+modes]
        
    return output, output_pars

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


def fit_Nmodal_distibution(FIMS_file, modes, start_time, end_time, cloud_flag, CVI_flag):
    
    Dp_lower, Dp_upper, measured_N, measured_N_error = read_FIMS(FIMS_file, start_time, end_time, cloud_flag, CVI_flag)
    
    p0=[]
    lower_bounds=[]
    upper_bounds=[]
    for i in range(modes):
        p0.append(10*(i+1))
        p0.append(100*(i+1))
        p0.append(1.2)
        lower_bounds.append(0)
        lower_bounds.append(0)
        lower_bounds.append(1.0)
        upper_bounds.append(np.inf)
        upper_bounds.append(np.inf)
        upper_bounds.append(np.inf)
        
    pars, cov = curve_fit(Nmodal_lognormal, xdata=Dp_upper, ydata=measured_N, p0=p0, bounds=[lower_bounds, upper_bounds])
    
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

def measured_gas_phase(trace_gas_folder, gas_names):
    
    # get the altitude grid used for interpolating
    filename = os.path.join(trace_gas_folder, os.listdir(trace_gas_folder)[0])
    raw_data = np.loadtxt(filename, delimiter = ',', dtype='str')
    gas_data = {}
    for ii in range(0, len(raw_data[0])): 
        gas_data[str(raw_data[0, ii])] = np.array(raw_data[1:, ii], dtype = 'float64')
    alt_grid = np.linspace(np.min(gas_data['Alt']), np.max(gas_data['Alt']), 16)
    alt_mids = 0.5*(alt_grid[range(len(alt_grid)-1)] + alt_grid[range(1,len(alt_grid))])
    
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
            
            NOx_dat = list(np.zeros(len(alt_grid)-1))
            for rr in range(1, len(alt_grid)):
                idx, = np.nonzero((gas_data['Alt']>alt_grid[rr-1]) & (gas_data['Alt']<=alt_grid[rr]) & (gas_data['Value_ppb']>=0.0))
                NOx_dat[rr-1] = gas_data['Value_ppb'][idx]

            medianprops = dict(linestyle='-', linewidth=2, color='k')
            meanprops = dict(markerfacecolor='k',markeredgecolor='k')   
            
            widths = 0.6*(alt_grid[range(1,len(alt_grid))] - alt_grid[range(len(alt_grid)-1)])

            bplot = plt.boxplot(NOx_dat, positions=alt_mids,
                      showfliers=False,whis=[5,95],widths=widths,
                      medianprops=medianprops,meanprops=meanprops,patch_artist=True,
                      capprops = {'color': 'k'}, whiskerprops = {'color': 'k'}, vert=False)
            plt.close()
            
            NOx_medians = []
            NOx_alts=[]
            for ii, (med) in enumerate(bplot['medians']):
                if med.get_xdata()[0]>=0:
                    NOx_medians.append(med.get_xdata()[0])
                    NOx_alts.append(med.get_ydata()[0])
                   
            raw_data = np.loadtxt(filename2, delimiter = ',', dtype='str')
            NO_data = {}
            for ii in range(0, len(raw_data[0])): 
                NO_data[str(raw_data[0, ii])] = np.array(raw_data[1:, ii], dtype = 'float64')
            
            NO_dat = list(np.zeros(len(alt_grid)-1))
            for rr in range(1, len(alt_grid)):
                idx, = np.nonzero((gas_data['Alt']>alt_grid[rr-1]) & (gas_data['Alt']<=alt_grid[rr]) & (gas_data['Value_ppb']>=0.0))
                NO_dat[rr-1] = gas_data['Value_ppb'][idx]

            medianprops = dict(linestyle='-', linewidth=2, color='k')
            meanprops = dict(markerfacecolor='k',markeredgecolor='k')   
            
            widths = 0.6*(alt_grid[range(1,len(alt_grid))] - alt_grid[range(len(alt_grid)-1)])

            bplot = plt.boxplot(NO_dat, positions=alt_mids,
                      showfliers=False,whis=[5,95],widths=widths,
                      medianprops=medianprops,meanprops=meanprops,patch_artist=True,
                      capprops = {'color': 'k'}, whiskerprops = {'color': 'k'}, vert=False)
            plt.close()
            
            NO_medians = []
            NO_alts=[]
            for ii, (med) in enumerate(bplot['medians']):
                if med.get_xdata()[0]>=0:
                    NO_medians.append(med.get_xdata()[0])
                    NO_alts.append(med.get_ydata()[0])
            
            if np.nansum(NOx_medians)>0 and np.nansum(NO_medians)>0:
                gas_data_all[gas]={}
                gas_data_all[gas]['ppb']=np.array((NOx_medians))-np.array((NO_medians))
                gas_data_all[gas]['alt']=NOx_alts 
            else:
                try:
                    alts, medians = get_AGFL_profile(gas, alt_mids, trace_gas_folder)
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
            
            if file_to_read:
                raw_data = np.loadtxt(file_to_read, delimiter = ',', dtype='str')
                gas_data = {}
                for ii in range(0, len(raw_data[0])): 
                    gas_data[str(raw_data[0, ii])] = np.array(raw_data[1:, ii], dtype = 'float64')
                
                gas_dat = list(np.zeros(len(alt_grid)-1))
                for rr in range(1, len(alt_grid)):
                    idx, = np.nonzero((gas_data['Alt']>alt_grid[rr-1]) & (gas_data['Alt']<=alt_grid[rr]) & (gas_data['Value_ppb']>=0.0))
                    gas_dat[rr-1] = gas_data['Value_ppb'][idx]
    
                medianprops = dict(linestyle='-', linewidth=2, color='k')
                meanprops = dict(markerfacecolor='k',markeredgecolor='k')   
                
                widths = 0.6*(alt_grid[range(1,len(alt_grid))] - alt_grid[range(len(alt_grid)-1)])

                bplot = plt.boxplot(gas_dat, positions=alt_mids,
                          showfliers=False,whis=[5,95],widths=widths,
                          medianprops=medianprops,meanprops=meanprops,patch_artist=True,
                          capprops = {'color': 'k'}, whiskerprops = {'color': 'k'}, vert=False)
                plt.close()
                
                medians = []
                alts=[]
                for ii, (med) in enumerate(bplot['medians']):
                    if med.get_xdata()[0]>=0:
                        medians.append(med.get_xdata()[0])
                        alts.append(med.get_ydata()[0])
                
                if np.nansum(medians)>0:
                    gas_data_all[gas]={}
                    gas_data_all[gas]['ppb']=medians
                    gas_data_all[gas]['alt']=alts 
                else:
                    try:
                        alts, medians = get_AGFL_profile(gas, alt_mids, trace_gas_folder)
                        gas_data_all[gas]={}
                        gas_data_all[gas]['ppb']=medians
                        gas_data_all[gas]['alt']=alts
                    except:
                        print('WARNING: No data for', gas, '!')
                        sys.exit()
                    
            else:
                try:
                    alts, medians = get_AGFL_profile(gas, alt_mids, trace_gas_folder)
                    gas_data_all[gas]={}
                    gas_data_all[gas]['ppb']=medians
                    gas_data_all[gas]['alt']=alts
                except:
                    print('WARNING: No data for', gas, '!')
                    sys.exit()
    
    return gas_data_all

def get_AGFL_profile(gas, alt_mids, trace_gas_folder):
    
    filename = os.path.join(trace_gas_folder, 'AGFL_atmosphere.txt')
    raw_data = np.loadtxt(filename, dtype='str')
    AGFL = {}
    AGFL[str(raw_data[0, 0])] = np.array(raw_data[1:, 0], dtype='float64')
    for i in range(0, len(raw_data[0])): 
        AGFL[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
        
    return 1000*AGFL['z'], 1000*AGFL[gas]
    

# ================================ main function ===============================

total_Np = int(sys.argv[1])
particles_per_run = int(sys.argv[2])

if total_Np%particles_per_run != 0:
    print('WARNING: The total number of particles is not divisible by the number of particles per run.')
    sys.exit()

splat_species = {'BC': ['soot'],
                  'OIN': ['Dust'],
                  'AS': ['sulfate_nitrate_org'],
                  'AN': ['nitrate_amine_org'], 
                  'OC': ['org28', 'org30_43', 'BB_SOA', 'org_amines', 'BB', 'pyridine'], 
                  'IEPOX': ['IEPOX_SOA']}

mass_fractions={'IEPOX': [[0.5,0.75,0.1], ['IEPOX_OS','tetrol','tetrol_olig']],
                'AS': [[0.5,0.75,0.1], ['SO4']],
                'AN': [[0.5,0.75,0.1], ['NO3']], 
                'OC': [[0.5,0.75,0.1], ['OC']], 
                'BC': [[0.5,0.75,0.1], ['BC']],
                'OIN': [[0.5,0.75,0.1], ['OIN']]}

gas_names = ['SO2', 'O3', 'H2O2', 'NO2', 'IEPOX']
diameters, num_concs, aero_spec_names, aero_spec_fracs, pHs, gas_data=splat_setup(Npart=total_Np, 
                      optimization_points=100, modes=2, mass_thresholds=mass_fractions,
                      size_distribution_file='../datasets/HISCALE_data_0425/BEASD_G1_20160425155810_R2_HISCALE_001s.txt',
                      splat_file='../datasets/HISCALE_data_0425/Splat_Composition_25-Apr-2016.txt',
                      aimms_file='../datasets/HISCALE_data_0425/AIMMS20_G1_20160425155810_R2_HISCALE020h.txt',
                      trace_gas_folder='../datasets/HISCALE_data_0425/CIMS_data',
                      splat_species=splat_species,
                      mass_fractions=mass_fractions,
                      gas_names=gas_names,
                      ams_file='../datasets/HISCALE_data_0425/HiScaleAMS_G1_20160425_R0.txt',
                      start_time=960, end_time=59160, 
                      cloud_flag=0, CVI_flag=0, override_matching=True)

files = ['particles.py', 'constants.py', 'scenario.py', 'aerosol_species.py',
         'utilities.py', 'systems.py', 'driver.py', 'visualization.py', 
         'TraceGases.py', 'Reactions.py', 'MAIN_les.py', 'parcel.py']

directories = ['../multipart/processes']

les_number=np.array((2000*np.random.rand(1)), dtype='int64')
les_number=str(les_number[0]).zfill(6)
for i in range(0, int(total_Np/particles_per_run)):

    # make the sub-run directories
    os.mkdir('run_'+str(i+1))
    
    # write pickle files that save the initial aerosol properties
    f = open('run_'+str(i+1)+'/diameters', 'wb')
    pickle.dump(diameters[i*particles_per_run:i*particles_per_run+particles_per_run], f)
    
    f = open('run_'+str(i+1)+'/num_concs', 'wb')
    pickle.dump(num_concs[i*particles_per_run:i*particles_per_run+particles_per_run], f)
    
    f = open('run_'+str(i+1)+'/aero_spec_names', 'wb')
    pickle.dump(aero_spec_names[i*particles_per_run:i*particles_per_run+particles_per_run], f)
    
    f = open('run_'+str(i+1)+'/aero_spec_fracs', 'wb')
    pickle.dump(aero_spec_fracs[i*particles_per_run:i*particles_per_run+particles_per_run], f)
    
    f = open('run_'+str(i+1)+'/pHs', 'wb')
    pickle.dump(pHs[i*particles_per_run:i*particles_per_run+particles_per_run], f)
    
    f = open('run_'+str(i+1)+'/gas_data', 'wb')
    pickle.dump(gas_data, f)
    
    f = open('run_'+str(i+1)+'/trajectory_number', 'wb')
    pickle.dump(les_number, f)
    
    # copy model scripts into the sub-directories
    for file in files:
        shutil.copy(file, 'run_'+str(i+1)+'/'+file)
    for directory in directories:
        source = directory
        destination = source.replace('.', '')
        destination = destination.replace('/', '')
        destination = destination.replace('multipart', '')
        shutil.copytree(source, 'run_'+str(i+1)+'/'+destination)
    
for file in files:
    os.remove(file)
shutil.rmtree('processes')
shutil.rmtree('__pycache__')

