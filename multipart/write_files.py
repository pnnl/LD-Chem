#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce and Payton Beeler
"""
import numpy as np
import pickle

def write_original(time, state, filename, specdata_path='../species_data'):
    
    data_dict = last_state_array(time, state, specdata_path=specdata_path)
    if isinstance(data_dict['gases'], np.ndarray):
        data_dict['gases']=np.expand_dims(data_dict['gases'], axis=0)
    data_dict['particles']=np.expand_dims(data_dict['particles'], axis=0)
    pickle.dump(data_dict, open(filename, 'wb'))
    return
    
    
def overwrite(time, trajectory, filename, specdata_path='../species_data'):

    original_trajectory=pickle.load(open(filename, 'rb'))
    new_state=last_state_array(time, trajectory, specdata_path=specdata_path)
    
#    print()
#    print(new_state['times'])
#    print(np.sum(new_state['particles']))
#    print(str(np.sum(new_state['particles']))=='nan')
#    for ii in range(len(new_state['particle species'])):
#        print(new_state['particle species'][ii], new_state['particles'][:, ii])
#    print()
    
    new_trajectory={}
    new_trajectory['gas species']=original_trajectory['gas species']
    new_trajectory['particle species']=original_trajectory['particle species']
    for kk in ['times','x','y','z','S','T','P','activated fraction']:
        new_trajectory[kk]=np.append(original_trajectory[kk], new_state[kk])
    
    new_trajectory['particles']=np.concatenate((original_trajectory['particles'], new_state['particles'][np.newaxis, :, :]), axis=0)
    
    if isinstance(new_state['gases'], np.ndarray):
        new_trajectory['gases']=np.vstack((original_trajectory['gases'], new_state['gases']))
    else:
        new_trajectory['gases']=original_trajectory['gases']
    
    #for kk in original_trajectory.keys():
    #    try:
    #        print(kk, original_trajectory[kk].shape, new_trajectory[kk].shape)
    #    except:
    #        print(kk, original_trajectory[kk], new_trajectory[kk])
            
    #print()
    #print(new_trajectory['times'])
    #print()
    pickle.dump(new_trajectory, open(filename, 'wb'))
    
    return


def last_state_array(time, state, specdata_path='../species_data'):
    
    aq_order=np.zeros(0)
    aero_datafile = specdata_path+'/aero_data.dat'
    with open(aero_datafile) as data_file:
        for line in data_file:
            try:
                name_in_file,density,ions_in_solution,molar_mass,kappa = line.split()
                aq_order=np.append(aq_order, name_in_file)
            except:
                pass
    aq_order=np.append(aq_order, 'num conc')
    aq_order=np.append(aq_order, 'Ddry')
    aq_order=np.append(aq_order, 'Dwet')
    aq_order=np.append(aq_order, 'kappa')
    
    num_particles=len(state.particle_population.particles)
        
    output_dict={}
    output_dict['times']=np.array([time]) #trajectory.ts[-1]])
    output_dict['particles']=np.zeros((num_particles, len(aq_order)))
    output_dict['particle species']=aq_order
    output_dict['x']=np.array([state.x])
    output_dict['y']=np.array([state.y])
    output_dict['z']=np.array([state.z])
    output_dict['S']=np.array([state.S])
    output_dict['T']=np.array([state.T])
    output_dict['P']=np.array([state.P])
    output_dict['activated fraction']=np.array([state.get_activated_fraction()])
    
    particles = state.particle_population.particles
    num_concs = state.particle_population.num_concs

    for pNumber, (particle, num_conc) in enumerate(zip(particles, num_concs)):
        
        Ddry=particle.get_Ddry()
        traj_idx = np.where(aq_order=='Ddry')[0][0]
        output_dict['particles'][pNumber, traj_idx]=Ddry
    
        Dwet=particle.get_Dwet()
        traj_idx = np.where(aq_order=='Dwet')[0][0]
        output_dict['particles'][pNumber, traj_idx]=Dwet
        
        for species in aq_order:
            if species=='num conc':
                traj_idx = np.where(aq_order==species)[0][0]
                output_dict['particles'][pNumber, traj_idx]=num_conc
            elif species=='kappa':
                traj_idx = np.where(aq_order==species)[0][0]
                output_dict['particles'][pNumber, traj_idx]=particle.get_tkappa()
            else:
                traj_idx = np.where(aq_order==species)[0][0]
                particle_idx = particle.get_species_idx(species)
                if particle_idx!=None:
                    output_dict['particles'][pNumber, traj_idx]=particle.masses[particle_idx]
    
    if state.TraceGas_population:
        gas_order=np.zeros(0)
        for gas in state.TraceGas_population.gases:
            gas_order=np.append(gas_order,gas.name)
    
        output_dict['gas species']=gas_order
        output_dict['gases']=np.zeros(len(gas_order))
        # gases = state.TraceGas_population.gases
        # gas_concs = state.TraceGas_population.concs
        for ii, (species) in enumerate(gas_order):
            traj_idx = state.TraceGas_population.get_species_idx(species)
            output_dict['gases'][ii]=state.TraceGas_population.concs[traj_idx]
    else:
        output_dict['gas species']=None
        output_dict['gases']=None
        
    return output_dict
