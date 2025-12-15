#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce and Payton Beeler
"""
import numpy as np
import pickle

def write_original(time, state, filename, specdata_path='species_data'):
    data_dict = last_state_array(time, state, specdata_path=specdata_path)
    if isinstance(data_dict['gases'], np.ndarray):
        data_dict['gases']=np.expand_dims(data_dict['gases'], axis=0)
    data_dict['particles']=np.expand_dims(data_dict['particles'], axis=0)
    pickle.dump(data_dict, open(filename, 'wb'))
    return
    
def overwrite(time, ParcelState_New, filename, specdata_path='species_data'):
    original_trajectory=pickle.load(open(filename, 'rb'))
    new_state=last_state_array(time, ParcelState_New, specdata_path=specdata_path)
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
    pickle.dump(new_trajectory, open(filename, 'wb'))
    return

def last_state_array(time, state, specdata_path='species_data'):
    aq_order = np.empty(state.particles.spec_masses.shape[1], dtype='U20')
    for ii, (species) in enumerate(state.particles.species):
        aq_order[ii]=species.name
    aq_order=np.append(aq_order, 'num conc')
    aq_order=np.append(aq_order, 'Ddry')
    aq_order=np.append(aq_order, 'Dwet')
    aq_order=np.append(aq_order, 'kappa')
    num_particles=len(state.particles.spec_masses)
    output_dict={}
    output_dict['times']=np.array([time])
    output_dict['particles']=np.zeros((num_particles, len(aq_order)))
    output_dict['particle species']=aq_order
    output_dict['x']=np.array([state.x])
    output_dict['y']=np.array([state.y])
    output_dict['z']=np.array([state.z])
    output_dict['S']=np.array([state.S])
    output_dict['T']=np.array([state.T])
    output_dict['P']=np.array([state.P])
    output_dict['activated fraction']=np.array([state.get_activated_fraction()]) 
    output_dict['particles'][:,np.where(aq_order=='num conc')[0][0]]=state.particles.num_concs
    output_dict['particles'][:,np.where(aq_order=='Ddry')[0][0]]=state.particles.get_particle_var('dry_diameter')
    output_dict['particles'][:,np.where(aq_order=='Dwet')[0][0]]=state.particles.get_particle_var('wet_diameter')
    output_dict['particles'][:,np.where(aq_order=='kappa')[0][0]]=state.particles.get_particle_var('tkappa')
    output_dict['particles'][:,:state.particles.spec_masses.shape[1]]=state.particles.spec_masses
    if state.gas:
        gas_order=np.empty(state.gas.concs.shape, dtype='U20')
        for ii, (gas) in enumerate(state.gas.gases):
            gas_order[ii]=gas.name
        output_dict['gas species']=gas_order
        output_dict['gases']=state.gas.concs
    else:
        output_dict['gas species']=None
        output_dict['gases']=None
    return output_dict
