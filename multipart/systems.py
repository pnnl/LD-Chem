""" systems - types and functions used to create different model systems. 

@author: Laura Fierce
"""

import numpy as np
import numba as nb
from numba.pycc import CC
import sys

import matplotlib.pyplot as plt

from dataclasses import dataclass
from dataclasses import replace

from typing import Tuple
from typing import Callable

from particles import ParticlePopulation
from TraceGases import TraceGasPopulation, GasSpecies
from processes import water_uptake, cocondensation
from processes import air_thermo
from processes import fluctuations
from processes.water_uptake import dlnr_dt
import constants as c

from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
from scipy.integrate import solve_ivp, ode

from copy import copy


    
# should this be in here?
@dataclass
class ParcelState:
    # t: float
    
    x: float
    y: float
    z: float
    
    u: float
    v: float
    w: float
    
    S: float
    P: float
    T: float
    
    # gas_mixture: GasMixture
    particle_population: ParticlePopulation
    TraceGas_population: TraceGasPopulation
    
    
@dataclass
class Processes:
    """AerosolProcesses: a definition of a set of aerosol processes under consideration"""
    condensation: bool = True
    collisions: bool = False
    cocondensation: bool = False
    chemistry: bool = False
    freezing: bool = False
    settling: bool = False
    fluctuations: bool = False
    
    
@dataclass
class GasFeedback:
    names: Tuple[GasSpecies, ...]
    dc_dts: Tuple[float, ...]
    
@dataclass
class Feedbacks:
    dwc_dt: float = 0.
    dwc_dt_next: float = 0.
    dwc: float = 0.
    dwi_dt: float = 0.
    gases: Tuple[GasFeedback, ...] = None

@dataclass
class ParcelTrajectory:
    """ ParcelTrajectory: definition of aerosol parcel that evolves over time """
    ts: Tuple[float, ...]
    parcel_states: Tuple[ParcelState, ...]

@dataclass
class TrajectoryInteractions: # should some of these be set to "none"?
    uniform: bool = False # if ture, all particles across trajectories interact, regardless of their location in space
    neighbors: bool = False # if true, parcels located near each other in space interact
    gns: bool = False # if ture, neigbor
    homogeneous_mixing: bool = False # just throwing things in here for now...
    
@dataclass
class TrajectoryEnsemble:
    parcel_trajectories: Tuple[ParcelTrajectory, ...]
    trajectory_interactions: TrajectoryInteractions
        

def update_state(
        ParcelState_0, processes, dt, sigma=1.0, verbosity=50, 
        C0=3.,accom=1.,radius_scale='lin',solver='CVODE', 
        mechanism_data_path='../mechanisms/'): 
    
    # print()
    # print()
    # print(ParcelState_0)
    
    ParcelState_1, feedbacks_1 = update_particle_population(
        ParcelState_0, processes, dt, 
        radius_scale=radius_scale,solver=solver,
        accom=accom, sigma=sigma, verbosity=verbosity,mechanism_data_path=mechanism_data_path)            
    ParcelState_2 = update_air(
        ParcelState_1, processes, feedbacks_1, dt, 
        verbosity=verbosity, solver=solver)  
    
    # print()
    # print(ParcelState_2.particle_population.particles[0].masses-ParcelState_0.particle_population.particles[0].masses)
    # print()
    # print(feedbacks_1)
    # print()
    # sys.exit()
    
    ParcelState_next = replace(ParcelState_2)    
    return ParcelState_next

# put all the functions for particle state in another file?
def update_particle_population(
        ParcelState_0, processes, dt, accom=1., sigma=1.0,
        verbosity=50,radius_scale='lin',solver='CVODE',
        mechanism_data_path='../mechanisms/'): # fix later -- put in species
    t0 = 0.
    # switches0 = [False]
    # wrap all of the condensation functions in to its own updater? "condense"
    T0 = ParcelState_0.T
    P0 = ParcelState_0.P
    S0 = ParcelState_0.S # put this into gas mixture?
    wv0 = air_thermo.S_to_wv(S0,T0,P0)
    
    dwc_dt = 0.
    dwc_dt_next = 0.
    dwc = 0.
    dwi_dt = 0.
    # dwv_dt = 0.
    
    ParcelState_next = copy(ParcelState_0) # maybe??
    
    # note: I've computed this in several places -- need a more elegant way to deal with this
    T_c = T0 - 273.15  # convert temperature to Celsius
    pv_sat = water_uptake.es(T_c)  # saturation vapor pressure
    e = S0 * pv_sat  # water vapor pressure
    rho_air_dry = (P0 - e) / c.Rd / T0
    
    sw0 = False # if False, mh2o (or r, etc.) hasn't yet become negative
    # put this in a condensation function?
    
    if cocondensation:
        dCgas_dts_all = []
    
    particles = []
    for ii,(particle,num_conc) in enumerate(zip(ParcelState_0.particle_population.particles,ParcelState_0.particle_population.num_concs)):
        new_particle = copy(particle)
        if processes.condensation:            
            r_dry_i = particle.get_Ddry()/2.
            tkappa_i = particle.get_tkappa()
            r0=particle.get_Dwet()/2.                     
            if radius_scale == 'log':
                lnr0 = np.log(r0)
                if solver == 'CVODE':  
                    rhs = lambda t, lnr: water_uptake.dlnr_dt(lnr, r_dry_i, tkappa_i, P0, T0, S0, wv0, accom)
                    prob = Explicit_Problem(rhs, lnr0)
                    sim = CVode(prob)
                    sim.atol=1.0e-10
                    sim.rtol=1.0e-10
                    sim.verbosity = verbosity
                    output=sim.simulate(dt)
                    r_next = np.exp(output[1][-1,0])
                    
                    # dwc_dt += rhs(0., r0)*water_uptake.dh2o_dr(r0)*num_conc/rho_air_dry
                    # dwc_dt_next += rhs(0., r_next)*water_uptake.dh2o_dr(r_next)*num_conc/rho_air_dry
                    
                elif solver == 'ode15s':
                    ode15s = ode(water_uptake.water_uptake_wrapper).set_integrator('lsoda', method='bdf', rtol=1E-6, atol=1E-12, nsteps=5000)
                    ode15s.set_initial_value(lnr0, t0).set_f_params(r_dry_i, tkappa_i, P0, T0, S0, wv0, accom, radius_scale)
                    r_next = np.exp(ode15s.integrate(ode15s.t+dt)[0])
                    
                    # dwc_dt += water_uptake.dlnr_dt(r0, r_dry_i, tkappa_i, P0, T0, S0, wv0, accom=accom)*water_uptake.dh2o_dr(r0)*num_conc/rho_air_dry
                    # dwc_dt_next += water_uptake.dlnr_dt(r_next, r_dry_i, tkappa_i, P0, T0, S0, wv0, accom=accom)*water_uptake.dh2o_dr(r_next)*num_conc/rho_air_dry
                    
            elif radius_scale == 'lin':   
                if solver == 'CVODE':
                    rhs = lambda t, r: water_uptake.dr_dt(r, r_dry_i, tkappa_i, P0, T0, S0, wv0, accom=accom)
                    prob = Explicit_Problem(rhs, r0, sw0)
                    sim = CVode(prob)
                    sim.atol=1.0e-10
                    sim.rtol=1.0e-10
                    sim.verbosity=verbosity
                    output=sim.simulate(dt)
                    r_next=output[1][-1,0]  
                    
                    # dwc_dt += rhs(0., r0)*water_uptake.dh2o_dr(r0)*num_conc/rho_air_dry
                    # dwc_dt_next += rhs(0., r_next)*water_uptake.dh2o_dr(r_next)*num_conc/rho_air_dry
                    
                elif solver == 'ode15s':
                    ode15s = ode(water_uptake.water_uptake_wrapper).set_integrator('lsoda', method='bdf', 
                                                          rtol=1E-6, atol=1E-16, nsteps=5000)
                    ode15s.set_initial_value(r0, t0).set_f_params(r_dry_i, tkappa_i, P0, T0, S0, wv0, accom, radius_scale)
                    r_next = ode15s.integrate(ode15s.t+dt)[0]       
                    
                    # dwc_dt += water_uptake.dr_dt(r0, r_dry_i, tkappa_i, P0, T0, S0, wv0, accom=accom)*water_uptake.dh2o_dr(r0)*num_conc/rho_air_dry
                    # dwc_dt_next += water_uptake.dr_dt(r_next, r_dry_i, tkappa_i, P0, T0, S0, wv0, accom=accom)*water_uptake.dh2o_dr(r_next)*num_conc/rho_air_dry
            
            dwc_dt += ((4.0*np.pi*c.rho_w)/rho_air_dry)*num_conc*(r0**2)*(r_next-r0)
            
            # print(r_next-r0, dwc_dt) 
                
            m_h2o_0 = particle.get_mass_h2o()
            # new_particle = replace(particle)
            new_particle = water_uptake.update_particle(new_particle, r_next)
            
            # ----------------
            # particle = water_uptake.update_particle(particle, r0, np.exp(lnr_next))
            # dwc_dt += rhs(0., lnr0)*water_uptake.dh2o_dlnr(lnr0)*num_conc/rho_air_dry
            # dwc_dt_next += rhs(0., lnr_next)*water_uptake.dh2o_dlnr(lnr_next)*num_conc/rho_air_dry
            # ----------------
            
            m_h2o_next = new_particle.get_mass_h2o()
            dwc += num_conc*(m_h2o_next - m_h2o_0)/rho_air_dry
    
            # ----------------
            # new_particle = copy(particle)
            # new_particle.masses[new_particle.idx_h2o] = m_h2o_next
            # ----------------

        if processes.cocondensation:
            water_volume = particle.get_vol_tot()-particle.get_vol_dry() # m^3
            dCgas_dt = []
            if ParcelState_0.TraceGas_population.gases:
                for gas, gas_ppb in zip(ParcelState_0.TraceGas_population.gases, ParcelState_0.TraceGas_population.concs):
                    if gas.molar_mass > 0.0:
                        Cgas_0 = (gas_ppb*1e-9*P0)/(8.314*T0) # mol/m^3
                        radius = new_particle.get_Ddry()/2.0
                        Caq_0 = (new_particle.masses[new_particle.get_species_idx(gas.name)]/gas.molar_mass)/water_volume # mol/m^3                    
                        
                        if solver == 'CVODE':
                            rhs = lambda t, Caq: cocondensation.dCaq_dt(Caq, Cgas_0, radius, T0, gas.get_Heff(T0), gas.alpha, gas.molar_mass)
                            prob = Explicit_Problem(rhs, Caq_0)
                            sim = CVode(prob)
                            sim.atol=1.0e-10
                            sim.rtol=1.0e-10
                            sim.verbosity=verbosity
                            output=sim.simulate(dt)
                            Caq_next=output[1][-1,0] # mol/m^3
                        
                        elif solver == 'ode15s':
                            ode15s = ode(cocondensation.cocondensation_wrapper).set_integrator('lsoda', method='bdf', 
                                                                  rtol=1E-6, atol=1E-16, nsteps=5000)
                            ode15s.set_initial_value(Caq_0, t0).set_f_params(Cgas_0, r0, T0, gas.get_Heff(T0), gas.alpha, gas.molar_mass)
                            Caq_next = Caq_0 #ode15s.integrate(ode15s.t+dt)[0]  # mol/m^3
                    
                        dCgas_dt.append(-1.0*water_volume*num_conc*(Caq_next-Caq_0)) # mol/m^3
                        new_particle.masses[new_particle.get_species_idx(gas.name)] = Caq_next*water_volume*gas.molar_mass # kg
                    else:
                        dCgas_dt.append(0.0)
                        
                dCgas_dts_all.append(dCgas_dt)   

        particles.append(new_particle)
    
    # add up the total amount of gas species that condensed
    dCgas_dt_total = []
    gases = []

    dCgas_dts_all = np.array((dCgas_dts_all))
    if processes.cocondensation:
        if ParcelState_0.TraceGas_population.gases:
            for gas in ParcelState_0.TraceGas_population.gases:
                species = gas.name
                idx = ParcelState_0.TraceGas_population.get_species_idx(species)
                dCgas_dt_total.append(1e9*np.sum(dCgas_dts_all[:,idx])*((8.314*T0)/P0)) # ppb
                # if -1.0*1e9*np.sum(dCgas_dts_all[:,idx])*((8.314*T0)/P0) > ParcelState_0.TraceGas_population.concs[idx]:
                #     masses, new_feedback = cocondensation.handle_negative_event(ParticlePopulation(particles=particles, num_concs=ParcelState_0.particle_population.num_concs, ids=ParcelState_0.particle_population.ids), gas, ParcelState_0.TraceGas_population.concs[idx], T0, P0)
                #     print(masses)
                gases.append(gas.name) 
        else:
            gases=None
            dCgas_dt_total=None
        gas_feedbacks = GasFeedback(names=gases, dc_dts=dCgas_dt_total)    
    else:
        gas_feedbacks = None
        
    ParcelState_next.particle_population.particles = particles
    feedbacks = Feedbacks(dwc_dt=dwc_dt, dwc_dt_next=dwc_dt_next, dwc=dwc, dwi_dt=dwi_dt, gases=gas_feedbacks)
    
    return ParcelState_next, feedbacks

# ORIG (from chatGPT)
# Define an event to stop integration if y becomes negative
# def negative_event(t, y, sw):
#     return y[0] # no matter the state of sw, negative event if crosses 0

# Define an event to stop integration if y becomes negative
def negative_event(t, y, sw):
    if sw[0]: # if switch has been activated, was already negative (no event)
        return np.array([1.])
    else:
        return np.array([y[0]]) # no matter the state of sw, negative event if crosses 0



def handle_negative_event(solver, event_info):
    """
    Event handling. This functions is called when Assimulo finds an event as
    specified by the event functions.
    """
    state_info = event_info[0] #We are only interested in state events

    if state_info[0] != 0: #Check if the first event function has been triggered
        if solver.sw[0]: #If True, state was already negative
            solver.sw[0] = False #Flip the switch to say it crossed back to positive
        else:
            solver.y[0] = 0.#If started positive and went negative, make y[0]=0
            solver.sw[0] = True #Flip the switch to say it went negative
        
# def state_events(t,y,sw):
#     """
#     This is the function that keeps track of  events. When the sign
#     of any of the functions changed, we have an event.
#     """
#     if sw[0]:
#         e_0 = 0. # if it crossed the zero line
#     else:
#         e_0 = y[0] # it didn't?
    
#     return np.array([e_0])

# def handle_event(solver, event_info):
#     """
#     Event handling. This functions is called when Assimulo finds an event as
#     specified by the event functions.
#     """
#     state_info = event_info[0] #We are only interested in state events

#     if state_info[0] != 0: #Check if the first event function has been triggered
#         if solver.sw[0]: #If the switch is True
#             solver.y[1] = -0.9*solver.y[1] #Change the velocity and lose energy
            
#         solver.sw[0] = not solver.sw[0] #Change event function
        
        
# import numpy as np
# from assimulo.problem import Explicit_Problem
# from assimulo.solvers import CVode

# # Define the system of ODEs
# def system(t, y):
#     dydt = y * (1 - y)  # Example system with growth term
#     return [dydt]

# # Define an event function to check if y becomes zero
# def zero_event(y):
#     return y[0]  # We want to detect when y[0] == 0

# # Create an Assimulo problem instance
# y0 = [1.0]  # Initial condition
# t0 = 0.0    # Initial time
# problem = Explicit_Problem(system, y0, t0)

# # Create an Assimulo solver instance
# solver = CVode(problem)

# # Set solver parameters
# solver.atol = 1e-6  # Absolute tolerance
# solver.rtol = 1e-6  # Relative tolerance

# # Define a custom simulation loop with manual event handling
# def simulate_with_events(solver, tfinal, dt=0.1):
#     times = []
#     states = []
    
#     t = solver.t
#     y = solver.y.copy()
    
#     while t < tfinal:
#         times.append(t)
#         states.append(y.copy())
        
#         # Integrate up to the next step
#         t_next = min(t + dt, tfinal)
#         solver.integrate(t_next)
        
#         # Check for events
#         if zero_event(solver.y) == 0:
#             # Handle the event by resetting the state
#             solver.y[0] = 0.0
#             print(f"Event detected at time {solver.t}. State reset to zero.")
        
#         t = solver.t
#         y = solver.y.copy()
    
#     times.append(t)
#     states.append(y.copy())
    
#     return np.array(times), np.array(states)

# # Simulate the ODE
# tfinal = 10.0
# t, y = simulate_with_events(solver, tfinal)

# # Print the solution
# print('t:', t)
# print('y:', y)

# =============================================================================
# Lessons so far:
#     - attach some functions to Particle to compute params from masses and AeroSpecs
#     - put water vapor with the gas mixtures?
#     - more collision processes to another file
# 
# =============================================================================
    
# # =============================================================================
# # move to a separate "coagulation" file        
# # =============================================================================
# def coagulate_SDM(ParcelState_0):
#     T0 = ParcelState.T
#     P0 = ParcelState.P
    
#     Mks0 = ParcelState_0.particle_population.get_Mks()
#     Ns0 = ParcelState_0.particle_population.num_concs
    
#     state_0, state_varnames, state_vardims_tuple = ravel_variables((Mks0,Ns0), ('Mks0','Ns0'))
    
    
#     # put this in a different function
#     dMks_dt = np.zeros(Mks0.shape)
#     dNs_dt = np.zeros(Ns0.shape)
#     for ii,(Mk0,N0) in enumerate([Mks0,Ns0]):
        
#         vol_i = get_vol(Mk,N,densities)
#         den_i = Particle_i.get_effective_density()
#         for jj,(Particle_j,N_j) in enumerate(ParcelState_0.particle_population.particles[(ii+1):],ParcelState_0.particle_population.num_concs[(ii+1):]):
#             vol_j = Particle_j.get_vol()
#             den_j = Particle_j.get_effective_density()
#             K_ij = brownian_kernel(vol_i, vol_j, den_i, den_j, T0, P0)
#             dNdt_ij = K_ij*N_i*N_j
            
#             if vol_i >= vol_j:
#                 dMks_dt[ii,:] = dNdt_ij*Particle_j.masses
#                 dNs_dt[jj] -= dNdt_ij
#             else:    
#                 dMks_dt[jj,:] = dNdt_ij*Particle_i.masses
#                 dNs_dt[ii] -= dNdt_ij
    
#     rhs, rhs_varnames, rhs_vardims_tuple = ravel_variables((dMks_dt,dNs_dt), ('dMks_dt','dNs_dt'))


# def get_vol(Mk,N,densities):
#     Vtot = np.sum(Mk/densities)
#     return Vtot/N

# def get_dens(Mk,densities):
#     Vtot = np.sum(Mk/densities)
#     Mtot = np.sum(Mk)
#     return Mtot/Vtot
    
# def ravel_variables(varvals_tuple, varnames_tuple):
#     varvals = []
#     varnames = []
    
#     vardims_tuple = ()
#     for ii,(varval,varname) in enumerate(zip(varvals_tuple, varnames_tuple)):
#         try:
#             varvals.append(varval.ravel())
#             for ii in range(len(varval.ravel())):
#                 varnames.append(varname)
#             vardims_tuple += (varval.shape)
#         except:
#             varvals.append(varval)
#             varnames.append(varname)
#             vardims_tuple += (1,)
#     return varvals, varnames, vardims_tuple

# def unravel_variables(varvals, varnames, vardims_tuple):
#     varnames_tuple = tuple(np.unique(varnames))
#     varvals_tuple = ()
#     for ii,varname in enumerate(varnames_tuple):
#         idx, = np.where([one_varname == varname for one_varname in varnames])
#         if len(idx) == 1.:
#             # variable is a float
#             one_varval = varvals[idx[0]]
#         else:
#             one_vardim = vardims_tuple[ii]
#             one_varval = varvals[idx].reshape(one_vardim)
#         varvals_tuple.append(one_varval)
#     return varvals_tuple, varnames_tuple
            

# def dxdt_sdm(Ns,rs):
#     vols = np.pi*4./3.*rs**3.
#     idx=np.argsort(rs)
    
#     dNs_dt = np.zeros_like()
#     for ii in idx:
#         Ni = Ns[ii]
#         for jj in idx[ii:]:
#             Nj = Ns[jj]
#             dNdt_ij = K_ij*Ni*Nj
            
#             # vol_i <= vol_j
#             dNs_dt[ii] -= dNdt_ij
#             dVs_dt[jj] += vol_i*dNdt_ij
    
#     drs_dt = (dVs_dt * 3./4. * np.pi - rs**3. * dNs_dt)/(3. * rs**2. * Ns)
    
#     return drs_dt, dNs_dt


# def SDM(Ns,Vks,spec_densities,N_quad=5,temp=293.15,press=101325.):
#     # check this!
#     dVkdt_i = np.zeros(Vks.shape)
#     dNdt_i = np.zeros(Ns.shape)
#     for ii,(Vk_i,num_i) in enumerate(zip(Vks,Ns)):
#         vk_i = Vk_i/num_i
#         vol_i = sum(vk_i)
#         den_i = aero_props.vk_to_density(vk_i,spec_densities)
#         for jj,(Vk_j,num_j) in enumerate(zip(Vks,Ns)):
#             vk_j = Vk_j/num_j
#             vol_j = sum(vk_j)
#             den_j = aero_props.vk_to_density(vk_j,spec_densities)
#             if num_i>0. and num_j>0.:
#                 dNdt_coag_ij = num_i*num_j*brownian_kernel(vol_i, vol_j, den_i, den_j, temp, press)
#                 if vol_i>vol_j:
#                     # removed from j, added to i
#                     # dNi_dt = 0, but dVki_dt>0
#                     # dNj_dt = -dNdt_coag_ij and dVkj_dt=-dVki_dt
#                     dNdt_i[jj] -= dNdt_coag_ij
#                     dVkdt_i[jj,:] -= dNdt_coag_ij*vk_j
#                     dVkdt_i[ii,:] += dNdt_coag_ij*vk_j
#                 elif vol_i<vol_j:
#                     # removed from i, added to j
#                     # dNj_dt = 0, but dVkj_dt> = dNdt_coag_ij 
#                     # dNi_dt = -dNdt_coag_ij and dVki_dt=-dVkj_dt
#                     dNdt_i[ii] -= dNdt_coag_ij
#                     dVkdt_i[ii,:] -= dNdt_coag_ij*vk_i
#                     dVkdt_i[jj,:] += dNdt_coag_ij*vk_i
#                 elif vol_i == vol_j:
#                     dNdt_i[ii] -= 0.5*dNdt_coag_ij
# #                print('ii',ii,'jj',jj,'dNdt_i[ii]',dNdt_i[ii],'dNdt_i[jj]',dNdt_i[jj])
#     return dNdt_i, dVkdt_i

# def brownian_kernel(vol_1, vol_2, den_1, den_2, temp, press):
#     boltz = 1.3806505e-23 * 1e7 # J/K to erg/K
#     avogad = 6.02214179e23 # 1/mol
#     mwair = 2.89644e-2 * 1e3 # kg/mol to g/mol
#     rgas = 8.314472 * 1e-2 # J/mole/K to atmos/(mol/liter)/K
    
#     rhoair = 0.001 * ((press/1.01325e5)*mwair/(rgas*temp))
    
#     viscosd = (1.8325e-04*(296.16+120)/(temp+120)) * (temp/296.16)**1.5
#     viscosk = viscosd/rhoair
#     gasspeed = np.sqrt(8*boltz*temp*avogad/(np.pi*mwair))
#     gasfreepath = 2*viscosk/gasspeed
        
#     den_i = den_1*1e-3 # particle wet density (g/cm3)
#     vol_i = vol_1*1e6 # particle wet density (g/cm3)    
#     rad_i     = (vol_i*6/np.pi)**(1/3)/2 # particle wet radius (cm)
    
#     knud      = gasfreepath/rad_i
#     cunning   = 1.0 + knud*(1.249 + 0.42*np.exp(-0.87/knud))
#     diffus_i  = boltz*temp*cunning/(6*np.pi*rad_i*viscosd)
#     speedsq_i = 8*boltz*temp/(np.pi*den_i*vol_i)
#     freepath  = 8*diffus_i/(np.pi*np.sqrt(speedsq_i))
#     tmp1      = (2*rad_i + freepath)**3
#     tmp2      = (4*rad_i*rad_i + freepath*freepath)**1.5
#     deltasq_i = ((tmp1-tmp2)/(6*rad_i*freepath) - 2*rad_i )**2
    
    
#     den_j = den_2*1e-3 # particle wet density (g/cm3)
#     vol_j = vol_2*1e6 # particle wet density (g/cm3)    
#     rad_j     = (vol_j*6/np.pi)**(1/3)/2 # particle wet radius (cm)
    
#     knud      = gasfreepath/rad_j
#     cunning   = 1.0 + knud*(1.249 + 0.42*np.exp(-0.87/knud))
#     diffus_j  = boltz*temp*cunning/(6*np.pi*rad_j*viscosd)
#     speedsq_j = 8*boltz*temp/(np.pi*den_j*vol_j)
#     freepath  = 8*diffus_j/(np.pi*np.sqrt(speedsq_j))
#     tmp1      = (2*rad_j + freepath)**3
#     tmp2      = (4*rad_j*rad_j + freepath*freepath)**1.5
#     deltasq_j = ((tmp1-tmp2)/(6*rad_j*freepath) - 2*rad_j )**2
    
#     rad_sum    = rad_i + rad_j
#     diffus_sum = diffus_i + diffus_j
#     tmp1       = rad_sum/(rad_sum + np.sqrt(deltasq_i + deltasq_j))
#     tmp2       = 4.0*diffus_sum/(rad_sum*np.sqrt(speedsq_i + speedsq_j))
#     bckernel1  = 4.0*np.pi*rad_sum*diffus_sum/(tmp1 + tmp2)
    
#     bckernel = bckernel1 * 1e-6
    
#     return bckernel

# def update_gas_mixture(ParcelState_0, processes):
#     pass

def update_air(ParcelState_0, processes, feedbacks, dt, verbosity=50,C0=3.,accom=0.3,solver='CVODE'):
    T0 = ParcelState_0.T
    P0 = ParcelState_0.P
    S0 = ParcelState_0.S # put this into gas mixture?    
    wv0 = air_thermo.S_to_wv(S0,T0,P0)
    t0 = 0.0
        
    state0 = np.array([T0,P0,S0,wv0])
    updraft_velocity = ParcelState_0.w

    if processes.fluctuations:
        if len(ParcelState_0.population.particles)==1:
            r = ParcelState_0.population.particles[0].get_Dwet()/2.
            N = ParcelState_0.population.particles[0].num_conc
            ds_turb = fluctuations.ds_fluctuation(
                S0-1.,dt,T0,P0,r,N,V=updraft_velocity,C0=C0,accom=accom)
    else:
        ds_turb = 0.
    
    ParcelState_next = copy(ParcelState_0) # maybe??
    if solver == 'CVODE':
        rhs = lambda t, state: air_thermo.dstate_dt(
            state, updraft_velocity, feedbacks.dwc_dt, feedbacks.dwi_dt)
        prob = Explicit_Problem(rhs, state0)
        sim = CVode(prob)
        sim.atol=1.0e-15
        sim.rtol=1.0e-15
        sim.verbosity=verbosity
        # lnr_next=sim.simulate(dt)
        state_next=sim.simulate(dt)
        ParcelState_next.T = state_next[1][-1][0]
        ParcelState_next.P = state_next[1][-1][1]
        ParcelState_next.S = state_next[1][-1][2] #+ ds_turb
        ParcelState_next.wv = state_next[1][-1][3]

    elif solver == 'ode15s':
        ode15s = ode(air_thermo.dstate_dt_wrapper).set_integrator('lsoda', method='bdf', 
                                              rtol=1E-6, atol=1E-12, nsteps=5000)
        ode15s.set_initial_value(state0, t0).set_f_params(updraft_velocity, feedbacks.dwc_dt, feedbacks.dwi_dt)
        state_next = ode15s.integrate(ode15s.t+dt)
        ParcelState_next.T = state_next[0]
        ParcelState_next.P = state_next[1]
        ParcelState_next.S = state_next[2] + ds_turb
        ParcelState_next.wv = state_next[3]    
    
    if processes.cocondensation:
        if ParcelState_next.TraceGas_population.gases:
            for ii in range(len(feedbacks.gases.names)):
                gas = feedbacks.gases.names[ii]
                dppb_dt = feedbacks.gases.dc_dts[ii]
                TraceGas_idx = ParcelState_next.TraceGas_population.get_species_idx(gas)
                ParcelState_next.TraceGas_population.concs[TraceGas_idx] += dppb_dt    
    
    # except:
    #     output = solve_ivp(rhs, [0.,dt], state0)
    #     ParcelState_next.T = output.y[0]
    #     ParcelState_next.P = output.y[1]
    #     ParcelState_next.S = output.y[2] #+ ds_turb
    #     ParcelState_next.wv = output.y[3]        
    # #     output = solve_ivp(rhs, [0.,dt], state0)
    #     # lnr_next=output.y[-1]
    #     # output = solve_ivp(rhs, [0.,dt], np.array([r0]))
    #     # r_next=output.y[-1]
    #     print()
    #     print()
    #     print("============ IN HERE =====================")
    #     print(state0)
    #     print()
    #     sys.exit()
    
    
    return ParcelState_next
    # put this in a condensation function?
    # if processes.water_uptake: # changes this to "condense"?
        


# @dataclass
# class CoupledParcel:
    
# @dataclass
# class PopulationUpdater:
#     population0: ParticlePopulation
#     population_next: ParticlePopulation
#     dt: float()
#     x0: [float, ...]
#     x_next: [float, ...]
#     rhs: [float, ...]

# @dataclass
# class ParticleUpdater:
    
