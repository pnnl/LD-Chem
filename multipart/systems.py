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
from processes import water_uptake, cocondensation, aqueous_chemistry
from processes import air_thermo
from processes import fluctuations
# from processes.water_uptake import dlnr_dt
import constants as c

from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
from scipy.integrate import solve_ivp, ode
from scipy.optimize import fminbound

from copy import copy, deepcopy
    
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
    
    def get_activated_fraction(self):
        
        def Seq(r, r_dry, T, kappa):
            """ Saturation ratio over the aqueous droplet. From pyrcel. """
            a_w = np.power(1.0+kappa*(np.power(r_dry,3)/(np.power(r, 3)-np.power(r_dry, 3))), -1)    
            sigma_w=0.0761 - (1.55e-4) * (T - 273.15)
            Seq = a_w*np.exp((2.0*sigma_w*(18.0 / 1e3))/(8.314*T*1000*r))
            return Seq
                
        particle_population = self.particle_population
        T = self.T
        sizes = np.zeros(0)
        Ns = np.zeros(0)
        for particle,num_conc in zip(particle_population.particles,particle_population.num_concs):
            r=particle.get_Dwet()/2.
            r_dry=particle.get_Ddry()/2.
            kappa=particle.get_tkappa()
            neg_Seq = lambda r: -1.0 * Seq(r, r_dry, T, kappa)
            out = fminbound(neg_Seq, r_dry, r_dry * 1e4, xtol=1e-10, full_output=True, disp=0)
            r_crit, s_crit = out[:2]
            s_crit *= -1.0  # multiply by -1 to undo negative flag for Seq
            if r>=r_crit:
                sizes=np.append(sizes, r)
                Ns=np.append(Ns, num_conc)
        
        return np.sum(Ns)/np.sum(particle_population.num_concs)  
    
    
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
    
    def get_max_S(self):
        S=[]
        for i in range(0, len(self.parcel_states)):
            S.append(self.parcel_states[i].S)
        return np.max(np.array((S)))
    
    def get_avg_droplet_radius(self):
        
        def Seq(r, r_dry, T, kappa):
            """ Saturation ratio over the aqueous droplet. From pyrcel. """
            a_w = np.power(1.0+kappa*(np.power(r_dry,3)/(np.power(r, 3)-np.power(r_dry, 3))), -1)    
            sigma_w=0.0761 - (1.55e-4) * (T - 273.15)
            Seq = a_w*np.exp((2.0*sigma_w*(18.0 / 1e3))/(8.314*T*1000*r))
            return Seq
                
        idx=len(self.parcel_states)-1
        particle_population = self.parcel_states[idx].particle_population
        T = self.parcel_states[idx].T
        sizes = np.zeros(0)
        Ns = np.zeros(0)
        for particle,num_conc in zip(particle_population.particles,particle_population.num_concs):
            r=particle.get_Dwet()/2.
            r_dry=particle.get_Ddry()/2.
            kappa=particle.get_tkappa()
            neg_Seq = lambda r: -1.0 * Seq(r, r_dry, T, kappa)
            out = fminbound(neg_Seq, r_dry, r_dry * 1e4, xtol=1e-10, full_output=True, disp=0)
            r_crit, s_crit = out[:2]
            s_crit *= -1.0  # multiply by -1 to undo negative flag for Seq
            if r>=r_crit:
                sizes=np.append(sizes, r)
                Ns=np.append(Ns, num_conc)
        if np.sum(Ns) > 0:
            return np.average(sizes, weights=Ns)      
        else:
            return 0
    
    def get_activated_fraction(self):
        
        def Seq(r, r_dry, T, kappa):
            """ Saturation ratio over the aqueous droplet. From pyrcel. """
            a_w = np.power(1.0+kappa*(np.power(r_dry,3)/(np.power(r, 3)-np.power(r_dry, 3))), -1)    
            sigma_w=0.0761 - (1.55e-4) * (T - 273.15)
            Seq = a_w*np.exp((2.0*sigma_w*(18.0 / 1e3))/(8.314*T*1000*r))
            return Seq
                
        idx=len(self.parcel_states)-1
        particle_population = self.parcel_states[idx].particle_population
        T = self.parcel_states[idx].T
        sizes = np.zeros(0)
        Ns = np.zeros(0)
        for particle,num_conc in zip(particle_population.particles,particle_population.num_concs):
            r=particle.get_Dwet()/2.
            r_dry=particle.get_Ddry()/2.
            kappa=particle.get_tkappa()
            neg_Seq = lambda r: -1.0 * Seq(r, r_dry, T, kappa)
            out = fminbound(neg_Seq, r_dry, r_dry * 1e4, xtol=1e-10, full_output=True, disp=0)
            r_crit, s_crit = out[:2]
            s_crit *= -1.0  # multiply by -1 to undo negative flag for Seq
            if r>=r_crit:
                sizes=np.append(sizes, r)
                Ns=np.append(Ns, num_conc)
        
        return np.sum(Ns)/np.sum(particle_population.num_concs)      
        

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
        

def update_state(t1, t2,
        ParcelState_0, processes, dt, verbosity=50, 
        C0=3.,accom=1.,radius_scale='lin',solver='CVODE', 
        mechanism_data_path='../mechanisms/', aq_reactions=None): 
    ParcelState_1, feedbacks_1 = update_particle_population(
        ParcelState_0, processes, dt, 
        radius_scale=radius_scale,solver=solver,
        accom=accom, verbosity=verbosity,mechanism_data_path=mechanism_data_path,
        aq_reactions=aq_reactions)  
    ParcelState_2 = update_air(t2,
        ParcelState_1, processes, feedbacks_1, dt, 
        verbosity=verbosity, solver=solver)  
    ParcelState_next = replace(ParcelState_2)    
    return ParcelState_next

# put all the functions for particle state in another file?
def update_particle_population(
        ParcelState_0, processes, dt, accom=1.,
        verbosity=50,radius_scale='lin',solver='CVODE',
        mechanism_data_path='../mechanisms/', aq_reactions=None): # fix later -- put in species
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
    
    
    # put this in a condensation function?
    
    if cocondensation:
        dCgas_dts_all = []
    
    # print()

    new_particles = []
    
    # do all of the water condensation at the same time        
    if processes.condensation:
        water_masses_next=water_condensation_solver(ParcelState_0.particle_population, P0, T0, S0, wv0, accom, processes, radius_scale, solver, verbosity, dt)
        
        for ii,(particle,num_conc) in enumerate(zip(ParcelState_0.particle_population.particles,ParcelState_0.particle_population.num_concs)):
            new_particle = deepcopy(particle)
            new_particle.masses[particle.get_species_idx('H2O')]=water_masses_next[ii]
            
            r0=particle.get_Dwet()/2
            r_next=new_particle.get_Dwet()/2.
            dwc_dt += ((4.0*np.pi*c.rho_w)/rho_air_dry)*num_conc*(r0**2)*(r_next-r0)
                         
            m_h2o_0 = particle.get_mass_h2o()
            m_h2o_next = new_particle.get_mass_h2o()
            dwc += num_conc*(m_h2o_next - m_h2o_0)/rho_air_dry
            
            new_particles.append(new_particle)
        
        # r_nexts=[]
        # for particle in ParcelState_0.particle_population.particles:  
        #     r_nexts.append(particle.get_Dwet()/2.)
        
        # particle_population=water_uptake.equilibrate_water(ParcelState_0.particle_population, S0, T0, P0, 7.0)
        # r_nexts_eq=[]
        # for particle in particle_population.particles:  
        #     r_nexts_eq.append(particle.get_Dwet()/2.)
            
        # print(np.array((r_nexts))*1e6, np.array((r_nexts_eq))*1e6)
                    
        # --------------------------------------------------------------
        
        # if processes.condensation:            
        #     r_dry_i = particle.get_Ddry()/2.
        #     tkappa_i = particle.get_tkappa()
        #     r0=particle.get_Dwet()/2.                     
        #     if radius_scale == 'log':
        #         lnr0 = np.log(r0)
        #         if solver == 'CVODE':  
        #             rhs = lambda t, lnr: water_uptake.dlnr_dt(lnr, r_dry_i, tkappa_i, P0, T0, S0, wv0, accom)
                    # prob = Explicit_Problem(rhs, lnr0)
                    # sim = CVode(prob)
                    # sim.atol=1.0e-10
                    # sim.rtol=1.0e-10
                    # sim.verbosity = verbosity
                    # output=sim.simulate(dt)
                    # r_next = np.exp(output[1][-1,0])
                    
                    # print('old function:', rhs(0.0, lnr0))
                    
            #     elif solver == 'ode15s':
            #         ode15s = ode(water_uptake.water_uptake_wrapper).set_integrator('lsoda', method='bdf', rtol=1E-6, atol=1E-12, nsteps=5000)
            #         ode15s.set_initial_value(lnr0, t0).set_f_params(r_dry_i, tkappa_i, P0, T0, S0, wv0, accom, radius_scale)
            #         r_next = np.exp(ode15s.integrate(ode15s.t+dt)[0])
                    
                    
            # elif radius_scale == 'lin':   
            #     if solver == 'CVODE':
            #         rhs = lambda t, r: water_uptake.dr_dt(r, r_dry_i, tkappa_i, P0, T0, S0, wv0, accom=accom)
            #         prob = Explicit_Problem(rhs, r0, sw0)
            #         sim = CVode(prob)
            #         sim.atol=1.0e-10
            #         sim.rtol=1.0e-10
            #         sim.verbosity=verbosity
            #         output=sim.simulate(dt)
            #         r_next=output[1][-1,0]  
                    
                    
            #     elif solver == 'ode15s':
            #         ode15s = ode(water_uptake.water_uptake_wrapper).set_integrator('lsoda', method='bdf', 
            #                                               rtol=1E-6, atol=1E-16, nsteps=5000)
            #         ode15s.set_initial_value(r0, t0).set_f_params(r_dry_i, tkappa_i, P0, T0, S0, wv0, accom, radius_scale)
            #         r_next = ode15s.integrate(ode15s.t+dt)[0]       
                    
            # dwc_dt += ((4.0*np.pi*c.rho_w)/rho_air_dry)*num_conc*(r0**2)*(r_next-r0)
                         
            # m_h2o_0 = particle.get_mass_h2o()
            # new_particle = water_uptake.update_particle(new_particle, r_next)

            # m_h2o_next = new_particle.get_mass_h2o()
            # dwc += num_conc*(m_h2o_next - m_h2o_0)/rho_air_dry
            
            # print('old function:', m_h2o_next)


    '''
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
                            Caq_next = ode15s.integrate(ode15s.t+dt)[0]  # mol/m^3
                    
                        dCgas_dt.append(-1.0*water_volume*num_conc*(Caq_next-Caq_0)) # mol/m^3
                        new_particle.masses[new_particle.get_species_idx(gas.name)] = Caq_next*water_volume*gas.molar_mass # kg
                    else:
                        dCgas_dt.append(0.0)
                        
                dCgas_dts_all.append(dCgas_dt)

        
        if processes.chemistry:
            water_volume = particle.get_vol_tot()-particle.get_vol_dry() # m^3
            Caq_0 = []
            for species,mass in zip(particle.species, particle.masses):
                Caq_0.append((mass/species.molar_mass)/(1000*water_volume)) # mol/L
            if solver == 'CVODE':
                rhs = lambda t, Caq: aqueous_chemistry.dCaq_dt(Caq, particle, aq_reactions, T0)
                prob = Explicit_Problem(rhs, Caq_0)
                sim = CVode(prob)
                sim.atol=1.0e-10
                sim.rtol=1.0e-10
                sim.verbosity=verbosity
                output=sim.simulate(dt)
                Caq_next=output[1][-1] # mol/m^3
        
            for species in particle.species:
                new_particle.masses[new_particle.get_species_idx(species.name)]+=Caq_next[new_particle.get_species_idx(species.name)]*water_volume*species.molar_mass # kg

        # print()
        # for species in new_particle.species:
        #     print(species.name, particle.masses[particle.get_species_idx(species.name)], new_particle.masses[new_particle.get_species_idx(species.name)])
        # print()
        '''
        # particles.append(new_particle)

        # print()
        # for i in range(len(particle.species)):
        #     print(particle.species[i].name, new_particle.masses[i], particle.masses[i])
            
    # print()
    # sys.exit()
    
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
        
    ParcelState_next.particle_population.particles = new_particles
    feedbacks = Feedbacks(dwc_dt=dwc_dt, dwc_dt_next=dwc_dt_next, dwc=dwc, dwi_dt=dwi_dt, gases=gas_feedbacks)
    
    return ParcelState_next, feedbacks



def water_condensation_solver(particle_population, P, T, S, wv, accom, processes, radius_scale, solver, verbosity, dt):
    
    t0 = 0       
    r_drys=[]
    tkappas=[]
    r0s=[]
    for particle in particle_population.particles:  
        r_drys.append(particle.get_Ddry()/2.)
        tkappas.append(particle.get_tkappa())
        r0s.append(particle.get_Dwet()/2.)
  
    r_drys=np.array((r_drys))
    tkappas=np.array((tkappas))
    r0s=np.array((r0s))
    
    sw0 = False # if False, mh2o (or r, etc.) hasn't yet become negative                     
    if radius_scale == 'log':
        lnr0s = np.log(r0s)
        if solver == 'CVODE':
            rhs = lambda t, lnr: water_uptake.dlnr_dt(lnr, r_drys, tkappas, P, T, S, accom)            
            prob = Explicit_Problem(rhs, lnr0s)
            sim = CVode(prob)
            sim.atol=1.0e-10
            sim.rtol=1.0e-10
            sim.verbosity = verbosity
            output=sim.simulate(dt)
            r_nexts = np.exp(output[1][-1,:])
            
            
            
        elif solver == 'ode15s':
            ode15s = ode(water_uptake.water_uptake_wrapper).set_integrator('lsoda', method='bdf', rtol=1E-6, atol=1E-12, nsteps=5000)
            ode15s.set_initial_value(lnr0s, t0).set_f_params(r_drys, tkappas, P, T, S, accom, radius_scale)
            r_nexts = np.exp(ode15s.integrate(ode15s.t+dt))
                           
    elif radius_scale == 'lin':   
        if solver == 'CVODE':
            rhs = lambda t, r: water_uptake.dr_dt(r, r_drys, tkappas, P, T, S, accom=accom)
            prob = Explicit_Problem(rhs, r0s, sw0)
            sim = CVode(prob)
            sim.atol=1.0e-10
            sim.rtol=1.0e-10
            sim.verbosity=verbosity
            output=sim.simulate(dt)
            r_nexts=output[1][-1,:] 
            
        elif solver == 'ode15s':
            ode15s = ode(water_uptake.water_uptake_wrapper).set_integrator('lsoda', method='bdf', 
                                                  rtol=1E-6, atol=1E-16, nsteps=5000)
            ode15s.set_initial_value(r0s, t0).set_f_params(r_drys, tkappas, P, T, S, accom, radius_scale)
            r_nexts=ode15s.integrate(ode15s.t+dt)     
    
    
    new_masses=[]
    water_density=particle_population.particles[0].get_rho_w()
    for dry_radius, radius in zip(r_drys, r_nexts):
        water_volume=(4.0/3.0)*np.pi*(radius**3-dry_radius**3)
        new_masses.append(water_density*water_volume)
    
    return new_masses
    





def update_air(t2, ParcelState_0, processes, feedbacks, dt, verbosity=50,C0=3.,accom=0.3,solver='CVODE'):
    T0 = ParcelState_0.T
    P0 = ParcelState_0.P
    S0 = ParcelState_0.S # put this into gas mixture?   
    z0 = ParcelState_0.z
    wv0 = air_thermo.S_to_wv(S0,T0,P0)
    t0 = 0.0
        
    state0 = np.array([z0,T0,P0,S0,wv0])
    if ParcelState_0.w:
        updraft_velocity = ParcelState_0.w  
    else:
        updraft_velocity = None    
    
    if processes.fluctuations:
        if len(ParcelState_0.population.particles)==1:
            r = ParcelState_0.population.particles[0].get_Dwet()/2.
            N = ParcelState_0.population.particles[0].num_conc
            ds_turb = fluctuations.ds_fluctuation(
                S0-1.,dt,T0,P0,r,N,V=updraft_velocity,C0=C0,accom=accom)
    else:
        ds_turb = 0.

    
    ParcelState_next = deepcopy(ParcelState_0) # maybe??
    
    if updraft_velocity:
        if solver == 'CVODE':
            rhs = lambda t, state: air_thermo.dstate_dt(
                state, updraft_velocity, feedbacks.dwc_dt, feedbacks.dwi_dt)
            prob = Explicit_Problem(rhs, state0)
            sim = CVode(prob)
            sim.atol=1.0e-15
            sim.rtol=1.0e-15
            sim.verbosity=verbosity
            state_next=sim.simulate(dt)
            ParcelState_next.z = state_next[1][-1][0]
            ParcelState_next.T = state_next[1][-1][1]
            ParcelState_next.P = state_next[1][-1][2]
            ParcelState_next.S = state_next[1][-1][3] #+ ds_turb
            ParcelState_next.wv = state_next[1][-1][4]
            
        elif solver == 'ode15s':
            ode15s = ode(air_thermo.dstate_dt_wrapper).set_integrator('lsoda', method='bdf', 
                                                  rtol=1E-6, atol=1E-12, nsteps=5000)
            ode15s.set_initial_value(state0, t0).set_f_params(updraft_velocity, feedbacks.dwc_dt, feedbacks.dwi_dt)
            state_next = ode15s.integrate(ode15s.t+dt)
            ParcelState_next.z = state_next[0]
            ParcelState_next.T = state_next[1]
            ParcelState_next.P = state_next[2]
            ParcelState_next.S = state_next[3] + ds_turb
            ParcelState_next.wv = state_next[4]  

    
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
        

    
