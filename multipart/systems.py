""" systems - types and functions used to create different model systems. 

@author: Laura Fierce
"""

import numpy as np
import numba as nb
from numba.pycc import CC
import sys, utilities

import matplotlib.pyplot as plt

from dataclasses import dataclass
from dataclasses import replace

from typing import Tuple
from typing import Callable
from numba.typed import Dict
from numba import types
import scipy.optimize as opt
from particles import ParticlePopulation
from TraceGases import TraceGasPopulation, GasSpecies
from processes import water_uptake, cocondensation, aqueous_chemistry, gas_chemistry
from processes import air_thermo
from processes import fluctuations
from processes.air_thermo import compute_thermo_props, S_to_wv, wv_to_S, H2O_gas_conc
import constants as c

#from assimulo.problem import Explicit_Problem
#from assimulo.solvers import CVode
from scipy.integrate import ode
from scipy.optimize import fminbound

    
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
    
    def clone_detached(self):
        return ParcelState(
            x=self.x, y=self.y, z=self.z,
            u=self.u, v=self.v, w=self.w,
            S=self.S, P=self.P, T=self.T,
            particle_population=self.particle_population.clone_detached(),
            TraceGas_population=(None if self.TraceGas_population is None
                                 else self.TraceGas_population.clone_detached())
        )
    
    
@dataclass
class Processes:
    """AerosolProcesses: a definition of a set of aerosol processes under consideration"""
    condensation: bool = True
    collisions: bool = False
    cocondensation: bool = False
    aq_chemistry: bool = False
    gas_chemistry: bool = False
    freezing: bool = False
    settling: bool = False
    fluctuations: bool = False
    entrainment: bool = False
    
    
@dataclass
class GasFeedback:
    names: Tuple[GasSpecies, ...]
    dc_dts: Tuple[float, ...]
    
@dataclass
class Feedbacks:
    dwc_dt: float = 0.
    dwv_dt: float = 0.
    # dwc_dt_next: float = 0.
    # dwc: float = 0. # change in water mass, kg water / m^3 air
    # dwi: float = 0. # change in ice mass, kg ice / m^3 air
    gases: Tuple[GasFeedback, ...] = None  # change trace gases, ppb

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
        

# @dataclass
# class TrajectoryInteractions: # should some of these be set to "none"?
#     uniform: bool = False # if ture, all particles across trajectories interact, regardless of their location in space
#     neighbors: bool = False # if true, parcels located near each other in space interact
#     gns: bool = False # if ture, neigbor
#     homogeneous_mixing: bool = False # just throwing things in here for now...
    
# @dataclass
# class TrajectoryEnsemble:
#     parcel_trajectories: Tuple[ParcelTrajectory, ...]
#     trajectory_interactions: TrajectoryInteractions
        


def update_state(t1, t2,
        ParcelState_0, processes, dt, verbosity=50, 
        accom=1.,radius_scale='lin',solver='CVODE', 
        mechanism_data_path='../mechanisms/', aq_reactions=None,
        gas_reactions=None, rtol=1e-10, atol=1e-10):
       
    # print()
    # print('Initital parcel:')
    # for gas, conc in zip(ParcelState_0.TraceGas_population.gases, ParcelState_0.TraceGas_population.concs):
    #     print(gas.name, conc)
    # print()
    # for particle in ParcelState_0.particle_population.particles[10:11]:
    #     for spec in particle.species:
    #         print(spec.name, particle.masses[particle.get_species_idx(spec.name)])
    # print()
    
    ParcelState_1, feedbacks_1 = update_particle_population(
        ParcelState_0, processes, dt, 
        radius_scale=radius_scale,solver=solver,
        accom=accom, verbosity=verbosity,mechanism_data_path=mechanism_data_path,
        aq_reactions=aq_reactions, rtol=rtol, atol=atol)
    
    # print()
    # print('After particles change:')
    # for gas, conc in zip(ParcelState_1.TraceGas_population.gases, ParcelState_1.TraceGas_population.concs):
    #     print(gas.name, conc)
    # print()
    # print()
    # for particle in ParcelState_1.particle_population.particles[10:11]:
    #     for spec in particle.species:
    #         print(spec.name, particle.masses[particle.get_species_idx(spec.name)])
    # print()
    # import sys
    # sys.exit()
    
    ParcelState_2 = update_air(t2,
        ParcelState_1, processes, feedbacks_1, dt, 
        verbosity=verbosity, solver=solver, gas_reactions=gas_reactions, rtol=rtol, atol=atol)
    
    # print('After gases change:')
    # for gas, conc in zip(ParcelState_2.TraceGas_population.gases, ParcelState_2.TraceGas_population.concs):
    #     print(gas.name, conc)
    # print()
    # for particle in ParcelState_2.particle_population.particles[1:2]:
    #     for spec in particle.species:
    #         print(spec.name, particle.masses[particle.get_species_idx(spec.name)])
    #     print()
    
    # sys.exit()
    
    ParcelState_next = replace(ParcelState_2)
    return ParcelState_next





# put all the functions for particle state in another file?
def update_particle_population(
        ParcelState_0, processes, dt, accom=1.,
        verbosity=50,radius_scale='lin',solver='CVODE',
        mechanism_data_path='../mechanisms/', aq_reactions=None,
        rtol=1e-10, atol=1e-10): # fix later -- put in species
    
    T0 = ParcelState_0.T
    P0 = ParcelState_0.P
    S0 = ParcelState_0.S # put this into gas mixture?
    wv0 = air_thermo.S_to_wv(S0,T0,P0)
    dwc_dt = 0.
    
    
    #ParcelState_Next = deepcopy(ParcelState_0) # maybe??
    ParcelState_Next = ParcelState_0.clone_detached()
    
    # water condensation   
    if processes.condensation:
        water_masses_next=water_condensation_solver(ParcelState_Next.particle_population, P0, T0, S0, wv0, accom, processes, radius_scale, solver, verbosity, dt)
        for ii,(old_particle, new_particle, num_conc) in enumerate(zip(ParcelState_0.particle_population.particles, ParcelState_Next.particle_population.particles, ParcelState_Next.particle_population.num_concs)):
            new_particle.masses[new_particle.idx_h2o]=water_masses_next[ii]
            m_h2o_0 = old_particle.get_mass_h2o()
            m_h2o_next = new_particle.get_mass_h2o()
            dwc_dt += num_conc*(m_h2o_next - m_h2o_0) # mass of water to particle phase, kg/m^3
    else:
        water_masses_next=np.zeros(len(ParcelState_0.particle_population.particles))
        for ii, (particle) in enumerate(ParcelState_0.particle_population.particles):
            water_masses_next[ii]=particle.masses[particle.get_species_idx('H2O')]
    for ii, (particle) in enumerate(ParcelState_Next.particle_population.particles):
        particle.masses[particle.idx_h2o]=water_masses_next[ii]

    utilities.check_water_condensation(ParcelState_0, ParcelState_Next, dwc_dt)

    # gas condensation
    if processes.cocondensation:
        ParcelState_Next.particle_population, gas_feedback=cocondensation_solver(
            ParcelState_Next.particle_population, 
            ParcelState_Next.TraceGas_population, 
            P0, T0, S0, solver=solver, 
            verbosity=verbosity, dt=dt,
            rtol=rtol, atol=atol)
        gas_feedback = utilities.check_gas_condensation(ParcelState_0, ParcelState_Next, gas_feedback)        
    else:
        gas_feedback=None

    # aqueous chemistry
    if processes.aq_chemistry:
        particles_before_chemistry=ParcelState_Next.particle_population
        ParcelState_Next.particle_population=aq_chemistry_solver(
            ParcelState_Next.particle_population, aq_reactions, T0,
            solver=solver, verbosity=verbosity, dt=dt, rtol=rtol, atol=atol)
        utilities.check_mass_balance(particles_before_chemistry, ParcelState_Next.particle_population)

    feedbacks = Feedbacks(dwc_dt=dwc_dt, dwv_dt=-1.0*dwc_dt, gases=gas_feedback)    
    
    return ParcelState_Next, feedbacks



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
                                                  rtol=1E-6, atol=1E-12, nsteps=5000)
            ode15s.set_initial_value(r0s, t0).set_f_params(r_drys, tkappas, P, T, S, accom, radius_scale)
            r_nexts=ode15s.integrate(ode15s.t+dt)     
    
    new_masses=[]
    water_density=particle_population.particles[0].get_rho_w()
    for dry_radius, radius in zip(r_drys, r_nexts):
        water_volume=(4.0/3.0)*np.pi*(radius**3-dry_radius**3)
        new_masses.append(water_density*water_volume)
    
    return new_masses



def cocondensation_solver(particle_population, gas_population, P, T, S,
                          dt=1.0, solver='CVODE', verbosity=50, 
                          atol=1e-10, rtol=1e-10):
    
    ParticlePopulation_Next=particle_population.clone_detached()
    gas_feedback=GasFeedback(names=[], dc_dts=[])
    
    for gas, gas_ppb in zip(gas_population.gases, gas_population.concs):
        
        if gas.H0 > 0.0 and gas.alpha > 0.0:
            # initial condition
            X0 = [(gas_ppb*1e-9*P)/(8.314*T)] # mol/m^3
            radii = []
            water_volumes = []
            for particle in particle_population.particles:
                part_idx = particle.get_species_idx(gas.name)
                water_volume=particle.get_vol_tot()-particle.get_vol_dry()
                X0.append((particle.masses[part_idx]/particle.species[part_idx].molar_mass)/water_volume) # mol/m^3
                radii.append(particle.get_Dwet()/2.0)
                water_volumes.append(water_volume)
            radii=np.array(radii)
            X0=np.array(X0)
            water_volumes=np.array(water_volumes)
            
            # define function
            if gas.name == 'IEPOX':
                H2O_concs=np.zeros(len(particle_population.particles))
                Hplus_concs=np.zeros(H2O_concs.shape)
                HSO4_concs=np.zeros(H2O_concs.shape)
                NH4_concs=np.zeros(H2O_concs.shape)
                SO4_concs=np.zeros(H2O_concs.shape)
                l_orgs=np.zeros(H2O_concs.shape)
                inorganic_radii=np.zeros(H2O_concs.shape)
                for ii, (particle) in enumerate(particle_population.particles):
                    H2O_concs[ii]=(particle.masses[particle.idx_h2o]/particle.species[particle.idx_h2o].molar_mass)/(1000*water_volumes[ii]) # mol/L
                    Hplus_concs[ii]=(particle.masses[particle.get_species_idx('H+')]/particle.species[particle.get_species_idx('H+')].molar_mass)/(1000*water_volumes[ii]) # mol/L
                    try:
                        HSO4_concs[ii]=(particle.masses[particle.get_species_idx('HSO4')]/particle.species[particle.get_species_idx('HSO4')].molar_mass)/(1000*water_volumes[ii]) # mol/L
                    except:
                        HSO4_concs[ii]=0.0
                    try:
                        NH4_concs[ii]=(particle.masses[particle.get_species_idx('NH4')]/particle.species[particle.get_species_idx('NH4')].molar_mass)/(1000*water_volumes[ii]) # mol/L
                    except:
                        NH4_concs[ii]=0.0
                    try:
                        SO4_concs[ii]=(particle.masses[particle.get_species_idx('SO4')]/particle.species[particle.get_species_idx('SO4')].molar_mass)/(1000*water_volumes[ii]) # mol/L
                    except:
                        SO4_concs[ii]=0.0
                    V_Org = 0
                    for species, mass in zip(particle.species, particle.masses):
                        if species.name in ['IEPOX_OS', 'tetrol', 'tetrol_olig', 'IEPOX_OH_SOA']:
                            V_Org += mass/species.density
                    V_NonOrg = particle.get_vol_dry() - V_Org
                    h2o_NonOrg_radius = ((3*(water_volumes[ii]+V_NonOrg))/(4.0*np.pi))**(1.0/3.0)
                    inorg_radius = ((3*V_NonOrg)/(4.0*np.pi))**(1.0/3.0)
                    radius = particle.get_Dwet()/2.0
                    l_orgs[ii] = radius-h2o_NonOrg_radius # m
                    inorganic_radii[ii] = inorg_radius
                
                rhs = lambda t, X: cocondensation.IEPOX_condensation(X, H2O_concs, Hplus_concs,
                                                                     HSO4_concs, NH4_concs,
                                                                     SO4_concs, radii, T, S,
                                                                     l_orgs, inorganic_radii, 
                                                                     np.array(particle_population.num_concs), 
                                                                     water_volumes, 
                                                                     gas.molar_mass, gas.alpha)
            elif gas.name in ['HNO3', 'H2SO4']: # these are super soluble and fully dissociate, so treat the concentration at the surface of the particle as = 0.0
                if gas.name == 'HNO3':
                    Dl0 = 1.25e-9 # m^2/s (reference value at 298 K, Newman et.al. 1973)
                elif gas.name == 'H2SO4':
                    Dl0 = 0.5e-10 # m^2/s (reference value at 298 K, Leaist et.al. 1984)
                    
                rhs = lambda t, X: cocondensation.dCaq_dt_diffusion_limited(X, radii, water_volumes,
                                                                            np.array(particle_population.num_concs),
                                                                            gas.molar_mass, gas.alpha, gas.get_Heff(T),
                                                                            T, P, Dl0)
            else:
                rhs = lambda t, X: cocondensation.dCaq_dt(X, radii, water_volumes,
                                                          np.array(particle_population.num_concs), 
                                                          gas.molar_mass, gas.alpha, 
                                                          gas.get_Heff(T), T)        
    
            # solve
            if solver == 'CVODE': 
                prob = Explicit_Problem(rhs, X0)
                sim = CVode(prob)
                sim.atol=atol
                sim.rtol=rtol
                sim.verbosity=verbosity
                output=sim.simulate(dt)
                X_next=output[1][-1] # mol/m^3
                
            elif solver == 'ode15s':
                ode15s = ode(rhs).set_integrator('lsoda', method='bdf',
                                                  rtol=rtol, atol=atol, nsteps=5000)
                ode15s.set_initial_value(X0, 0.0)
                X_next = ode15s.integrate(ode15s.t+dt)  # mol/m^3
                
            # update the particles
            for particle, Caq, water_volume in zip(ParticlePopulation_Next.particles, X_next[1:], water_volumes):
                idx=particle.get_species_idx(gas.name)
                molar_mass=particle.species[idx].molar_mass
                mass_next = Caq*water_volume*molar_mass # kg
                particle.masses[idx]=mass_next # kg
    
            # add to the feedbacks
            gas_feedback.names.append(gas.name)
            gas_feedback.dc_dts.append(1e9*(X_next[0]-X0[0])*((8.314*T)/P)) # ppb
            
    return ParticlePopulation_Next, gas_feedback



def aq_chemistry_solver(particle_population, aq_reactions, T,
                     dt=1.0, solver='CVODE', verbosity=50, 
                     atol=1e-10, rtol=1e-10):
    
    ParticlePopulation_Next = particle_population.clone_detached()
    
    for particle_0, new_particle in zip(particle_population.particles, ParticlePopulation_Next.particles):
        
        # set up the initial aqueous concentrations
        X0 = np.zeros(len(particle_0.species))
        water_volume = particle_0.get_vol_tot()-particle_0.get_vol_dry() # m^3
        aq_names = Dict.empty(key_type=types.unicode_type, value_type=types.int32)
        for ii, (species) in enumerate(particle_0.species):
            aq_names[species.name]=ii
            X0[ii]=(particle_0.masses[particle_0.get_species_idx(species.name)]/species.molar_mass)/water_volume # mol/m^3
                
        # set up an array of reactants and products that
        # can be passed into njit function
        reactants = Dict.empty(key_type=types.int32, value_type=types.string)
        products = Dict.empty(key_type=types.int32, value_type=types.string)
        rates = np.empty(0)
        for ii, (reaction) in enumerate(aq_reactions.reactions):
            temp=str(reaction.reactants)
            temp=temp.replace(',','')
            temp=temp.replace('[','')
            temp=temp.replace(']','')
            temp=temp.replace("'",'')
            reactants[ii]=temp
            temp=str(reaction.products)
            temp=temp.replace(',','')
            temp=temp.replace('[','')
            temp=temp.replace(']','')
            temp=temp.replace("'",'')
            products[ii]=temp
            rates=np.append(rates, reaction.get_rate(T))  
    
        # define function    
        rhs = lambda t, X: aqueous_chemistry.dCaq_dt(X, reactants, products, rates, 
                                                     aq_names, T) # mol/m^3*s
        
        # dX_dt = rhs(0.0, X0)
        # print()
        # for kk in aq_names.keys():
        #     print(kk, X0[aq_names[kk]], dX_dt[aq_names[kk]])
        # print()
        # sys.exit()
        
        # solve
        if solver == 'CVODE': 
            prob = Explicit_Problem(rhs, X0)
            sim = CVode(prob)
            sim.atol=atol
            sim.rtol=rtol
            sim.verbosity=verbosity
            output=sim.simulate(dt)
            X_next=output[1][-1] # mol/m^3
            
        elif solver == 'ode15s':
                        
            ode15s = ode(rhs).set_integrator('lsoda', method='bdf',
                                              rtol=rtol, atol=atol, nsteps=5000)
            ode15s.set_initial_value(X0, 0.0)
            X_next = ode15s.integrate(ode15s.t+dt)  # mol/m^3
            
            # print()
            # for kk in aq_names.keys():
            #     print(kk, X0[aq_names[kk]], X_next[aq_names[kk]])
            # print()
            # sys.exit()

        # adjust the OH concentration based on the pH
        Hplus_conc_next=0.001*X_next[np.where(np.array(aq_names)=='H+')[0][0]] # mol/L
        pH_next=-1.0*np.log10(Hplus_conc_next)
        OH_conc = 10**(-14.0+pH_next) # mol/L
        X_next[np.where(np.array(aq_names)=='OH-')[0][0]]=1000*OH_conc
    
        for Caq, species in zip(X_next, aq_names):
            idx=new_particle.get_species_idx(species)
            molar_mass=new_particle.species[idx].molar_mass
            new_particle.masses[idx] = Caq*water_volume*molar_mass # kg

    return ParticlePopulation_Next


def update_air(t2, ParcelState_0, processes, feedbacks, dt, 
               verbosity=50,C0=3.,accom=0.3,solver='CVODE',
               gas_reactions=None, rtol=1e-10, atol=1e-10):
    
    T0 = ParcelState_0.T
    P0 = ParcelState_0.P
    S0 = ParcelState_0.S # put this into gas mixture?   
    z0 = ParcelState_0.z
    wv0 = air_thermo.S_to_wv(S0,T0,P0)
    
    dz_dt=0
    dT_dt=0
    dP_dt=0
    dS_dt=0
    dwv_dt=0

    if processes.fluctuations:
        if len(ParcelState_0.population.particles)==1:
            r = ParcelState_0.population.particles[0].get_Dwet()/2.
            N = ParcelState_0.population.particles[0].num_conc
            ds_turb = fluctuations.ds_fluctuation(
                S0-1.,dt,T0,P0,r,N,V=ParcelState_0.w,C0=C0,accom=accom)
    else:
        ds_turb = 0.

    ParcelState_next = ParcelState_0.clone_detached()
    
    if processes.condensation:
        state0 = np.array([z0,T0,P0,S0,wv0])
        if ParcelState_0.w:
            
            rhs = lambda t, state: air_thermo.dstate_dt(state, ParcelState_0.w, feedbacks.dwc_dt)
            
            if solver == 'CVODE':
                prob = Explicit_Problem(rhs, state0)
                sim = CVode(prob)
                sim.atol=atol
                sim.rtol=rtol
                sim.verbosity=verbosity
                state_next=sim.simulate(dt)
                dz_dt = state_next[1][-1][0]-z0
                dT_dt = state_next[1][-1][1]-T0
                dP_dt = state_next[1][-1][2]-P0
                dS_dt = state_next[1][-1][3]-S0+ds_turb
                dwv_dt = state_next[1][-1][4]-wv0
                
            elif solver == 'ode15s':
                ode15s = ode(rhs).set_integrator('lsoda', method='bdf',
                                                 rtol=rtol, atol=atol, nsteps=5000)
                ode15s.set_initial_value(state0, 0.0)
                state_next = ode15s.integrate(ode15s.t+dt)
                dz_dt = state_next[0]-z0
                dT_dt = state_next[1]-T0
                dP_dt = state_next[2]-P0
                dS_dt = state_next[3]-S0+ds_turb
                dwv_dt = state_next[4]-wv0    
        else:
            # assumes that temperature and pressure are constant over time step
            X0 = (S0*water_uptake.es(T0-273.15))/(c.R*T0) # mol/m^3
            X_next = X0 + (feedbacks.dwv_dt/c.Mw) # change in water vapor moles, mol/m^3
            Ph2o_next = X_next*c.R*T0
            dT_dt = -1.0*(c.L/c.Cp)*feedbacks.dwv_dt
            S_next = Ph2o_next/water_uptake.es((T0+dT_dt)-273.15)
            dS_dt = S_next-S0
            dwv_dt = air_thermo.S_to_wv(S0+dS_dt, T0+dT_dt, P0)-wv0
            
            # pv_sat, rho_air, rho_air_dry = compute_thermo_props(T0, P0, S0)
            # gamma=(P0*c.Ma)/(pv_sat*c.Mw)+(c.Mw*c.L**2)/(c.Cp*c.R*T0**2)
            # dS_dt=-1.0*gamma*(feedbacks.dwc_dt/rho_air_dry)

    if processes.cocondensation:
        if ParcelState_next.TraceGas_population:
            for ii in range(len(feedbacks.gases.names)):
                gas = feedbacks.gases.names[ii]
                dppb_dt = feedbacks.gases.dc_dts[ii]
                TraceGas_idx = ParcelState_next.TraceGas_population.get_species_idx(gas)
                ParcelState_next.TraceGas_population.concs[TraceGas_idx] += dppb_dt
    
    if processes.gas_chemistry:
        P = ParcelState_0.P
        T = ParcelState_0.T
        S = ParcelState_0.S
        
        # set up the initial gas concentrations
        X0 = np.zeros(len(ParcelState_0.TraceGas_population.gases)+2)
        gas_names = Dict.empty(key_type=types.unicode_type, value_type=types.int32)
        for ii, (species, gas_ppb) in enumerate(zip(ParcelState_0.TraceGas_population.gases, ParcelState_0.TraceGas_population.concs)):
            gas_names[species.name]=ii
            X0[ii]=(gas_ppb*1e-9*P)/(c.R*T) # mol/m^3
        X0[ii+1]=H2O_gas_conc(S,T,P) # water vapor conc
        gas_names['H2O']=ii+1
        X0[ii+2]=P/(c.R*T)
        gas_names['M']=ii+2

        # set up an array of reactants and products that
        # can be passed into njit function
        reactants = Dict.empty(key_type=types.int32, value_type=types.string)
        products = Dict.empty(key_type=types.int32, value_type=types.string)
        rates = np.empty(0)
        for ii, (reaction) in enumerate(gas_reactions.reactions):
            temp=str(reaction.reactants)
            temp=temp.replace(',','')
            temp=temp.replace('[','')
            temp=temp.replace(']','')
            temp=temp.replace("'",'')
            reactants[ii]=temp
            temp=str(reaction.products)
            temp=temp.replace(',','')
            temp=temp.replace('[','')
            temp=temp.replace(']','')
            temp=temp.replace("'",'')
            products[ii]=temp
            rates=np.append(rates, reaction.get_rate(S, T, P))
        
        # define function    
        rhs = lambda t, X: gas_chemistry.dCgas_dt(X, reactants, products, rates, 
                                                  gas_names, T, P) 
        
        # solve
        if solver == 'CVODE': 
            prob = Explicit_Problem(rhs, X0)
            sim = CVode(prob)
            sim.atol=atol
            sim.rtol=rtol
            sim.verbosity=verbosity
            output=sim.simulate(dt)
            X_next=output[1][-1] # mol/m^3
            
        elif solver == 'ode15s':
            ode15s = ode(rhs).set_integrator('lsoda', method='bdf',
                                              rtol=rtol, atol=atol, nsteps=5000)
            
            ode15s.set_initial_value(X0, 0.0)
            X_next = ode15s.integrate(ode15s.t+dt)  # mol/m^3
            #print(X0[ii+1], X_next[ii+1])
            #print(X_next[-3]-X0[-3])
            
        # convert to ppb
        for ii in range(len(ParcelState_0.TraceGas_population.gases)):
            ParcelState_next.TraceGas_population.concs[ii]=X_next[ii]*1e9*((c.R*T)/P)
        ParcelState_next.S = (X_next[-2]*c.R*T)/water_uptake.es(T-273.15) # S change from chemistry

        # do mass balance
        utilities.check_gas_chemistry(ParcelState_0, ParcelState_next)
    
    ParcelState_next.z = z0+dz_dt
    ParcelState_next.T = T0+dT_dt
    ParcelState_next.P = P0+dP_dt
    ParcelState_next.S = S0+dS_dt
    ParcelState_next.wv = wv0+dwv_dt
    
    return ParcelState_next



def air_from_les(ParcelState_0, processes, t2, one_trajectory_settings, 
                 relaxation_time, dt, solver, gas_data, gas_names, atol, 
                 rtol, verbosity=50):
    
    #ParcelState_Next=deepcopy(ParcelState_0)
    ParcelState_Next=ParcelState_0.clone_detached()
    t0=0.0
    
    if processes.entrainment:
        if not relaxation_time:
            raise ValueError("Must enter an entrainment rate if entrainment = True!")
        
        # update the parcel location, temp, and pressure
        ParcelState_Next.z = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.z_data)
        ParcelState_Next.x = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.x_data)
        ParcelState_Next.y = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.y_data)
        ParcelState_Next.P = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.P_data)
        ParcelState_Next.T = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.T_data)
        
        # update the saturation ratio
        S_env = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.S_data)
        X_h2o_env = (S_env*water_uptake.es(ParcelState_Next.T-273.15))/(c.R*ParcelState_Next.T) # mol/m^3
        X0_h2o_parcel = (ParcelState_0.S*water_uptake.es(ParcelState_Next.T-273.15))/(c.R*ParcelState_Next.T) # mol/m^3
        rhs = lambda t, X_parcel: (1/relaxation_time)*(X_h2o_env-X_parcel)        
        if solver == 'CVODE':
            prob = Explicit_Problem(rhs, X0_h2o_parcel)
            sim = CVode(prob)
            sim.atol=atol
            sim.rtol=rtol
            sim.verbosity=verbosity
            output=sim.simulate(dt)
            X_h2o_next=output[1][-1] # mol/m^3
            P_h2o_next = X_h2o_next[-1]*c.R*ParcelState_Next.T
            S_next = P_h2o_next/water_uptake.es(ParcelState_Next.T-273.15)
            ParcelState_Next.S=S_next
        elif solver == 'ode15s':
            ode15s = ode(rhs).set_integrator('lsoda', method='bdf',
                                            rtol=rtol, atol=atol, nsteps=5000)
            ode15s.set_initial_value(X0_h2o_parcel, t0)
            X_h2o_next = ode15s.integrate(ode15s.t+dt) # mol/m^3
            P_h2o_next = X_h2o_next[-1]*c.R*ParcelState_Next.T
            S_next = P_h2o_next/water_uptake.es(ParcelState_Next.T-273.15)
            ParcelState_Next.S=S_next
            
        # update the gas concentrations
        if ParcelState_0.TraceGas_population:
            X0 = []
            X_env = []
            for gas in ParcelState_0.TraceGas_population.gases:
                idx = ParcelState_0.TraceGas_population.get_species_idx(gas.name)
                X0.append(ParcelState_0.TraceGas_population.concs[idx])
                if gas.name in gas_names:
                    if ParcelState_Next.z < np.min(gas_data[gas.name]['alt']):
                        f = lambda x, a, b: a*x**b
                        params, covariance = opt.curve_fit(f, gas_data[gas.name]['alt'][:2], gas_data[gas.name]['ppb'][:2], p0=[1, 0.1])
                        X_env.append(f(ParcelState_Next.z, params[0], params[1]))
                    else:
                        X_env.append(np.interp(ParcelState_Next.z, xp=gas_data[gas.name]['alt'], fp=gas_data[gas.name]['ppb']))
                elif gas.name == 'N2':
                    X_env.append(1e9*0.7808)
                elif gas.name == 'O2':
                    X_env.append(1e9*0.2095)   
                else:
                    X_env.append(0.0)

            X_env = np.array(X_env)
            X0 = np.array(X0)
            
            rhs = lambda t, X_parcel: (1/relaxation_time)*(X_env-X_parcel)
            
            if solver == 'CVODE':
                prob = Explicit_Problem(rhs, X0)
                sim = CVode(prob)
                sim.atol=atol
                sim.rtol=rtol
                sim.verbosity=verbosity
                output=sim.simulate(dt)
                ParcelState_Next.TraceGas_population.concs=output[1][-1] # mol/m^3
            elif solver == 'ode15s':
                ode15s = ode(rhs).set_integrator('lsoda', method='bdf',
                                                 rtol=rtol, atol=atol, nsteps=5000)
                ode15s.set_initial_value(X0, t0)
                X_next = ode15s.integrate(ode15s.t+dt)
                ParcelState_Next.TraceGas_population.concs = X_next               
    else:
        ParcelState_Next.z = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.z_data)
        ParcelState_Next.x = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.x_data)
        ParcelState_Next.y = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.y_data)
        ParcelState_Next.P = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.P_data)
        ParcelState_Next.S = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.S_data)
        ParcelState_Next.T = np.interp(t2, one_trajectory_settings.t_data, one_trajectory_settings.T_data)
                    
        # update the gas concentrations
        if cocondensation and ParcelState_Next.TraceGas_population:
            new_gas_conc = []
            for gas in ParcelState_Next.TraceGas_population.gases:
                if ParcelState_Next.z < np.min(gas_data[gas.name]['alt']):
                    f = lambda x, a, b: a*x**b
                    params, covariance = opt.curve_fit(f, gas_data[gas.name]['alt'][:2], gas_data[gas.name]['ppb'][:2], p0=[1, 0.1])
                    new_gas_conc.append(f(ParcelState_Next.z, params[0], params[1]))
                #elif ParcelState_Next.z > np.max(gas_data[gas.name]['alt']):
                    #f = lambda x, a, b: a*x**b
                    #params, covariance = opt.curve_fit(f, gas_data[gas.name]['alt'][-2:], gas_data[gas.name]['ppb'][-2:], p0=[1, 0.1])
                    #new_gas_conc.append(f(ParcelState_Next.z, params[0], params[1]))
                else:
                    new_gas_conc.append(np.interp(ParcelState_Next.z, xp=gas_data[gas.name]['alt'], fp=gas_data[gas.name]['ppb']))
            ParcelState_Next.TraceGas_population.concs=new_gas_conc
    return ParcelState_Next


