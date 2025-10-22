""" scenario - types and functions supporting the creation of aerosol particles
and particle populations

@author: Laura Fierce
"""

# Multi-scale particle-based microphysics model (multipart)
import sys, warnings
import numpy as np
from dataclasses import dataclass
# from .aerosol_species import retrieve_one_species, AerosolSpecies
from typing import Tuple, Callable # need this for Python versions earlier than 3.9
# Callable[[float], float] takes as input a float and returns a float

warnings.filterwarnings('ignore')

@dataclass(frozen=True)
class AerosolSpecies:
    """AerosolSpecies: the definition of an aerosol species in terms of species-
    specific parameters (no state information)"""
    name: str          # name of the species
    density: float
    kappa: float
    molar_mass: float
    surface_tension: float
    # other parameters controlling phase partitioning
    # refractive_index: float
    
@dataclass
class Particle:
    """Particle: the definition of an individual aerosol particle
    in terms of the amounts of different constituent species """
    species: Tuple[AerosolSpecies, ...]
    masses: Tuple[float, ...]
    idx_h2o: int = -1
    
    def idx_dry(self):
        idx_all = np.arange(len(self.species))        
        idx_h2o = self.idx_h2o
        
        if idx_h2o == -1:
            idx_not_h2o = idx_all[:-1]
        elif idx_h2o >= 0:
            idx_not_h2o = np.array([idx for idx in idx_all if idx != idx_h2o])
        else:    
            idx_not_h2o = np.hstack([idx_all[:idx_h2o],idx_all[idx_h2o:][1:]])
        return idx_not_h2o
        
    def get_spec_rhos(self):
        spec_rhos = np.array([one_spec.density for one_spec in self.species])
        return spec_rhos
    
    def get_spec_kappas(self):
        spec_kappas = np.array([one_spec.kappa for one_spec in self.species])
        return spec_kappas
    
    def get_mass_dry(self):
        mks = self.masses
        mass_dry = np.sum(mks[self.idx_dry()])
        return mass_dry

    def get_mass_tot(self):
        mks = self.masses
        mass_tot = np.sum(mks)
        return mass_tot
        
    def get_rho_h2o(self):
        return self.species[self.idx_h2o].density
    
    def get_mass_h2o(self):
        return self.masses[self.idx_h2o]
        
    def get_vks(self):
        mks = self.masses
        rhos = self.get_spec_rhos() 
        vks = np.zeros(len(mks))
        for i in range(len(vks)):
            if rhos[i] > 0.0:
                vks[i] = mks[i]/rhos[i]
        return vks
        
    def get_vol_dry(self):
        vks = self.get_vks()
        vol_dry = np.sum(vks[self.idx_dry()])
        return vol_dry

    def get_vol_tot(self):
        vks = self.get_vks()
        vol_tot = np.sum(vks)
        return vol_tot
    
    def get_Ddry(self):
        vol_dry = self.get_vol_dry()
        Ddry = (vol_dry*6./np.pi)**(1./3.)
        return Ddry

    def get_Dwet(self):
        vol_wet = self.get_vol_tot()
        Dwet = (vol_wet*6./np.pi)**(1./3.)
        return Dwet
    
    def get_rho_w(self):
        return 1000. # kg/m^3 -- todo: fix this later
    
    def get_Hplus_conc(self):
        Hplus_molar_mass=self.species[self.get_species_idx('H+')].molar_mass
        Hplus_moles=self.masses[self.get_species_idx('H+')]/Hplus_molar_mass
        return Hplus_moles/(1000*(self.get_vol_tot()-self.get_vol_dry())) # mol/L
               
    # def get_rho_w(self):
    #     idx_h2o, = np.where([one_spec.name.upper()=='H2O' for one_spec in self.species])
    #     print(idx_h2o)
    #     rho_w = float(self.species[idx_h2o].density)
    #     return rho_w
    
    def get_tkappa(self):
        # compute effective kappa
        vks = self.get_vks()
        spec_kappas = self.get_spec_kappas()
        idx_not_h2o, = np.where([one_spec.name.upper()!='H2O' for one_spec in self.species])
        tkappa = np.nansum(vks[idx_not_h2o]*spec_kappas[idx_not_h2o])/np.nansum(vks[idx_not_h2o])
        return tkappa
    
    def get_trho(self):
        # compute effective density
        mks = self.masses
        vks = self.get_vks()
        trho = np.nansum(mks)/np.nansum(vks)
        
        # # alternative:
        # spec_rhos = self.get_spec_rhos()
        # trho = np.sum(vks*spec_rhos)/np.sum(vks)
        return trho
    
    def get_species_idx(self, species):
        idx = None
        for ii in range(len(self.species)):
            if self.species[ii].name == species:
                return ii
        return idx
    
    def get_pH(self):
        water_volume=1000*(self.get_vol_tot()-self.get_vol_dry())
        idx=self.get_species_idx('H+')
        Hplus_conc=(self.masses[idx]/self.species[idx].molar_mass)/water_volume
        return -1.0*np.log10(Hplus_conc)
        
    def clone_detached(self):
        # New Particle object; copy arrays you may mutate
        return Particle(
            species=self.species,        # shared: treat as immutable metadata
            masses=self.masses.copy(),   # IMPORTANT: new array, no aliasing
            idx_h2o=self.idx_h2o,
        )
        
    
    
@dataclass
class ParticlePopulation:
    """ParticlePopulation: the definition of a population of particles
    in terms of the number concentrations of different particles """
    
    particles: Tuple[Particle, ...]
    num_concs: Tuple[float, ...]
    ids: Tuple[int, ...] # can we make this an optional argument?
    # later... shape parameters?
    
    def clone_detached(self):
        # New ParticlePopulation object; particles are newly cloned
        return ParticlePopulation(
            particles=tuple(p.clone_detached() for p in self.particles),
            num_concs=tuple(self.num_concs),  # safe shallow copy
            ids=tuple(self.ids),
        )
    

def retrieve_one_species(name, specdata_path='../species_data/',surface_tension=0.072):
    aero_datafile = specdata_path + 'aero_data.dat'
    with open(aero_datafile) as data_file:
        for line in data_file:
            if line.split()[0]==name:
                name_in_file,density,ions_in_solution,molar_mass,kappa = line.split()
    
    return AerosolSpecies(
        name=name,
        density=float(density),
        kappa=float(kappa),
        molar_mass=float(molar_mass.replace('d','e')),
        surface_tension=surface_tension)
    

def make_particle(D, aero_spec_names, aero_spec_fracs, specdata_path='../species_data/',surface_tension=0.072, reactions=None, gases=None):
    # add in species that take part in aqueous reactions
    if reactions:
        for reaction in reactions.reactions:
            for reactant in reaction.reactants:
                if reactant not in aero_spec_names and reactant not in ['S(IV)', 'S(VI)', 'O2']:
                    aero_spec_names = np.append(aero_spec_names, reactant)
                    aero_spec_fracs = np.append(aero_spec_fracs, 0.0)
            for product in reaction.products:
                if product not in aero_spec_names and product not in ['S(IV)', 'S(VI)', 'O2']:
                    aero_spec_names = np.append(aero_spec_names, product)
                    aero_spec_fracs = np.append(aero_spec_fracs, 0.0)
    
    # add in species that condense into aqueous phase
    if gases:
        if gases.gases:
            for gas in gases.gases:
                if gas.H0 > 0 and gas.name not in aero_spec_names:
                    aero_spec_names = np.append(aero_spec_names, gas.name)
                    aero_spec_fracs = np.append(aero_spec_fracs, 0.0)
    
    if 'H+' not in aero_spec_names:
        aero_spec_names = np.append(aero_spec_names, 'H+')
        aero_spec_fracs = np.append(aero_spec_fracs, 0.0)
    
    AeroSpecs = []
    for name in aero_spec_names:
        AeroSpecs.append(retrieve_one_species(name, specdata_path=specdata_path, surface_tension=surface_tension))
    
    vol = np.pi/6.*D**3.
    mass = effective_density(aero_spec_fracs,AeroSpecs)*vol
    spec_masses = np.array([mass*spec_frac for spec_frac in aero_spec_fracs]) 
    
    return Particle(species=AeroSpecs,masses=spec_masses)

def effective_density(aero_spec_fracs,AeroSpecs):
    denominator = 0
    for kk in range(len(AeroSpecs)):
        if AeroSpecs[kk].density > 0.0:
            denominator += aero_spec_fracs[kk]/AeroSpecs[kk].density
    return 1./denominator

    
    
    
    
    
