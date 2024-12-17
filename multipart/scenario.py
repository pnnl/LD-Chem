""" scenario - types and functions supporting the creation of single
simulation scenarios.

@author: Laura Fierce
"""

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import interp1d
from particles import ParticlePopulation
from particles import make_particle
from TraceGases import retrieve_gas_species
from TraceGases import TraceGasPopulation
# from Reactions import retrieve_eq_reactions, retrieve_aq_species, AqueousPopulation
from Reactions import Reaction, AqueousReactions
from processes.water_uptake import equilibrate_water
from aerosol_species import retrieve_one_species
from typing import Tuple, Callable, Optional
import mat73, sys, pickle, warnings
from scipy.special import erfinv
from systems import Processes
import matplotlib.pyplot as plt

@dataclass
class TrajectorySettings: # settings driving on trajectory simulation (trajectories can interact)
    population0: ParticlePopulation
    gas0: TraceGasPopulation 
    
    x0: Optional[float]
    y0: Optional[float]
    z0: Optional[float]
    
    u0: Optional[float]
    v0: Optional[float]
    w0: Optional[float]
    
    S0: Optional[float]
    P0: Optional[float]
    T0: Optional[float]
    
    t_data: Optional[float] = None
    x_data: Optional[float] = None
    y_data: Optional[float] = None
    z_data: Optional[float] = None
    
    u_data: Optional[float] = None
    v_data: Optional[float] = None
    w_data: Optional[float] = None
    
    S_data: Optional[float] = None
    P_data: Optional[float] = None
    T_data: Optional[float] = None

@dataclass
class Scenario:
    # settings needed to simulate ensemble of trajectories
    trajectories_settings: Tuple[TrajectorySettings, ...]
    start_times: Tuple[float, ...]
    end_times: Tuple[float, ...]
    dt: float
  
# maybe turn this into a class? and store the file, etc. along with the scenarios
def create_scenario_from_DNS(
        case_num=2,dns_dir='/Users/fier887/Downloads/New_cases (7-27-22)/',
        Ddry=100e-9, Nper=1e6, species_names=['NaCl'], mass_fractions=np.array([1.]),
        dt = None, specdata_path='../species_data/',this_many=None):
    dns_filename = dns_dir + 'case' + str(case_num) + '.mat'
    data_dict = mat73.loadmat(dns_filename)
    
    aero_spec_names, aero_spec_fracs = get_aero_spec_fracs(
        molecule_names=species_names, molecule_mass_fracs=mass_fractions,
        specdata_path=specdata_path)
    monodisperse_population = make_monodisperse_population(Ddry, Nper, aero_spec_names, aero_spec_fracs, specdata_path='../species_data/')
    
    trajectories_settings = []
    start_times = []
    end_times = []
    
    all_trj_idx = np.arange(len(data_dict['trajectories']))
    if this_many == None:
        all_dns_trajectories = data_dict['trajectories']
        trj_idx = all_trj_idx
    else:
        trj_idx = np.random.choice(all_trj_idx,size=this_many)
        all_dns_trajectories = [data_dict['trajectories'][ii] for ii in trj_idx]
    
    for dns_trajectory in all_dns_trajectories:
        ts = dns_trajectory[0][:,1]
        one_settings = TrajectorySettings(
            x0=None,y0=None,z0=None,u0=None,v0=None,w0=None,
            S0=None, T0=None, P0=None,
            u_fun=lambda t: np.interp(t,ts[1:],dns_trajectory[0][1:,5]),
            v_fun=lambda t: np.interp(t,ts[1:],dns_trajectory[0][1:,6]),
            w_fun=lambda t: np.interp(t,ts[1:],dns_trajectory[0][1:,7]),
            S_fun=lambda t: np.interp(t,ts[1:],dns_trajectory[0][1:,13])/100.,
            P_fun=lambda t: 101325.,
            T_fun=lambda t: np.interp(t,ts[1:],dns_trajectory[0][1:,10]),
            population0=monodisperse_population)
        trajectories_settings.append(one_settings)
        start_times.append(min(ts[1:]))
        end_times.append(max(ts[1:]))
        
    if dt == None:
        dt = ts[1] - ts[0]
    
    return Scenario(
        trajectories_settings=trajectories_settings,# processes=processes,
        start_times=start_times, end_times=end_times, dt=dt)

def create_parcel_scenario(
        aerosol_population = None, TraceGas_population=None,
            Ddry=100e-9,sigma=1.0,Ntot=1e6,Npart=1,updraft_velocity=0.5,
            S0=-0.15,P0=101325,T0=298,pH0=7.0,z_start=0.0,z_end=1000,
            species_names=['NaCl'],mass_fractions=np.array([1.]),
            gas_names=None, gas_conc=None, 
            dt=1.0, specdata_path='../species_data/',
            mechanism_data_path='../mechamisms/',
            chemistry=None, cocondensation=False):

    if Npart > 1 and sigma == 1.0:
        print('WARNING: Sigma = 1.0 and Npart > 1! Setting Npart to 1 to speed up calculations.')
        Npart = 1
        
    if cocondensation:
        TraceGas_population = make_TraceGas_population(gas_names, gas_conc, specdata_path=specdata_path)
    else:
        TraceGas_population=None
    
    if chemistry:
        aq_reactions = make_AqReactions(chemistry=chemistry, mechanism_data_path=mechanism_data_path)
    else:
        aq_reactions = None
    
    if aerosol_population == None:
        
        aero_spec_names, aero_spec_fracs = get_aero_spec_fracs(
            molecule_names=species_names, molecule_mass_fracs=mass_fractions,
            specdata_path=specdata_path)        
        
        if 'H+' not in aero_spec_names:
            aero_spec_names.append('H+')
            aero_spec_fracs=np.append(aero_spec_fracs, 0.0)
        
        if np.iterable(Ddry):
            aerosol_population = make_polydisperse_population(Ddry, Ntot, aero_spec_names, aero_spec_fracs, aq_reactions=aq_reactions, gases=TraceGas_population, specdata_path=specdata_path, surface_tension=0.072)
        elif sigma==1.0:
            aerosol_population = make_monodisperse_population(Ddry, Ntot, aero_spec_names, aero_spec_fracs, aq_reactions=aq_reactions, gases=TraceGas_population, specdata_path=specdata_path, surface_tension=0.072)        
        else:
            Dpg = Ddry*np.exp(np.log(sigma)*np.log(sigma))
            Dmin = Dpg*sigma**(-np.sqrt(2)*erfinv(0.999)) 
            Dmax = Dpg*sigma**(np.sqrt(2)*erfinv(0.999)) 
            model_Dps = np.logspace(np.log10(Dmin), np.log10(Dmax), Npart) # nm        
            model_Ns = lognormal_distribution(model_Dps, Ntot, Dpg, sigma) # m^-3 
            mult = Ntot/np.sum(model_Ns)
            model_Ns *= mult 
            aerosol_population = make_polydisperse_population(model_Dps, model_Ns, aero_spec_names, aero_spec_fracs, specdata_path=specdata_path, aq_reactions=aq_reactions, gases=TraceGas_population)        
        
        aerosol_population = equilibrate_water(aerosol_population,S0,T0,P0,pH0) 
    
    trajectories_settings = [TrajectorySettings(
            x0=None,y0=None,z0=z_start,
            u0=None,v0=None,w0=updraft_velocity,
            S0=S0, T0=T0, P0=P0,
            t_data=np.array((0.0, (z_end-z_start)/updraft_velocity)),
            w_data=np.array((updraft_velocity, updraft_velocity)),
            population0=aerosol_population,
            gas0=TraceGas_population)]
    start_times=[0.0]
    end_times=[(z_end-z_start)/updraft_velocity]
    return Scenario(
        trajectories_settings=trajectories_settings,
        start_times=start_times, end_times=end_times, dt=dt)


def create_hysplit_scenario(hysplit_trajectory_file,
            scenario_numbers='all', aerosol_population=None, TraceGas_population=None,
            Ddry=100e-9,sigma=1.0,Ntot=1e6,Npart=1,
            pH0=7.0,species_names=['NaCl'],mass_fractions=np.array([1.]),
            gas_names=None, gas_conc=None, 
            dt=None, specdata_path='../species_data/',
            mechanism_data_path='../mechamisms/',
            chemistry=None, cocondensation=False):
    
    # set up the S, T, and P drivers
    N_traj=0
    data=[]
    read=False
    columns=['trajectory num','??','year','month','day','hour','minute','second','relative time','latitude','longitude','altitude']
    with open(hysplit_trajectory_file) as data_file:
        for line in data_file:
            if len(line.split())==7:
                N_traj+=1
            if read:
                data.append(line.split())
            if 'PRESSURE' in line.split():
                read=True
                for variable in line.split()[1:]:
                    columns.append(variable)
    
    if 'RELHUMID' not in columns:
        print('ERROR: Relative humidity is not tracked in this HYSPLIT trajectory!')
        sys.exit()
    elif 'AIR_TEMP' not in columns:
        print('ERROR: Temperature is not tracked in this HYSPLIT trajectory!')
        sys.exit()
    
    data=np.array((data), dtype='float64')
    combined_data={}
    for i in range(len(columns)):
        combined_data[columns[i]]=data[:,i]    
    
    all_hysplit_trajectories=[]
    for t in range(N_traj):
        temp={}
        idx=np.where(combined_data['trajectory num']==t+1)
        for variable in combined_data.keys():
            temp[variable]=combined_data[variable][idx[0]]
        temp['time']=3600*(temp['relative time']+abs(np.min(temp['relative time'])))
        temp['RELHUMID']/=100
        all_hysplit_trajectories.append(temp)
    
    trajectories_settings = []
    start_times = []
    end_times = []
    
    if scenario_numbers=='all':
        scenarios=np.arange(0, len(all_hysplit_trajectories))
    else:
        scenarios=np.array((scenario_numbers))-1
        
    for i in scenarios:
        
        one_trajectory=all_hysplit_trajectories[i]
        aerosol_population=None
        
        # set up the aerosol and gas populations
        if Npart > 1 and sigma == 1.0:
            print('WARNING: Sigma = 1.0 and Npart > 1! Setting Npart to 1 to speed up calculations.')
            Npart = 1
            
        if cocondensation:
            TraceGas_population = make_TraceGas_population(gas_names, gas_conc, specdata_path=specdata_path)
        else:
            TraceGas_population=None
        
        if chemistry:
            aq_reactions = make_AqReactions(chemistry=chemistry, mechanism_data_path=mechanism_data_path)
        else:
            aq_reactions = None
        
        if aerosol_population == None:
            
            aero_spec_names, aero_spec_fracs = get_aero_spec_fracs(
                molecule_names=species_names, molecule_mass_fracs=mass_fractions,
                specdata_path=specdata_path)
            
            if 'H+' not in aero_spec_names:
                aero_spec_names.append('H+')
                aero_spec_fracs=np.append(aero_spec_fracs, 0.0)
            
            if np.iterable(Ddry):
                aerosol_population = make_polydisperse_population(Ddry, Ntot, aero_spec_names, aero_spec_fracs, aq_reactions=aq_reactions, gases=TraceGas_population, specdata_path=specdata_path, surface_tension=0.072)
            elif sigma==1.0:
                aerosol_population = make_monodisperse_population(Ddry, Ntot, aero_spec_names, aero_spec_fracs, aq_reactions=aq_reactions, gases=TraceGas_population, specdata_path=specdata_path, surface_tension=0.072)        
            else:
                Dpg = Ddry*np.exp(np.log(sigma)*np.log(sigma))
                Dmin = Dpg*sigma**(-np.sqrt(2)*erfinv(0.999)) 
                Dmax = Dpg*sigma**(np.sqrt(2)*erfinv(0.999)) 
                model_Dps = np.logspace(np.log10(Dmin), np.log10(Dmax), Npart) # nm        
                model_Ns = lognormal_distribution(model_Dps, Ntot, Dpg, sigma) # m^-3 
                mult = Ntot/np.sum(model_Ns)
                model_Ns *= mult 
                aerosol_population = make_polydisperse_population(model_Dps, model_Ns, aero_spec_names, aero_spec_fracs, specdata_path=specdata_path, aq_reactions=aq_reactions, gases=TraceGas_population)        
            
            idx=np.where(one_trajectory['time']==0)
            S0=one_trajectory['RELHUMID'][idx[0][0]]
            T0=one_trajectory['AIR_TEMP'][idx[0][0]]
            P0=one_trajectory['PRESSURE'][idx[0][0]]
            aerosol_population = equilibrate_water(aerosol_population,S0,T0,P0,pH0)
                    
        ts = np.flip(one_trajectory['time'])
        one_settings = TrajectorySettings(
            x0=None,y0=None,z0=np.flip(one_trajectory['altitude'])[0],u0=None,
            v0=None,w0=None,
            S0=None, T0=None, P0=None,
            u_data=None,
            v_data=None,
            w_data=None,
            t_data=ts,
            z_data=np.flip(one_trajectory['altitude']),
            S_data=np.flip(one_trajectory['RELHUMID']),
            P_data=np.flip(one_trajectory['PRESSURE']),
            T_data=np.flip(one_trajectory['AIR_TEMP']),
            population0=aerosol_population, gas0=TraceGas_population)
        
        trajectories_settings.append(one_settings)
        start_times.append(min(ts))
        end_times.append(max(ts))
        
        # plt.plot(one_trajectory['time'], one_trajectory['RELHUMID'])
        # plt.title(str(i+1))
        # plt.show()
    
    return Scenario(
        trajectories_settings=trajectories_settings,
        start_times=start_times, end_times=end_times, dt=dt)



def create_les_scenario(les_trajectory_file,         
            aerosol_population=None, TraceGas_population=None,
            diameters=np.array([100e-9]),N_concs=np.array([1e6]),
            pHs=np.array([7.0]),species_names=np.array(['NaCl']),
            mass_fractions=np.array([1.]),
            gas_names=None, gas_data=None,
            dt=None, specdata_path='../species_data/',
            mechanism_data_path='../mechamisms/',
            chemistry=None, cocondensation=False):
    
    # set up the S, T, and P drivers
    LES_data = pickle.load(open(les_trajectory_file, 'rb'))
    LES_data['hour'] = np.zeros(len(LES_data['time']))
    LES_data['minute'] = np.zeros(len(LES_data['time']))
    LES_data['second'] = np.zeros(len(LES_data['time']))
    for i in range(len(LES_data['time'])):
        LES_data['hour'][i] = float(str(LES_data['time'][i])[8:10])
        LES_data['minute'][i] = float(str(LES_data['time'][i])[10:12])
        LES_data['second'][i] = float(str(LES_data['time'][i])[12:])
    LES_data['simulation time']=3600*LES_data['hour']+60*LES_data['minute']+LES_data['second']
    LES_data['simulation time']-=np.min(LES_data['simulation time'])
    P0 = 101325*np.exp(-0.14586*LES_data['z'][0])
    T0 = LES_data['T'][0]
    LES_data['z']*=1000
    LES_data['P']=(P0*LES_data['T'])/T0    
    
    trajectories_settings = []
    start_times = []
    end_times = []
    aerosol_population=None
    
    
    
    if cocondensation and gas_names:
        gas_conc=np.zeros(len(gas_names))
        for ii, (gas) in enumerate(gas_names):
            gas_conc[ii]=np.interp(LES_data['z'][0], xp=gas_data[gas]['alt'], fp=gas_data[gas]['ppb'])
        TraceGas_population = make_TraceGas_population(gas_names, gas_conc, specdata_path=specdata_path)
    else:
        TraceGas_population=None
    
    if chemistry:
        aq_reactions = make_AqReactions(chemistry=chemistry, mechanism_data_path=mechanism_data_path)
    else:
        aq_reactions = None
    
    if aerosol_population == None:
        Npart = len(diameters)
        
        # check the input shapes
        if len(N_concs) != Npart:
            print('WARNING: Shape of number concentrations is not consistent with the number of particles!')
            sys.exit()
        elif len(pHs) != Npart:
            print('WARNING: Shape of pHs is not consistent with the number of particles!')
            sys.exit()
        elif len(species_names) != Npart:
            print('WARNING: Shape of species names is not consistent with the number of particles!')
            sys.exit()
        elif len(mass_fractions) != Npart:
            print('WARNING: Shape of mass fractions is not consistent with the number of particles!')
            sys.exit()
            
        particles = [None]*Npart
        num_concs = [None]*Npart
        ids = [None]*Npart 
        
        for ii in range(len(species_names)):
            
            aero_spec_names, aero_spec_fracs = get_aero_spec_fracs(
                molecule_names=species_names[ii], molecule_mass_fracs=mass_fractions[ii],
                specdata_path=specdata_path)            
        
            if 'H+' not in aero_spec_names:
                aero_spec_names.append('H+')
                aero_spec_fracs=np.append(aero_spec_fracs, 0.0)
            
            if np.round(np.sum(aero_spec_fracs),9) != 1.0:
                print(aero_spec_fracs)
                print('WARNING: Mass fractions for particle', ii, 'does not equal 1!')
                sys.exit()
            
            OneParticle = make_particle(diameters[ii], aero_spec_names, aero_spec_fracs, specdata_path=specdata_path, reactions=aq_reactions, gases=TraceGas_population)
            particles[ii] = OneParticle
            num_concs[ii] = N_concs[ii]
            ids[ii] = ii
        
        aerosol_population = ParticlePopulation(particles=particles, num_concs=num_concs, ids=ids)
        idx=np.where(LES_data['simulation time']==0)
        S0=LES_data['SS'][idx[0][0]]+1
        T0=LES_data['T'][idx[0][0]]
        P0=LES_data['P'][idx[0][0]]
        aerosol_population = equilibrate_water(aerosol_population,S0,T0,P0,pHs)
        
        ts = LES_data['simulation time']
        one_settings = TrajectorySettings(
            x0=None,y0=None,z0=LES_data['z'][0],u0=None,
            v0=None,w0=None,
            S0=None, T0=None, P0=None,
            u_data=None,
            v_data=None,
            w_data=None,
            t_data=ts,
            z_data=LES_data['z'],
            S_data=LES_data['SS']+1,
            P_data=LES_data['P'],
            T_data=LES_data['T'],
            population0=aerosol_population, gas0=TraceGas_population)
        
        trajectories_settings.append(one_settings)
        start_times.append(min(ts))
        end_times.append(max(ts))
        
        # plt.plot(one_trajectory['time'], one_trajectory['RELHUMID'])
        # plt.title(str(i+1))
        # plt.show()
    
    return Scenario(
        trajectories_settings=trajectories_settings,
        start_times=start_times, end_times=end_times, dt=dt)





def lognormal_distribution(x, Ntot, Dpg, sigma):
    prefactor = Ntot/(np.sqrt(2.0*np.pi)*np.log(sigma)*x)
    numerator = -1.0*np.power(np.log(x)-np.log(Dpg), 2)
    denominator = 2.0*np.log(sigma)*np.log(sigma)
    return prefactor*np.exp(numerator/denominator)



def get_partmc_population(filename, spec_path='../species_data/'):
    pass

# def get_monodisperse_population(D, N, aero_spec_names, aero_spec_fracs, specdata_path='../species_data/'):
#     if len([spec_name.upper('H2O') for spec_name in aero_spec_names]) == 0:
#         aero_spec_names.append('H2O')
#         aero_spec_fracs = np.append(aero_spec_fracs, 0.)
#     OneParticle = make_particle(D, aero_spec_names, aero_spec_fracs, specdata_path=specdata_path)
#     return ParticlePopulation(particles=[OneParticle], num_concs=[N], ids=[0])

def get_aero_spec_fracs(molecule_names=['NaCl'], molecule_mass_fracs=np.array([1.]),specdata_path='../species_data/'):
    aero_names_temp = []
    aero_fracs_temp = np.array([])    
    for (molecule_name,molecule_fraction) in zip(molecule_names,molecule_mass_fracs):
        one_aero_spec_names, one_aero_spec_fracs = molecules_to_fracs(
            molecule_name,molecule_fraction=molecule_fraction,specdata_path=specdata_path)
        for (onename,onefrac) in zip(one_aero_spec_names, one_aero_spec_fracs):
            aero_names_temp.append(onename)
            aero_fracs_temp = np.append(aero_fracs_temp,onefrac)
    
    aero_spec_names=[]
    aero_spec_fracs=np.array([])
    for name in np.unique(np.array((aero_names_temp))):
        idx=np.where(np.array((aero_names_temp))==name)
        aero_spec_names.append(name)
        aero_spec_fracs=np.append(aero_spec_fracs, np.sum(aero_fracs_temp[idx[0]]))
        
    # print(aero_spec_names, np.sum(aero_spec_fracs))
    
    # if len([spec_name.upper() == 'H2O' for spec_name in aero_spec_names]) == 0:
    if 'H2O' not in aero_spec_names:
        aero_spec_names.append('H2O')
        aero_spec_fracs = np.append(aero_spec_fracs, 0.)    
    
    return aero_spec_names, aero_spec_fracs
    
def molecules_to_fracs(molecule_name,molecule_fraction=1.,specdata_path='../species_data/',surface_tension=0.072):
    
    if molecule_name == 'NaCl':
        ion_names = ['Na','Cl']
        num_ions_per_molecule = [1,1]
    elif molecule_name == 'AS':
        ion_names = ['NH4','SO4']
        num_ions_per_molecule = [2,1]
    elif molecule_name == 'ABS':
        ion_names = ['NH4','HSO4']
        num_ions_per_molecule = [1,1]
    elif molecule_name == 'OC':
        ion_names = ['OC']
        num_ions_per_molecule = [1]
    elif molecule_name == 'AN':
        ion_names = ['NH4','NO3']
        num_ions_per_molecule = [1,1]
    elif molecule_name == 'BC':
        ion_names = ['BC']
        num_ions_per_molecule = [1]
    elif molecule_name == 'OIN':
        ion_names = ['OIN']
        num_ions_per_molecule = [1]
    elif molecule_name == 'tetrol':
        ion_names = ['tetrol']
        num_ions_per_molecule = [1]
    elif molecule_name == 'IEPOX_OS':
        ion_names = ['IEPOX_OS']
        num_ions_per_molecule = [1]
    elif molecule_name == 'tetrol_olig':
        ion_names = ['tetrol_olig']
        num_ions_per_molecule = [1]
        
    else:
        warnings.warn('warning: ' + molecule_name + ' is not (yet) included as a molecule; returning original')

    # double-check this!
    mass_tot = 0.
    aero_spec_fracs = []
    for (ion_name, num_ion) in zip(ion_names, num_ions_per_molecule):
        AeroSpec = retrieve_one_species(ion_name, specdata_path=specdata_path,surface_tension=surface_tension)
        mass_tot += num_ion*AeroSpec.molar_mass
        aero_spec_fracs.append(num_ion*AeroSpec.molar_mass)
        
    aero_spec_names = ion_names
    aero_spec_fracs = molecule_fraction*np.array(aero_spec_fracs)/mass_tot 
    return aero_spec_names, aero_spec_fracs
    
def make_monodisperse_population(D, N, aero_spec_names, aero_spec_fracs, aq_reactions=None, gases=None, specdata_path='../species_data/',surface_tension=0.072):
    if 'H2O' not in aero_spec_names:
    # if len([spec_name.upper()=='H2O' for spec_name in aero_spec_names]) == 0:
        aero_spec_names.append('H2O')
        aero_spec_fracs = np.append(aero_spec_fracs, 0.)
    OneParticle = make_particle(D, aero_spec_names, aero_spec_fracs, specdata_path=specdata_path, reactions=aq_reactions, gases=gases)
    
    return ParticlePopulation(particles=[OneParticle], num_concs=[N], ids=[0])

# def make_monodisperse_population(D, N, aero_spec_names, aero_spec_fracs, specdata_path='../species_data/',surface_tension=0.072):
#     if len([spec_name.upper()=='H2O' for spec_name in aero_spec_names]) == 0:
#         aero_spec_names.append('H2O')
#         aero_spec_fracs = np.append(aero_spec_fracs, 0.)
#     OneParticle = make_particle(D, aero_spec_names, aero_spec_fracs, specdata_path=specdata_path, surface_tension=surface_tension)
#     return ParticlePopulation(particles=[OneParticle], num_concs=[N], ids=[0])

def make_polydisperse_population(Ds, Ns, aero_spec_names, aero_spec_fracs, aq_reactions=None, gases=None, specdata_path='../species_data/',surface_tension=0.072):
    if 'H2O' not in aero_spec_names: #len([spec_name.upper()=='H2O' for spec_name in aero_spec_names]) == 0:
        aero_spec_names.append('H2O')
        aero_spec_fracs_withH2O = np.zeros([aero_spec_fracs.shape[0],aero_spec_fracs.shape[1]+1])
        for ii in range(len(Ds)):
            aero_spec_fracs_withH2O[ii,:] = np.append(aero_spec_fracs[ii,:],0.)
        aero_spec_fracs = aero_spec_fracs_withH2O
            # aero_spec_fracs = np.append(aero_spec_fracs, 0.)
    
    Npart = len(Ds)
    particles = [None]*Npart
    num_concs = [None]*Npart
    ids = [None]*Npart        
    for ii in range(0,Npart):
        try:
            OneParticle = make_particle(Ds[ii], aero_spec_names, aero_spec_fracs[ii,:], specdata_path=specdata_path, surface_tension=surface_tension, reactions=aq_reactions, gases=gases)
        except:
            OneParticle = make_particle(Ds[ii], aero_spec_names, aero_spec_fracs, specdata_path=specdata_path, surface_tension=surface_tension, reactions=aq_reactions, gases=gases)

        particles[ii] = OneParticle
        num_concs[ii] = Ns[ii]
        ids[ii] = ii
        
    return ParticlePopulation(particles=particles, num_concs=num_concs, ids=ids)


def make_TraceGas_population(gas_names, gas_conc, aq_reactions=None, specdata_path='../species_data/'):
    
    # if aq_reactions:
    #     for reaction in aq_reactions.reactions:
    #         for reactant in reaction.reactants:
    #             try:
    #                 OneGas = retrieve_gas_species(reactant, specdata_path=specdata_path)
    #                 if reactant not in gas_names and reactant not in ['H2O','O2']:
    #                     gas_names.append(reactant)
    #                     gas_conc.append(0.0)
    #             except:
    #                 x=1
            
    #         for product in reaction.products:
    #             try:
    #                 OneGas = retrieve_gas_species(product, specdata_path=specdata_path)
    #                 if product not in gas_names and product!='H2O':
    #                     gas_names.append(product)
    #                     gas_conc.append(0.0)
    #             except:
    #                 x=1
    
    if gas_names:
        Nspec = len(gas_conc)
        gases = [None]*Nspec
        concs = [None]*Nspec
        ids = [None]*Nspec 
        for ii in range(0,Nspec):
            OneGas = retrieve_gas_species(gas_names[ii], specdata_path=specdata_path)
            gases[ii] = OneGas
            concs[ii] = gas_conc[ii]
            ids[ii] = ii  
    else:
        gases = None
        concs = None
        ids = None
    return TraceGasPopulation(gases=gases, concs=concs, ids=ids)


def make_AqReactions(chemistry=None, mechanism_data_path='../mechanisms/'):
    reaction_datafile = mechanism_data_path + 'aq_reactions.dat'
    Nreactions=0
    with open(reaction_datafile) as data_file:
        for line in data_file:
            reactants,products,rate,dH_R,group = line.split()
            if group in chemistry:
                Nreactions+=1
    if Nreactions > 0:
        reactions = [None]*Nreactions
        ids = [None]*Nreactions
        ii=0
        while ii < Nreactions:
            with open(reaction_datafile) as data_file:
                for line in data_file:
                    reactants,products,rate,dH_R,group = line.split()
                    if group in chemistry:
                        reactants=reactants.split(',')
                        products=products.split(',')
                        OneReaction = Reaction(reactants=reactants,
                                               products=products,
                                               rate0=float(rate),
                                               neg_dH_R=float(dH_R))
                        reactions[ii]=OneReaction
                        ids[ii]=ii
                        ii+=1 
    
    else:
        reactions = None
        ids = None
    return AqueousReactions(reactions=reactions, ids=ids)


# def make_AqSpecies(aq_reactions, specdata_path='../species_data/'):
#     print()
#     aerosol_species=[]
#     aero_datafile = specdata_path + 'aero_data.dat'
#     with open(aero_datafile) as data_file:
#         for line in data_file:
#             try:
#                 name_in_file,density,ions,molecular_weight,kappa = line.split()
#                 aerosol_species.append(name_in_file)
#             except:
#                 temp=1
    
#     aq_species = []
#     for reaction in aq_reactions.reactions:
#         reactants=reaction.reactants
#         for name in reactants:
#             if name not in aq_species:
#                 aq_species.append(name)
#         products=reaction.products
        
        
#     print(len(aerosol_species))
#     print()
#     sys.exit()

#     return 1



# def make_monodisperse_population(D, N, aero_spec_names, aero_spec_fracs, specdata_path='../species_data/',surface_tension=0.072):
#     if len([spec_name.upper()=='H2O' for spec_name in aero_spec_names]) == 0:
#         aero_spec_names.append('H2O')
#         aero_spec_fracs = np.append(aero_spec_fracs, 0.)
#     OneParticle = make_particle(D, aero_spec_names, aero_spec_fracs, specdata_path=specdata_path, surface_tension=surface_tension)
#     return ParticlePopulation(particles=[OneParticle], num_concs=[N], ids=[0])

# def make_polydisperse_population(Ds, Ns, Npart, aero_spec_names, aero_spec_fracs, specdata_path='../species_data/',surface_tension=0.072):
#     if len([spec_name.upper()=='H2O' for spec_name in aero_spec_names]) == 0:
#         aero_spec_names.append('H2O')
#         aero_spec_fracs = np.append(aero_spec_fracs, 0.)
#     particles = [None]*Npart
#     num_concs = [None]*Npart
#     ids = [None]*Npart        
#     for ii in range(0,Npart):
#         OneParticle = make_particle(Ds[ii], aero_spec_names, aero_spec_fracs[ii,:], specdata_path=specdata_path, surface_tension=surface_tension)
#         particles[ii] = OneParticle
#         num_concs[ii] = Ns[ii]
#         ids[ii] = ii
#     return ParticlePopulation(particles=particles, num_concs=num_concs, ids=ids)
