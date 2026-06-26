import pickle, argparse, os, warnings
from part2pop import build_population
import numpy as np
import part2pop.population.factory.helpers.hiscale as hiscale_helpers
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

def main(args):
    
    # STEP 1: Sample the aerosol population using part2pop
    population_settings = {
        "type": "hiscale_observations",
        "N_particles": args.N_particles,
        "beasd_file": args.size_distribution_file,
        "aimms_file": args.AIMMS_file,
        "splat_file": args.SPLAT_file,
        "ams_file": args.AMS_file,
        "z": args.z,
        "dz": args.dz,
        "splat_cutoff_nm": args.SPLAT_cutoff,
        "splat_species": args.SPLAT_species,
        "mass_thresholds": args.mass_thresholds
    }

    aerosol_population = build_population(population_settings)
    aero_spec_names = np.array([species.name for species in aerosol_population.species]) # Get species names
    aero_spec_masses = np.array(aerosol_population.spec_masses) # Get mass of species in each particle
    num_concs = np.array(aerosol_population.num_concs) # Get number concentration of each particle
    pHs=np.random.normal(size=num_concs.shape[0], loc=2.28, scale=0.78) # sample pHs
    diameters = aerosol_population.get_particle_var("dry_diameter")
    total_masses = np.repeat(np.sum(aero_spec_masses, axis=1)[:, np.newaxis], aero_spec_masses.shape[1], axis=1)
    aero_spec_fracs = aero_spec_masses/total_masses
    assert np.isclose(np.sum(aero_spec_fracs, axis=1), 1.0).all()

    evaluate_size_distribution(
        args, diameters, num_concs
    )
    evaluate_mass_concentrations(
        args, aero_spec_names,
        aero_spec_masses, num_concs
    )


    # STEP 2: load FLEXPART trajectory information
    flexpart_trajectory = read_flexpart_output(args)


    # STEP 3: interpolate vertical gas concentrations along
    #         trajectory to get background gas trajectory
    vertical_gas_data = vertical_gas_profiles(args)
    flexpart_trajectory['gas'] = build_gas_trajectory(flexpart_trajectory['z'], vertical_gas_data)


    # STEP 4: write files  
    os.makedirs(args.output_directory, exist_ok=True)  
    pickle.dump(aero_spec_names, open(f"{args.output_directory}/aero_spec_names.pkl", "wb"))
    pickle.dump(aero_spec_masses, open(f"{args.output_directory}/aero_spec_masses.pkl", "wb"))
    pickle.dump(aero_spec_fracs, open(f"{args.output_directory}/aero_spec_fracs.pkl", "wb"))
    pickle.dump(num_concs, open(f"{args.output_directory}/number_concentrations.pkl", "wb"))
    pickle.dump(diameters, open(f"{args.output_directory}/diameters.pkl", "wb"))
    pickle.dump(pHs, open(f"{args.output_directory}/pHs.pkl", "wb"))
    pickle.dump(flexpart_trajectory, open(f"{args.output_directory}/FLEXPART_trajectory.pkl", "wb"))

    return

def evaluate_size_distribution(
        args, sampled_Dps, sampled_Ns
):
    Dp_lo_nm, Dp_hi_nm, N_cm3, _ = hiscale_helpers._read_beasd_avg_size_dist(
        beasd_file=args.size_distribution_file, 
        aimms_file=args.AIMMS_file, 
        z=args.z, 
        dz=args.dz,
    )
    Dp_mid_nm = Dp_lo_nm + 0.5 * (Dp_hi_nm - Dp_lo_nm)
    Dp_mid_m = Dp_mid_nm * 1e-9
    N_m3 = N_cm3 * 1e6  # cm^-3 -> m^-3
    dln = np.log(Dp_hi_nm / Dp_lo_nm)
    if np.any(~np.isfinite(dln)) or np.any(dln <= 0):
        raise ValueError("Invalid FIMS bin edges; cannot compute dln widths.")
    measured_SA = np.sum(4.0*np.pi*np.power((Dp_mid_m)/2, 2)*(N_m3))
    measured_Vtot = np.sum((4.0/3.0)*np.pi*np.power((Dp_mid_m)/2, 3)*N_m3)
    measured_mean_size = np.average(1e-9*Dp_mid_nm, weights=N_m3)
    
    sampled_SA = np.sum(4.0*np.pi*np.power((sampled_Dps)/2, 2)*(sampled_Ns))
    sampled_Vtot = np.sum((4.0/3.0)*np.pi*np.power((sampled_Dps)/2, 3)*sampled_Ns)
    sampled_mean_size = np.average(sampled_Dps, weights=sampled_Ns)

    print("SIZE DISTRIBUTION DIAGNOSTICS: (measured, sampled)")
    print(f"Total surface area (m^2/m^3): {measured_SA:.2e}, {sampled_SA:.2e}")
    print(f"Total volume (m^3/m^3): {measured_Vtot:.2e}, {sampled_Vtot:.2e}")
    print(f"Mean size (m): {measured_mean_size:.2e}, {sampled_mean_size:.2e}\n")
    
    return
    

def evaluate_mass_concentrations(
        args, aero_spec_names, 
        aero_spec_masses, num_concs
):
    measured_mass_frac, _, _, _ = hiscale_helpers._read_ams_mass_fractions(
        ams_file=args.AMS_file, aimms_file=args.AIMMS_file, 
        size_dist_type=args.size_dist_type, 
        size_dist_file=args.size_distribution_file,
        z=args.z, dz=args.dz
    )

    sampled_masses = {'total': 0.0}
    for kk in measured_mass_frac.keys():
        if kk == "OC":
            indices = [list(aero_spec_names).index(x) for x in args.mass_thresholds['IEPOX_SOA'][1]]
            SOA_masses=np.sum(np.sum(aero_spec_masses[:,indices], axis=1)*num_concs)
            OC_idx=np.where(aero_spec_names==kk)[0][0]
            OC_masses=np.sum(aero_spec_masses[:,OC_idx]*num_concs)
            sampled_masses["OC"]=SOA_masses+OC_masses
            sampled_masses["total"]=SOA_masses+OC_masses
        else:
            idx = np.where(aero_spec_names==kk)[0][0]
            sampled_masses[kk]=np.sum(aero_spec_masses[:,idx]*num_concs)
            sampled_masses['total']+=np.sum(aero_spec_masses[:,idx]*num_concs)

    print("BULK MASS FRACTION DIAGNOSTICS: (measured, sampled)")
    for kk in measured_mass_frac.keys():
        print(f"{kk}: {measured_mass_frac[kk]:.3f}, {sampled_masses[kk]/sampled_masses['total']:.3f}")
    print()

    return 

def build_gas_trajectory(trajectory_z, vertical_gas_data):
    gas_trajectory={}
    for gas in vertical_gas_data.keys():
        ppb_interp=np.zeros(len(trajectory_z))
        for ii in range(len(trajectory_z)):
            if trajectory_z[ii] < np.min(vertical_gas_data[gas]['alt']):
                f = lambda x, a, b: a*x**b
                params, _ = curve_fit(f, vertical_gas_data[gas]['alt'][:2], vertical_gas_data[gas]['ppb'][:2], p0=[1, 0.1])
                ppb_interp[ii]=f(trajectory_z[ii], params[0], params[1])
            else:
                ppb_interp[ii]=np.interp(trajectory_z[ii], xp=vertical_gas_data[gas]['alt'], fp=vertical_gas_data[gas]['ppb'])
        gas_trajectory[gas]=ppb_interp
    
    return gas_trajectory

def read_flexpart_output(args):

    # EXPECTED COLUMN NUMBERS
    # IN FLEXPART OUTPUT FILE
    # trajectory number = 0 
    # time (utc) = 1 
    # long = 2 
    # lat = 3 
    # altitude (agl) = 4 
    # altitude (msl) = 5 
    # pressure (mb) = 6
    # temperature (K) = 7
    # RH (%) = 8
    # w (m/s) = 9
    # water vapor mixing ratio (g/kg) = 10 
    # cloud mixing ratio (g/kg) = 11

    data = np.loadtxt(args.FLEXPART_file)
    output_dict = {}
    output_dict['t'] = data[:,1]*3600
    output_dict['x'] = data[:,2]
    output_dict['y'] = data[:,3]
    output_dict['z'] = data[:,4]
    output_dict['P'] = data[:,6]*100 # Pa
    output_dict['T'] = data[:,7]
    qvapor=data[:,10]
    temp=data[:,7]-273.15 # degrees C
    pres=data[:,6]*100 # Pa
    es = 611.2 * np.exp(17.67 * temp / (temp + 243.5)) # Pa
    qs = 622*es/(pres-(1.-0.622)*es)
    traj_rh=100.0*(qvapor/qs) # %
    output_dict['s'] = traj_rh/100

    return output_dict


def vertical_gas_profiles(args):
    
    # get the altitude grid used for interpolating
    filename = os.listdir(args.gas_phase_directory)[0]
    raw_data = np.loadtxt(f"{args.gas_phase_directory}/{filename}", delimiter = ',', dtype='str')
    gas_data = {}
    for ii in range(0, len(raw_data[0])):
       gas_data[str(raw_data[0, ii])] = np.array(raw_data[1:, ii], dtype = 'float64')
    alt_grid = np.linspace(np.min(gas_data['Alt']), np.max(gas_data['Alt']), 16)
    alt_mids = 0.5*(alt_grid[range(len(alt_grid)-1)] + alt_grid[range(1,len(alt_grid))])

    gas_data_all={}    
    for gas in args.gas_names:       
        if gas == 'NO2': # NO2 = NOx-NO
            for f in os.listdir(args.gas_phase_directory):
                filename = os.path.join(args.gas_phase_directory, f)
                if os.path.isfile(filename):
                    temp_name=filename.split('_')
                    if temp_name[-2]=='NOx':
                        filename1=os.path.join(args.gas_phase_directory, f)
                    elif temp_name[-2]=='NO':
                        filename2=os.path.join(args.gas_phase_directory, f)
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
                        alts, medians = get_AGFL_profile(gas, args.gas_phase_directory)
                        gas_data_all[gas]={}
                        gas_data_all[gas]['ppb']=medians
                        gas_data_all[gas]['alt']=alts
                    except:
                        raise ValueError(f"No gas phase data for {gas}")
            else:
                try:
                    alts, medians = get_AGFL_profile(gas, args.gas_phase_directory)
                    gas_data_all[gas]={}
                    gas_data_all[gas]['ppb']=medians
                    gas_data_all[gas]['alt']=alts
                except:
                    raise ValueError(f"No gas phase data for {gas}")
        else: # this is for all the other gases
            file_to_read=None
            for f in os.listdir(args.gas_phase_directory):
                filename = os.path.join(args.gas_phase_directory, f)
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
                                          & (gas_data['Long']>args.Lon_min)
                                          & (gas_data['Long']<args.Lon_max)
                                          & (gas_data['Lat']>args.Lat_min)
                                          & (gas_data['Lat']<args.Lat_max))[0]
                        if len(idx)>0:
                            alts=np.append(alts,alt_mids[rr-1])
                            medians = np.append(medians, np.percentile(gas_data['Value_ppb'][idx], 50))
                
                if len(medians)>0:
                    gas_data_all[gas]={}
                    gas_data_all[gas]['ppb']=medians
                    gas_data_all[gas]['alt']=alts
                else:
                    try:
                        alts, medians = get_AGFL_profile(gas, args.gas_phase_directory)                        
                        gas_data_all[gas]={}
                        gas_data_all[gas]['ppb']=medians
                        gas_data_all[gas]['alt']=alts
                    except:
                        raise ValueError(f"No gas phase data for {gas}")
            else:
                try:
                    alts, medians = get_AGFL_profile(gas, args.gas_phase_directory)
                    gas_data_all[gas]={}
                    gas_data_all[gas]['ppb']=medians
                    gas_data_all[gas]['alt']=alts
                except:
                    raise ValueError(f"No gas phase data for {gas}")
    
    return gas_data_all


def get_AGFL_profile(gas, trace_gas_folder):
    filename = os.path.join(trace_gas_folder, 'AGFL_atmosphere.txt')
    raw_data = np.loadtxt(filename, dtype='str')
    AGFL = {}
    AGFL[str(raw_data[0, 0])] = np.array(raw_data[1:, 0], dtype='float64')
    for i in range(0, len(raw_data[0])): 
        AGFL[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
        
    return 1000*AGFL['z'], 1000*AGFL[gas]


if __name__ == "__main__":

    default_splat_species = {
        'BC': ['soot'], 
        'OIN': ['Dust'], 
        'SO4': ['sulfate_nitrate_org'], 
        'NO3': ['nitrate_amine_org'],
        'OC': ['org28', 'org30_43', 'BB_SOA', 'org_amines', 'BB', 'pyridine'], 
        'IEPOX_SOA': ['IEPOX_SOA']
    }

    # mass thresholds[class][0] is min mass fraction, mean initial mass fraction, std of initial mass fraction
    # mass thresholds[class][1] are tehe species that are included in that class 
    default_mass_thresholds = {
        'IEPOX_SOA': [[0.3,0.5,0.1], ['IEPOX_OS','tetrol','tetrol_olig', 'IEPOX_OH_SOA']],
        'SO4': [[0.5,0.7,0.1], ['SO4']],
        'NO3': [[0.5,0.7,0.1], ['NO3']],
        'OC': [[0.5,0.7,0.1], ['OC']],
        'BC': [[0.5,0.7,0.1], ['BC']],
        'OIN': [[0.5,0.7,0.1], ['OIN']]
    }

    parser = argparse.ArgumentParser(description="Get inputs for one WRF/FLEXPART driven LD-Chem simulation of HI-SCALE field campaign.")
    parser.add_argument("--N_particles", type=int, required=True,
                        help="Number of particles to include in simulation.")
    parser.add_argument("--size_distribution_file", type=str, required=True,
                        help="File where size distribution information is stored.")
    parser.add_argument("--AIMMS_file", type=str, required=True,
                        help="File where location/altitude of aricraft is stored.")
    parser.add_argument("--SPLAT_file", type=str, required=True,
                        help="File where time series of miniSPLAT number fraction in each class is stored.")
    parser.add_argument("--AMS_file", type=str, required=True,
                        help="File where AMS-measured mass concentrations are stored.")
    parser.add_argument("--FLEXPART_file", type=str, required=True,
                        help="File where FLEXPART output is stored.")
    parser.add_argument("--gas_phase_directory", type=str, required=True,
                        help="Directory where gas phase measurements are stored. Expected format is xxxx_NAME_xxxx.txt.")
    parser.add_argument("--z", type=float, default=100.0,
                        help="Altitude where initial values are drawn from. Will be averaged between z-dz and z+dz.")
    parser.add_argument("--dz", type=float, default=100.0,
                        help="Altitude where initial values are drawn from. Will be averaged between z-dz and z+dz.")
    parser.add_argument("--SPLAT_species", type=dict, default=default_splat_species,
                        help="Map between LD-Chem species and SPLAT classes.")
    parser.add_argument("--mass_thresholds", type=dict, default=default_mass_thresholds,
                        help="Rules for class assignment and initial per-particle mass fractions.")
    parser.add_argument("--mean_pH", type=float, default=2.28,
                        help="Mean pH of particles (randomly assigned).")
    parser.add_argument("--std_pH", type=float, default=0.78,
                        help="Standard deviation in pH of particles (randomly assigned).")
    parser.add_argument("--SPLAT_cutoff", type=float, default=85.0,
                        help="Cutoff size for SPLAT instrument in nm.")
    parser.add_argument("--output_directory", type=str, default='.',
                        help="Folder where pickle files will be written.")
    parser.add_argument("--size_dist_type", type=str, default='BEASD',
                        help="BEASD or FIMS size distribution.")
    parser.add_argument("--gas_names", type=list, default=['SO2','O3','H2O2','IEPOX','OH','HNO3','NO2','NO','NH3'],
                        help="Names of gases to include in simulations.")
    parser.add_argument("--Lon_min", type=float, default=-97.5,
                        help="Minimum latitude of G-1 sampling box where gas data is averaged.")
    parser.add_argument("--Lon_max", type=float, default=-97.4,
                        help="Minimum latitude of G-1 sampling box where gas data is averaged.")
    parser.add_argument("--Lat_min", type=float, default=36.05,
                        help="Minimum latitude of G-1 sampling box where gas data is averaged.")
    parser.add_argument("--Lat_max", type=float, default=36.81,
                        help="Minimum latitude of G-1 sampling box where gas data is averaged.")

    args = parser.parse_args()


    main(args)
