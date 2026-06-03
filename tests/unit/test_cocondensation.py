import pytest
import numpy as np
from ld_chem.processes.cocondensation import (
    dCaq_dt, IEPOX_condensation, dCaq_dt_diffusion_limited,
    beta_FS, water_viscosity, cocondensation_solver, GasFeedback
)
# from ld_chem.particles import AerosolSpecies
# from ld_chem.gases import GasSpecies, TraceGasPopulation
# from part2pop.population import ParticlePopulation
from ld_chem.scenario import create_parcel_scenario
from pathlib import Path
import warnings

def test_gasfeedback_dataclass():
    """Test GasFeedback dataclass creation."""
    names = ("CO2", "SO2")
    dc_dts = np.array([1.0, 2.0])
    feedback = GasFeedback(names=names, dc_dts=dc_dts)
    
    assert feedback.names == names
    assert np.allclose(feedback.dc_dts, dc_dts)
    assert len(feedback.names) == len(feedback.dc_dts)


def test_dCaq_dt():
    """Test basic cocondensation rate calculation."""
    X = np.array([1e0, 1e0, 1e0])  # Gas conc, two particle concs
    radii = np.array([1e-6, 1e-6])  # m
    water_volumes = np.array([1e-18, 1e-18])  # m^3
    num_concs = np.array([1e6, 1e6])  # 1/m^3
    molar_mass = 0.1  # kg/mol
    alpha = 0.1
    Heff = 1e5  # mol/m^3/Pa
    T = 298.15
    result = dCaq_dt(X, radii, water_volumes, num_concs, molar_mass, alpha, Heff, T)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == X.shape
    assert np.all(np.isfinite(result))
    assert np.isclose(result[0], -1.0*np.sum(water_volumes*result[1:]*num_concs))


def test_IEPOX_condensation():
    """Test IEPOX condensation rate."""
    X = np.array([1e0, 1e0, 1e0])
    H2O_concs = np.array([55e3, 55e3])
    Hplus_concs = np.array([1e-4, 1e-4])
    HSO4_concs = np.array([1e-3, 1e-3])
    NH4_concs = np.array([1e-3, 1e-3])
    SO4_concs = np.array([1e-3, 1e-3])
    radii = np.array([1e-6, 1e-6])
    T = 298.15
    S = 1.0
    l_orgs = np.array([1e-7, 1e-7])
    inorganic_radii = np.array([[5e-7, 5e-7]])
    num_concs = np.array([1e6, 1e6])
    water_volumes = np.array([1e-18, 1e-18])
    molar_mass = 0.1
    alpha = 0.001
    result = IEPOX_condensation(
        X, H2O_concs, Hplus_concs, HSO4_concs, NH4_concs,
        SO4_concs, radii, T, S, l_orgs, inorganic_radii,
        num_concs, water_volumes, molar_mass, alpha
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == X.shape
    assert np.isclose(result[0], -1.0*np.sum(water_volumes*result[1:]*num_concs))



def test_dCaq_dt_diffusion_limited():
    """Test diffusion-limited cocondensation."""
    X = np.array([1e0, 1e0, 1e0])
    radii = np.array([1e-6, 1e-6])
    water_volumes = np.array([1e-18, 1e-18])
    num_concs = np.array([1e6, 1e6])
    molar_mass = 0.1
    alpha = 0.1
    Heff = 1e5
    T = 298.15
    P = 101325
    Dl_0 = 1e-9

    result = dCaq_dt_diffusion_limited(
        X, radii, water_volumes, num_concs, molar_mass, alpha, Heff, T, P, Dl_0
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == X.shape
    assert np.isclose(result[0], -1.0*np.sum(water_volumes*result[1:]*num_concs))


def test_beta_FS():
    """Test Fuchs-Sutugin correction factor."""
    r = 1e-6  # m
    T = 298.15
    P = 101325
    alpha = 0.1
    result = beta_FS(r, T, P, alpha)
    assert isinstance(result, (float, np.floating))
    assert result > 0


def test_water_viscosity():
    """Test water viscosity calculation."""
    T = 298.15
    result = water_viscosity(T)
    assert isinstance(result, (float, np.floating))
    assert result > 0

def test_cocondensation_solver():
    """Test cocondensation solver."""
    num_concs = np.array([1e6])
    pHs = np.array([7.0])
    species_names = np.array(['SO4','NH4','IEPOX_OS','H2O'])
    species_masses = np.array([[1e-25,1e-25,1e-25,1e-25]])
    gas_names = ['SO2','IEPOX','HNO3','H2SO4']
    gas_concs = [1e-6,1e-6,1e-6,1e-6]

    mechanisms_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "mechanisms"
    species_data_path = Path(__file__).parent.parent.parent / "src" / "ld_chem" / "species_data"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore") # ignore divide by zero warnings
        parcel_state, aq_reactions, gas_reactions = create_parcel_scenario(
            num_concs=num_concs,
            pHs=pHs,
            species_names=species_names,
            species_masses=species_masses,
            gas_names=gas_names,
            gas_concs=gas_concs,
            z_end=10.0,
            specdata_path=str(species_data_path) + "/",
            mechanism_data_path=str(mechanisms_path) + "/",
            gas_chemistry=False,
            cocondensation=True,
            aq_chemistry=['sulfate']
        )

        population_next, gas_feedback = cocondensation_solver(
            parcel_state.particles, parcel_state.gas, 101325, 298, 0.85)
        
    assert isinstance(population_next, type(parcel_state.particles))
    assert isinstance(gas_feedback, GasFeedback)
    assert len(gas_feedback.names) == len(gas_feedback.dc_dts)
    assert np.all(np.isfinite(gas_feedback.dc_dts))

