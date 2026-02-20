import pytest
import numpy as np
from numba.typed import Dict
from numba import types
from ld_chem.processes.aqueous_chemistry import (
    dCaq_dt, O3_sulfur_oxidation_rate, H2O2_sulfur_oxidation_rate,
    NO2_sulfur_oxidation_rate, HNO2_sulfur_oxidation_rate,
    O2_sulfur_oxidation_rate, IEPOX_OH_chemistry
)


def test_dCaq_dt_basic():
    """Test basic functionality of dCaq_dt."""
    # Set up mock data
    Caq_0 = np.array([1e-3, 1e-4, 1e-5])  # Concentrations for H2O2, S(IV), OH-
    aq_names = np.array(['H2O2', 'S(IV)', 'OH-'])
    
    # Mock reaction: H2O2 + S(IV) -> products
    reactants_all = np.array(['H2O2 S(IV)'])
    products_all = np.array(['products'])
    rates = np.array([1e-4])  # Mock rate constant
    T = 298.15  # Temperature
    
    result = dCaq_dt(Caq_0, reactants_all, products_all, rates, aq_names, T)
    assert isinstance(result, np.ndarray)
    assert result.shape == Caq_0.shape
    assert np.all(np.isfinite(result))


def test_O3_sulfur_oxidation_rate():
    """Test O3 sulfur oxidation rate calculation."""
    Caq_0 = np.array([1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0 ,1e0])  # SO2, HSO3, SO3, O3, H2SO4, HSO4, SO4, H+
    dCaq_dt_all = np.zeros(len(Caq_0))
    aq_names = np.array(['SO2', 'HSO3', 'SO3', 'O3', 'H2SO4', 'HSO4', 'SO4', 'H+'])
    T = 298.15
    result = O3_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T)
    assert isinstance(result, np.ndarray)
    assert result.shape == dCaq_dt_all.shape
    assert np.isclose(result[3], -1.0*(result[4]+result[5]+result[6]))


def test_H2O2_sulfur_oxidation_rate():
    """Test H2O2 sulfur oxidation rate calculation."""
    Caq_0 = np.array([1e0, 1e0, 1e0, 1e0, 1e0, 1e0])  # HSO3, H2O2, H2SO4, HSO4, SO4, H+
    dCaq_dt_all = np.zeros(len(Caq_0))
    aq_names = np.array(['HSO3', 'H2O2', 'H2SO4', 'HSO4', 'SO4', 'H+'])
    T = 298.15
    result = H2O2_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T)
    assert isinstance(result, np.ndarray)
    assert result.shape == dCaq_dt_all.shape
    assert np.isclose(result[1], -1.0*(result[2]+result[3]+result[4]))


def test_NO2_sulfur_oxidation_rate():
    """Test NO2 sulfur oxidation rate calculation."""
    Caq_0 = np.array([1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0])  # SO2, HSO3, SO3, NO2, NO, H2SO4, HSO4, SO4, H+
    dCaq_dt_all = np.zeros(len(Caq_0))
    aq_names = np.array(['SO2', 'HSO3', 'SO3', 'NO2', 'NO', 'H2SO4', 'HSO4', 'SO4', 'H+'])
    T = 298.15
    result = NO2_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T)
    assert isinstance(result, np.ndarray)
    assert result.shape == dCaq_dt_all.shape
    assert np.isclose(result[3], -1.0*(result[5]+result[6]+result[7]))


def test_HNO2_sulfur_oxidation_rate():
    """Test HNO2 sulfur oxidation rate calculation."""
    Caq_0 = np.array([1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0])  # SO2, HSO3, SO3, HNO2, H2SO4, HSO4, SO4, H+
    dCaq_dt_all = np.zeros(len(Caq_0))
    aq_names = np.array(['SO2', 'HSO3', 'SO3', 'HNO2', 'H2SO4', 'HSO4', 'SO4', 'H+'])
    T = 298.15
    result = HNO2_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T)
    assert isinstance(result, np.ndarray)
    assert result.shape == dCaq_dt_all.shape
    assert np.isclose(result[3], -1.0*(result[4]+result[5]+result[6]))


def test_O2_sulfur_oxidation_rate():
    """Test O2 sulfur oxidation rate calculation."""
    Caq_0 = np.array([1e0, 1e0, 1e0, 1e0, 1e0, 1e0, 1e0])  # SO2, HSO3, SO3, H2SO4, HSO4, SO4, H+    
    dCaq_dt_all = np.zeros(len(Caq_0))
    aq_names = np.array(['SO2', 'HSO3', 'SO3', 'H2SO4', 'HSO4', 'SO4', 'H+'])
    T = 298.15
    result = O2_sulfur_oxidation_rate(Caq_0, dCaq_dt_all, aq_names, T)
    assert isinstance(result, np.ndarray)
    assert result.shape == dCaq_dt_all.shape
    assert np.isclose(result[0]+result[1]+result[2], -1.0*(result[3]+result[4]+result[5]))


def test_IEPOX_OH_chemistry():
    """Test IEPOX OH chemistry rate calculation."""
    Caq_0 = np.array([1e0, 1e0, 1e0, 1e0])  # OHrad, IEPOX, IEPOX_OH_SOA, HO2
    dCaq_dt_all = np.zeros(len(Caq_0))
    aq_names = np.array(['OHrad', 'IEPOX', 'IEPOX_OH_SOA', 'HO2'])
    T = 298.15
    result = IEPOX_OH_chemistry(Caq_0, dCaq_dt_all, aq_names, T)
    assert isinstance(result, np.ndarray)
    assert result.shape == dCaq_dt_all.shape
    assert np.isclose(result[1], -1.0*result[2])
