import pytest
import numpy as np
from multipart.processes.gas_chemistry import dCgas_dt


def test_dCgas_dt_basic():
    """Test basic gas chemistry rate calculation."""
    # Set up mock gas concentrations
    Cgas_0 = np.array([1e5, 1e5, 1e5])  # mol/m^3 for three species
    gas_names = np.array(['A', 'B', 'C'])

    # Mock reaction: A + B -> C
    reactants_all = np.array(['A B'])
    products_all = np.array(['C'])
    rates = np.array([1e3])  # mol/m^3/s (rate constant)
    T = 298.15  # K
    P = 101325  # Pa
    result = dCgas_dt(Cgas_0, reactants_all, products_all, rates, gas_names, T, P)

    assert isinstance(result, np.ndarray)
    assert result.shape == Cgas_0.shape
    assert np.all(np.isfinite(result))
    assert result[0] < 0  # A consumed
    assert result[1] < 0  # B consumed
    assert result[2] > 0  # C produced
    assert np.isclose(result[0], -1.0*result[2])


def test_dCgas_dt_multiple_reactions():
    """Test gas chemistry with multiple reactions."""
    # Set up mock gas concentrations
    Cgas_0 = np.array([1e-6, 1e-6, 1e-6, 1e-6])  # mol/m^3 for four species
    gas_names = np.array(['A', 'B', 'C', 'D'])

    # Mock reactions: A + B -> C and C -> D
    reactants_all = np.array(['A B', 'C'])
    products_all = np.array(['C', 'D'])
    rates = np.array([1e-4, 1e-5])  # mol/L/s
    T = 298.15  # K
    P = 101325  # Pa
    result = dCgas_dt(Cgas_0, reactants_all, products_all, rates, gas_names, T, P)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == Cgas_0.shape
    assert np.all(np.isfinite(result))
    assert result[0] < 0  # A consumed
    assert result[1] < 0  # B consumed
    assert result[3] > 0  # D produced
    assert np.isclose(result[0], -1.0*result[2])
    assert np.isclose(result[2], -1.0*result[3])
