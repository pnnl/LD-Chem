import pytest
import numpy as np
import ld_chem.constants as c

def test_gravitational_constant():
    """Test gravitational constant."""
    assert isinstance(c.g, (int, float))
    assert np.isclose(c.g, 9.81, rtol=1e-2)


def test_specific_heats():
    """Test specific heat constants."""
    assert isinstance(c.Cp, (int, float))
    assert np.isclose(c.Cp, 1004.0, rtol=1e-3)

    assert isinstance(c.Cpv, (int, float))
    assert np.isclose(c.Cpv, 1952.0, rtol=1e-3)

    assert isinstance(c.Cl, (int, float))
    assert np.isclose(c.Cl, 4179.0, rtol=1e-3)


def test_latent_heat():
    """Test latent heat of condensation."""
    assert isinstance(c.L, (int, float))
    assert np.isclose(c.L, 2.25e6, rtol=1e-3)


def test_water_density():
    """Test water density."""
    assert isinstance(c.rho_w, (int, float))
    assert np.isclose(c.rho_w, 1000.0, rtol=1e-3)


def test_gas_constants():
    """Test gas constants."""
    assert isinstance(c.R, (int, float))
    assert np.isclose(c.R, 8.314, rtol=1e-3)

    assert isinstance(c.Mw, (int, float))
    assert np.isclose(c.Mw, 0.018, rtol=1e-3)

    assert isinstance(c.Ma, (int, float))
    assert np.isclose(c.Ma, 0.0289, rtol=1e-3)


def test_derived_gas_constants():
    """Test derived gas constants."""
    # Rd = R / Ma
    expected_Rd = c.R / c.Ma
    assert np.isclose(c.Rd, expected_Rd, rtol=1e-10)

    # Rv = R / Mw
    expected_Rv = c.R / c.Mw
    assert np.isclose(c.Rv, expected_Rv, rtol=1e-10)


def test_transport_properties():
    """Test transport property constants."""
    assert isinstance(c.Dv, (int, float))
    assert np.isclose(c.Dv, 3.0e-5, rtol=1e-3)

    assert isinstance(c.Ka, (int, float))
    assert np.isclose(c.Ka, 0.02, rtol=1e-3)


def test_accommodation_coefficients():
    """Test accommodation coefficients."""
    assert isinstance(c.ac, (int, float))
    assert np.isclose(c.ac, 1.0, rtol=1e-10)

    assert isinstance(c.at, (int, float))
    assert np.isclose(c.at, 0.96, rtol=1e-3)

    assert isinstance(c.accom, (int, float))
    assert np.isclose(c.accom, 1.0, rtol=1e-10)


def test_molecular_weight_ratio():
    """Test molecular weight ratio."""
    assert isinstance(c.epsilon, (int, float))
    assert np.isclose(c.epsilon, 0.622, rtol=1e-3)


def test_fundamental_constants():
    """Test fundamental physical constants."""
    assert isinstance(c.e, (int, float))
    assert np.isclose(c.e, 1.602176634e-19, rtol=1e-10)

    assert isinstance(c.kb, (int, float))
    assert np.isclose(c.kb, 1.380649e-23, rtol=1e-10)

    assert isinstance(c.Na, (int, float))
    assert np.isclose(c.Na, 6.022e23, rtol=1e-3)


def test_model_parameters():
    """Test model-specific parameters."""
    assert isinstance(c.N_STATE_VARS, int)
    assert c.N_STATE_VARS == 5
