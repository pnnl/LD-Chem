import pytest
import numpy as np
from multipart.processes.air_thermo import es, S_to_wv, H2O_gas_conc, H2O_mole_fraction, compute_thermo_props


def test_es():
    """Test saturation vapor pressure calculation."""
    # Test at 0°C (273.15 K)
    T_c = 0.0
    expected_es = 611.2  # Pa
    assert np.isclose(es(T_c), expected_es, rtol=1e-3)

    # Test at 25°C (298.15 K)
    T_c = 25.0
    expected_es = 3168.0  # Approximate value
    assert np.isclose(es(T_c), expected_es, rtol=1e-2)


def test_S_to_wv():
    """Test conversion from saturation ratio to water vapor mixing ratio."""
    S = 1.0  # Saturated
    T = 298.15  # 25°C
    P = 101325  # 1 atm
    wv = S_to_wv(S, T, P)
    assert wv > 0
    assert wv < 1  # Mixing ratio should be less than 1

    # Test with S=0
    wv_dry = S_to_wv(0.0, T, P)
    assert wv_dry == 0.0


def test_H2O_gas_conc():
    """Test water vapor gas concentration calculation."""
    S = 1.0
    T = 298.15
    P = 101325
    conc = H2O_gas_conc(S, T, P)
    assert conc > 0
    # Should be approximately P_H2O / (R*T)
    Psat = es(T - 273.15)
    expected = (S * Psat) / (8.314 * T)
    assert np.isclose(conc, expected, rtol=1e-3)


def test_H2O_mole_fraction():
    """Test water vapor mole fraction calculation."""
    S = 1.0
    T = 298.15
    P = 101325
    mole_frac = H2O_mole_fraction(S, T, P)
    assert mole_frac > 0
    assert mole_frac < 1

    # At saturation, mole fraction should be Psat/P
    Psat = es(T - 273.15)
    expected = Psat / P
    assert np.isclose(mole_frac, expected, rtol=1e-3)


def test_compute_thermo_props():
    """Test computation of thermodynamic properties."""
    T = 298.15
    P = 101325
    S = 1.0

    pv_sat, rho_air, rho_air_dry = compute_thermo_props(T, P, S)

    assert pv_sat > 0
    assert rho_air > 0
    assert rho_air_dry > 0
    assert rho_air_dry < rho_air  # Dry air density should be less than moist air

    # Test with S=0 (dry air)
    pv_sat_dry, rho_air_dry_only, rho_air_dry_dry = compute_thermo_props(T, P, 0.0)
    assert rho_air_dry_only == rho_air_dry_dry
