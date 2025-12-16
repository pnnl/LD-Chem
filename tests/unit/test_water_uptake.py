import pytest
import numpy as np
from multipart.processes.water_uptake import (
    sigma_w, ka, dv, es, Seq, dr_dt, dlnr_dt
)


def test_sigma_w():
    """Test surface tension of water calculation."""
    T = 298.15  # 25°C
    sigma = sigma_w(T)
    assert isinstance(sigma, (float, np.floating))
    assert sigma > 0
    assert sigma < 1  # Surface tension in N/m

    # Should decrease with temperature
    T_higher = 308.15  # 35°C
    sigma_higher = sigma_w(T_higher)
    assert sigma_higher < sigma


def test_ka():
    """Test thermal conductivity calculation."""
    T = 298.15
    r = 1e-6  # 1 micron radius
    rho = 1.2  # kg/m^3 air density

    k = ka(T, r, rho)
    assert isinstance(k, (float, np.floating))
    assert k > 0

    # Should be close to continuum value for large particles
    r_large = 1e-3  # 1 mm
    k_large = ka(T, r_large, rho)
    k_cont = 1e-3 * (4.39 + 0.071 * T)
    assert np.isclose(k_large, k_cont, rtol=1e-2)


def test_dv():
    """Test water vapor diffusivity calculation."""
    T = 298.15
    r = 1e-6  # 1 micron
    P = 101325  # 1 atm
    accom = 1.0  # Accommodation coefficient

    D = dv(T, r, P, accom)
    assert isinstance(D, (float, np.floating))
    assert D > 0

    # Should be close to continuum value for large particles
    r_large = 1e-3  # 1 mm
    D_large = dv(T, r_large, P, accom)
    D_cont = 1e-4 * (0.211 / (P * 1.01325e-5)) * ((T / 273.0) ** 1.94)
    assert np.isclose(D_large, D_cont, rtol=1e-2)


def test_es():
    """Test saturation vapor pressure calculation."""
    T = 298.15  # 25°C
    P_sat = es(T)
    assert isinstance(P_sat, (float, np.floating))
    assert P_sat > 0


def test_Seq():
    """Test equilibrium saturation ratio calculation."""
    kappa_low = 0.1  # hygroscopicity
    kappa_high = 0.65
    S_eq = Seq(1e-6, 5e-7, 298.15, kappa_high)
    S_eq_kappa0 = Seq(1e-6, 5e-7, 298.15, kappa_low)
    assert S_eq_kappa0 > S_eq

    # Seq should be higher for smaller particles
    D_low = 1e-7
    D_high = 5e-7
    S_eq = Seq(1.1*D_high, D_high, 298.15, kappa_high)
    S_eq_smaller = Seq(1.1*D_low, D_low, 298.15, kappa_high)
    assert S_eq_smaller > S_eq


def test_dr_dt():
    """Test droplet radius growth rate."""
    r_i = 1e-6  # initial radius
    r_dry_i = 5e-7  # dry radius
    kappa_i = 0.1  # hygroscopicity
    P = 101325  # pressure
    T = 298.15  # temperature
    S = 1.01  # slight supersaturation
    accom = 1.0  # accommodation coefficient

    growth_rate = dr_dt(r_i, r_dry_i, kappa_i, P, T, S, accom)
    assert isinstance(growth_rate, (float, np.floating))

    # Should be positive for supersaturated conditions
    assert growth_rate > 0

    # Should be negative for no humidity
    S_sub = 0.0
    growth_rate_sub = dr_dt(r_i, r_dry_i, kappa_i, P, T, S_sub, accom)
    assert growth_rate_sub < 0


def test_dlnr_dt():
    """Test logarithmic radius growth rate."""
    lnr_i = np.log(1e-6)  # log of initial radius
    r_dry_i = 5e-7  # dry radius
    kappa_i = 0.1  # hygroscopicity
    P = 101325  # pressure
    T = 298.15  # temperature
    S = 1.01  # slight supersaturation
    accom = 1.0  # accommodation coefficient

    growth_rate = dlnr_dt(lnr_i, r_dry_i, kappa_i, P, T, S, accom)
    assert isinstance(growth_rate, (float, np.floating))

    # Should be positive for supersaturated conditions
    assert growth_rate > 0

    # Should be negative for subsaturated conditions
    S_sub = 0.01
    growth_rate_sub = dlnr_dt(lnr_i, r_dry_i, kappa_i, P, T, S_sub, accom)
    assert growth_rate_sub < 0
