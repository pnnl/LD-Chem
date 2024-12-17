#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""
import numpy as np
import constants as c


def ds_fluctuation(s,dt,T,P,r,N,V=0.,C0=3.,accom=0.3):
#     expressions from:
#         Chandrakar, K. K., Grabowski, W. W., Morrison, H., & Bryan, G. H. (2021). 
#         Impact of entrainment mixing and turbulent fluctuations on droplet size 
#         distributions in a cumulus cloud: An investigation using Lagrangian microphysics 
#         with a subgrid-scale model. Journal of the Atmospheric Sciences, 78(9), 2983-3005.
# @article{chandrakar2021impact,
#   title={Impact of entrainment mixing and turbulent fluctuations on droplet size distributions in a cumulus cloud: An investigation using Lagrangian microphysics with a subgrid-scale model},
#   author={Chandrakar, Kamal Kant and Grabowski, Wojciech W and Morrison, Hugh and Bryan, George H},
#   journal={Journal of the Atmospheric Sciences},
#   volume={78},
#   number={9},
#   pages={2983--3005},
#   year={2021}
# }

# more here: https://journals.ametsoc.org/view/journals/atsc/75/10/jas-d-18-0078.1.xml


    if V>0.:
        eu=V*4. # SGS turbulent kinetic energy
        eps=eu/50.
    else:
        eu=4. # SGS turbulent kinetic energy
        eps=0.05 # kinetic energy dissipation rate
    sigma_u_i = sigma_u(eu)
    tau_t_i = tau_t(sigma_u_i,eps,C0=C0)
    tau_c_i = tau_c(r,N,T,P,accom=accom)
    sigma_s_i = sigma_s(sigma_u_i,tau_t_i,T)
    alpha = get_alpha(T)
    ds = - (s/tau_t_i + s/tau_c_i)*dt + alpha*V*dt + np.sqrt(2*sigma_s_i**2*dt/tau_t_i)*np.random.normal()
    return ds

def sigma_u(eu):
    return np.sqrt(2./3. * eu)

def tau_t(sigma_u_i,eps,C0=3.):
    return 2.*sigma_u_i**2./(C0*eps)

def tau_c(r,N,T,P,accom=0.3): 
    return 1./(4.*np.pi*dv(T, r, P, accom)*r*N)

def sigma_s(sigma_w_i,tau_t_i,T):
    alpha = get_alpha(T)
    return alpha * sigma_w_i * tau_t_i / np.sqrt(2)

def get_alpha(T):
    alpha = (c.g * c.Mw * c.L) / (c.Cp * c.R * (T**2))
    alpha -= (c.g * c.Ma) / (c.R * T)
    return alpha

def dv(T, r, P, accom):
    """See :func:`pyrcel.thermo.dv` for full documentation"""
    P_atm = P * 1.01325e-5  # Pa -> atm
    dv_cont = 1e-4 * (0.211 / P_atm) * ((T / 273.0) ** 1.94)
    denom = 1.0 + (dv_cont / (accom * r)) * np.sqrt((2 * np.pi * c.Mw) / (c.R * T))
    return dv_cont / denom
