# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:19:53 2024

@author: Nahin Ferdousi
"""

""" 
Based on constants function from pyrcel: https://github.com/darothen/pyrcel/blob/master/pyrcel/constants.py

Commonly used constants in microphysics and aerosol thermodynamics equations as
well as important model parameters.
================= ============= ========== ==========        ======================
Symbol            Variable      Value      Units             Description
================= ============= ========== ==========        ======================

:math:`\\rho_w`    ``rho_w``     1000.0     kg m**-3          density of water at STP
:math:`R`         ``R``         8.314      J/mol/K           universal gas constant
:math:`M_w`       ``Mw``        0.018      kg/mol            molecular weight of water

================= ============= ========== ==========        ======================

"""

rho_w = 1e3  #: Density of water, kg/m^3
R = 8.314  #: Universal gas constant, J/(mol K)
Mw = 18.02 / 1e3  #: Molecular weight of water, kg/mol
T = 295 #Temperature of air, K
ST_w = 0.072 #Surface tension of water, N/m