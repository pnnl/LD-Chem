#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Laura Fierce
"""

import numpy as np
import numba as nb
from numba.pycc import CC
import sys
from scipy.optimize import fminbound


from dataclasses import dataclass
from dataclasses import replace

from typing import Tuple
from typing import Callable

from copy import copy

'''
# should this be in here?
@dataclass
class ParcelState:
    # t: float
    
    x: float
    y: float
    z: float
    
    u: float
    v: float
    w: float
    
    S: float
    P: float
    T: float
    
    # gas_mixture: GasMixture
    particle_population: ParticlePopulation
    
    
@dataclass
class ProcessControls:
    """AerosolProcesses: a definition of a set of processes under consideration"""
    # processes that can be enabled/disabled
    condensation: bool = True # condensation/evaporation of water vapor
    collisions: bool = False # particle collisions
    cocondensation: bool = False # co-condensation of organic gases
    chemistry: bool = False # gas and aqueous chemistry
    freezing: bool = False # homog freezing and growth
    settling: bool = False # removal through graviational settling
    fluctuations: bool = False # sub-grid turbulent fluctuations

@dataclass
class SimulationSettings:
    dt: float = 1. # seconds --> note, not yet set up for process sub-stepping (but could be)
    solver: str = 'BDF' # good for stiff problems. must be one of the solvers in scipy's solve_ivp (for now)
    accom: float = 0.3 # accomodation coefficient
    C0: float = 3. # used in SGS supersaturation fluctuation param
    sigma: float = 1.0 # for now, each particle is a delta function. Could also be a distribution.

    
@dataclass
class Feedbacks:
    dwc_dt: float = 0.
    dwc_dt_next: float = 0.
    dwc: float = 0.
    dwi_dt: float = 0.

@dataclass
class ParcelTrajectory:
    """ ParcelTrajectory: definition of air parcel that evolves over time """
    ts: Tuple[float, ...]
    parcel_states: Tuple[ParcelState, ...]
    
# def get_next_state_and_feedbacks(ParcelState_0, processes, settings):
# get next state (and feedbacks) after a set of coupled processes
# need to loop through all of the sub-stepped proceses separately
'''
