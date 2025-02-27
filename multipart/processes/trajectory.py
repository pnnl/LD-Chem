#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:05:02 2024

@author: Laura Fierce
"""
from .particles import ParticlePopulation
from dataclasses import dataclass

@dataclass
class ParcelTrajectory:
    """ ParcelTrajectory: definition of aerosol parcel that evolves over time """
    ts: tuple[float, ...]
    Ss: tuple[float, ...]
    Ts: tuple[float, ...]
    Ps: tuple[float, ...]
    particle_populations: tuple[ParticlePopulation, ...]
    
    # other parameters controlling phase partitioning
    # refractive_index: float

@dataclass
class ParcelState:
    S: float
    P: float
    T: float
    # gas_mixture: GasMixture
    particle_population: ParticlePopulation