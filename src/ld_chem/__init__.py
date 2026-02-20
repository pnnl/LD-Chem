#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LD-Chem: Lagrangian Droplets with Chemistry Model

A comprehensive model for simulating aerosol-cloud interactions and 
atmospheric microphysics processes.

@author: fier887
"""

__version__ = "0.1.0"
__author__ = "Laura Fierce"
__email__ = "laura.fierce@pnnl.gov"
# Import main components for easy access
try:
    from .scenario import (
        create_les_scenario,
        create_parcel_scenario,
        make_AqReactions,
        make_GasReactions,
    )
    from .systems import (
        ParcelState,
        Processes,
    )
    from .particles import (
        ParticlePopulation
    )
    
    __all__ = [
        'create_les_scenario',
        'create_parcel_scenario',
        'make_AqReactions',
        'make_GasReactions',
        'ParcelState',
        'Processes',
        'ParticlePopulation',
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import all multipart components: {e}")
