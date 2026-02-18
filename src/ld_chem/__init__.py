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
        # create_scenario_from_DNS,
        # create_parcel_scenario,
        # create_hysplit_scenario,
        create_les_scenario,
        # create_pichamber_scenario,
        make_AqReactions,
        make_GasReactions,
    )
    from .systems import (
        ParcelState,
        Processes,
        # update_state,
    )
    from .particles import (
        ParticlePopulation,
        # make_particle,
    )
    # from .optics import (
    #     mie_calculation,
    # )
    
    __all__ = [
        # 'create_scenario_from_DNS',
        # 'create_parcel_scenario',
        # 'create_hysplit_scenario',
        'create_les_scenario',
        # 'create_pichamber_scenario',
        'make_AqReactions',
        'make_GasReactions',
        # 'ParcelState',
        # 'Processes',
        # 'update_state',
        # 'ParticlePopulation',
        # 'make_particle',
        # 'mie_calculation',
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import all multipart components: {e}")
