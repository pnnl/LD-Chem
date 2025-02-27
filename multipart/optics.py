#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""

import PyMieScatt
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Callable

@dataclass
class CoreShellOptics(particle):
    """CoreShellOptics: defines optical properties for "particle" assuming core-shell morphology """
    
    core_specs: Tuple[str,...]
    shell_specs: Tuple[str,...]
    wavelengths: Tuple[float,...]
    shell_ris: Tuple[complex,...]
    core_ris: Tuple[complex,...]
    abs_crossects: Tuple[float,...]
    scat_crossects: Tuple[float,...]
    core_abs_crossects: Tuple[float,...]
    core_scat_crossects: Tuple[float,...]
    
    def _add_specs(particle,core_specs=['BC']):
        
    def _add_RIs(particle,wavelengths):
        # code to compute 
        