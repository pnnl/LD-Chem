#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Laura Fierce and Payton Beeler
"""
import numpy as np
from driver import restart_les_trajectories
import matplotlib.pyplot as plt
import pickle, sys

output_path=str(sys.argv[1])
ParcelState_file=str(sys.argv[2])[2:]
trajectory_file=str(sys.argv[3])[2:]

trajectory = restart_les_trajectories(output_path=output_path, ParcelState_file=ParcelState_file, trajectory_file=trajectory_file)
    

