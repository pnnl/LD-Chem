#!/bin/bash


python3 ../../multipart/SPLAT_initialization.py 100 ../../datasets/parcel_traces_0425_15utc /Users/beel083/Library/CloudStorage/OneDrive-PNNL/Desktop/codes/multipart_archived/test/trajectory_7 928
python3 ../../multipart/MAIN_les.py ../../datasets/parcel_traces_0425_15utc_constant_updraft /Users/beel083/Library/CloudStorage/OneDrive-PNNL/Desktop/codes/multipart_archived/test/trajectory_7
