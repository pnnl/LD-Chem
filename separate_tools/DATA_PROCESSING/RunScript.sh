#!/bin/bash

source /share/apps/python/miniforge25.3.0/etc/profile.d/conda.sh
conda activate multipart
module load python/3.11.13

python3 Processing_OneDir.py
