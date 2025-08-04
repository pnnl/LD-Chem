#!/bin/bash

#SBATCH -A partikkel
#SBATCH -p slurm
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -J T0
#SBATCH -o run.out
#SBATCH -e run.err
#SBATCH --cpus-per-task=4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source /share/apps/python/anaconda3.6/etc/profile.d/conda.sh
conda activate multipart
module load python/3.7.2

python3 MAIN_multipart.py

