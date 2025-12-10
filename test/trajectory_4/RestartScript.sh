#!/bin/bash

#SBATCH -A partikkel
#SBATCH -p shared
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -J CU_4
#SBATCH -o /dev/null
#SBATCH -e trajectory_4.err
#SBATCH --cpus-per-task=6

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source /share/apps/python/anaconda3.6/etc/profile.d/conda.sh
conda activate multipart
module load python/3.7.2

directory="${1:-.}"
restart_file=$(find "$directory" -type f -name "*RESTART.pkl" -print -quit)
trajectory_file=$(find "$directory" -type f -name "*.pkl" ! -name "*RESTART.pkl" -print -quit)

python3 ../../multipart/RESTART_les.py /Users/beel083/Library/CloudStorage/OneDrive-PNNL/Desktop/codes/multipart_archived/test/trajectory_4 ${restart_file} ${trajectory_file}
