#!/bin/bash

#SBATCH -A partikkel
#SBATCH -p slurm
#SBATCH -t 80:00:00
#SBATCH -N 1
#SBATCH -J multipart
#SBATCH -o multipart_250p.out
#SBATCH -e multipart_250p.err

#source /share/apps/python/anaconda3.6/etc/profile.d/conda.sh
#conda activate multipart
#module load python/3.7.2

working_directory=wdir2
total_particles=100
particles_per_run=10
output_directory=output_run2


# =====================================================
# get rid of this before doing the actual runs
rm -r $working_directory
rm -r $output_directory
# =====================================================


if [ -d "$working_directory" ] || [ -d "$output_directory" ]; then
    echo "$working_directory or $output_directory already exists. Make sure all data is saved properly and delete, or rename the working directory above."
else
    mkdir $working_directory
    mkdir $output_directory
    cp multipart/SPLAT_initialization.py $working_directory
    cp multipart/aerosol_species.py $working_directory
    cp multipart/constants.py $working_directory
    cp multipart/driver.py $working_directory
    cp multipart/MAIN_les.py $working_directory
    cp multipart/parcel.py $working_directory
    cp multipart/particles.py $working_directory
    cp -r multipart/processes $working_directory
    cp multipart/Reactions.py $working_directory
    cp multipart/scenario.py $working_directory
    cp multipart/systems.py $working_directory
    cp multipart/TraceGases.py $working_directory
    cp multipart/utilities.py $working_directory
    cp multipart/visualization.py $working_directory
    
    # get the initial aerosol properties and make the sub-directories
    cd $working_directory
    python3 SPLAT_initialization.py ${total_particles} ${particles_per_run}
    echo ""
    
    # run the model for each sub-directory
    for dir in */; do
        dir=${dir%/}
        cd $dir
            python3 MAIN_les.py ${output_directory} ${dir}
        cd ..
    done
    
fi







#cd multipart
#python3 MAIN_les.py
