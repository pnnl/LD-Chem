#!/bin/bash

num_trajectories=1000
particles_per_trajectory=100
output_directory='0425_100p_tau24'
LES_path='../../datasets/parcel_traces_0425_15utc'
CU_path='../../datasets/parcel_traces_0425_15utc_constant_updraft'
model_path='../../multipart'
partition_name='shared'
time_limit='24:00:00'

# =====================================================
# get rid of this before doing the actual runs
# rm -r $output_directory
# =====================================================

# Make an array of 0..999
random_numbers=()
for ((i=0; i<1000; i++)); do
  random_numbers[i]=$i
done

# Fisher–Yates shuffle (in-place)
for ((i=999; i>0; i--)); do
  j=$((RANDOM % (i + 1)))
  tmp=${random_numbers[i]}
  random_numbers[i]=${random_numbers[j]}
  random_numbers[j]=$tmp
done

if [ -d "$output_directory" ]; then
    echo "One of the output directories already exists. Make sure all data is saved properly and delete, or rename the working directory above."
else
    mkdir $output_directory

    cd $output_directory
    for (( traj=0; traj<$num_trajectories; traj++ ))
    do
        echo 'trajectory' $traj
        mkdir trajectory_${traj}
        cd trajectory_${traj}
        
        # make the run script (regular LES)
        errname=trajectory_${traj}.err
        jobname=LES_${traj}
        cwd=$(pwd)
        setupfile=${model_path}/SPLAT_initialization.py
        driverfile=${model_path}/MAIN_les.py
        restartfile=${model_path}/RESTART_les.py
        echo '#!/bin/bash' >> RunScript_LES.sh
        echo '' >> RunScript_LES.sh
        echo '#SBATCH -A partikkel' >> RunScript_LES.sh
        echo '#SBATCH -p' $partition_name>> RunScript_LES.sh
        echo '#SBATCH -t' $time_limit >> RunScript_LES.sh
        echo '#SBATCH -N 1' >> RunScript_LES.sh
        echo '#SBATCH -J' $jobname >> RunScript_LES.sh
        echo '#SBATCH -o /dev/null' >> RunScript_LES.sh
        echo '#SBATCH -e' $errname >> RunScript_LES.sh
        echo '#SBATCH --cpus-per-task=1' >> RunScript_LES.sh
        echo '' >> RunScript_LES.sh
        echo 'export OMP_NUM_THREADS=1' >> RunScript_LES.sh
        echo 'export MKL_NUM_THREADS=1' >> RunScript_LES.sh
        echo 'export OPENBLAS_NUM_THREADS=1' >> RunScript_LES.sh
        echo '' >> RunScript_LES.sh
        echo 'source /share/apps/python/miniforge25.3.0/etc/profile.d/conda.sh' >> RunScript_LES.sh
        echo 'conda activate multipart' >> RunScript_LES.sh
        echo 'module load python/3.11.13' >> RunScript_LES.sh
        echo '' >> RunScript_LES.sh
        random_number=${random_numbers[$traj]}
        echo 'python3' ${setupfile} ${particles_per_trajectory} ${LES_path} ${cwd} ${random_number} >> RunScript_LES.sh
        echo 'python3' ${driverfile} ${LES_path} ${cwd} >> RunScript_LES.sh
        chmod +x RunScript_LES.sh


        # make the run script (constant updraft)
        errname=trajectory_${traj}.err
        jobname=CU_${traj}
        cwd=$(pwd)
        setupfile=${model_path}/SPLAT_initialization.py
        driverfile=${model_path}/MAIN_les.py
        restartfile=${model_path}/RESTART_les.py
        echo '#!/bin/bash' >> RunScript_CU.sh
        echo '' >> RunScript_CU.sh
        echo '#SBATCH -A partikkel' >> RunScript_CU.sh
        echo '#SBATCH -p' $partition_name>> RunScript_CU.sh
        echo '#SBATCH -t' $time_limit >> RunScript_CU.sh
        echo '#SBATCH -N 1' >> RunScript_CU.sh
        echo '#SBATCH -J' $jobname >> RunScript_CU.sh
        echo '#SBATCH -o /dev/null' >> RunScript_CU.sh
        echo '#SBATCH -e' $errname >> RunScript_CU.sh
        echo '#SBATCH --cpus-per-task=1' >> RunScript_CU.sh
        echo '' >> RunScript_CU.sh
        echo 'export OMP_NUM_THREADS=1' >> RunScript_CU.sh
        echo 'export MKL_NUM_THREADS=1' >> RunScript_CU.sh
        echo 'export OPENBLAS_NUM_THREADS=1' >> RunScript_CU.sh
        echo '' >> RunScript_CU.sh
        echo 'source /share/apps/python/miniforge25.3.0/etc/profile.d/conda.sh' >> RunScript_CU.sh
        echo 'conda activate multipart' >> RunScript_CU.sh
        echo 'module load python/3.11.13' >> RunScript_CU.sh
        echo '' >> RunScript_CU.sh
        random_number=${random_numbers[$traj]}
        echo 'python3' ${setupfile} ${particles_per_trajectory} ${LES_path} ${cwd} ${random_number} >> RunScript_CU.sh
        echo 'python3' ${driverfile} ${CU_path} ${cwd} >> RunScript_CU.sh
        chmod +x RunScript_CU.sh
        
        
        jobname=RS_${traj}
        echo '#!/bin/bash' >> RestartScript.sh
        echo '' >> RestartScript.sh
        echo '#SBATCH -A partikkel' >> RestartScript.sh
        echo '#SBATCH -p' $partition_name >> RestartScript.sh
        echo '#SBATCH -t' $time_limit >> RestartScript.sh
        echo '#SBATCH -N 1' >> RestartScript.sh
        echo '#SBATCH -J' $jobname >> RestartScript.sh
        echo '#SBATCH -o /dev/null' >> RestartScript.sh
        echo '#SBATCH -e' $errname >> RestartScript.sh
        echo '#SBATCH --cpus-per-task=1' >> RestartScript.sh
        echo '' >> RestartScript.sh
        echo 'export OMP_NUM_THREADS=1' >> RestartScript.sh
        echo 'export MKL_NUM_THREADS=1' >> RestartScript.sh
        echo 'export OPENBLAS_NUM_THREADS=1' >> RestartScript.sh
        echo '' >> RestartScript.sh
        echo 'source /share/apps/python/miniforge25.3.0/etc/profile.d/conda.sh' >> RestartScript.sh
        echo 'conda activate multipart' >> RestartScript.sh
        echo 'module load python/3.11.13' >> RestartScript.sh
        echo '' >> RestartScript.sh
        echo 'directory="${1:-.}"' >> RestartScript.sh
        echo 'restart_file=$(find "$directory" -type f -name "*RESTART.pkl" -print -quit)' >> RestartScript.sh
        echo 'trajectory_file=$(find "$directory" -type f -name "*.pkl" ! -name "*RESTART.pkl" -print -quit)' >> RestartScript.sh
        echo '' >> RestartScript.sh
        echo 'python3' ${restartfile} ${cwd} '${restart_file} ${trajectory_file}' >> RestartScript.sh
        chmod +x RestartScript.sh
        cd ..
    done

fi

