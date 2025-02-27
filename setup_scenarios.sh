#!/bin/bash

#source /share/apps/python/anaconda3.6/etc/profile.d/conda.sh
#conda activate multipart
#module load python/3.7.2

num_trajectories=1
particles_per_trajectory=5
output_directory='/Users/beel083/Library/CloudStorage/OneDrive-PNNL/Desktop/multipart_archived-main/TEST'
LES_path='/Users/beel083/Library/CloudStorage/OneDrive-PNNL/Desktop/multipart_archived-main/datasets/parcel_traces_0425_15utc'
model_path='/Users/beel083/Library/CloudStorage/OneDrive-PNNL/Desktop/multipart_archived-main/multipart'
partition_name='shared'
time_limit='03:00:00'

# =====================================================
# get rid of this before doing the actual runs
# rm -r $output_directory
# =====================================================


if [ -d "$output_directory" ]; then
    echo "$output_directory already exists. Make sure all data is saved properly and delete, or rename the working directory above."
else
    mkdir $output_directory
    cd $output_directory

    for (( traj=0; traj<$num_trajectories; traj++ ))
    do
        echo 'trajectory' $traj
        mkdir trajectory_${traj}
        cd trajectory_${traj}
        
        # make the run script
        outname=trajectory_${traj}.out
        errname=trajectory_${traj}.err
        jobname=trajectory_${traj}
        cwd=$(pwd)
        setupfile=${model_path}/SPLAT_initialization.py
        driverfile=${model_path}/MAIN_les.py
        restartfile=${model_path}/RESTART_les.py
        echo '#!/bin/bash' >> RunScript.sh
        echo '' >> RunScript.sh
        #echo '#SBATCH -A partikkel' >> RunScript.sh
        #echo '#SBATCH -p' $partition_name>> RunScript.sh
        #echo '#SBATCH -t' $time_limit >> RunScript.sh
        #echo '#SBATCH -N 1' >> RunScript.sh
        #echo '#SBATCH -J' $jobname >> RunScript.sh
        #echo '#SBATCH -o' $outname >> RunScript.sh
        #echo '#SBATCH -e' $errname >> RunScript.sh
        #echo '' >> RunScript.sh
        #echo 'source /share/apps/python/anaconda3.6/etc/profile.d/conda.sh' >> RunScript.sh
        echo 'conda activate multipart' >> RunScript.sh
        #echo 'module load python/3.7.2' >> RunScript.sh
        echo '' >> RunScript.sh
        echo 'python3' ${setupfile} ${particles_per_trajectory} ${LES_path} ${cwd} ${traj} >> RunScript.sh
        echo 'python3' ${driverfile} ${LES_path} ${cwd} >> RunScript.sh
        chmod +x RunScript.sh
        
        
        #echo '#!/bin/bash' >> RestartScript.sh
        #echo '' >> RestartScript.sh
        #echo '#SBATCH -A partikkel' >> RestartScript.sh
        #echo '#SBATCH -p' $partition_name >> RestartScript.sh
        #echo '#SBATCH -t' $time_limit >> RestartScript.sh
        #echo '#SBATCH -N 1' >> RestartScript.sh
        #echo '#SBATCH -J' $jobname >> RestartScript.sh
        #echo '#SBATCH -o' $outname >> RestartScript.sh
        #echo '#SBATCH -e' $errname >> RestartScript.sh
        #echo '' >> RestartScript.sh
        #echo 'source /share/apps/python/anaconda3.6/etc/profile.d/conda.sh' >> RestartScript.sh
        #echo 'conda activate multipart' >> RestartScript.sh
        #echo 'module load python/3.7.2' >> RestartScript.sh
        #echo '' >> RestartScript.sh
        #echo 'directory="${1:-.}"' >> RestartScript.sh
        #echo 'restart_file=$(find "$directory" -type f -name "*RESTART.pkl" -print -quit)' >> RestartScript.sh
        #echo 'trajectory_file=$(find "$directory" -type f -name "*.pkl" -print -quit)' >> RestartScript.sh
        #echo '' >> RestartScript.sh
        #echo 'python3' ${restartfile} ${cwd} '${restart_file} ${trajectory_file}' >> RestartScript.sh
        #chmod +x RestartScript.sh
        cd ..
    done
    

fi

