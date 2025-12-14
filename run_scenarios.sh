#!/bin/bash

working_directory='/rcfs/projects/partikkel/multipart/0425_100p_tau24'

cd ${working_directory}

for (( traj=0; traj<50; traj++ ))
do
    cd trajectory_${traj}
    pwd
    sbatch RunScript_CU.sh
    cd ..
done

cd ..
