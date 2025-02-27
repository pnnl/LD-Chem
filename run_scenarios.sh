#!/bin/bash

working_directory='/rcfs/projects/partikkel/multipart/0425_trajectories_50p'

cd ${working_directory}

for dir in */; do
    dir=${dir%/}
    cd $dir
    sbatch RunScript.sh
    cd ..
done

cd ..
