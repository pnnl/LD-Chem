#!/bin/bash

working_directory='/rcfs/projects/partikkel/multipart/0425_15utc_single_activation'

cd ${working_directory}

#for dir in */; do
for i in {750..849}; do
    dir=trajectory_${i}
    #dir=${dir%/}
    cd $dir
    pwd
    sbatch RunScript.sh
    sleep 2
    cd ..
done

cd ..
