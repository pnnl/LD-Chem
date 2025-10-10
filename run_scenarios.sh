#!/bin/bash

working_directory='/Users/beel083/Library/CloudStorage/OneDrive-PNNL/Desktop/multipart_archived-main/entrainment_tests'

cd ${working_directory}

for dir in */; do
    dir=${dir%/}
    cd $dir
    pwd
    echo ''
    ./RunScript.sh
    cd ..
done

cd ..
