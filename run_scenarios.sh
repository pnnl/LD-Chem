#!/bin/bash

working_directory='test'

cd ${working_directory}

for dir in */; do
    dir=${dir%/}
    cd $dir
    pwd
    echo ''
    ./RunScript_LES.sh
    cd ..
done

cd ..
