#!/bin/bash

working_directory=/rcfs/projects/partikkel/multipart/0425_100p_tau24
still_running=0
completed=0
restarted=0

#cd ${working_directory}
#echo ''
#total_trajectories=$(find */ -type d | wc -l)
#echo 'checking' $total_trajectories 'runs stored in' $(pwd)

#for dir in */; do
for (( traj=0; traj<100; traj++ ))
do
    # find the name of the status file and get the flags
    dir="${working_directory}/trajectory_${traj}"
    cd $dir
    pwd

    directory="${1:-.}"
    status_filename=$(find "$directory" -type f -name "*STATUS" -print -quit)
    error_filename=$(find "$directory" -type f -name "*.err" -print -quit)

    if grep -q "in progress" "$status_filename"; then
        
        # these ones got cancelled due to time limit
        if grep -q "CANCELLED" "$error_filename"; then
            echo ''
            echo 'CANCELED' $dir
            #sbatch RestartScript.sh
            echo ''
            restarted=$((restarted+1))
        
        # these ones had an I/O error after initialization
        elif grep -q "Remote I/O error" "$error_filename"; then
            echo ''
            echo 'I/O' $dir
            #sbatch RestartScript.sh
            echo ''
            restarted=$((restarted+1))
        
        # these ones are still actively running
        else
            still_running=$((still_running+1))
            echo ''
            echo 'RUNNING' $dir
            #sbatch RunScript_LES.sh
            echo ''
        fi
    
    # these had an I/O error during initialization
    elif grep -q "Remote I/O error" "$error_filename"; then
        echo ''
        echo 'I/O' $dir
        #sbatch RunScript.sh
        echo ''
        restarted=$((restarted+1))
    
    # these ones finished
    elif grep -q "complete" "$status_filename"; then
        completed=$((completed+1))
        trajectory_filename=$(find "$directory" -type f -name "*.pkl" ! -name "*RESTART.pkl" -print -quit)
        cp $trajectory_filename ..
    
    fi

done


if [[ "$completed" == "$total_trajectories" ]]; then
    echo''
    echo 'all trajectories finished running...'
    echo 'cleaning up working directory...'
    #for dir in */; do
        #echo $dir
        #rm -r $dir
    #done
    echo''
else
    echo ''
    echo $still_running 'trajectories are still running'
    echo $restarted 'trajectories were restarted'
    echo $completed 'trajectories finished running'
    echo ''
fi

