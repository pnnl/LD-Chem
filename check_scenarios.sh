#!/bin/bash

working_directory='/rcfs/projects/partikkel/multipart/0425_15utc_single_activation'
still_running=0
completed=0
restarted=0

cd ${working_directory}

echo ''
total_trajectories=$(find */ -type d | wc -l)
echo 'checking' $total_trajectories 'runs stored in' $(pwd)

for i in {750..999}
#for dir in */; do
do
    dir=trajectory_${i}

    # find the name of the status file and get the flags
    cd $dir
    pwd
    directory="${1:-.}"
    status_filename=$(find "$directory" -type f -name "*STATUS" -print -quit)
    restart_filename=$(find "$directory" -type f -name "*RESTART.pkl" -print -quit)
    error_filename=$(find "$directory" -type f -name "*.err" -print -quit)
    output_filename=$(find "$directory" -type f -name "*.out" -print -quit)
        
    if test -s "$error_filename"; then
        
        if [[ -f "${status_filename}" && -f "${restart_filename}" ]]; then
            echo ""
            echo $dir "Error during running"
            #sbatch RunScript.sh
            #sleep 2
            echo ""
            restarted=$((restarted+1))
        else
            echo ""
            echo $dir "Error during initialization"
            #sbatch RestartScript.sh
            #sleep 2
            echo ""
            restarted=$((restarted+1))
        fi

    # these ones are still actively running
    elif [ -f "${status_filename}" ]; then
        if grep -q "in progress" "$status_filename"; then
            still_running=$((still_running+1))
            echo ""
            echo $dir 'running'
            #sbatch RestartScript.sh
            #sleep 2
            echo ""
        
        # these ones finished
        elif grep -q "complete" "$status_filename"; then
            completed=$((completed+1))
            trajectory_filename=$(find "$directory" -type f -name "*.pkl" -print -quit)
            cp $trajectory_filename ..
        fi
    
    fi
    cd ..
done


if [[ "$completed" == "$total_trajectories" ]]; then
    echo''
    echo 'all trajectories finished running...'
    echo 'cleaning up working directory...'
    #for dir in */; do
    #    echo $dir
    #    rm -r $dir
    #done
    echo''
else

    echo ''
    echo $still_running 'trajectories are still running'
    echo $restarted 'trajectories were restarted'
    echo $completed 'trajectories finished running'
    echo ''
fi
