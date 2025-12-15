#!/bin/bash

BASE_DIR="/rcfs/projects/partikkel/multipart/0425_100p_tau24"

matches=$(find "$BASE_DIR" -type f -name '*000028.pkl')

if [ -n "$matches" ]; then
    echo "Found the following files:"
    echo "$matches"
else
    echo "No files ending with '000028.pkl' found."
fi


#PATTERN="^python3 ../../multipart/SPLAT_initialization.py 100 ../../datasets/parcel_traces_0425_15utc"

#for (( traj=0; traj<1000; traj++ ))
#do
#    # Path to the target file
#    FILE="0425_100p_tau24/trajectory_${traj}/RunScript_CU.sh"
#    # Check if the file exists
#    if [[ -f "$FILE" ]]; then
#        # Use sed to replace "#SBATCH -p shared" with "#SBATCH -p TEST"
#        sed -i.bak "/${PATTERN//\//\\/}/d" "$FILE"
#        echo "Line deleted in in $FILE"
#    else
#        echo "Error: File not found at $FILE"
#        exit 1
#    fi
#done
