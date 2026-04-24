#!/bin/bash

config="./config/latent_diffusion/F16K512Z32S8_cosine/config.yaml"
trainingOutput="./models/F16K512Z32S8_cosine"

# Monitoring
TENSORBOARD_PORT=6778

# Set Hardware
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES="3"

# Python
source ./venv/bin/activate

# Training
vv fit --config ${config} &
fit_pid=$!
trap "kill $fit_pid" INT



sleep 60



# Monitor training
highest_version=0
version=0

for file in "$trainingOutput"/version_*; do
    version=$(basename "$file" | cut -d'_' -f2)
    if [[ "$version" -gt "$highest_version" ]]; then
        highest_version="$version"
    fi
done

logFile="${trainingOutput}/version_${highest_version}"
tensorboardRunningLog="${logFile}/tensorboard.log"

echo "Using TensorBoard port: $TENSORBOARD_PORT"
echo "Using log file: $logFile"

tensorboard --logdir=${logFile} --port=$TENSORBOARD_PORT --samples_per_plugin images=1000 > "${tensorboardRunningLog}" 2>&1 &
tensorboard_pid=$!

# Set a trap to kill both processes if this script is interrupted
trap "kill $fit_pid $tensorboard_pid" INT

# Wait for both processes to finish
wait $fit_pid $tensorboard_pid
