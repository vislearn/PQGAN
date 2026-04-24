#!/bin/bash

config="./config/latent_diffusion/VQ_GAN_f16/disc_weight=0.8_disc_num_layers=2.yaml"
trainingOutput="./models/VQ_GAN_f16_256_1024_1_disc_weight=0.8_disc_num_layers=2"

# Monitoring
TENSORBOARD_PORT=6777

# Set Hardware
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES="2,3"

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
