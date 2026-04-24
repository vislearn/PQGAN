#!/bin/bash

data_config="./config/latent_diffusion/256_1024_64/data_ImageNet.yaml"
model_config="./config/latent_diffusion/256_1024_64/latent_diffusion_model.yaml"
trainer_config="./config/latent_diffusion/256_1024_64/base_trainer_steps.yaml"
trainingOutput="./models/VQ_256_1024_64"

# Monitoring
TENSORBOARD_PORT=6779

# Set GPU
export CUDA_VISIBLE_DEVICES="7"

# Python
source ./venv/bin/activate

# Training
vv fit --config ${data_config} \
       --config ${model_config} \
       --config ${trainer_config} &
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

tensorboard --logdir=${logFile} --port=$TENSORBOARD_PORT > "${tensorboardRunningLog}" 2>&1 &
tensorboard_pid=$!

# Set a trap to kill both processes if this script is interrupted
trap "kill $fit_pid $tensorboard_pid" INT

# Wait for both processes to finish
wait $fit_pid $tensorboard_pid
