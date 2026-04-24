#!/bin/bash

data_config="./config/experimental/base_data.yaml"
model_config="./config/experimental/base_model_autoencoder_res.yaml"
trainer_config="./config/experimental/base_trainer.yaml"
trainingOutput="./models/TEST"

# Training
vv fit --config ${data_config} \
       --config ${model_config} \
       --config ${trainer_config} &
fit_pid=$!
trap "kill $fit_pid" INT



sleep 10



# Monitor training
highest_version=0

for file in "$trainingOutput"/version_*; do
    version=$(basename "$file" | cut -d'_' -f2)
    if [[ "$version" -gt "$highest_version" ]]; then
        highest_version="$version"
    fi
done

logFile="${trainingOutput}/version_${highest_version}"

tensorboard --logdir=${logFile} > /dev/null 2>&1 &
tensorboard_pid=$!

# Set a trap to kill both processes if this script is interrupted
trap "kill $fit_pid $tensorboard_pid" INT

# Wait for both processes to finish
wait $fit_pid $tensorboard_pid
