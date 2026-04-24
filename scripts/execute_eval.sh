#!/bin/bash

# Set Hardware
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES="0"

source_path="/path/to/models"
#source_path="/path/to/tmp/copy_models"
destination_path="/path/to/additional_evaluations/inference_speed"
#destination_path="/path/to/tmp"
eval_configuration="./config/eval/eval_config_speed.yaml"
#eval_configuration="./config/eval/eval_config_testing.yaml"

# Training
vv eval --source_path ${source_path} \
        --destination_path ${destination_path} \
        --eval_config ${eval_configuration} \
        --overwrite_possible "False"