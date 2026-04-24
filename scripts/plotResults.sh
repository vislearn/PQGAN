#!/bin/bash

# Set Hardware
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES="3"

destination_path="/path/to/additional_evaluations/codebook_util"
#destination_path="/path/to/cluster/output"
#destination_path="/path/to/tmp"

# Training
vv plot --destination_path ${destination_path}