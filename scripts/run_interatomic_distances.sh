#!/usr/bin/env bash

# extract first element of ROCR_VISIBLE_DEVICES (which is comma-separated)
DEVICE=$(echo $ROCR_VISIBLE_DEVICES | cut -d',' -f1)

MODEL_NAME="big_model_gamma_75_16_epochs"

python -m analysis.generate_interatomic_distance_plots \
--checkpoint_path "/mnt/trip/models/${MODEL_NAME}.pth" \
--species "Li" "F" \
--device "cuda:0" \
--output_dir "/mnt/trip/analysis/interatomic_distance_plots/${MODEL_NAME}" \
