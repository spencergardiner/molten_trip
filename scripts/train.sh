#!/usr/bin/env bash

# CLI args
BATCH_SIZE=${1:-8}
AMP=${2:-true}
NUM_EPOCHS=${3:-10}
LEARNING_RATE=${4:-1e-3}
WEIGHT_DECAY=${5:-0.1}
 
python -m trip.runtime.training \
 --amp "$AMP" \
 --batch_size "$BATCH_SIZE" \
 --epochs "$NUM_EPOCHS" \
 --lr "$LEARNING_RATE" \
 --gamma 0.5 \
 --cutoff 4.6\
 --weight_decay "$WEIGHT_DECAY" \
 --use_layer_norm \
 --norm \
 --save_ckpt_path results/model_trip2_4_6_cutoff_flibenak.pth \
 --seed 42 \
 --num_workers 4 \
 --gradient_clip 10.0 \
 --wandb \
 --eval_interval 1 \
 --force_weight 0.1 \
 --ckpt_interval 1 \
 --data_file results/processed_flibenak.h5 \
 --log_dir results/logs \
 --dllogger_name 4_6.json\
 --device_rank 1 \
 
  