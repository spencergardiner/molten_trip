#!/usr/bin/env bash

# CLI args with defaults
BATCH_SIZE=${1:-8}
AMP=${2:-true}
NUM_EPOCHS=${3:-30}
LEARNING_RATE=${4:-2e-3}
WEIGHT_DECAY=${5:-0.1}

python -m trip.runtime.training \
  --amp "$AMP" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$NUM_EPOCHS" \
  --lr "$LEARNING_RATE" \
  --gamma 0.5 \
  --cutoff 4.6 \
  --weight_decay "$WEIGHT_DECAY" \
  --use_layer_norm \
  --norm \
  --save_ckpt_path models/frontier_2_default.pth \
  --seed 42 \
  --gradient_clip 10.0 \
  --eval_interval 1 \
  --force_weight 0.1 \
  --data_file datasets/processed_flibenak.h5 \
  --dllogger_name frontier_2_default.json \
  --log_dir logs/frontier_2_default \
