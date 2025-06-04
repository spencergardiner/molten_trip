#!/usr/bin/env bash

# CLI args
BATCH_SIZE=${1:-12}
AMP=${2:-true}
NUM_EPOCHS=${3:-10}
LEARNING_RATE=${4:-1e-3}
WEIGHT_DECAY=${5:-0.75}
 
python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
 trip.runtime.training \
 --amp "$AMP" \
 --batch_size "$BATCH_SIZE" \
 --epochs "$NUM_EPOCHS" \
 --lr "$LEARNING_RATE" \
 --gamma 0.5 \
 --cutoff 5 \
 --weight_decay "$WEIGHT_DECAY" \
 --use_layer_norm \
 --norm \
 --save_ckpt_path results/frontier_model_default.pth \
 --seed 42 \
 --num_workers 4 \
 --gradient_clip 10.0 \
 --eval_interval 1 \
 --force_weight 0.5 \
 --ckpt_interval 1 \
 --data_file results/processed_flibenak.h5 \
 --dllogger_name frontier_model_default.json \
 
  