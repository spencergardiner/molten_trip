#!/usr/bin/env bash

# CLI args with defaults

python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
  trip.runtime.training \
  --amp false \
  --batch_size 8 \
  --epochs 200 \
  --lr 1e-5 \
  --gamma 0.97 \
  --cutoff 5 \
  --weight_decay 1e-3 \
  --use_layer_norm \
  --norm \
  --save_ckpt_path models/200k_2layers_fw0p1_batch8/200k_2layers_fw0p1_batch8.pth \
  --load_ckpt_path models/200k_2layers_fw10_batch8/200k_2layers_fw10_batch8.pth \
  --seed 42 \
  --gradient_clip 10.0 \
  --eval_interval 1 \
  --ckpt_interval 1 \
  --force_weight 0.1 \
  --data_file datasets/processed_flibenak_it121.h5 \
  --dllogger_name frontier_2_default.json \
  --log_dir logs/frontier_2_default \
  --amd \
  --num_layers 2 \
  --num_heads 8 \
  --num_channels 16 \
  --load_weights_only \
  --r2_lr 1e-4 \
  --r2_gamma 0.97 \
  --r2_fw 0.1 \
  --r2_epoch_start 100 \