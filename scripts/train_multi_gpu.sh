#!/usr/bin/env bash

# CLI args with defaults

python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
  trip.runtime.training \
  --amp false \
  --batch_size 32 \
  --epochs 80 \
  --lr 1e-6 \
  --gamma 0.97 \
  --cutoff 5 \
  --weight_decay 1e-3 \
  --use_layer_norm \
  --norm \
  --save_ckpt_path models/frontier_2_400k_3layers.pth \
  --load_ckpt_path models/400k_3layers_fw_10/400k_3layers_fw_10.pth \
  --seed 42 \
  --gradient_clip 10.0 \
  --eval_interval 1 \
  --ckpt_interval 1 \
  --force_weight 0.5 \
  --data_file datasets/processed_flibenak.h5 \
  --dllogger_name frontier_2_default.json \
  --log_dir logs/frontier_2_default \
  --amd \
  --num_layers 3 \
  --num_heads 8 \
  --num_channels 16 \
  --load_weights_only \