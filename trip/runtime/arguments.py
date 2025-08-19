# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import argparse
import pathlib

from se3_transformer.runtime.utils import str2bool

from trip.data_loading import TrIPDataModule
from trip.model import TrIP


PARSER = argparse.ArgumentParser(description='TrIP')

paths = PARSER.add_argument_group('Paths')
paths.add_argument('--log_dir', type=pathlib.Path, default=pathlib.Path('/results'),
                   help='Directory where the results logs should be saved')
paths.add_argument('--dllogger_name', type=str, default='dllogger_results.json',
                   help='Name for the resulting DLLogger JSON file')
paths.add_argument('--save_ckpt_path', type=pathlib.Path, default=None,
                   help='File where the checkpoint should be saved')
paths.add_argument('--load_ckpt_path', type=pathlib.Path, default=None,
                   help='File of the checkpoint to be loaded')

PARSER.add_argument('--load_weights_only', type=str2bool, nargs='?', const=True, default=False,
                    help='Load only the weights from the checkpoint, ignore scheduler and optimizer states.')

PARSER.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
PARSER.add_argument('--batch_size', type=int, default=240, help='Batch size')
PARSER.add_argument('--seed', type=int, default=None, help='Set a seed globally')
PARSER.add_argument('--num_workers', type=int, default=8, help='Number of dataloading workers')
PARSER.add_argument('--add_atoms', type=str2bool, nargs='?', const=True, default=True,
                    help='Add lone atoms during training')

PARSER.add_argument('--amp', type=str2bool, nargs='?', const=True, default=False, help='Use Automatic Mixed Precision')
PARSER.add_argument('--gradient_clip', type=float, default=None, help='Clipping of the gradient norms')
PARSER.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation')
PARSER.add_argument('--ckpt_interval', type=int, default=-1, help='Save a checkpoint every N epochs')
PARSER.add_argument('--eval_interval', dest='eval_interval', type=int, default=20,
                    help='Do an evaluation round every N epochs')
PARSER.add_argument('--silent', type=str2bool, nargs='?', const=True, default=False,
                    help='Minimize stdout output')
PARSER.add_argument('--wandb', type=str2bool, nargs='?', const=True, default=False,
                    help='Enable W&B logging')
PARSER.add_argument('--benchmark', type=str2bool, nargs='?', const=True, default=False,
                    help='Benchmark mode')
PARSER.add_argument('--amd', type=str2bool, nargs='?', const=True, default=False,
                    help="Run with AMD GPU's (also turns off amp). NOTE: must manually change dgl.copy_edge() --> dgl.copy_e() for updated DGL versions.")

PARSER.add_argument('--r2_lr', type=float, default=1e-5,
                    help='Initial learning rate for round 2 loss')
PARSER.add_argument('--r2_gamma', type=float, default=0.97,
                    help='Learning rate decay factor for round 2 loss')
PARSER.add_argument('--r2_fw', type=float, default=0.1,
                    help='scaling factor for the force component in the round 2 loss')
PARSER.add_argument('--r2_epoch_start', type=int, default=-1,
                    help='Epoch to start the round 2 loss (default: -1, which means to not initiate round 2 training). \
                        NOTE: "_r2" will be appended to the model name if this is set to a value >= 0.')
PARSER.add_argument('--r2_base_model_ckpt_path', default=None, type=pathlib.Path, nargs='?',
                    const=None,
                    help='Path to the base model checkpoint to be used for round 2 training. \
                        If not set, the model will be trained from scratch. \
                        NOTE: this should be a model trained with --r2_epoch_start=-1.')

TrIP.add_argparse_args(PARSER)
TrIPDataModule.add_argparse_args(PARSER)
