import argparse
import datetime
import os
import platform
import random
import sys
import time

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary, LearningRateMonitor
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from utils.prepare import experiment_from_args

random.seed(1)
torch.manual_seed(1)
torch.set_float32_matmul_precision('high')


def define_args(parent_parser):
    parser = parent_parser.add_argument_group('train.py')
    parser.add_argument('--wandb',
                        help='log to wandb',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--tensorboard',
                        help='log to tensorboard',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--fp16',
                        help='use 16 bit precision',
                        type=bool,
                        default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--name',
                        help='experiment name',
                        type=str,
                        default=None)
    return parent_parser


def main():
    data_module, model, args = experiment_from_args(sys.argv, add_argparse_args_fn=define_args)

    plugins = []

    run_name = args.name
    if run_name is None:
        run_name = f'{time.strftime("%Y-%m-%d_%H:%M:%S")}-{platform.node()}'
    print('Run name:', run_name)

    loggers = []
    if args.tensorboard:
        loggers.append(TensorBoardLogger(save_dir='logs/', name=run_name))
    if args.wandb:
        loggers.append(WandbLogger(project='elastic_glimpse', entity="ideas_cv", name=run_name))

    callbacks = [
        ModelCheckpoint(dirpath=f"checkpoints/{run_name}", monitor="val/loss"),
        RichProgressBar(leave=True, theme=RichProgressBarTheme(metrics_format='.2e')),
        RichModelSummary(max_depth=3),
        LearningRateMonitor(logging_interval='epoch')
    ]

    if 'SLURM_NTASKS' in os.environ:
        num_nodes = int(os.environ['SLURM_NNODES'])
        devices = int(os.environ['SLURM_NTASKS'])
        if num_nodes * devices > 1:
            strategy = DDPStrategy(find_unused_parameters=True, timeout=datetime.timedelta(seconds=3600))
        else:
            strategy = 'auto'
        print(f'Running on slurm, {num_nodes} nodes, {devices} gpus')
    else:
        strategy = 'auto'
        num_nodes = 1
        devices = 'auto'

    if not args.fp16:
        precision = None
    elif torch.cuda.is_bf16_supported():
        precision = 'bf16-mixed'
    else:
        precision = '16-mixed'

    trainer = Trainer(plugins=plugins,
                      max_epochs=args.epochs,
                      accelerator='gpu',
                      logger=loggers,
                      callbacks=callbacks,
                      enable_model_summary=False,
                      strategy=strategy,
                      num_nodes=num_nodes,
                      devices=devices,
                      precision=precision,
                      benchmark=True,
                      check_val_every_n_epoch=2
                      )

    trainer.fit(model=model, datamodule=data_module)
    if data_module.has_test_data:
        trainer.test(ckpt_path='best', datamodule=data_module)


if __name__ == "__main__":
    main()
