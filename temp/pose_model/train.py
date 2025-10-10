from html import parser
from typing import Dict
import argparse
from pathlib import Path
import yaml
import wandb
from trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgmentParser()
    parser.add_argument('dataset_dir', help='Directory containing the preprocessed dataset')
    return parser.parse_args()


def make_trainer_config(run_config: Dict, ds_path: Path) -> TrainerConfig:
    return TrainerConfig(
        # From wandb_config
        lr                      = run_config.get('lr'),
        batch_size              = run_config.get('batch_size'),
        total_steps             = run_config.get('total_steps'),
        denoiser_depth          = run_config.get('denoiser_depth'),
        unet_kernels            = run_config.get('unet_kernels'),
        unet_strides            = run_config.get('unet_strides'),
        ref_kernels             = run_config.get('ref_kerenels'),
        ref_strides             = run_config.get('ref_strides'),

        # From script args
        preprocess_dataset_dir  = ds_path
    )


def main():
    args = parse_args()

    with open('pose_model/wandb_config.yaml') as fp:
        wandb_config = yaml.load(fp)

    with wandb.init(config=wandb_config) as run:
        tr_config = make_trainer_config(run.config, args.dataset_dir)
        tr = Trainer(tr_config)

        for epoch in range(run.config['epochs']):
            train_loss = tr.train_one_epoch()
            val_loss = tr.evaluate_one_epoch()
            run.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
