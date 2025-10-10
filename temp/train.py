from pathlib import Path
import os
from Typing import Dict
import yaml
import click
import wandb
from pixelpose.pose_model import TrainerConfig, Trainer


def load_wandb_config(config_path: Path) -> Dict:
    with open(config_path, 'r') as fp:
        return yaml.load(fp)


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


@click.command()
@click.argument('dataset', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--config', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=Path('wandb.yaml'), help="Weigth & Biases training YAML path")
@click.option('--checkpoints', type=click.Path(exists=False, file_okay=False, path_type=Path),
              default= Path('checkpoints'), help="Directory to store checkpoints")
def train(dataset: Path, config: Path, checkpoints: Path):
    wandb_config = load_wandb_config(config)

    with wandb.init(config=wandb_config) as run:
        tr_config = make_trainer_config(run.config, dataset)
        tr = Trainer(tr_config)

        for epoch in range(run.config['epochs']):
            # TODO: Add progress visualization (probably needs to be done inside the trainer class)
            train_loss = tr.train_one_epoch()
            val_loss = tr.evaluate_one_epoch()
            run.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
        
        os.makedirs(checkpoints, exist_okay=True)
        tr.save_checkpoints(f"checkpoint_epoch_{run.config['epochs']}.pt", checkpoints)
