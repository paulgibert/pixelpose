import click
from .dataset import dataset


@click.group()
def cli():
    """PixelPose - Pose-guided character rendering"""
    pass


# Add subcommands
cli.add_command(dataset)
# cli.add_command(train)


if __name__ == '__main__':
    cli()
