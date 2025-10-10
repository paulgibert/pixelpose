from pathlib import Path
from typing import Optional
import time
import os
import click
from pixelpose.mixamo import MixamoClient, MissingTokenError


@click.group()
def dataset():
    """Dataset related commands"""
    pass


# ============================================================================
# Download Command
# ============================================================================

@dataset.command()
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False, path_type=Path),
    default=Path('assets'),
    help="Downloaded assets output directory"
)
@click.option(
    '--character-id',
    type=str,
    help="Downloads assets for only a specific character ID"
)
@click.option(
    '--small',
    is_flag=True,
    help="Downloads a small subset of the assets"
)
def download(
    output_dir: Path,
    character_id: Optional[str],
    small: bool
):
    """Download character and animation assets from Mixamo"""
    try:
        client = MixamoClient()

        os.makedirs(output_dir, exist_ok=True)

        # Download characters
        _download_characters(client, output_dir, character_id, small)

        click.echo() # Print a space between each set of outputs
        
        # Download animations
        _download_animations(client, output_dir, small)

    except MissingTokenError:
        raise click.ClickException("Missing MIXAMO_TOKEN env variable. Did you set it?")


def _download_characters(
    client: MixamoClient,
    output_dir: Path,
    character_id: Optional[str],
    small: bool
):
    """Helper function to download character assets"""
    # Determine which characters to download
    try:
        if character_id is None:
            character_ids = client.fetch_character_ids(show_progress=True)
            if small:
                character_ids = character_ids[:10]
        else:
            character_ids = [character_id]

        click.echo(f"✅ Fetched character asset IDs")
    except Exception as e:
        raise click.ClickException(f"❌ Failed to fetch character asset IDs") from e

    total = len(character_ids)

    # Create output directory
    character_dir = output_dir / 'characters'
    os.makedirs(character_dir, exist_ok=True)

    # Download each character
    for i, char_id in enumerate(character_ids, start=1):
        try:
            download_path = client.download_character_fbx(
                char_id,
                character_dir,
                show_progress=True
            )
            click.echo(f"[{i}/{total}] ✅ - {download_path.name}")
        except Exception as e:
            click.echo(f"[{i}/{total}] ❌ - {char_id} - ERROR: {e}")


def _download_animations(
    client: MixamoClient,
    output_dir: Path,
    small: bool
):
    """Helper function to download animation assets"""
    # Fetch animation IDs
    try:
        animation_ids = client.fetch_animation_ids(show_progress=True)
        if small:
            animation_ids = animation_ids[:10]

        click.echo(f"✅ Gathered animation asset IDs")
    
    except Exception as e:
        raise click.ClickException(f"❌ Failed to fetch animation asset IDs") from e

    total = len(animation_ids)

    # Create output directory
    animation_dir = output_dir / 'animations'
    os.makedirs(animation_dir, exist_ok=True)

    # Download each animation
    for i, anim_id in enumerate(animation_ids, start=1):
        try:
            download_path = client.download_animation_fbx(
                anim_id,
                animation_dir,
                show_progress=True
            )
            click.echo(f"[{i}/{total}] ✅ - {download_path.name}")
            time.sleep(1) # Delay to avoid rate limits
        
        except Exception as e:
            click.echo(f"[{i}/{total}] ❌ - {anim_id} - {e}")


# ============================================================================
# Render Command
# ============================================================================

@dataset.command()
@click.argument(
    'assets_dir',
    type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False, path_type=Path),
    default=Path('rendered'),
    help="Directory to save rendered frames"
)
def render(assets_dir: Path, output_dir: Path):
    """Render character animations to PNG frames"""
    click.echo("Not implemented yet")
    # TODO: Implement rendering pipeline


# ============================================================================
# Preprocess Command
# ============================================================================

# @dataset.command()
# @click.argument(
#     'assets_dir',
#     type=click.Path(exists=True, file_okay=False, path_type=Path)
# )
# @click.option(
#     '--output-dir',
#     type=click.Path(file_okay=False, path_type=Path),
#     default=Path('preprocessed_dataset'),
#     help="Directory to save the final preprocessed dataset"
# )
# @click.option(
#     '--samples-per-file',
#     type=int,
#     default=1000,
#     help="The number of samples to store in each dataset .pt file"
# )
# def preprocess(
#     assets_dir: Path,
#     output_dir: Path,
#     samples_per_file: int
# ):
#     """Preprocess rendered assets into training-ready format"""
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Load dataset and compile
#     ds = AssetsDataset(assets_dir)
    
#     click.echo(f"Compiling dataset to {output_dir}")
#     compile_dataset(
#         ds,
#         output_dir,
#         samples_per_file=samples_per_file,
#         show_progress=True
#     )
