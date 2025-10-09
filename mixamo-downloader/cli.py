import argparse
import os
from pathlib import Path
import time
from .client.client import MixamoClient


def parse_args() -> argparse.Namespace:
    """Parse cli arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=['animations', 'characters'],
                        help="the asset type to download")
    parser.add_argument('--output_dir', '-o', default='assets', help="asset download directory (default: ./mixamo/)")
    parser.add_argument('--limit', type=int, default=4096, help="max number of assets to download (default: 4096)")
    parser.add_argument('--count_only', action='store_true', help="count the number of assets that this command would download without performing any downloads")
    return parser.parse_args()


def main():
    """Cli entrypoint"""
    # Parse arguments
    args = parse_args()

    # Create the mixamo client
    client = MixamoClient()

    # Get IDs
    if args.type == 'animations':
        ids = client.fetch_animation_ids(show_progress=True)
    else: # args.type == 'characters'
        ids = client.fetch_character_ids(show_progress=True)
    ids = ids[:args.limit]

    if args.count_only:
        # Exit early if user only wants to count
        print(f"Counted {len(ids)} assets for this command")
        return
        
    # Setup the output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # download each asset to [output_dir]/[id].fbx
    for i, id in enumerate(ids):
        filepath = output_dir / f'{id}.fbx'
        progress_label = f"Downloading {i+1}/{len(ids)}"

        if args.type == 'animations':
            client.fetch_animation_fbx(id, filepath, show_progress=True,
                                        progress_label=progress_label)
        else: # args.type == 'characters'
            client.fetch_character_fbx(id, filepath, show_progress=True,
                                        progress_label=progress_label)
        
        time.sleep(1) # Avoids rate limits
