from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List
import os
import requests
from tqdm import tqdm


def download_file(file_url: str, output_dir: Path, show_progress: bool = True) -> Path:
    """Download file from URL to output path."""
    response = requests.get(file_url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    # Get the file name from the url
    parsed = urlparse(file_url)
    filename = Path(parsed.path).name

    with open(output_dir / filename, 'wb') as fp:
        with tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            leave=False,
            disable=not show_progress
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                fp.write(chunk)
                pbar.update(len(chunk))

    return output_dir / filename


class MissingTokenError(Exception):
    pass


def get_auth_token() -> str:
    """Get MIXAMO_TOKEN from environment."""
    token = os.environ.get('MIXAMO_TOKEN', None)
    if token is None:
        raise MissingTokenError("MIXAMO_TOKEN is not defined")
    return token


def parse_gms_hash(animation_details: Dict) -> List:
    """Extract and process gms_hash from animation details."""
    try:
        gms_hash = animation_details['details']['gms_hash']

        # Convert params to comma-separated string
        if 'params' not in gms_hash:
            raise ValueError("Response gms_hash is missing 'params'")
        gms_hash['params'] = ','.join([str(param[1]) for param in gms_hash['params']])

        # Add overdrive field if missing
        if 'overdrive' not in gms_hash:
            gms_hash['overdrive'] = 0
        
        return [gms_hash]
    
    except KeyError as e:
        raise Exception(f"Error in animation details format: {e}") from e
