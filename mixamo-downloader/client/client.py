import json
import os
from pathlib import Path
from typing import Dict, List, Callable
import time

import requests
from tqdm import tqdm
from yaspin import yaspin
from yaspin.spinners import Spinners

# Downloading animations requires specefying a character id even if we don't request the skin
# We use the id of X Bot, which has a standard skeleton as the default character
DEFAULT_CHARACTER_ID = '2dee24f8-3b49-48af-b735-c6377509eaac'

class MixamoClient:
    """
    A client for downloading animations and characters from Mixamo.
    
    This client provides methods to fetch animation/character IDs from Mixamo's API and download
    individual products as FBX files. It handles authentication, export requests,
    job monitoring, and file downloads with progress tracking.
    
    Attributes:
        _session (requests.Session): Authenticated session for API requests
        
    Example:
        >>> client = MixamoClient()
        >>> animation_ids = client.fetch_animation_ids()
        >>> client.fetch_animation_fbx(animation_ids[0], Path('animation.fbx'))
        >>> character_ids = client.fetch_character_ids()
        >>> client.fetch_character_fbx(character_ids[0], Path('character.fbx'))
    """
    
    def __init__(self):
        """
        Initialize the MixamoClient with authentication.
        
        Loads headers and cookies from JSON files and sets up an authenticated
        session using the MIXAMO_TOKEN environment variable.
        
        Raises:
            ValueError: If MIXAMO_TOKEN environment variable is not set
            FileNotFoundError: If headers.json or cookies.json files are missing
        """
        self._session = requests.Session()

        headers = {
            'X-Api-Key': 'mixamo2',
            'Authorization': f'Bearer {_get_auth_token()}'
        }
        self._session.headers.update(headers)
    
    def fetch_animation_ids(self, show_progress: bool = True) -> List[str]:
        """
        Fetch all available animation IDs from Mixamo.
        
        Args:
            show_progress (bool, optional): Whether to show a progress bar during fetching.
                Defaults to True.
        
        Returns:
            List[str]: A list of animation IDs available for download.
            
        Raises:
            ValueError: If the API response is missing required fields
            requests.RequestException: If the API request fails
        """
        return self._fetch_product_ids('Motion,MotionPack', show_progress=show_progress)

    def fetch_character_ids(self, show_progress: bool = True) -> List[str]:
        """
        Fetch all available character IDs from Mixamo.
        
        Args:
            show_progress (bool, optional): Whether to show a progress bar during fetching.
                Defaults to True.
        
        Returns:
            List[str]: A list of character IDs available for download.
            
        Raises:
            ValueError: If the API response is missing required fields
            requests.RequestException: If the API request fails
        """
        return self._fetch_product_ids('Character', show_progress=show_progress)

    def fetch_animation_fbx(self, id: str, output_path: Path,
                            show_progress: bool = True,
                            progress_label: str = None) -> None:
        """
        Download an animation as an FBX file.
        
        Downloads a specific animation from Mixamo by its ID and saves it as an FBX file.
        The process involves exporting the animation, monitoring the export job, and then
        downloading the resulting file. Progress is shown with spinners and progress bars.
        
        Args:
            id (str): The animation ID to download
            output_path (Path): Path where the FBX file will be saved
            show_progress (bool, optional): Whether to show progress indicators.
                Defaults to True.
            progress_label (str, optional): Custom label for the download progress bar.
                Defaults to None.
                
        Raises:
            ValueError: If the animation ID is invalid or export fails
            RuntimeError: If the animation export job fails
            requests.RequestException: If network requests fail
        """
        return self._fetch_product_fbx(id, DEFAULT_CHARACTER_ID, self._export_animation,
                                       output_path, show_progress=show_progress,
                                       progress_label=progress_label)
    
    def fetch_character_fbx(self, id: str, output_path: Path,
                            show_progress: bool = True,
                            progress_label: str = None) -> None:
        """
        Download a character as an FBX file.
        
        Downloads a specific character from Mixamo by its ID and saves it as an FBX file.
        The process involves exporting the character, monitoring the export job, and then
        downloading the resulting file. Progress is shown with spinners and progress bars.
        
        Args:
            id (str): The character ID to download
            output_path (Path): Path where the FBX file will be saved
            show_progress (bool, optional): Whether to show progress indicators.
                Defaults to True.
            progress_label (str, optional): Custom label for the download progress bar.
                Defaults to None.
                
        Raises:
            ValueError: If the character ID is invalid or export fails
            RuntimeError: If the character export job fails
            requests.RequestException: If network requests fail
        """
        return self._fetch_product_fbx(id, id, self._export_character,
                                       output_path, show_progress=show_progress,
                                       progress_label=progress_label)

    def _fetch_product_ids(self, type: str, show_progress: bool = True) -> List[str]:
        """Fetch product IDs from Mixamo API with pagination."""
        url = f'https://www.mixamo.com/api/v1/products'
        params = {
            'page': 1,
            'limit': 96,
            'type': type
        }

        pbar = tqdm(total=0,
                    disable=not show_progress,
                    desc="Fetching asset IDs")

        ids = []
        while True:
            # Fetch a new page
            response = self._session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Validate the response
            if 'results' not in data:
                raise ValueError("Response is missing 'results'")
            
            if 'pagination' not in data:
                raise ValueError("Response is missing 'pagination'")
            
            if 'num_results' not in data['pagination']:
                raise ValueError("Response pagination is missing 'num_results'")

            # Store the new animation ids
            results = data['results']

            if len(results) == 0:
                break

            for r in results:
                if 'id' not in r:
                    raise ValueError("Response animation is missing 'id'")
                ids.append(r['id'])

            # Update the progress bar
            pbar.total = data['pagination']['num_results']
            pbar.update(len(results))
            pbar.refresh()

            # Increment page number
            params['page'] += 1
        
        return ids

    def _fetch_product_fbx(self, product_id: str, status_id: str, export_func: Callable[[str], None],
                           output_path: Path, show_progress: bool = True, progress_label: str = None) -> None:
        """Download a product (animation or character) as FBX file."""
        spinner = yaspin(text=f"Queueing {product_id}", spinner=Spinners.dots)
        if show_progress:
            spinner.start()
        
        # 1. Create the export job
        try:
            export_func(product_id)
        except RuntimeError as e: # Only observed error is when the gms_hash does not exist. This is not a common enough occurence to debug yet
            spinner.stop()
            spinner.write(f"❌ Error: {e}")
            return

        # 2. Monitor the job until completion or failure
        while True:
            poll = self._check_export_status(status_id)

            if 'status' not in poll:
                raise ValueError("Request missing 'poll'")
            

            if poll['status'] == 'completed':
                # 3. If the job completes successfully, download the file
                spinner.stop()
                if 'job_result' not in poll:
                    raise ValueError("Response missing 'job_result'")
                
                _download_file(poll['job_result'], output_path,
                                show_progress=show_progress,
                                progress_label=progress_label)
                spinner.write(f"✅ {product_id}")
                return

            elif poll['status'] == 'failed':
                spinner.stop()
                spinner.write("❌ FAILED!")
                raise RuntimeError(f"Failed to fetch product {product_id}")

            elif poll['status'] == 'processing':
                time.sleep(1) # Wait before polling again

            else:
                spinner.stop()
                spinner.write("❌ FAILED!")
                raise ValueError(f"Unrecognized job status: {poll['status']}")

    def _fetch_animation_details(self, id: str) -> Dict:
        """Fetch detailed information about an animation including gms_hash."""
        url = f'https://www.mixamo.com/api/v1/products/{id}'
        params = {
            'similiar': 0,
            'character_id': DEFAULT_CHARACTER_ID # Use a dummy character id
        }
        response = self._session.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def _fetch_animation_gms_hash(self, id: str) -> Dict:
        """Fetch and process the gms_hash for an animation export."""
        # Fetch animation details which includes gms_hash
        details = self._fetch_animation_details(id)
        
        # Validate response
        if 'details' not in details:
            raise ValueError("Response missing 'details'")
        
        if 'gms_hash' not in details['details']:
            raise ValueError("Response details missing 'gms_hash'")
        
        gms_hash = details['details']['gms_hash']

        # Convert the params field to a comma separated list of values
        if 'params' not in gms_hash:
            raise ValueError("Response gms_hash is missing 'params'")
        gms_hash['params'] = ','.join([str(param[1]) for param in gms_hash['params']])

        # Add an overdrive field if needed
        if 'overdrive' not in gms_hash:
            gms_hash['overdrive'] = 0
        
        return gms_hash

    def _export_animation(self, id: str) -> Dict:
        """Request export of an animation from Mixamo."""
        # Get thge gms hash for this animation (not sure what gms stands for but we need it)
        gms_hash = self._fetch_animation_gms_hash(id)
        url = f'https://www.mixamo.com/api/v1/animations/export'

        # Configure the export
        payload = {
            'gms_hash': [gms_hash],
            'preferences': {
                'format': 'fbx7_2019',  # Downloads as .fbx
                'skin': 'false',        # We don't want the mesh, just the animation
                'fps': '30',
                'reducekf': '0'         # Disable keyframe reduction
            },
            'character_id': DEFAULT_CHARACTER_ID, # Use a dummy character id
            'type': 'Motion',
            'product_name': 'MY_PRODUCT' # Just a filler name
        }

        # Perform the POST
        response = self._session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def _export_character(self, id: str) -> Dict:
        """Request export of a character from Mixamo."""
        url = f'https://www.mixamo.com/api/v1/animations/export'

        # Configure the export
        payload = {
            'gms_hash': None,
            'preferences': {
                'format': 'fbx7_2019',              # Downloads as .fbx
                'mesh': 't-pose',                   # Character mesh in t-pose
            },
            'character_id': id,                     # Use the actual character id
            'type': 'Character',
            'product_name': f'Character_{id[:8]}'   # Use character id as name
        }

        # Perform the POST
        response = self._session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def _check_export_status(self, character_id: str) -> Dict:
        """Check the status of an export job."""
        url = f'https://www.mixamo.com/api/v1/characters/{character_id}/monitor'
        response = self._session.get(url)
        response.raise_for_status()
        return response.json()


def _load_local_json(filename: Path) -> Dict:
    """Load a JSON file from the same directory as this file."""
    current_dir = Path(__file__).parent.resolve()
    json_path = current_dir / filename
    with open(json_path, 'r') as f:
        return json.load(f)
    
def _get_auth_token() -> str:
    """Get MIXAMO_TOKEN from the environment."""
    token = os.environ.get('MIXAMO_TOKEN', None)
    if token is None:
        raise RuntimeError("MIXAMO_TOKEN is not defined")
    return token

def _download_file(file_url: str, output_path: Path, show_progress: bool=True,
                   progress_label: str=None):
    """Downloads a file from a url to output_path."""
    # Perform the GET
    response = requests.get(file_url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    # Download in chunks
    with open(output_path, 'wb') as fp:
        with tqdm(total=total_size, unit='B', unit_scale=True,
                  leave=False, desc=progress_label,
                  disable=not show_progress) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                fp.write(chunk)
                pbar.update(len(chunk))
