from pathlib import Path
import os
from typing import List
import logging
import requests
from tqdm import tqdm
from .export_job import MixamoExportJob
from .utils import get_auth_token


logger = logging.getLogger(__name__)


MIXAMO_API_BASE = 'https://www.mixamo.com/api/v1'


class MixamoClient:
    """Client for downloading animations and characters from Mixamo API."""
    def __init__(self):
        """Initialize authenticated session using MIXAMO_TOKEN env var."""
        self._session = requests.Session()

        headers = {
            'X-Api-Key': 'mixamo2',
            'Authorization': f'Bearer {get_auth_token()}'
        }
        self._session.headers.update(headers)

    def fetch_character_ids(self, show_progress: bool = True) -> List[str]:
        """Fetch all available character IDs from Mixamo."""
        return self._fetch_product_ids('Character', show_progress=show_progress)


    def fetch_animation_ids(self, show_progress: bool = True) -> List[str]:
        """Fetch all available animation IDs from Mixamo."""
        return self._fetch_product_ids('Motion,MotionPack', show_progress=show_progress)


    def download_character_fbx(
        self,
        character_id: str,
        output_dir: Path,
        show_progress: bool = True
    ) -> Path:
        """Download character as FBX file."""
        os.makedirs(output_dir, exist_ok=True)
        job = MixamoExportJob.character(character_id, self._session)
        return job.execute(output_dir, show_progress=show_progress)


    def download_animation_fbx(
        self,
        animation_id: str,
        output_dir: Path,
        show_progress: bool = True
    ) -> Path:
        """Download animation as FBX file."""
        os.makedirs(output_dir, exist_ok=True)
        job = MixamoExportJob.animation(animation_id, self._session)
        return job.execute(output_dir, show_progress=show_progress)


    def _fetch_product_ids(self, type: str, show_progress: bool = True) -> List[str]:
        """Fetch product IDs from Mixamo API with pagination."""
        url = f'{MIXAMO_API_BASE}/products'
        params = {'page': 1, 'limit': 96, 'type': type}

        ids = []
        with tqdm(
            total=0,
            desc="Feteching asset IDs",
            disable=not show_progress,
            leave=False
        ) as pbar:
            while True:
                try:
                    response = self._session.get(url, params=params)
                    response.raise_for_status()

                    data = response.json()
                    results = data['results']
                    total = data['pagination']['num_results']

                    if len(results) == 0:
                        break
                    
                    ids.extend(r['id'] for r in results)

                    # Update the progress bar
                    pbar.total = total
                    pbar.update(len(results))
                    pbar.refresh()
                    params['page'] += 1
                
                except KeyError as e:
                    # Log response format failures as warnings
                    logger.warning(f"Failed to fetch page {params['page']}: {e}")
                    params['page'] += 1
                continue

        return ids
