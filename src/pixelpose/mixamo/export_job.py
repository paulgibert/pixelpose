from typing import Dict
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed, 
    retry_if_result,
    RetryError
)
from requests import Session
from yaspin import yaspin
from yaspin.spinners import Spinners
from .utils import download_file, parse_gms_hash


# TODO: Extrracted this shared constant
MIXAMO_API_BASE = 'https://www.mixamo.com/api/v1'

# X Bot with standard skeleton
DEFAULT_CHARACTER_ID = '2dee24f8-3b49-48af-b735-c6377509eaac'


class MixamoExportJob:
    def __init__(
        self,
        product_id: str,
        is_character: bool,
        session: Session,
    ):
        self.product_id = product_id
        self.export_func = (
            self._export_character
            if is_character
            else self._export_animation
        )
        self.session = session


    @classmethod
    def character(cls, character_id: str, session: Session) -> 'MixamoExportJob':
        return cls(character_id, True, session)


    @classmethod
    def animation(cls, animation_id: str, session: Session) -> 'MixamoExportJob':
        return cls(animation_id, False, session)


    def execute(self, output_dir: Path, show_progress: bool = True) -> Path:
        # Does the full export/poll/download loop
        spinner = yaspin(
            text=f"Preparing download for {self.product_id[:8]}...",
            spinner=Spinners.dots
        )
        
        if show_progress:
            spinner.start()
        
        try:
            # Step 1. Export a job
            status_id = self.export_func(self.product_id)

            # Step 2. Wait for the job to complete
            job_result = self._wait_for_job(status_id, self.product_id)

            # Step 3. Download the result
            spinner.stop()
            return download_file(job_result, output_dir, show_progress=show_progress)

        except Exception as e:
            spinner.stop()
            raise Exception(f"Error executing job: {e}") from e


    def _export_character(self, character_id: str) -> str:
        """Request character export from Mixamo API."""
        url = f'{MIXAMO_API_BASE}/animations/export'

        payload = {
            'gms_hash': None,
            'preferences': {
                'format': 'fbx7_2019',
                'mesh': 't-pose',
            },
            'character_id': character_id,
            'type': 'Character',
            'product_name': f'Character_{character_id[:8]}'
        }

        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return character_id # The character ID is used to poll the job status


    def _export_animation(self, animation_id: str) -> str:
        """Request animation export from Mixamo API."""
        # Fetch and process gms_hash
        details = self._fetch_animation_details(animation_id)
        gms_hash = parse_gms_hash(details)

        # Export the job
        url = f'{MIXAMO_API_BASE}/animations/export'

        # Configure the export
        payload = {
            'gms_hash': gms_hash,
            'preferences': {
                'format': 'fbx7_2019',
                'skin': 'false',
                'fps': '30',
                'reducekf': '0'
            },
            'character_id': DEFAULT_CHARACTER_ID,
            'type': 'Motion',
            'product_name': 'MY_PRODUCT'
        }

        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return DEFAULT_CHARACTER_ID # The character ID is used to poll the job status


    def _fetch_animation_details(self, animation_id: str) -> Dict:
        """Fetch animation details"""
        url = f'{MIXAMO_API_BASE}/products/{animation_id}'
        params = {
            'similar': 0,
            'character_id': DEFAULT_CHARACTER_ID
        }
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()
    

    def _wait_for_job(self, status_id: str, product_id: str) -> str:
        """Poll export status until completion. Returns job result URL."""
        @retry(
            stop=stop_after_attempt(15),
            wait=wait_fixed(1),
            retry=retry_if_result(lambda result: result is None),
            reraise=True
        )
        def poll_until_complete():
            """Poll once, return job_result if complete, None if processing."""
            try:
                poll = self._fetch_job_status(status_id)
                status = poll['status']
                
                if status == 'completed':
                    return poll['job_result']  # Success - stop retrying
                
                elif status == 'failed':
                    raise RuntimeError(f"Export job failed for {product_id[:8]}")
                
                elif status == 'processing':
                    return None  # Retry
                
                else:
                    raise ValueError(f"Unknown status: {status}")
                    
            except KeyError as e:
                raise ValueError(f"Invalid response: missing {e}") from e
        
        try:
            return poll_until_complete()
        except RetryError:
            raise TimeoutError(f"Export timed out for {product_id[:8]}")
    

    def _fetch_job_status(self, status_id: str) -> Dict:
        """Poll export job status."""
        # Note: What this method calls 'status_id' is actually just the character ID
        # involved in the job.
        url = f'{MIXAMO_API_BASE}/characters/{status_id}/monitor'
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
