import os
from pathlib import Path
from typing import Iterator, List
import subprocess
from tqdm import tqdm
from PIL import Image
from .model import RenderJob


BLENDER_EXE = 'blender'
RENDER_SCRIPT_PATH = 'renderer/scripts/render.py'
COUNT_FRAMES_SCRIPT_PATH = 'renderer/scripts/count_frames.py'


def render_job_iter(source_dir: Path, target_dir: Path,
                    output_dir: Path, resolution: int,
                    fps: int, pixel_size: int) -> Iterator[RenderJob]:
    # Check that source and target directories exist
    if not source_dir.exists():
        raise FileNotFoundError(f"{source_dir} does not exist")
    
    if not target_dir.exists():
        raise FileNotFoundError(f"{target_dir} does not exist")

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Yield a job for every taget and source FBX file combo
    for target_path in target_dir.glob('*.fbx'):
        for source_path in source_dir.glob('*.fbx'):
            render_dir = output_dir / target_path.stem / source_path.stem
            yield RenderJob(source_path, target_path, render_dir,
                            resolution, fps, pixel_size)


def count_job_frames(jobs: List[RenderJob], show_progress: bool = True) -> int:
    total_frames = 0
    for j in tqdm(jobs, disable=not show_progress):
        frames = call_count_frames_script(j.source_path, j.fps)
        total_frames += frames

def call_render_script(job: RenderJob) -> Path:
    cmd = [
        BLENDER_EXE,
        '--background',
        '--python', RENDER_SCRIPT_PATH,
        '--',
        str(job.source_path),
        str(job.target_path),
        str(job.output_dir),
        '--resolution', str(job.resolution),
        '--fps', str(job.fps)
    ]
    
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    return job.output_dir / 'frames'


def call_count_frames_script(job: RenderJob) -> int:
    cmd = [
        BLENDER_EXE,
        '--background',
        '--python', COUNT_FRAMES_SCRIPT_PATH,
        '--',
        str(job.source_path),
        '--fps', str(job.fps)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(result.stderr.strip()) # The result is printed to stderr


def pixelize_frames(frames_dir: Path, output_dir: Path, pixel_size) -> Path:
    pixels_dir = output_dir / 'pixels'
    os.makedirs(pixels_dir, exist_ok=True)
    for frame_path in frames_dir.glob('*.png'):
        frame = Image.open(frame_path)
        width, height = frame.size
        pixels = frame.resize((width // pixel_size, height // pixel_size), resample=Image.NEAREST)
        pixels = pixels.resize((width, height), resample=Image.NEAREST)
        pixels.save(pixels_dir / f'{frame_path.stem}.png', format='PNG')
    return pixels_dir
