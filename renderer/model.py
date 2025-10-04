from pathlib import Path
from dataclasses import dataclass


@dataclass
class RenderJob:
    source_path: Path
    target_path: Path
    output_dir: Path
    resolution: int
    fps: int

@dataclass
class RenderJobResult:
    job: RenderJob
    frames_rendered: int
    error: bool
    error_message: str
