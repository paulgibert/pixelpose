import multiprocessing as mp
from .model import RenderJobResult
from .utils import call_count_frames_script, call_render_script
from .pose import render_stick_poses


def worker_process(job_queue: mp.Queue, result_queue: mp.Queue, count_only: bool):
    while True:
        job = job_queue.get() # Blocks until job is available

        # Exit on poison pill
        if job is None:
            break
        
        # Count the number of frames we will render
        frames_to_render = call_count_frames_script(job)

        error = False
        error_message = ""

        # Error if we found 0 frames to render
        if frames_to_render == 0:
            error = True
            error_message = "0 frames to render"

        # Skip render if we only want to count the frames
        elif not count_only:
            # Render mesh frames and export poses.json
            call_render_script(job)

            # Render pose frames from poses.json
            poses_json_path = job.output_dir / 'poses.json'
            poses_output_dir = job.output_dir / 'poses'
            render_stick_poses(poses_json_path, poses_output_dir, job.resolution, job.resolution)

        # Enqueue the result
        result = RenderJobResult(job, frames_to_render, error, error_message)
        result_queue.put(result)
