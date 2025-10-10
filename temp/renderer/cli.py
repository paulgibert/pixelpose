import argparse
import multiprocessing as mp
import os
import shutil
from typing import List
from pathlib import Path
from tqdm import tqdm
from .worker import worker_process
from .utils import render_job_iter
from .model import RenderJob, RenderJobResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir', help="Source FBX directory")
    parser.add_argument('target_dir', help="Target FBX directory")
    parser.add_argument('--output_dir', '-o', default='dataset', help="Output directory for renders (default: dataset/)")
    parser.add_argument('--resolution', type=int, default=128, help=f"Render resolution (default: 128)")
    parser.add_argument('--fps', type=int, default=8, help=f"Render FPS (default: 8)")
    parser.add_argument('--pixel_size', type=int, default=4, help=f"The pixel size to use for pixelization (default: 4)")
    parser.add_argument('--num_workers', '-n', type=int, default=1, help=f"Number of workers (default: 1)")
    parser.add_argument('--count_only', '-c', action='store_true', help=f"Disables rendering and counts the number of frames that would have been rendered")
    parser.add_argument('--purge_errors', '-p', action='store_true', help=f"Relocated files that cause errors")
    parser.add_argument('--purge_dir', default='purged', help=f"Purged files are stored here (default: purged/)")
    return parser.parse_args()


def main():
    # Parse argments
    args = parse_args()

    # Setup queues
    job_queue = mp.Queue()
    result_queue = mp.Queue()

    # Create and start workers
    workers = _spawn_workers(args, job_queue, result_queue)

    # Create jobs
    jobs = _create_jobs(args)
    for j in jobs:
        job_queue.put(j)
    
    # Collect results until all jobs are processed
    results = []
    with tqdm(total=len(jobs), desc="Processing jobs") as pbar:
        completed = 0
        while completed < len(jobs):
            r = result_queue.get()
            results.append(r)
            completed += 1
            pbar.update(1)
    
    _kill_workers(workers, job_queue)

    # Purge errors
    if args.purge_errors:
        count = _purge_errors(args, results)
        print(f"Purged {count} files")

    # Display results
    _display_results(results)


def _spawn_workers(args: argparse.Namespace, job_queue: mp.Queue, result_queue: mp.Queue):
    workers = []
    for i in range(args.num_workers):
        w = mp.Process(target=worker_process,
                       args=(job_queue, result_queue, args.count_only))
        w.start()
        workers.append(w)
        print(f"Started worker {i}")
    return workers


def _kill_workers(workers: List[mp.Process], job_queue: mp.Queue):
    # Stop workers by enqueing poison pills
    for _ in workers:
        job_queue.put(None)
    
    for worker in workers:
        worker.join()

def _create_jobs(args: argparse.Namespace) -> List[RenderJob]:
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    output_dir = Path(args.output_dir)

    return list(render_job_iter(source_dir, target_dir, output_dir,
                                args.resolution, args.fps,
                                args.pixel_size))

def _purge_errors(args: argparse.Namespace, results: List[RenderJobResult]) -> int:
    # Setup the purge directory
    os.makedirs(args.purge_dir, exist_ok=True)

    # Search results for errors
    count = 0
    purged = []
    for r in results:
        if r.error:
            bad_file = r.job.source_path

            if bad_file not in purged:                  # Check if file has been purged yet
                shutil.copy2(bad_file, args.purge_dir)  # Copy file to purge directory
                os.remove(bad_file)                     # Remove the file for its original location

                # Count the purge
                purged.append(bad_file)
                count += 1

    return count


def _display_results(results: List[RenderJobResult]):
    total_frames = sum([r.frames_rendered for r in results])
    print(f"Rendered {total_frames} frames")
    
    total_combos = sum([1 for r in results if not r.error])
    print(f"{total_combos} combos rendered")

    total_errors = sum([1 for r in results if r.error])
    print(f"{total_errors} errors")
