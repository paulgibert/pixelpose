"""Frame player for handling animation playback."""

import time
import tkinter as tk
from pathlib import Path
from typing import Callable, Optional


class FramePlayer:
    """Handles animation frame playback with timing control."""
    
    def __init__(
        self,
        animation_path: Path,
        fps: float = 24.0,
        loop: bool = True,
        on_frame_change: Optional[Callable[[Path, int, int], None]] = None,
        on_status_change: Optional[Callable[[str], None]] = None,
        root: Optional[tk.Tk] = None
    ):
        self.animation_path = animation_path
        self.fps = fps
        self.loop = loop
        self.on_frame_change = on_frame_change
        self.on_status_change = on_status_change
        self.root = root
        
        # Animation state
        self.frame_paths = self._find_frame_files()
        self.total_frames = len(self.frame_paths)
        self.current_frame = 0
        self.is_playing = False
        self.delay_ms = max(1, int(1000 / fps)) if fps > 0 else 41
        
        # Timing
        self.last_frame_time = 0.0
        self._after_id = None
        self._is_destroyed = False  # Flag to prevent updates after cleanup
        
        if self.total_frames == 0:
            raise ValueError(f"No PNG frames found in {animation_path}")
            
        # Don't show first frame automatically - let the caller control when to display
    
    def _find_frame_files(self) -> list[Path]:
        """Find all PNG frame files in the animation directory."""
        if not self.animation_path.exists() or not self.animation_path.is_dir():
            return []
        
        # Look for frames in a 'frames' subdirectory first, then in the animation directory itself
        frames_dir = self.animation_path / "frames"
        if frames_dir.exists() and frames_dir.is_dir():
            search_path = frames_dir
        else:
            search_path = self.animation_path
        
        # Collect only .png files
        files = [p for p in search_path.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
        
        # Sort by numeric value in the stem if available, else lexicographically
        def sort_key(p: Path):
            stem = p.stem
            digits = "".join(ch for ch in stem if ch.isdigit())
            return (int(digits) if digits else float("inf"), stem)
        
        files.sort(key=sort_key)
        return files
    
    def _show_frame(self, frame_index: int):
        """Show the frame at the given index."""
        if 0 <= frame_index < self.total_frames:
            frame_path = self.frame_paths[frame_index]
            if self.on_frame_change:
                self.on_frame_change(frame_path, frame_index, self.total_frames)
    
    def _schedule_next_frame(self):
        """Schedule the next frame update."""
        if not self.is_playing or not self.root:
            return
            
        now = time.time()
        elapsed_ms = int((now - self.last_frame_time) * 1000)
        wait_ms = max(0, self.delay_ms - elapsed_ms) if self.last_frame_time else self.delay_ms
        
        # Cancel any existing timer
        if self._after_id:
            self.root.after_cancel(self._after_id)
        
        # Schedule the next frame
        self._after_id = self.root.after(wait_ms, self._advance_frame)
    
    def _advance_frame(self):
        """Advance to the next frame."""
        if not self.is_playing or self._is_destroyed:
            return
            
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            if self.loop:
                self.current_frame = 0
            else:
                self.is_playing = False
                # Reset to first frame when animation finishes
                self.current_frame = 0
                self._show_frame(0)
                if self.on_status_change:
                    self.on_status_change("Animation finished")
                return
        
        self._show_frame(self.current_frame)
        self.last_frame_time = time.time()
        self._schedule_next_frame()
    
    def play(self):
        """Start or resume playback."""
        if not self.is_playing:
            self.is_playing = True
            self.last_frame_time = time.time()
            self._schedule_next_frame()
            if self.on_status_change:
                self.on_status_change("Playing")
    
    def pause(self):
        """Pause playback."""
        self.is_playing = False
        if self._after_id and self.root:
            # Cancel any pending frame updates
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if self.on_status_change:
            self.on_status_change("Paused")
    
    def stop(self):
        """Stop playback and reset to first frame."""
        self.pause()
        self.current_frame = 0
        self._show_frame(0)
        if self.on_status_change:
            self.on_status_change("Stopped")
    
    def cleanup(self):
        """Aggressively clean up all timers and state."""
        self._is_destroyed = True  # Mark as destroyed first
        self.is_playing = False
        if self._after_id and self.root:
            try:
                self.root.after_cancel(self._after_id)
            except:
                pass  # Ignore errors if timer was already cancelled
            self._after_id = None
        self.current_frame = 0
        # Clear callbacks to prevent old animation from updating UI
        self.on_frame_change = None
        self.on_status_change = None
    
    def goto_frame(self, frame_index: int):
        """Go to a specific frame."""
        if 0 <= frame_index < self.total_frames:
            self.current_frame = frame_index
            self._show_frame(frame_index)
            if self.on_status_change:
                self.on_status_change(f"Frame {frame_index + 1}/{self.total_frames}")
    
    def prev_frame(self):
        """Go to previous frame."""
        if self.current_frame > 0:
            self.goto_frame(self.current_frame - 1)
        elif self.loop:
            self.goto_frame(self.total_frames - 1)
    
    def next_frame(self):
        """Go to next frame."""
        if self.current_frame < self.total_frames - 1:
            self.goto_frame(self.current_frame + 1)
        elif self.loop:
            self.goto_frame(0)
    
    def set_fps(self, fps: float):
        """Update the playback FPS."""
        self.fps = fps
        self.delay_ms = max(1, int(1000 / fps)) if fps > 0 else 41
    
    def set_loop(self, loop: bool):
        """Set whether to loop the animation."""
        self.loop = loop
    
    def is_at_end(self) -> bool:
        """Check if the animation is at the last frame."""
        return self.current_frame >= self.total_frames - 1
