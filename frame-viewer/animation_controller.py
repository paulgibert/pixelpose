"""Central animation controller to prevent conflicts between multiple animations."""

import time
import tkinter as tk
from pathlib import Path
from typing import Callable, Optional


class AnimationController:
    """Central controller for all animations to prevent conflicts."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.current_animation_path = None
        self.frame_paths = []
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False
        self.loop = True
        self.fps = 24.0
        self.delay_ms = 41
        
        # Callbacks
        self.on_frame_change: Optional[Callable[[Path, int, int], None]] = None
        self.on_status_change: Optional[Callable[[str], None]] = None
        
        # Timer management
        self._after_id = None
        self.last_frame_time = 0.0
        
    def load_animation(self, animation_path: Path) -> bool:
        """Load a new animation, stopping any current one."""
        # Stop current animation
        self.stop()
        
        # Find frame files
        frames_dir = animation_path / "frames"
        if frames_dir.exists() and frames_dir.is_dir():
            search_path = frames_dir
        else:
            search_path = animation_path
        
        # Collect PNG files
        files = [p for p in search_path.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
        
        if not files:
            return False
        
        # Sort by numeric value
        def sort_key(p: Path):
            stem = p.stem
            digits = "".join(ch for ch in stem if ch.isdigit())
            return (int(digits) if digits else float("inf"), stem)
        
        files.sort(key=sort_key)
        
        # Update state
        self.current_animation_path = animation_path
        self.frame_paths = files
        self.total_frames = len(files)
        self.current_frame = 0
        
        # Show first frame
        self._show_frame(0)
        return True
    
    def play(self):
        """Start or resume playback."""
        if not self.frame_paths:
            return
            
        self.is_playing = True
        self.last_frame_time = time.time()
        self._schedule_next_frame()
        
        if self.on_status_change:
            self.on_status_change("Playing")
    
    def pause(self):
        """Pause playback."""
        self.is_playing = False
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
            
        if self.on_status_change:
            self.on_status_change("Paused")
    
    def stop(self):
        """Stop playback and reset to first frame."""
        self.pause()
        self.current_frame = 0
        if self.frame_paths:
            self._show_frame(0)
            
        if self.on_status_change:
            self.on_status_change("Stopped")
    
    def goto_frame(self, frame_index: int):
        """Go to a specific frame."""
        if 0 <= frame_index < self.total_frames:
            self.current_frame = frame_index
            self._show_frame(frame_index)
    
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
    
    def first_frame(self):
        """Go to first frame."""
        self.goto_frame(0)
    
    def last_frame(self):
        """Go to last frame."""
        self.goto_frame(self.total_frames - 1)
    
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
    
    def _show_frame(self, frame_index: int):
        """Show the frame at the given index."""
        if 0 <= frame_index < self.total_frames:
            frame_path = self.frame_paths[frame_index]
            if self.on_frame_change:
                self.on_frame_change(frame_path, frame_index, self.total_frames)
    
    def _schedule_next_frame(self):
        """Schedule the next frame update."""
        if not self.is_playing or not self.frame_paths:
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
        if not self.is_playing or not self.frame_paths:
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
