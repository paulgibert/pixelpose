"""Main GUI application for the animation player."""

import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox
from typing import Optional

from .animation_controller import AnimationController


class AnimationPlayerApp:
    """Main GUI application for browsing and playing animations."""
    
    def __init__(self, directory: Path, default_fps: float = 24.0, default_loop: bool = True):
        self.directory = directory
        self.default_fps = default_fps
        self.default_loop = default_loop
        
        # Animation data
        self.characters = []
        self.animations = []
        self.current_character = None
        self.current_animation = None
        self.animation_controller = None
        self.current_animation_index = 0  # Track current animation index
        
        # GUI components
        self.root = None
        self.character_var = None
        self.animation_var = None
        self.fps_var = None
        self.loop_var = None
        self.play_button = None
        self.frame_label = None
        self.status_label = None
        
        self.setup_gui()
        self.animation_controller = AnimationController(self.root)
        self.animation_controller.on_frame_change = self.on_frame_change
        self.animation_controller.on_status_change = self.on_status_change
        self.load_animations()
        
    def setup_gui(self):
        """Set up the GUI components."""
        self.root = tk.Tk()
        self.root.title("Animation Player")
        self.root.geometry("1200x700")
        
        # Configure grid weights for main window
        self.root.columnconfigure(0, weight=1)  # Left panel
        self.root.columnconfigure(1, weight=0)   # Separator (no weight)
        self.root.columnconfigure(2, weight=2)  # Right panel gets more space
        self.root.rowconfigure(0, weight=1)
        
        # Left panel - File browser
        left_panel = ttk.Frame(self.root, padding="10")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(5, weight=1)  # Allow space to grow
        
        # Separator
        separator = ttk.Separator(self.root, orient="vertical")
        separator.grid(row=0, column=1, sticky=(tk.N, tk.S), padx=5)
        
        # Right panel - Animation viewer
        right_panel = ttk.Frame(self.root, padding="10")
        right_panel.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(2, weight=1)  # Animation display area
        
        # Left panel - File browser components
        browser_label = ttk.Label(left_panel, text="File Browser", font=("Arial", 12, "bold"))
        browser_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Character selection
        ttk.Label(left_panel, text="Character:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.character_var = tk.StringVar()
        self.character_combo = ttk.Combobox(
            left_panel, 
            textvariable=self.character_var,
            state="readonly"
        )
        self.character_combo.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        self.character_combo.bind("<<ComboboxSelected>>", self.on_character_selected)
        
        # Animation selection
        ttk.Label(left_panel, text="Animation:").grid(row=3, column=0, sticky=tk.W, pady=(10, 2))
        self.animation_var = tk.StringVar()
        self.animation_combo = ttk.Combobox(
            left_panel,
            textvariable=self.animation_var,
            state="readonly"
        )
        self.animation_combo.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=2)
        self.animation_combo.bind("<<ComboboxSelected>>", self.on_animation_selected)
        
        # Help panel
        help_frame = ttk.LabelFrame(left_panel, text="Keyboard Shortcuts", padding="5")
        help_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(20, 0))
        help_frame.columnconfigure(0, weight=1)
        
        # Help text
        help_text = """Playback:
Space/P - Play/Pause
← → - Previous/Next Frame
Home/End - First/Last Frame

Navigation:
W/S - Previous/Next Character
A/D - Previous/Next Animation

FPS Control:
↑ ↓ - Increase/Decrease FPS

Other:
Q/Esc - Quit"""
        
        help_label = ttk.Label(help_frame, text=help_text, justify=tk.LEFT, font=("Courier", 9))
        help_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Right panel - Animation viewer components
        viewer_label = ttk.Label(right_panel, text="Animation Viewer", font=("Arial", 12, "bold"))
        viewer_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Control frame
        control_frame = ttk.Frame(right_panel)
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # FPS control
        ttk.Label(control_frame, text="FPS:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.fps_var = tk.DoubleVar(value=self.default_fps)
        fps_spinbox = ttk.Spinbox(
            control_frame,
            from_=1.0,
            to=60.0,
            increment=1.0,
            textvariable=self.fps_var,
            width=10
        )
        fps_spinbox.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        fps_spinbox.bind("<FocusOut>", self.on_fps_changed)
        fps_spinbox.bind("<Return>", self.on_fps_changed)
        
        # Loop checkbox
        self.loop_var = tk.BooleanVar(value=self.default_loop)
        loop_check = ttk.Checkbutton(
            control_frame,
            text="Loop",
            variable=self.loop_var,
            command=self.on_loop_changed
        )
        loop_check.grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        
        # Play/Pause button
        self.play_button = ttk.Button(
            control_frame,
            text="Play",
            command=self.toggle_play
        )
        self.play_button.grid(row=0, column=3, padx=(0, 10))
        
        # Frame navigation
        nav_frame = ttk.Frame(control_frame)
        nav_frame.grid(row=0, column=4, sticky=tk.W)
        
        ttk.Button(nav_frame, text="⏮", command=self.first_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="⏪", command=self.prev_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="⏩", command=self.next_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="⏭", command=self.last_frame).pack(side=tk.LEFT, padx=2)
        
        # Animation display area
        display_frame = ttk.LabelFrame(right_panel, text="Animation & Poses", padding="5")
        display_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        display_frame.columnconfigure(0, weight=1)  # Animation column
        display_frame.columnconfigure(1, weight=1)  # Poses column
        display_frame.rowconfigure(0, weight=1)
        
        # Animation frame display
        animation_frame = ttk.LabelFrame(display_frame, text="Animation", padding="5")
        animation_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        animation_frame.columnconfigure(0, weight=1)
        animation_frame.rowconfigure(0, weight=1)
        
        self.frame_label = ttk.Label(animation_frame, text="Select an animation to play")
        self.frame_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Poses frame display
        poses_frame = ttk.LabelFrame(display_frame, text="Poses", padding="5")
        poses_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        poses_frame.columnconfigure(0, weight=1)
        poses_frame.rowconfigure(0, weight=1)
        
        self.pose_label = ttk.Label(poses_frame, text="Poses will appear here")
        self.pose_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_label = ttk.Label(right_panel, text="Ready", relief=tk.SUNKEN)
        self.status_label.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Keyboard bindings
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("p", lambda e: self.toggle_play())
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("<Home>", lambda e: self.first_frame())
        self.root.bind("<End>", lambda e: self.last_frame())
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("q", lambda e: self.root.destroy())
        
        # New keyboard shortcuts
        self.root.bind("a", lambda e: self.prev_animation())
        self.root.bind("d", lambda e: self.next_animation())
        self.root.bind("w", lambda e: self.prev_character())
        self.root.bind("s", lambda e: self.next_character())
        self.root.bind("<Up>", lambda e: self.increase_fps())
        self.root.bind("<Down>", lambda e: self.decrease_fps())
        
    def load_animations(self):
        """Load available characters and animations from the directory."""
        try:
            # Find all character directories
            self.characters = []
            for item in self.directory.iterdir():
                if item.is_dir():
                    # Check if this character has animations
                    animations = []
                    for anim_dir in item.iterdir():
                        if anim_dir.is_dir():
                            # Check if this directory contains PNG files (either directly or in frames/ subdirectory)
                            png_files = list(anim_dir.glob("*.png"))
                            frames_dir = anim_dir / "frames"
                            if frames_dir.exists():
                                png_files.extend(list(frames_dir.glob("*.png")))
                            
                            if png_files:
                                animations.append(anim_dir.name)
                    
                    if animations:
                        self.characters.append({
                            'name': item.name,
                            'path': item,
                            'animations': sorted(animations)
                        })
            
            # Update character combo
            character_names = [char['name'] for char in self.characters]
            self.character_combo['values'] = character_names
            
            if character_names:
                self.character_combo.current(0)
                self.on_character_selected()
                # Auto-start the first animation
                if self.animation_controller and self.animations:
                    self.animation_controller.play()
                    self.play_button.config(text="Pause")
                
            # Debug info
            total_animations = sum(len(char['animations']) for char in self.characters)
            self.update_status(f"Loaded {len(self.characters)} characters with {total_animations} total animations")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load animations: {e}")
            self.update_status("Error loading animations")
    
    def on_character_selected(self, event=None):
        """Handle character selection."""
        if not self.characters:
            return
            
        selected_name = self.character_var.get()
        character = next((char for char in self.characters if char['name'] == selected_name), None)
        
        if not character:
            return
            
        # Remember current play state
        was_playing = False
        if self.animation_controller:
            was_playing = self.animation_controller.is_playing
            
        self.current_character = character
        self.animations = character['animations']
        
        # Update animation combo
        self.animation_combo['values'] = self.animations
        if self.animations:
            # Preserve animation index if possible, otherwise use 0
            target_index = min(self.current_animation_index, len(self.animations) - 1)
            self.animation_combo.current(target_index)
            # Store the play state to restore after animation loads
            self._pending_play_state = was_playing
            self.on_animation_selected()
        else:
            self.animation_combo.set("")
            self.animation_var.set("")
            
        self.update_status(f"Selected character: {character['name']}")
    
    def on_animation_selected(self, event=None):
        """Handle animation selection."""
        if not self.current_character or not self.animations:
            return
            
        selected_animation = self.animation_var.get()
        if not selected_animation:
            return
            
        # Remember current play state (use pending state if available, otherwise current state)
        was_playing = False
        if hasattr(self, '_pending_play_state'):
            was_playing = self._pending_play_state
            delattr(self, '_pending_play_state')
        elif self.animation_controller:
            was_playing = self.animation_controller.is_playing
            
        # Load new animation
        animation_path = self.current_character['path'] / selected_animation
        try:
            # Load animation into the controller
            if self.animation_controller.load_animation(animation_path):
                self.current_animation = selected_animation
                
                # Update current animation index
                if self.animations:
                    self.current_animation_index = self.animations.index(selected_animation)
                
                # Set FPS and loop settings
                self.animation_controller.set_fps(self.fps_var.get())
                self.animation_controller.set_loop(self.loop_var.get())
                
                # Restore play state if it was playing, or auto-start if this is the first animation
                if was_playing or (not hasattr(self, '_has_loaded_animation')):
                    self.animation_controller.play()
                    self.play_button.config(text="Pause")
                    self._has_loaded_animation = True
                
                self.update_status(f"Loaded animation: {selected_animation}")
            else:
                messagebox.showerror("Error", f"No frames found in {animation_path}")
                self.update_status("Error loading animation")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load animation: {e}")
            self.update_status("Error loading animation")
    
    def toggle_play(self):
        """Toggle play/pause."""
        if not self.animation_controller:
            return
            
        if self.animation_controller.is_playing:
            self.animation_controller.pause()
            self.play_button.config(text="Play")
        else:
            self.animation_controller.play()
            self.play_button.config(text="Pause")
    
    def first_frame(self):
        """Go to first frame."""
        if self.animation_controller:
            self.animation_controller.first_frame()
    
    def prev_frame(self):
        """Go to previous frame."""
        if self.animation_controller:
            self.animation_controller.prev_frame()
    
    def next_frame(self):
        """Go to next frame."""
        if self.animation_controller:
            self.animation_controller.next_frame()
    
    def last_frame(self):
        """Go to last frame."""
        if self.animation_controller:
            self.animation_controller.last_frame()
    
    def on_frame_change(self, frame_path: Path, frame_index: int, total_frames: int):
        """Handle frame change events."""
        try:
            # Load and display the animation frame
            photo = tk.PhotoImage(file=str(frame_path))
            self.frame_label.configure(image=photo)
            self.frame_label.image = photo  # Keep a reference
            
            # Try to load and display the corresponding pose frame
            self._load_pose_frame(frame_path, frame_index)
            
            # Update status
            self.update_status(f"Frame {frame_index + 1}/{total_frames}")
            
        except Exception as e:
            print(f"Error loading frame {frame_path}: {e}")
    
    def _load_pose_frame(self, frame_path: Path, frame_index: int):
        """Load and display the corresponding pose frame."""
        try:
            # Construct the pose frame path
            # frame_path is something like: /path/to/character/animation/frames/0001.png
            # We need: /path/to/character/animation/poses/poses/0001.png
            
            # Get the animation directory (parent of frames directory)
            animation_dir = frame_path.parent.parent
            pose_dir = animation_dir / "poses" / "poses"
            
            # Get the frame filename
            frame_filename = frame_path.name
            
            # Construct pose frame path
            pose_frame_path = pose_dir / frame_filename
            
            if pose_frame_path.exists():
                # Load and display the pose frame
                pose_photo = tk.PhotoImage(file=str(pose_frame_path))
                self.pose_label.configure(image=pose_photo)
                self.pose_label.image = pose_photo  # Keep a reference
            else:
                # Show a message if no corresponding pose exists
                self.pose_label.configure(image="", text="No pose available")
                self.pose_label.image = None
                
        except Exception as e:
            print(f"Error loading pose frame: {e}")
            # Show error message
            self.pose_label.configure(image="", text="Error loading pose")
            self.pose_label.image = None
    
    def on_status_change(self, status: str):
        """Handle status change events."""
        self.update_status(status)
        # Update play button text when animation finishes
        if status == "Animation finished":
            self.play_button.config(text="Play")
    
    def on_fps_changed(self, event=None):
        """Handle FPS change."""
        if self.animation_controller:
            self.animation_controller.set_fps(self.fps_var.get())
    
    def on_loop_changed(self):
        """Handle loop setting change."""
        if self.animation_controller:
            self.animation_controller.set_loop(self.loop_var.get())
            # If loop is disabled and animation is at the end, reset to beginning
            if not self.loop_var.get() and self.animation_controller.is_at_end():
                self.animation_controller.goto_frame(0)
                self.play_button.config(text="Play")
    
    def prev_animation(self):
        """Go to previous animation."""
        if not self.animations:
            return
            
        current_animation = self.animation_var.get()
        if not current_animation:
            return
            
        # Remember current play state
        was_playing = False
        if self.animation_controller:
            was_playing = self.animation_controller.is_playing
            
        try:
            current_index = self.animations.index(current_animation)
            prev_index = (current_index - 1) % len(self.animations)
            self.animation_var.set(self.animations[prev_index])
            # Update animation index
            self.current_animation_index = prev_index
            # Store play state to restore after animation loads
            self._pending_play_state = was_playing
            self.on_animation_selected()
        except ValueError:
            pass
    
    def next_animation(self):
        """Go to next animation."""
        if not self.animations:
            return
            
        current_animation = self.animation_var.get()
        if not current_animation:
            return
            
        # Remember current play state
        was_playing = False
        if self.animation_controller:
            was_playing = self.animation_controller.is_playing
            
        try:
            current_index = self.animations.index(current_animation)
            next_index = (current_index + 1) % len(self.animations)
            self.animation_var.set(self.animations[next_index])
            # Update animation index
            self.current_animation_index = next_index
            # Store play state to restore after animation loads
            self._pending_play_state = was_playing
            self.on_animation_selected()
        except ValueError:
            pass
    
    def prev_character(self):
        """Go to previous character."""
        if not self.characters:
            return
            
        current_character = self.character_var.get()
        if not current_character:
            return
            
        # Remember current play state
        was_playing = False
        if self.animation_controller:
            was_playing = self.animation_controller.is_playing
            
        try:
            current_index = self.characters.index(next(char for char in self.characters if char['name'] == current_character))
            prev_index = (current_index - 1) % len(self.characters)
            self.character_var.set(self.characters[prev_index]['name'])
            # Store play state to restore after character loads
            self._pending_play_state = was_playing
            self.on_character_selected()
        except (ValueError, StopIteration):
            pass
    
    def next_character(self):
        """Go to next character."""
        if not self.characters:
            return
            
        current_character = self.character_var.get()
        if not current_character:
            return
            
        # Remember current play state
        was_playing = False
        if self.animation_controller:
            was_playing = self.animation_controller.is_playing
            
        try:
            current_index = self.characters.index(next(char for char in self.characters if char['name'] == current_character))
            next_index = (current_index + 1) % len(self.characters)
            self.character_var.set(self.characters[next_index]['name'])
            # Store play state to restore after character loads
            self._pending_play_state = was_playing
            self.on_character_selected()
        except (ValueError, StopIteration):
            pass
    
    def increase_fps(self):
        """Increase FPS by 1."""
        current_fps = self.fps_var.get()
        new_fps = min(60.0, current_fps + 1.0)
        self.fps_var.set(new_fps)
        self.on_fps_changed()
    
    def decrease_fps(self):
        """Decrease FPS by 1."""
        current_fps = self.fps_var.get()
        new_fps = max(1.0, current_fps - 1.0)
        self.fps_var.set(new_fps)
        self.on_fps_changed()
    
    def update_status(self, message: str):
        """Update the status bar."""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()
