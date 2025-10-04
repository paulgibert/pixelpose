# Animation Viewer

A GUI application for browsing and playing retargetted animations.

## Usage

```bash
python3 -m animation-viewer retargetted
```

## Features

- **Character Browser**: Select from available characters
- **Animation Browser**: Choose animations for the selected character
- **Playback Controls**: Play, pause, and navigate through frames
- **FPS Control**: Adjust playback speed (1-60 FPS)
- **Loop Control**: Enable/disable animation looping
- **Frame Navigation**: Jump to first, previous, next, or last frame
- **Keyboard Shortcuts**:
  - `Space` or `P`: Play/Pause
  - `←` `→`: Previous/Next frame
  - `Home`/`End`: First/Last frame
  - `Q` or `Esc`: Quit

## Directory Structure

The application expects a directory structure like:
```
retargetted/
├── character1/
│   ├── animation1/
│   │   ├── frame001.png
│   │   ├── frame002.png
│   │   └── ...
│   └── animation2/
│       ├── frame001.png
│       └── ...
└── character2/
    └── ...
```

## Requirements

- Python 3.8+
- tkinter (usually included with Python)
- PNG image files for animation frames
