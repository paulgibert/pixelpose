"""Main entry point for the animation-viewer module."""

import argparse
import sys
from pathlib import Path

from .app import AnimationPlayerApp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GUI application for browsing and playing retargetted animations"
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing retargetted animations (e.g., retargetted/)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Default FPS for animation playback (default: 24)"
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=True,
        help="Enable looping by default (default: True)"
    )
    return parser.parse_args(argv if argv is not None else sys.argv[1:])


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    try:
        args = parse_args(argv)
        
        if not args.directory.exists():
            print(f"Error: Directory '{args.directory}' does not exist", file=sys.stderr)
            return 1
            
        if not args.directory.is_dir():
            print(f"Error: '{args.directory}' is not a directory", file=sys.stderr)
            return 1
            
        app = AnimationPlayerApp(args.directory, default_fps=args.fps, default_loop=args.loop)
        app.run()
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
