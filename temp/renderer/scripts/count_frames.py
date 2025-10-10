import argparse
import sys
import bpy


def parse_args() -> argparse.Namespace:
    # Remove all leading blender args
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('source_path', help="Path to source FBX file")
    parser.add_argument('--fps', type=int, default=8, help="Intended render FPS (default: 8)")
    return parser.parse_args(argv)


def main():
    # Parse arguments
    args = parse_args()

    # Init the scene with the source FBX file
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=args.source_path)

    # Find the source Armature
    source_arm = bpy.data.objects.get('Armature')
    if source_arm is None:
        raise RuntimeError(f"Failed to find source Armature")
    
    # Ensure the Armature has animation data
    if not (source_arm.animation_data and source_arm.animation_data.action):
        raise RuntimeError('The source Armature does not have animation data')
    
    # Count the total frames in the animation
    action = source_arm.animation_data.action
    start_frame = int(action.frame_range[0])
    end_frame = int(action.frame_range[1])
    total_frames = end_frame - start_frame + 1

    # Calculate the total frames to be rendered
    scene_fps = bpy.context.scene.render.fps
    render_fps = args.fps
    total_frames_to_render = int(total_frames / scene_fps * render_fps)

    # Print the result to the console
    print(total_frames_to_render, file=sys.stderr)


if __name__ == "__main__":
    main()
