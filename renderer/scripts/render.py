import sys
import os
import json
import argparse
import mathutils
import bpy


def main():
    """Main entry point for the Blender rendering script"""
    args = parse_args(sys.argv)

    init_scene(args.resolution, args.resolution, args.fps, args.render_padding)
    source_arm, target_arm = import_assets(args.source, args.target)
    frames_dir = prepare_frames_directory(args.output_dir)

    poses = render_sequence(source_arm, target_arm, frames_dir, args.render_padding, args.fps)
    write_poses_json(args.output_dir, args.source, args.target, poses)


def parse_args(argv):
    """Parse command line arguments"""
    # Remove all leading blender args
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]

    # Parse remaining args
    p = argparse.ArgumentParser(description='Render animation frames and export poses (Blender helper)')
    p.add_argument('source', help='The source .fbx containing the animated rig')
    p.add_argument('target', help='The target .fbx file containing the character rig to apply the animation to')
    p.add_argument('output_dir', help='The directory to output rendered frames')
    p.add_argument('--resolution', type=int, default=128, help='Square output resolution (default: 128)')
    p.add_argument('--fps', type=int, default=12, help='Frame rate (default: 12)')
    p.add_argument('--render-padding', type=int, default=0, help='Padding in pixels for left/right/top of rendered frames (default: 0)')

    return p.parse_args(argv)


def init_scene(res_w: int, res_h: int, fps: int, render_padding: int):
    """Initialize the Blender scene with camera, lighting, and render settings"""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = True
    scene.render.resolution_x = int(res_w)
    scene.render.resolution_y = int(res_h)
    
    # Set FPS
    scene.render.fps = fps
    scene.render.fps_base = 1.0
    
    _create_camera(scene)
    _create_lighting(scene)


def import_assets(source_filepath: str, target_filepath: str):
    """Import source and target .fbx files and return their armatures"""
    # Import .fbx files
    bpy.ops.import_scene.fbx(filepath=source_filepath) # Armature id wil default to 'Armature'
    bpy.ops.import_scene.fbx(filepath=target_filepath) # Armature id Will default to 'Armature.001'

    # Load the source Armature
    source_arm = bpy.data.objects.get('Armature')
    if source_arm is None:
        raise RuntimeError('Failed to find source Armature')
    
    # Ensure the source Armature is animated
    if not (source_arm.animation_data and source_arm.animation_data.action):
        raise RuntimeError('The source Armature does not have animation data')

    # Load the target Armature
    target_arm = bpy.data.objects.get('Armature.001')
    if target_arm is None:
        raise RuntimeError('Failed to find target Armature')

    # Check if there are any meshes in the scene
    scene_meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    if len(scene_meshes) == 0:
        raise RuntimeError('No meshes found in the target file')

    return source_arm, target_arm


def prepare_frames_directory(output_dir: str):
    """Create and return the frames output directory"""
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    return frames_dir


def render_sequence(source_arm, target_arm, frames_dir: str, render_padding: int, fps: int):
    """Render animation sequence with consistent sprite sheet framing"""
    camera = bpy.context.scene.camera

    # Get the frame bounds of the target Armature
    action = source_arm.animation_data.action
    frame_start, frame_end = map(int, action.frame_range)
    
    # Calculate global bounding box and collect pose data
    global_bbox, frame_poses = _calculate_global_bbox_and_poses(source_arm, target_arm, frame_start, frame_end, fps)
    
    # Position camera once using global bounding box
    _position_camera_for_sprite_sheet(camera, global_bbox, render_padding=render_padding)
    
    # Render all frames using consistent retargeting
    poses = _render_frames_with_poses(frame_poses, target_arm, frames_dir, source_arm)
    
    return poses


def write_poses_json(output_root: str, source: str, target: str, poses):
    """Write pose data to JSON file"""
    scene = bpy.context.scene
    poses_path = os.path.join(output_root, 'poses.json')
    with open(poses_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source': os.path.basename(source),
            'target': os.path.basename(target),
            'resolution': {'width': scene.render.resolution_x, 'height': scene.render.resolution_y},
            'frames': poses,
        }, f, indent=4)


def _create_camera(scene):
    """Create an orthographic camera for the scene"""
    # Creat the camera
    cam_data = bpy.data.cameras.new('Camera')
    cam_data.type = 'ORTHO' # Orthagraphic camera makes the character look 2D
    
    # Fix clipping planes to prevent character parts from disappearing
    cam_data.clip_start = 0.1    # Very close near plane
    cam_data.clip_end = 1000.0   # Very far far plane
    
    cam_obj = bpy.data.objects.new('Camera', cam_data)

    # Assign the camera to the scene
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj


def _create_lighting(scene):
    """Set up lighting for the scene"""
    # Use Eevee for better material support while still being fast
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    
    # Optimize Eevee for maximum speed
    scene.eevee.taa_render_samples = 8  # Very low samples for speed
    
    # Optimize render settings
    scene.render.use_motion_blur = False
    scene.render.use_freestyle = False
    scene.render.use_sequencer = False
    scene.render.use_compositing = False
    
    # Create an new World
    scene.world = bpy.data.worlds.new('World')
    scene.world.use_nodes = True

    # Create a white background
    # Note: This is not the background of the scene, just the world used to light it
    bg = scene.world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (1, 1, 1, 1)  # white
    bg.inputs[1].default_value = 1.0           # brightness
    
    # Add simple lighting for materials (minimal setup)
    light_data = bpy.data.lights.new('Light', type='SUN')
    light_data.energy = 1.5  # Lower energy for speed
    light_obj = bpy.data.objects.new('Light', light_data)
    light_obj.location = (0, 0, 5)
    bpy.context.collection.objects.link(light_obj)


def _matrix_to_trs(m: mathutils.Matrix):
    """Decompose matrix into translation, rotation, and scale"""
    loc, rot, scale = m.decompose()
    return [loc.x, loc.y, loc.z], [rot.w, rot.x, rot.y, rot.z], [scale.x, scale.y, scale.z]


def _collect_pose_world(arm_obj):
    """Collect world space pose data for all bones in the armature"""
    pose = {}
    arm_world = arm_obj.matrix_world.copy()
    for pbone in arm_obj.pose.bones:
        bone_world = arm_world @ pbone.matrix
        loc, rotq, scl = _matrix_to_trs(bone_world)
        pose[pbone.name] = {'location': loc, 'rotation_quaternion': rotq, 'scale': scl}
    return pose


def _get_armature_bone_prefix(arm_obj):
    """Get the bone prefix from the first bone name"""
    return arm_obj.data.bones[0].name.split(':')[0]


def _get_bone_suffix(bone):
    """Get the bone suffix from the bone name"""
    return bone.name.split(':')[1]


def _retarget_armature(frame, source_arm, target_arm):
    """Retarget animation from source to target armature for a specific frame"""
    source_prefix = _get_armature_bone_prefix(source_arm)

    for target_bone in target_arm.pose.bones:
        # Get the coresponding source bone for this target bone
        source_bone_name = ':'.join([source_prefix, _get_bone_suffix(target_bone)])

        try:
            source_bone = source_arm.pose.bones[source_bone_name]
        except KeyError:
            continue # Skip bones not found in the source Armature

        # Set the target bone transforms to match the source bone
        target_bone.location            = source_bone.location
        target_bone.rotation_quaternion = source_bone.rotation_quaternion
        target_bone.scale               = source_bone.scale

        # Inserts keyframes into the target
        target_bone.keyframe_insert(data_path='location', frame=frame)
        target_bone.keyframe_insert(data_path='rotation_quaternion', frame=frame)
        target_bone.keyframe_insert(data_path='scale', frame=frame)


def _calculate_bbox_for_frame(target_arm):
    """Calculate bounding box for the current frame using bone positions"""
    # Get all bone positions in world space
    bone_positions = []
    for bone in target_arm.pose.bones:
        world_pos = target_arm.matrix_world @ bone.matrix.translation
        bone_positions.append(world_pos)
    
    if not bone_positions:
        # Fallback to armature bound_box if no bones
        bbox = [target_arm.matrix_world @ mathutils.Vector(corner) for corner in target_arm.bound_box]
        min_x = min(v.x for v in bbox)
        max_x = max(v.x for v in bbox)
        min_y = min(v.y for v in bbox)
        max_y = max(v.y for v in bbox)
        min_z = min(v.z for v in bbox)
        max_z = max(v.z for v in bbox)
    else:
        # Use bone positions
        min_x = min(v.x for v in bone_positions)
        max_x = max(v.x for v in bone_positions)
        min_y = min(v.y for v in bone_positions)
        max_y = max(v.y for v in bone_positions)
        min_z = min(v.z for v in bone_positions)
        max_z = max(v.z for v in bone_positions)
    
    return min_x, max_x, min_y, max_y, min_z, max_z


def _calculate_global_bbox_and_poses(source_arm, target_arm, frame_start, frame_end, fps):
    """Calculate global bounding box and collect pose data for frames at specified FPS"""
    scene = bpy.context.scene
    frame_poses = []
    global_min_x = float('inf')
    global_max_x = float('-inf')
    global_min_y = float('inf')
    global_max_y = float('-inf')
    global_min_z = float('inf')
    global_max_z = float('-inf')
    
    # Calculate frame step based on FPS
    # Get the original animation FPS from the action (typically 30 FPS for Mixamo)
    original_fps = 30  # Mixamo animations are typically 30 FPS
    frame_step = max(1, round(original_fps / fps))  # Step size to achieve desired FPS
    
    print(f"Original FPS: {original_fps}, Target FPS: {fps}, Frame step: {frame_step}")
    print(f"Frame range: {frame_start} to {frame_end}, will render every {frame_step} frames")
    
    rendered_frames = []
    for frame in range(frame_start, frame_end + 1, frame_step):
        rendered_frames.append(frame)
        scene.frame_set(frame)
        _retarget_armature(frame, source_arm, target_arm)
        
        # Calculate bounding box for this frame
        min_x, max_x, min_y, max_y, min_z, max_z = _calculate_bbox_for_frame(target_arm)
        
        # Update global bounds
        global_min_x = min(global_min_x, min_x)
        global_max_x = max(global_max_x, max_x)
        global_min_y = min(global_min_y, min_y)
        global_max_y = max(global_max_y, max_y)
        global_min_z = min(global_min_z, min_z)
        global_max_z = max(global_max_z, max_z)
        
        # Store the pose data for later use
        pose_data = _collect_pose_world(target_arm)
        frame_poses.append({'frame': int(frame), 'pose': pose_data})
    
    # Calculate final global bounding box
    width = global_max_x - global_min_x
    height = global_max_z - global_min_z
    depth = global_max_y - global_min_y  # Y movement range (depth in camera view)
    center = mathutils.Vector(((global_min_x + global_max_x) / 2, 
                              (global_min_y + global_max_y) / 2, 
                              (global_min_z + global_max_z) / 2))
    global_bbox = (width, height, center, global_min_z, global_max_z, depth)
    
    print(f"Actually rendered {len(rendered_frames)} frames: {rendered_frames[:10]}...{rendered_frames[-10:] if len(rendered_frames) > 10 else ''}")
    
    return global_bbox, frame_poses


def _render_frames_with_poses(frame_poses, target_arm, frames_dir, source_arm):
    """Render all frames using consistent retargeting"""
    scene = bpy.context.scene
    poses = []

    for frame_data in frame_poses:
        frame = frame_data['frame']
        scene.frame_set(frame)
        
        # Use the same retargeting logic as bbox calculation
        _retarget_armature(frame, source_arm, target_arm)
        
        # Update camera to follow character (dynamic camera following)
        _update_camera_for_frame(target_arm)
        
        # Render
        scene.render.filepath = os.path.join(frames_dir, f'{frame:04d}.png')
        bpy.ops.render.render(write_still=True)
        
        poses.append(frame_data)
    
    return poses


def _update_camera_for_frame(target_arm):
    """Update camera position to follow the character for this frame"""
    # Get the camera object
    cam_obj = bpy.context.scene.camera
    
    # Calculate current frame bounding box
    min_x, max_x, min_y, max_y, min_z, max_z = _calculate_bbox_for_frame(target_arm)
    
    # Use a more stable reference point - the character's root position
    # This reduces jitter by using a single point rather than bounding box center
    root_bone = target_arm.pose.bones[0] if target_arm.pose.bones else None
    if root_bone:
        # Use the root bone's world position as the stable reference
        root_world_pos = target_arm.matrix_world @ root_bone.matrix.translation
        center = root_world_pos
    else:
        # Fallback to bounding box center if no root bone
        center = mathutils.Vector((
            (min_x + max_x) / 2,
            (min_y + max_y) / 2,
            (min_z + max_z) / 2
        ))
    
    # Update camera position to follow the character
    # Keep the same distance but update Y position to follow character
    current_distance = abs(cam_obj.location.x)  # Keep current distance
    cam_obj.location = (-current_distance, center.y, center.z)
    
    # Update camera rotation to look at the character
    direction = center - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def _position_camera_for_sprite_sheet(cam_obj, global_bbox, padding=1.1, distance=10, render_padding=0):
    """Position camera once for consistent sprite sheet rendering"""
    width, height, center, min_z, max_z, depth = global_bbox
    
    # Calculate camera scale based on character size (not movement range)
    # Use only width and height to keep character properly sized
    character_size = max(width, height)
    base_scale = character_size * padding
    
    # Add render padding in world units (convert pixels to world units)
    padding_world_units = render_padding * 0.01
    cam_obj.data.ortho_scale = base_scale + padding_world_units
    
    # Position camera to look at the center of the global bounding box
    required_distance = character_size * padding * 0.5  # Half the scale to get distance
    cam_obj.location = (-required_distance, center.y, center.z)
    direction = center - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    # Adjust Z position to ensure the bottom of the character is visible
    half_scale = cam_obj.data.ortho_scale / 2
    cam_obj.location.z = min_z + half_scale


if __name__ == "__main__":
    main()