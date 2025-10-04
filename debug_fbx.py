#!/usr/bin/env python3
"""
Debug script to analyze FBX files and print bone information.
Usage: blender --background --python debug_fbx.py -- <path_to_fbx_file>
"""

import sys
import bpy
import mathutils
from pathlib import Path


def clear_scene():
    """Clear the current scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def analyze_fbx(fbx_path):
    """Analyze an FBX file and print detailed information"""
    print(f"Analyzing FBX file: {fbx_path}")
    print("=" * 60)
    
    # Clear scene first
    clear_scene()
    
    # Import the FBX file
    try:
        bpy.ops.import_scene.fbx(filepath=str(fbx_path))
        print("✓ FBX file imported successfully")
    except Exception as e:
        print(f"✗ Failed to import FBX: {e}")
        return
    
    # Get all objects in the scene
    objects = bpy.context.scene.objects
    print(f"\nTotal objects in scene: {len(objects)}")
    
    # Find armatures
    armatures = [obj for obj in objects if obj.type == 'ARMATURE']
    print(f"Armatures found: {len(armatures)}")
    
    for i, armature in enumerate(armatures):
        print(f"\n--- ARMATURE {i+1}: {armature.name} ---")
        print(f"Location: {armature.location}")
        print(f"Rotation: {armature.rotation_euler}")
        print(f"Scale: {armature.scale}")
        
        # Bone information
        bones = armature.data.bones
        print(f"Total bones: {len(bones)}")
        
        # Print all bone names
        print("\nBone names:")
        for j, bone in enumerate(bones):
            print(f"  {j+1:3d}. {bone.name}")
            print(f"      Head: {bone.head}")
            print(f"      Tail: {bone.tail}")
            print(f"      Parent: {bone.parent.name if bone.parent else 'None'}")
            print(f"      Children: {len(bone.children)}")
        
        # Animation information
        if armature.animation_data and armature.animation_data.action:
            action = armature.animation_data.action
            print(f"\nAnimation: {action.name}")
            print(f"Frame range: {action.frame_range}")
            print(f"F-Curves: {len(action.fcurves)}")
            
            # Group f-curves by bone
            bone_curves = {}
            for fcurve in action.fcurves:
                bone_name = fcurve.data_path.split('"')[1] if '"' in fcurve.data_path else "Unknown"
                if bone_name not in bone_curves:
                    bone_curves[bone_name] = []
                bone_curves[bone_name].append(fcurve)
            
            print(f"\nAnimated bones: {len(bone_curves)}")
            for bone_name, curves in bone_curves.items():
                print(f"  {bone_name}: {len(curves)} curves")
                for curve in curves:
                    print(f"    - {curve.data_path} ({curve.array_index})")
        else:
            print("\nNo animation data found")
    
    # Find meshes
    meshes = [obj for obj in objects if obj.type == 'MESH']
    print(f"\nMeshes found: {len(meshes)}")
    
    for i, mesh in enumerate(meshes):
        print(f"\n--- MESH {i+1}: {mesh.name} ---")
        print(f"Location: {mesh.location}")
        print(f"Vertices: {len(mesh.data.vertices)}")
        print(f"Faces: {len(mesh.data.polygons)}")
        print(f"Materials: {len(mesh.data.materials)}")
        
        # Check if mesh has armature modifier
        for modifier in mesh.modifiers:
            if modifier.type == 'ARMATURE':
                print(f"Armature modifier: {modifier.object.name if modifier.object else 'None'}")
    
    # Find cameras
    cameras = [obj for obj in objects if obj.type == 'CAMERA']
    print(f"\nCameras found: {len(cameras)}")
    
    # Find lights
    lights = [obj for obj in objects if obj.type == 'LIGHT']
    print(f"Lights found: {len(lights)}")
    
    # Animation information for the scene
    if bpy.context.scene.animation_data and bpy.context.scene.animation_data.action:
        scene_action = bpy.context.scene.animation_data.action
        print(f"\nScene animation: {scene_action.name}")
        print(f"Scene frame range: {scene_action.frame_range}")
    else:
        print("\nNo scene animation data")
    
    print(f"\nScene frame range: {bpy.context.scene.frame_start} - {bpy.context.scene.frame_end}")
    print(f"Scene FPS: {bpy.context.scene.render.fps}")


def main():
    # Get the FBX file path from command line arguments
    # When run with blender --python, the script name is in sys.argv[0]
    # and the actual arguments come after '--'
    if '--' in sys.argv:
        fbx_path = Path(sys.argv[sys.argv.index('--') + 1])
    else:
        print("Usage: blender --background --python debug_fbx.py -- <path_to_fbx_file>")
        sys.exit(1)
    
    if not fbx_path.exists():
        print(f"Error: File {fbx_path} does not exist")
        sys.exit(1)
    
    if not fbx_path.suffix.lower() == '.fbx':
        print(f"Error: {fbx_path} is not an FBX file")
        sys.exit(1)
    
    try:
        analyze_fbx(fbx_path)
    except Exception as e:
        print(f"Error analyzing FBX: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
