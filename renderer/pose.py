from pathlib import Path
from typing import Dict, Tuple

try:
    from PIL import Image, ImageDraw
except Exception as exc:  # pragma: no cover
    raise


# Joint suffixes we want to find (everything after the colon)
JOINT_SUFFIXES = {
    "Hips",
    "Spine",
    "Spine1", 
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
}

# Bone connections using suffixes (will be matched flexibly)
BONE_CONNECTION_SUFFIXES = [
    ("Spine", "Hips"),
    ("Spine1", "Spine"),
    ("Spine2", "Spine1"),
    ("Neck", "Spine2"),
    ("Head", "Neck"),
    ("LeftShoulder", "Spine2"),
    ("LeftArm", "LeftShoulder"),
    ("LeftForeArm", "LeftArm"),
    ("LeftHand", "LeftForeArm"),
    ("RightShoulder", "Spine2"),
    ("RightArm", "RightShoulder"),
    ("RightForeArm", "RightArm"),
    ("RightHand", "RightForeArm"),
    ("LeftUpLeg", "Hips"),
    ("LeftLeg", "LeftUpLeg"),
    ("LeftFoot", "LeftLeg"),
    ("LeftToeBase", "LeftFoot"),
    ("RightUpLeg", "Hips"),
    ("RightLeg", "RightUpLeg"),
    ("RightFoot", "RightLeg"),
    ("RightToeBase", "RightFoot"),
]

# Body part mapping for color-coding
BODY_PART_MAPPING = {
    # Left leg
    "LeftUpLeg": "left_leg",
    "LeftLeg": "left_leg", 
    "LeftFoot": "left_leg",
    "LeftToeBase": "left_leg",
    
    # Right leg
    "RightUpLeg": "right_leg",
    "RightLeg": "right_leg",
    "RightFoot": "right_leg", 
    "RightToeBase": "right_leg",
    
    # Left arm
    "LeftShoulder": "left_arm",
    "LeftArm": "left_arm",
    "LeftForeArm": "left_arm",
    "LeftHand": "left_arm",
    
    # Right arm
    "RightShoulder": "right_arm",
    "RightArm": "right_arm",
    "RightForeArm": "right_arm",
    "RightHand": "right_arm",
    
    # Torso
    "Hips": "torso",
    "Spine": "torso",
    "Spine1": "torso",
    "Spine2": "torso",
    
    # Head/neck
    "Neck": "head",
    "Head": "head",
}

# Distinct colors for each body part (bones)
BODY_PART_COLORS = {
    "left_leg": (255, 0, 0),      # Red
    "right_leg": (0, 255, 0),     # Green  
    "left_arm": (0, 0, 255),      # Blue
    "right_arm": (255, 255, 0),   # Yellow
    "torso": (255, 0, 255),       # Magenta
    "head": (0, 255, 255),        # Cyan
}

# Contrasting colors for joints (distinct colors that pop against bones)
JOINT_COLORS = {
    "left_leg": (255, 128, 128),  # Light Red (contrasts with red bones)
    "right_leg": (128, 255, 128), # Light Green (contrasts with green bones)
    "left_arm": (128, 128, 255),  # Light Blue (contrasts with blue bones)
    "right_arm": (255, 255, 128), # Light Yellow (contrasts with yellow bones)
    "torso": (255, 128, 255),     # Light Magenta (contrasts with magenta bones)
    "head": (128, 255, 255),      # Light Cyan (contrasts with cyan bones)
}

def joint_color(suffix: str) -> Tuple[int, int, int]:
    """Get color for a joint based on its body part"""
    body_part = BODY_PART_MAPPING.get(suffix, "torso")  # Default to torso
    return JOINT_COLORS.get(body_part, (255, 255, 255))  # Default to white

def bone_color(suffix: str) -> Tuple[int, int, int]:
    """Get color for a bone based on its body part"""
    body_part = BODY_PART_MAPPING.get(suffix, "torso")  # Default to torso
    return BODY_PART_COLORS.get(body_part, (255, 255, 255))  # Default to white


def project_points_yz(joint_to_loc: Dict[str, Tuple[float, float, float]], width: int, height: int, padding: int):
    if not joint_to_loc:
        return {}
    ys = [loc[1] for loc in joint_to_loc.values()]
    zs = [loc[2] for loc in joint_to_loc.values()]
    min_z, max_z = min(zs), max(zs)
    hips = joint_to_loc.get("mixamorig1:Hips")
    hips_y = hips[1] if hips else (min(ys) + max(ys)) * 0.5
    max_abs_y_off = max(abs(y - hips_y) for y in ys) if ys else 1.0
    span_y_sym = max(2.0 * max_abs_y_off, 1e-5)
    span_z = max(max_z - min_z, 1e-5)
    scale_x = (width - 2 * padding) / span_y_sym
    scale_y = (height - 2 * padding) / span_z
    scale = min(scale_x, scale_y)

    def to_img(loc):
        y, z = loc[1], loc[2]
        u = (hips_y - y) * scale + (width / 2.0)
        v_scene = (z - min_z) * scale + padding
        v = height - v_scene
        return int(round(u)), int(round(v))

    return {name: to_img(loc) for name, loc in joint_to_loc.items()}


def find_joint_by_suffix(pose: Dict, suffix: str) -> str:
    """Find joint name that ends with the given suffix."""
    for name in pose.keys():
        if name.endswith(f":{suffix}"):
            return name
    return None


def render_stick_image(img_w: int, img_h: int, pose: Dict, line_width: int = 3, joint_radius: int = 1.5, padding: int = 16) -> Image.Image:
    # Map suffixes to actual joint names in the pose
    suffix_to_name = {}
    for suffix in JOINT_SUFFIXES:
        actual_name = find_joint_by_suffix(pose, suffix)
        if actual_name:
            suffix_to_name[suffix] = actual_name

    # Extract locations for found joints
    joint_to_loc: Dict[str, Tuple[float, float, float]] = {}
    for suffix, actual_name in suffix_to_name.items():
        entry = pose.get(actual_name)
        if entry:
            loc = entry.get("location")
            if loc and len(loc) == 3:
                joint_to_loc[actual_name] = (float(loc[0]), float(loc[1]), float(loc[2]))

    points = project_points_yz(joint_to_loc, img_w, img_h, padding)
    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw bones using actual joint names with body part colors
    for child_suffix, parent_suffix in BONE_CONNECTION_SUFFIXES:
        child_name = suffix_to_name.get(child_suffix)
        parent_name = suffix_to_name.get(parent_suffix)
        if child_name and parent_name:
            p0 = points.get(child_name)
            p1 = points.get(parent_name)
            if p0 and p1:
                # Use the child joint's body part color for the bone
                bone_color_rgb = bone_color(child_suffix)
                draw.line([p0, p1], fill=bone_color_rgb, width=line_width)
    
    # Draw joints with region-based colors
    for actual_name, (u, v) in points.items():
        # Find the suffix for this joint
        suffix = None
        for s in JOINT_SUFFIXES:
            if actual_name.endswith(f":{s}"):
                suffix = s
                break
        if suffix:
            color = joint_color(suffix)
            r = joint_radius
            draw.ellipse([u - r, v - r, u + r, v + r], fill=color, outline=None)
    
    return img


def render_stick_poses(poses_json_path: str, output_dir: str, img_w: int, img_h: int) -> None:
    """Process poses.json and generate stick pose images for each frame"""
    import json
    import os
    from pathlib import Path
    
    # Load poses data
    with open(poses_json_path, 'r') as f:
        poses_data = json.load(f)
    
    # Create poses output directory
    poses_dir = Path(output_dir) / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each frame
    for frame_data in poses_data.get("frames", []):
        frame_num = frame_data.get("frame", 1)
        pose = frame_data.get("pose", {})
        
        # Generate stick image for this pose
        stick_img = render_stick_image(img_w, img_h, pose)
        
        # Save the image
        output_path = poses_dir / f"{frame_num:04d}.png"
        stick_img.save(output_path)


